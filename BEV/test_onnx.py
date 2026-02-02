#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, time
import numpy as np
import cv2
import onnxruntime as ort
from project_mask_to_bev import project_mask_to_bev
#from find_triple_points_0_2_3 import find_triple_points_0_2_3
from view_mask import MaskBEVViewer

PALETTE = np.array([
    [255,   0,   0],  # class 0 - red - background
    [  0, 255,   0],  # class 1 - green - close
    [  0,   0, 255],  # class 2 - blue - mid
    [255, 255,   0],  # class 3 - yellow - far
    [255,   0, 255],  # class 4 - magenta - board
], dtype=np.uint8)

def overlay_mask(img_rgb_uint8, mask_hw_int, alpha=0.5):
    """把整数 mask (H,W) 映射成彩色并叠加到原图"""
    h, w = mask_hw_int.shape
    # 防御：负值或过大值裁掉
    mask_safe = np.clip(mask_hw_int, 0, len(PALETTE)-1)
    color_mask = PALETTE[mask_safe]                 # (H,W,3) RGB uint8
    # 半透明融合
    out = (img_rgb_uint8.astype(np.float32) * (1 - alpha) +
           color_mask.astype(np.float32) * alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def load_image_rgb(path, target_hw=(1200, 1920), norm="none", in_dtype=np.float32):
    """读取 BGR->RGB，必要时 resize 到 (H=1200,W=1920)；返回 (1,3,H,W)"""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    Ht, Wt = target_hw
    h, w = img_rgb.shape[:2]
    if (h, w) != (Ht, Wt):
        print(f"[warn] input image size {w}x{h}, resizing to {Wt}x{Ht}")
        img_rgb = cv2.resize(img_rgb, (Wt, Ht), interpolation=cv2.INTER_LINEAR)

    img = img_rgb.astype(np.float32) / 255.0
    if norm.lower() == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

    # (H,W,3) -> (1,3,H,W)
    x = np.transpose(img, (2, 0, 1))[None, ...]
    x = x.astype(in_dtype, copy=False)
    return img_rgb, x  # 返回原始 RGB(用于overlay) 和 NCHW 输入

def run_onnx(onnx_path, image_path, output_path, alpha=0.5, norm="none", try_cuda=False):
    # 选择 EP
    providers = ["CPUExecutionProvider"]
    if try_cuda:
        # 如果是 GPU 版 ORT 且 CUDA 依赖满足，会用到 CUDA，否则会 fallback 到 CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]

    print(f"[info] onnx  = {onnx_path}")
    print(f"[info] image = {image_path}")
    print(f"[info] EP    = {sess.get_providers()}")
    print(f"[info] input name={inp.name}, type={inp.type}, shape={inp.shape}")
    print(f"[info] output name={out.name}, type={out.type}, shape={out.shape}")

    # 目标分辨率（你的模型固定 1200x1920）
    H, W = 1200, 1920

    # 按模型输入 dtype 准备数据
    if inp.type == "tensor(float16)":
        in_dtype = np.float16
    elif inp.type == "tensor(float)":
        in_dtype = np.float32
    else:
        raise RuntimeError(f"Unexpected input dtype: {inp.type}")

    img_rgb_uint8, x = load_image_rgb(image_path, target_hw=(H, W), norm=norm, in_dtype=in_dtype)

    # 推理
    t0 = time.time()
    y = sess.run([out.name], {inp.name: x})[0]
    dt = (time.time() - t0) * 1000
    print(f"[time] ORT inference: {dt:.2f} ms")

    # 判定是 logits 还是已经 argmax 的 mask
    y_np = np.asarray(y)
    mask = None

    if y_np.dtype == np.int32 or y_np.dtype == np.int64:
        # 直接是 mask
        if y_np.ndim == 3 and y_np.shape[0] == 1:
            mask = y_np[0]
        elif y_np.ndim == 2:
            mask = y_np
        else:
            raise RuntimeError(f"Unexpected mask shape: {y_np.shape}")
        print(f"[debug] mask from ONNX, shape={mask.shape}, dtype={mask.dtype}")
    else:
        # 视为 logits
        if y_np.ndim == 4:
            # 可能是 (1,C,H,W) 或 (1,H,W,C)
            if y_np.shape[1] <= 16 and y_np.shape[1] != H:  # 通常 C<<H
                # NCHW
                mask = y_np.argmax(axis=1)[0]
            elif y_np.shape[-1] <= 16 and y_np.shape[-1] != W:
                # NHWC
                mask = y_np.argmax(axis=-1)[0]
            else:
                raise RuntimeError(f"Ambiguous logits shape: {y_np.shape}")
        elif y_np.ndim == 3:
            # (C,H,W) 或 (H,W,C)
            if y_np.shape[0] <= 16:
                mask = y_np.argmax(axis=0)
            elif y_np.shape[-1] <= 16:
                mask = y_np.argmax(axis=-1)
            else:
                raise RuntimeError(f"Ambiguous logits shape: {y_np.shape}")
        else:
            raise RuntimeError(f"Unexpected output shape for logits: {y_np.shape}")

        mask = mask.astype(np.int32, copy=False)
        mask_rgb = PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]
        print(f"[debug] mask from argmax, shape={mask.shape}, dtype={mask.dtype}")

    # 快速检查：是否出现 2147483647 之类的异常值
    uniq, cnt = np.unique(mask, return_counts=True)
    top = sorted(zip(cnt.tolist(), uniq.tolist()), reverse=True)[:10]
    print("[debug] top classes:", {k: v for v, k in top})

    # find BEV mask (Optional)
    bev_mask, src_pts_xy, H_final, dst_pts_bev_px, f_passed_4_corners_check = project_mask_to_bev(mask)

    np.save("H_final.npy", H_final)
    np.save("bev_mask.npy", bev_mask)
    np.save("dst_pts_bev_px.npy", dst_pts_bev_px)
    np.save("mask.npy", mask)

    #ear_pt1, ear_pt2 = find_triple_points_0_2_3(dst_pts_bev_px[1], dst_pts_bev_px[3])

    # view projection of point (Optional)
    viewer = MaskBEVViewer(mask, bev_mask, H_final,dst_pts_bev_px)
    viewer.run()

    # 生成 overlay 并保存
    overlay_rgb = overlay_mask(img_rgb_uint8, mask, alpha=float(alpha))
    # OpenCV 保存用 BGR
    cv2.imwrite(output_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    print(f"[done] saved overlay to: {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default='unimatch_field_1200x1920_logits_op17_839_33902.onnx', help="Path to ONNX model")

    ap.add_argument("--image", default='2025-09-22_18-00-4100002700.jpg', help="Path to an RGB PNG (expected 1920x1200)")
    #reflection on the ground
    # ap.add_argument("--image",
    #                 default='1102_outdoor_0531_00394.png',
    #                 help="Path to an RGB PNG (expected 1920x1200)")


    ap.add_argument("--output", default="overlay.png", help="Output overlay image path")
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha 0~1")
    ap.add_argument("--norm", choices=["none", "imagenet"], default="none",
                    help="Match your training/export preprocessing. Your export script used clamp[0,1], so choose 'none'.")
    ap.add_argument("--use_cuda", action="store_true", help="Try CUDA EP if available")
    args = ap.parse_args()

    run_onnx(args.onnx, args.image, args.output, args.alpha, args.norm, args.use_cuda)
