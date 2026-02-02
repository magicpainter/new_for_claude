#!/usr/bin/env python3
"""
Offline OD simulation for shooting feature.

- Uses your TensorRT YOLO11 engine (EfficientNMS version).
- Runs inference on BOTH full image and shooting crop.
- Merges detections (crop + full) like your online pipeline:
  * Crop boxes are shifted back to full-frame coordinates.
  * Full-frame boxes fully inside the crop region are suppressed.
- Processes a sequence of images from a folder.
- Prepares a `cur` dict per frame for your feature algorithm.

You need:
  - A TensorRT engine compatible with YOLO11 + EfficientNMS
  - corners_left.npy, corners_right.npy in the same folder,
    so the crop boxes match your runtime pipeline.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

import tensorrt as trt
import pycuda.autoinit  # creates CUDA context
import pycuda.driver as cuda

from features.shooting_feature_simu import ShootingFeature

FP32, I32 = 4, 4  # bytes per float32 / int32
SHOOT_MODES = ['mid-range baseline', '3-points baseline', 'mid-range star',
               '3-points star', 'close-range single-hand', 'free throw',
               '3-points challenge', 'layup']

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# ----------------------------------------------------------------------
#  Crop config (same as your online pipeline)
# ----------------------------------------------------------------------
corners_left = np.load("corners_left.npy")
corners_right = np.load("corners_right.npy")
left_rim_x, left_rim_y = int(corners_left[4][0]), int(corners_left[4][1])    # left side shifts to the right
right_rim_x, right_rim_y = int(corners_right[4][0]), int(corners_right[4][1])

# 640x640 shooting crop around LEFT/RIGHT_CX
crop_shoot_left = [left_rim_x - 320, 0, left_rim_x + 320, 640]
crop_shoot_right = [right_rim_x - 320, 0, right_rim_x + 320, 640]


# ----------------------------------------------------------------------
#  Letterbox helpers (numpy / OpenCV, no torch)
# ----------------------------------------------------------------------
def letterbox_np(img_bgr: np.ndarray, new_hw=(640, 640), pad_val=114):
    """
    Pure-CPU letterbox, similar to your torch version.

    img_bgr: H x W x 3, uint8 (BGR from cv2.imread)
    new_hw: (dh, dw)

    Returns:
      chw: (3, dh, dw) float32 in [0,1]
      meta: dict with scale/pad info to reverse later
    """
    H, W = img_bgr.shape[:2]
    dh, dw = new_hw
    r = min(dh / H, dw / W)
    nh, nw = int(round(H * r)), int(round(W * r))

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((dh, dw, 3), pad_val, dtype=np.uint8)
    top = (dh - nh) // 2
    left = (dw - nw) // 2
    canvas[top:top + nh, left:left + nw, :] = resized

    chw = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0

    meta = dict(r=r, pad=(left, top), in_hw=(H, W), out_hw=(dh, dw))
    return chw, meta


def scale_back_xyxy_np(xyxy: np.ndarray, meta: dict) -> np.ndarray:
    """
    Reverse letterbox.
    xyxy: (N,4) in letterbox space (dh,dw)
    Returns xyxy in original image space (H,W).
    """
    left, top = meta["pad"]
    r = meta["r"]
    H, W = meta["in_hw"]

    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / r

    out[:, 0] = np.clip(out[:, 0], 0, W - 1)
    out[:, 2] = np.clip(out[:, 2], 0, W - 1)
    out[:, 1] = np.clip(out[:, 1], 0, H - 1)
    out[:, 3] = np.clip(out[:, 3], 0, H - 1)
    return out


# ----------------------------------------------------------------------
#  TensorRT helpers (borrowed pattern from infer_images_w_eninge.py)
# ----------------------------------------------------------------------
def load_engine(engine_path: str):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine, context, batch_size: int, input_hw=(640, 640)):
    """
    Generic binding-based buffer allocator, similar to infer_images_w_eninge.py.

    Returns:
      inputs:  list of dicts {host, device, shape}
      outputs: list of dicts {host, device, shape}
      bindings: list of device pointer ints
      stream: cuda.Stream
    """
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()

    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        is_input = engine.binding_is_input(binding)

        if is_input:
            # Handle static vs dynamic
            shape0 = engine.get_binding_shape(binding_idx)
            if -1 in shape0:
                # dynamic: set shape explicitly
                _, c, h, w = shape0
                if h <= 0 or w <= 0:
                    h, w = input_hw
                input_shape = (batch_size, c, h, w)
                context.set_binding_shape(binding_idx, input_shape)
                shape = context.get_binding_shape(binding_idx)
            else:
                # static explicit batch
                shape = shape0
        else:
            # For outputs, use engine's binding shape (static) or context's (dynamic)
            shape0 = engine.get_binding_shape(binding_idx)
            if -1 in shape0:
                shape = context.get_binding_shape(binding_idx)
            else:
                shape = shape0

        size = trt.volume(shape)
        # Use the TRT10-style API like your working script
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        io_dict = {
            "host": host_mem,
            "device": device_mem,
            "shape": tuple(shape),
            "index": binding_idx,
            "name": binding,
        }

        if is_input:
            inputs.append(io_dict)
        else:
            outputs.append(io_dict)

    return inputs, outputs, bindings, stream


class YoloTrtEngine:
    """
    Thin wrapper around TensorRT execution, using the same pattern
    as infer_images_w_eninge.py (pycuda + bindings + async execute).
    Assumes a single input and a single detection output (B, N, 6).
    """

    def __init__(self, engine_path: str, batch_size: int = 2, input_hw=(640, 640)):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bs = batch_size
        self.input_hw = input_hw

        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.engine, self.context, self.bs, self.input_hw
        )

        if len(self.inputs) != 1:
            raise RuntimeError(f"Expected 1 input binding, got {len(self.inputs)}")
        if len(self.outputs) < 1:
            raise RuntimeError("No output bindings found")

        self.in_shape = self.inputs[0]["shape"]  # (B, 3, H, W)
        self.in_dtype = self.inputs[0]["host"].dtype
        self.bs = self.in_shape[0]
        self.dst_h = self.in_shape[2]
        self.dst_w = self.in_shape[3]

        # Assume first output is (B, det_max, det_dim)
        self.out_shape = self.outputs[0]["shape"]
        if len(self.out_shape) != 3:
            raise RuntimeError(f"Unexpected output shape: {self.out_shape}")
        self.det_max = self.out_shape[1]
        self.det_dim = self.out_shape[2]

    def infer(self, batch_chw: np.ndarray) -> np.ndarray:
        """
        batch_chw: numpy array of shape (B, 3, H, W), float32 in [0,1]
        Returns:
          detections: (B, det_max, det_dim) float32
        """
        assert batch_chw.shape == self.in_shape, (
            f"batch shape {batch_chw.shape} != expected {self.in_shape}"
        )

        # Make sure dtype matches engine input
        batch = batch_chw.astype(self.in_dtype, copy=False)

        # Copy host -> device
        np.copyto(self.inputs[0]["host"], batch.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        # Copy detections back from first output
        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"], self.outputs[0]["device"], self.stream
        )
        self.stream.synchronize()

        det = self.outputs[0]["host"].reshape(self.out_shape)  # (B, det_max, 6)
        return det.astype(np.float32, copy=False)


# ----------------------------------------------------------------------
#  Main simulation loop (one side: left or right)
# ----------------------------------------------------------------------
def run_sequence(
    engine_path: str,
    image_dir: str,
    side: str = "left",
    conf_thres: float = 0.10,
):
    """
    engine_path: path to TensorRT engine.
    image_dir: folder containing sequenced images (png/jpg).
    side: "left" or "right" (uses different shooting crop).
    conf_thres: confidence filter after merge.

    For each frame, builds:

        cur = {
            "frame_idx": int,
            "image_path": str,
            "side": "left" or "right",
            "bboxes": np.ndarray [N,4] (x1,y1,x2,y2 in full-frame coords),
            "scores": np.ndarray [N],
            "labels": np.ndarray [N] (int class indices),
            "original_size": (W, H),
            "category_map": {0: 'ball', 1: 'bib', 2: 'bob', 3: 'player'},
            "infer_time": float,
            "cuda_img": BGR image (for ShootingFeature drawing),
            "img_name": str,
        }
    """

    image_dir = Path(image_dir)
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    # Collect images (sorted by name)
    img_paths = sorted(
        [p for p in image_dir.glob("*.png")] + [p for p in image_dir.glob("*.jpg")]
    )
    if not img_paths:
        print(f"No images found in {image_dir}")
        return

    # Init engine using the same pattern as infer_images_w_eninge.py
    engine = YoloTrtEngine(engine_path, batch_size=2, input_hw=(640, 640))

    # Shooting crop for this side
    if side == "left":
        crop_region = crop_shoot_left
    else:
        crop_region = crop_shoot_right
    cx1, cy1, cx2, cy2 = crop_region

    category_map = {0: "ball", 1: "bib", 2: "player"}
    feature_shooting = ShootingFeature(1920, 1200, 0.067, side, 'free throw')

    for frame_idx, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read {img_path}, skipping.")
            continue

        H, W = img.shape[:2]

        metas = []  # one per batch slot
        # Prepare batch tensor in CHW layout
        batch = np.empty(engine.in_shape, dtype=engine.in_dtype)

        # ---------------- Prepare batch: [0]=full, [1]=crop ----------------
        for bidx, view in enumerate(("full", "crop")):
            if bidx >= engine.bs:
                break

            if view == "full":
                src_img = img  # BGR full frame
                crop_ofs = (0, 0)
            else:
                # shooting crop in original coords
                x1, y1, x2, y2 = cx1, cy1, cx2, cy2
                src_img = img[y1:y2, x1:x2, :]
                crop_ofs = (x1, y1)

            # BGR HWC -> CHW float [0,1] with letterbox
            chw, lbox_meta = letterbox_np(
                src_img,
                new_hw=(engine.dst_h, engine.dst_w)
            )
            batch[bidx] = chw.astype(engine.in_dtype, copy=False)

            metas.append(
                {
                    "view": view,  # "full" or "crop"
                    "crop_ofs": crop_ofs,
                    "letterbox": lbox_meta,
                }
            )

        # ---------------- Run TensorRT inference ----------------
        t0 = time.time()
        det_bs = engine.infer(batch)  # (B, det_max, 6)
        infer_time = time.time() - t0

        # ---------------- Merge detections (full + crop) ----------------
        acc_boxes = []
        acc_scores = []
        acc_labels = []

        for bidx, meta in enumerate(metas):
            det = det_bs[bidx]  # (det_max, 6)
            xyxy = det[:, :4]
            scores = det[:, 4]
            cls = det[:, 5].astype(np.int32)

            # Map from letterbox back to image / crop space
            xyxy_img = scale_back_xyxy_np(xyxy, meta["letterbox"])

            if meta["view"] == "full":
                # Suppress boxes fully inside the crop region
                x1, y1, x2, y2 = cx1, cy1, cx2, cy2
                inside = (
                    (xyxy_img[:, 0] > x1)
                    & (xyxy_img[:, 1] > y1)
                    & (xyxy_img[:, 2] < x2)
                    & (xyxy_img[:, 3] < y2)
                )
                scores = scores.copy()
                scores[inside] = 0.0
            else:  # "crop"
                # Shift crop coords into full-frame coords
                ox, oy = meta["crop_ofs"]
                xyxy_img[:, [0, 2]] += ox
                xyxy_img[:, [1, 3]] += oy

            # Apply confidence threshold AFTER merge behavior
            keep = scores >= conf_thres
            xyxy_keep = xyxy_img[keep]
            scores_keep = scores[keep]
            labels_keep = cls[keep]

            if xyxy_keep.shape[0] == 0:
                continue

            acc_boxes.append(xyxy_keep)
            acc_scores.append(scores_keep)
            acc_labels.append(labels_keep)

        if not acc_boxes:
            # No detections
            cur = {
                "frame_idx": frame_idx,
                "side": side,
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int32),
                "original_size": (W, H),
                "category_map": category_map,
                "infer_time": infer_time,
                "cuda_img": img,            # BGR frame for drawing
                "img_name": str(img_path),
            }
        else:
            Bx = np.concatenate(acc_boxes, axis=0)
            Sc = np.concatenate(acc_scores, axis=0)
            Lb = np.concatenate(acc_labels, axis=0)

            cur = {
                "frame_idx": frame_idx,
                "side": side,
                "bboxes": Bx,  # full-frame xyxy
                "scores": Sc,
                "labels": Lb,
                "original_size": (W, H),
                "category_map": category_map,
                "infer_time": infer_time,
                "cuda_img": img,
                "img_name": str(img_path),
            }

        feature_shooting.on_frame(cur)


# ----------------------------------------------------------------------
#  CLI entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Offline OD simulation for shooting feature (full+crop merge)."
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Path to TensorRT engine (YOLO11 + EfficientNMS).",
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Folder containing sequenced images.",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Camera side to simulate (affects shooting crop).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Confidence threshold after merge.",
    )

    args = parser.parse_args()

    run_sequence(
        engine_path=args.engine,
        image_dir=args.images,
        side=args.side,
        conf_thres=args.conf,
    )
