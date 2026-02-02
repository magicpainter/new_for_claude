#!/usr/bin/env python3
import os
import argparse
from glob import glob

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model.semseg.dpt import DPT


def strip_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def get_color_map():
    # BGR colors: class 0=black, 1=blue, 2=red
    return {
        0: (0,   0,   0),
        1: (0, 0,   255),   # red in BGR
        2: (0,   255, 0),   # green in BGR
        3: (255, 0,   0),
        4: (0, 255, 255),
    }


def put_title(img, title, font_scale=1.0, thickness=2):
    """
    Draws a title centered at the top of the image.
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
    x = (w - text_w) // 2
    y = int(text_h * 1.5)
    cv2.putText(img, title, (x, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)


def put_filename(img, filename, font_scale=0.8, thickness=2):
    """
    Draws the filename centered at the bottom of the image.
    """
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(filename, font, font_scale, thickness)
    x = (w - text_w) // 2
    y = h - int(text_h * 0.5)
    cv2.putText(img, filename, (x, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description='Inference with UniMatch‑V2 DPT model with overlays and optional GT')
    parser.add_argument('--checkpoint', type=str, default='/home/ubuntu2/yhe/Projects/UniMatch-V2/exp/field/unimatch_v2_minibatch/dinov2_base/138_3052_saved/latest.pth',
                        help='Path to latest.pth')
    parser.add_argument('--input-dir', type=str, default='/home/ubuntu2/yhe/Projects/data/field/train/unlabeled_rgbImages/',help='Folder of input images')

    parser.add_argument('--output-mask-dir', type=str, default='/home/ubuntu2/yhe/Projects/Outputs/Unimatch_v2/prediction/mask/',
                        help='Where to save raw PNG masks')
    parser.add_argument('--output-video', type=str, default='/home/ubuntu2/yhe/Projects/Outputs/Unimatch_v2/prediction/overlay.mp4',
                        help='Path for output overlay video (mp4)')
    parser.add_argument('--backbone', type=str, default='/home/ubuntu2/yhe/Projects/pretrained/dinov2_base.pth',
                        help='Which pretrained backbone (small, base, large, giant)')
    parser.add_argument('--alpha',           type=float, default=0.5,
                        help='Overlay transparency (0.0–1.0)')
    parser.add_argument('--multiplier',      type=int, default=14,
                        help='Resize multiplier used at eval time (matches training)')

    parser.add_argument('--show-gt',  default=False,       action='store_true',
                        help='Enable side-by-side predicted vs. ground-truth overlay')
    parser.add_argument('--gt-dir',          type=str, default='None',
                        help='Directory of ground-truth single-channel masks (.png)')
    args = parser.parse_args()

    if args.show_gt and not args.gt_dir:
        parser.error('--show-gt requires --gt-dir')

    os.makedirs(args.output_mask_dir, exist_ok=True)
    # determine video output path
    if os.path.isdir(args.output_video):
        os.makedirs(args.output_video, exist_ok=True)
        video_path = os.path.join(args.output_video, 'overlay.mp4')
    else:
        os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
        video_path = args.output_video

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model
    nclass = 5
    size_key = args.backbone.split('_')[-1]
    model_configs = {
        'small.pth': {'encoder_size':'small', 'features':64,  'out_channels':[48,96,192,384]},
        'base.pth':  {'encoder_size':'base',  'features':128, 'out_channels':[96,192,384,768]},
        'large.pth': {'encoder_size':'large', 'features':256, 'out_channels':[256,512,1024,1024]},
        'giant.pth': {'encoder_size':'giant', 'features':384, 'out_channels':[1536,1536,1536,1536]},
    }
    cfg = model_configs[size_key]
    model = DPT(**{**cfg, 'nclass': nclass})
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(strip_module_prefix(ckpt['model']))
    model.to(device).eval()

    # collect images
    img_paths = sorted(glob(os.path.join(args.input_dir, '*')))
    if not img_paths:
        raise RuntimeError(f'No images found in {args.input_dir}')

    # sample for dims
    sample = cv2.imread(img_paths[0])
    if sample is None:
        raise RuntimeError(f'Cannot read sample image: {img_paths[0]}')
    H, W = sample.shape[:2]

    # video writer dims: columns = 3 if show_gt else 2
    cols = 3 if args.show_gt else 2
    frame_w = W * cols
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 2.0, (frame_w, H))

    cmap = get_color_map()
    mult = args.multiplier

    for img_path in img_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Skipping unreadable file: {img_path}")
            continue
        H, W = img_bgr.shape[:2]

        # prepare tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).to(device)
        # resize
        new_h = int(H / mult + 0.5) * mult
        new_w = int(W / mult + 0.5) * mult
        t_rs = F.interpolate(tensor, (new_h, new_w), mode='bilinear', align_corners=True)
        # inference + resize back
        with torch.no_grad():
            logits = model(t_rs)
            logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)
            # ignore class 1: set its score to a very large negative so it's never chosen
            #logits[:, 1, :, :] = -1e9
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # save mask
        fn = os.path.basename(img_path)
        out_mask = os.path.join(args.output_mask_dir, fn.rsplit('.',1)[0] + '.png')
        Image.fromarray(mask, mode='L').save(out_mask)

        # predicted overlay only where mask>0
        color_mask = np.zeros_like(img_bgr, dtype=np.uint8)
        for cls_id, color in cmap.items():
            color_mask[mask == cls_id] = color
        full_overlay = cv2.addWeighted(img_bgr, 1-args.alpha, color_mask, args.alpha, 0)
        overlay_pred = img_bgr.copy()
        overlay_pred[mask != 0] = full_overlay[mask != 0]

        # assemble frames
        tiles = []
        # predicted
        tile_pred = overlay_pred.copy()
        put_title(tile_pred, 'predicted')
        tiles.append(tile_pred)

        if args.show_gt:
            # load GT mask
            base = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(args.gt_dir, base + '.png')
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_mask is None:
                raise RuntimeError(f"Cannot read GT mask: {gt_path}")
            # gt overlay
            gt_color = np.zeros_like(img_bgr, dtype=np.uint8)
            for cls_id, color in cmap.items():
                gt_color[gt_mask == cls_id] = color
            gt_full = cv2.addWeighted(img_bgr, 1-args.alpha, gt_color, args.alpha, 0)
            overlay_gt = img_bgr.copy()
            overlay_gt[gt_mask != 0] = gt_full[gt_mask != 0]
            put_title(overlay_gt, 'ground truth')
            tiles.append(overlay_gt)

        # raw image
        raw = img_bgr.copy()
        put_title(raw, 'raw image')
        tiles.append(raw)

        frame = np.hstack(tiles)
        # put filename at bottom center across full frame
        put_filename(frame, fn)

        writer.write(frame)

    writer.release()
    print(f"Done. Masks → {args.output_mask_dir}, Video → {video_path}")


if __name__ == '__main__':
    main()
