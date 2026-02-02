import os
import sys

from manual_pts_sel import select_critical_zone_corners
import cv2
import numpy as np
from jetson_utils import cudaToNumpy
import jetson_utils as jutils
from manual_bev import MANUAL_BEV
import subprocess


def is_valid_image(img, min_std=10.0):
    """Check if image is valid (not corrupted/blank)."""
    if img is None or img.size == 0:
        return False
    # Check if image has reasonable variance (not all black/white/uniform)
    if img.std() < min_std:
        return False
    # Check if image is not all zeros
    if np.all(img == 0):
        return False
    # Check if image is not all same value
    if np.all(img == img.flat[0]):
        return False
    return True


def capture_valid_frame(video_source, warmup_frames=10, max_retries=5, min_std=10.0):
    """Capture a valid frame after discarding warm-up frames."""
    # Discard warm-up frames to let camera stabilize
    for _ in range(warmup_frames):
        video_source.Capture()

    # Try to capture a valid frame
    for attempt in range(max_retries):
        cuda_frame = video_source.Capture()
        frame_np = cudaToNumpy(cuda_frame).astype(np.uint8)
        frame_np = frame_np[..., ::-1]  # Convert to BGR

        if is_valid_image(frame_np, min_std):
            print(f"Valid frame captured on attempt {attempt + 1}")
            return frame_np
        else:
            print(f"Attempt {attempt + 1}: Invalid frame (std={frame_np.std():.2f}), retrying...")

    # Return last frame even if validation failed, with warning
    print("Warning: Could not capture a validated frame, using last capture")
    return frame_np


# Accept --image <path> to skip opening the camera (used when called from the GUI)
_image_arg = None
if '--image' in sys.argv:
    _idx = sys.argv.index('--image')
    if _idx + 1 < len(sys.argv):
        _image_arg = sys.argv[_idx + 1]

f_use_cam = _image_arg is None
DEFAULT_CORNERS = np.array([[142, 459], [1003, 341], [1179, 396], [874, 459], [746, 180]])
if os.path.exists("corners_left.npy"):
    DEFAULT_CORNERS = np.load("corners_left.npy")

if f_use_cam:
    subprocess.run(
        ["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl=horizontal_flip=1"],
        capture_output=True, text=True, check=True  # check=True ¡ú ·Ç 0 ·µ»ØÂë»áÅ×Òì³£
    )
    cap_left = jutils.videoSource("/dev/video0", options={'width': 1920, 'height': 1200, 'framerate': 60})
    left_np = capture_valid_frame(cap_left, warmup_frames=10, max_retries=5)
else:
    left_np = cv2.imread(_image_arg if _image_arg else '2025-11-07_09-28-1000000009.png')

f_cals = False
side = 'left'
# Pass image path for refresh capability (only when using --image from GUI)
manual_bev_left = MANUAL_BEV(left_np, DEFAULT_CORNERS, f_cals, side, image_path=_image_arg)
# calculate homography and
manual_bev_left.calc_H()


