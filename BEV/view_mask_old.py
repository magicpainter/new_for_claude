#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Original � BEV � RBEV (Vertical Stack)

- Top:    original mask (single-channel class IDs, visualized with palette)
- Middle: BEV mask (single-channel class IDs, visualized)
- Bottom: RBEV_1400.png loaded from current working directory

Interaction:
- Left-click on the TOP (original) image -> project to BEV using H_src2bev and draw a red
  downward triangle on BEV; then map the BEV point to RBEV using H_RBEV and draw the same marker.
- 'r' : reset markers
- 's' : save a screenshot of the stacked canvas
- 'q' or 'Esc': quit

Notes:
- H_RBEV (BEV�RBEV) is estimated from template points on BEV (2/3 boundaries, triple points, etc.)
  and a fixed set of template pixels on RBEV_1400.png.
- If the required points cannot be found, H_RBEV is skipped gracefully and RBEV marker won't be drawn.
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Project helpers
from find_23_boundary_points import find_23_boundary_points
from find_23_boundary_vertical import find_23_boundary_vertical
from find_triple_points_0_2_3 import find_triple_points_0_2_3

PALETTE = np.array([
    [255, 0, 0],  # class 0 - red
    [0, 255, 0],  # class 1 - green
    [0, 0, 255],  # class 2 - blue
    [255, 255, 0],  # class 3 - yellow
    [255, 0, 255],  # class 4 - magenta
], dtype=np.uint8)

# RBEV template pixels (corresponding to 9 points on BEV)
RBEV_TEMPLATE_PTS = (
    (80, 0), (475, 0), (925, 0), (1320, 0),
    (205, 580), (475, 580), (925, 580), (1195, 580),
    (700, 840)
)


def colorize_mask(mask_hw_uint8: np.ndarray) -> np.ndarray:
    """Map class IDs to RGB then convert to BGR for OpenCV display."""
    if mask_hw_uint8.ndim != 2:
        raise ValueError(f"mask must be single-channel, got shape={mask_hw_uint8.shape}")
    rgb = PALETTE[np.clip(mask_hw_uint8, 0, len(PALETTE) - 1)]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def draw_down_triangle(img_bgr: np.ndarray, center_xy: tuple, size: int = 8,
                       color=(0, 0, 255), thickness: int = -1):
    """Draw a downward triangle centered at center_xy."""
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    pts = np.array([
        [x - size, y - size],
        [x + size, y - size],
        [x, y + size]
    ], dtype=np.int32).reshape(-1, 1, 2)
    if thickness < 0:
        cv2.fillConvexPoly(img_bgr, pts, color)
    else:
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color, thickness=thickness)


def draw_cross(img_bgr: np.ndarray, center_xy: tuple, half: int = 6, color=(0, 0, 255), thickness=2):
    """Draw a cross centered at center_xy."""
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.line(img_bgr, (x - half, y), (x + half, y), color, thickness, cv2.LINE_AA)
    cv2.line(img_bgr, (x, y - half), (x, y + half), color, thickness, cv2.LINE_AA)


def draw_square(img_bgr: np.ndarray, center_xy: tuple, half: int = 4,
                color=(255, 0, 255), thickness: int = -1):
    """Draw a filled/outlined square centered at center_xy (default filled, purple)."""
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    tl = (x - half, y - half)
    br = (x + half, y + half)
    cv2.rectangle(img_bgr, tl, br, color, thickness)


class MaskBEVViewer:
    def __init__(self,
                 original_mask: np.ndarray,
                 bev_mask: np.ndarray,
                 H_src2bev: np.ndarray,
                 dst_pts_bev_px,
                 window_name: str = "Original�BEV�RBEV (Vertical)",
                 gap: int = 16,
                 max_width: int = 1400,
                 max_height: int = 2400,
                 marker_size: int = 8,
                 rbev_path: str = "RBEV_1400.png"):
        """
        Args:
            original_mask: (H,W) uint8, class IDs
            bev_mask:      (H,W) uint8, class IDs
            H_src2bev:     3x3 homography from original to BEV
            dst_pts_bev_px: BEV corner points (pixels) from project_mask_to_bev (tuple of 4 points)
            window_name:   OpenCV window title
            gap:           vertical gap (pixels) between views
            max_width:     max canvas width
            max_height:    max canvas height
            marker_size:   size of the red downward triangle
            rbev_path:     path to RBEV image (loaded as BGR)
        """
        assert original_mask.ndim == 2 and bev_mask.ndim == 2, "expect single-channel masks"
        assert H_src2bev.shape == (3, 3), "H must be 3x3"

        self.orig_mask = original_mask
        self.bev_mask = bev_mask
        self.H = H_src2bev.astype(np.float64)

        # Base layers
        self.orig_bgr0 = colorize_mask(self.orig_mask)
        self.bev_bgr0 = colorize_mask(self.bev_mask)

        # RBEV image (BGR)
        self.rbev_bgr0 = cv2.imread(rbev_path, cv2.IMREAD_COLOR)
        if self.rbev_bgr0 is None:
            # Fallback blank RBEV if file not found
            self.rbev_bgr0 = np.full((1400, 1400, 3), 32, np.uint8)
            print(f"[warn] cannot read '{rbev_path}', using blank 1400x1400 RBEV.")

        self.window = window_name or "Original�BEV�RBEV (Vertical)"
        self.gap = int(gap)
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        self.marker_size = int(marker_size)

        # Interaction state
        self.marker_uv = None  # Marker in BEV (from top image click)
        self.marker_rbev = None  # Marker in RBEV (projected from BEV via H_RBEV)
        self.top_xy = None  # Click location on top image (original)
        self.scale = 1.0  # Global uniform display scale
        self.roi_top = (0, 0, 0, 0)  # (x, y, w, h) of top image on canvas
        self.roi_middle = (0, 0, 0, 0)  # middle (BEV)
        self.roi_bottom = (0, 0, 0, 0)  # bottom (RBEV)
        self.canvas_bgr = None

        # Keep BEV corner points (pixels)
        self.dst_pts_bev_px = dst_pts_bev_px

        # Precompute boundary-related points for H_RBEV
        self._A_default = dst_pts_bev_px[0]
        self._B_default = (dst_pts_bev_px[1][0], dst_pts_bev_px[1][1] + 200)
        self._boundary_pts = None
        self._compute_default_boundary_pts()

        # Vertical 2/3 boundary points: A, B, D
        try:
            self._boundary_vertical_pts = find_23_boundary_vertical(
                self.bev_mask, dst_pts_bev_px[0], dst_pts_bev_px[2]
            )
        except Exception as e:
            print(f"[warn] find_23_boundary_vertical failed: {e}")
            self._boundary_vertical_pts = {}

        # Triple points (0/2/3)
        try:
            self.ear_pt1, self.ear_pt2 = find_triple_points_0_2_3(
                self.bev_mask, dst_pts_bev_px[1], dst_pts_bev_px[3]
            )
        except Exception as e:
            print(f"[warn] find_triple_points_0_2_3 failed: {e}")
            self.ear_pt1, self.ear_pt2 = None, None

        # Estimate H_RBEV (BEV � RBEV)
        self.H_RBEV = None
        self._compute_H_RBEV()

        # Build and show initial canvas
        self._build_canvas()

        # Use ASCII-only window title to avoid Unicode title issues on some builds.
        safe_title = (self.window or "Original-BEV-RBEV-Vertical")
        safe_title = safe_title.encode("ascii", "ignore").decode()
        self.window = safe_title

        # Create window first, then immediately bind the mouse callback, then imshow.
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL is more tolerant than AUTOSIZE
        cv2.setMouseCallback(self.window, self._on_mouse)
        cv2.imshow(self.window, self.canvas_bgr)
        cv2.waitKey(1)

    def _compute_default_boundary_pts(self):
        """Compute four 2/3 boundary points from default A/B; on failure sets None."""
        H, W = self.bev_mask.shape
        Ax, Ay = self._A_default
        Bx, By = self._B_default

        def _in_bounds(x, y):
            return (0 <= int(x) < W and 0 <= int(y) < H)

        if not _in_bounds(Ax, Ay) or not _in_bounds(Bx, By):
            print(f"[warn] default A/B out of bounds for BEV size {W}x{H}: "
                  f"A={self._A_default}, B={self._B_default}")
            self._boundary_pts = None
            return
        try:
            self._boundary_pts = find_23_boundary_points(
                self.bev_mask, A_xy=self._A_default, B_xy=self._B_default, step=100
            )
            print(f"[ok] 2/3 boundary points: {self._boundary_pts}")
        except Exception as e:
            print(f"[warn] find_23_boundary_points failed: {e}")
            self._boundary_pts = None

    def _compute_H_RBEV(self):
        """
        Estimate H_RBEV using 9 BEV template points -> RBEV_TEMPLATE_PTS.
        If any required points are missing, skip gracefully.
        """
        if self._boundary_pts is None or self._boundary_vertical_pts is None:
            print("[warn] H_RBEV skipped due to missing template points.")
            self.H_RBEV = None
            return
        if self.ear_pt1 is None or self.ear_pt2 is None:
            print("[warn] H_RBEV skipped due to missing triple (ear) points.")
            self.H_RBEV = None
            return
        try:
            bev_template_pts = (
                self.ear_pt1,
                self.dst_pts_bev_px[1],  # 2nd corner
                self.dst_pts_bev_px[3],  # 4th corner
                self.ear_pt2,
                self._boundary_pts['A_left'],
                self.dst_pts_bev_px[0],  # 1st corner
                self.dst_pts_bev_px[2],  # 3rd corner
                self._boundary_pts['A_right'],
                self._boundary_vertical_pts['D'],
            )
            src = np.asarray(bev_template_pts, dtype=np.float32).reshape(-1, 1, 2)
            dst = np.asarray(RBEV_TEMPLATE_PTS, dtype=np.float32).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
            if H is None:
                print("[warn] cv2.findHomography returned None for H_RBEV.")
            else:
                self.H_RBEV = H.astype(np.float64)
                print("[ok] H_RBEV estimated.")
        except Exception as e:
            print(f"[warn] compute H_RBEV failed: {e}")
            self.H_RBEV = None

    def _project_bev_to_rbev(self, uv):
        """Project BEV pixel (u,v) to RBEV pixel (xr,yr) using H_RBEV; return None on failure."""
        if self.H_RBEV is None or uv is None:
            return None
        u, v = float(uv[0]), float(uv[1])
        q = self.H_RBEV @ np.array([u, v, 1.0], np.float64)
        if abs(q[2]) < 1e-9:
            return None
        xr = q[0] / q[2]
        yr = q[1] / q[2]
        return (int(round(xr)), int(round(yr)))

    def _build_canvas(self):
        """Compose a single canvas with three vertically stacked images."""
        purple = (255, 0, 255)  # BGR
        square_half = 15

        # Middle: BEV with overlays
        bev_draw = self.bev_bgr0.copy()

        # 2/3 boundary points (A_left, A_right, B_left, B_right)
        if self._boundary_pts is not None:
            for key in ("A_left", "A_right", "B_left", "B_right"):
                if key in self._boundary_pts and self._boundary_pts[key] is not None:
                    xk, yk = self._boundary_pts[key]
                    if 0 <= xk < bev_draw.shape[1] and 0 <= yk < bev_draw.shape[0]:
                        draw_square(bev_draw, (xk, yk), half=square_half, color=purple, thickness=-1)

        # Vertical 2/3 points: A, B, D
        if self._boundary_vertical_pts:
            for key in ("A", "B", "D"):
                if key in self._boundary_vertical_pts and self._boundary_vertical_pts[key] is not None:
                    xk, yk = self._boundary_vertical_pts[key]
                    if 0 <= xk < bev_draw.shape[1] and 0 <= yk < bev_draw.shape[0]:
                        draw_square(bev_draw, (xk, yk), half=square_half, color=purple, thickness=-1)

        # Triple points (ears)
        if self.ear_pt1 is not None:
            draw_square(bev_draw, self.ear_pt1, half=square_half, color=purple, thickness=-1)
        if self.ear_pt2 is not None:
            draw_square(bev_draw, self.ear_pt2, half=square_half, color=purple, thickness=-1)

        # Red marker on BEV
        if self.marker_uv is not None:
            u, v = self.marker_uv
            if 0 <= u < bev_draw.shape[1] and 0 <= v < bev_draw.shape[0]:
                draw_down_triangle(bev_draw, (u, v), size=self.marker_size, color=(0, 0, 255), thickness=-1)

        # Bottom: RBEV with red marker (if any)
        rbev_draw = self.rbev_bgr0.copy()
        if self.marker_rbev is not None:
            xr, yr = self.marker_rbev
            if 0 <= xr < rbev_draw.shape[1] and 0 <= yr < rbev_draw.shape[0]:
                draw_down_triangle(rbev_draw, (xr, yr), size=self.marker_size, color=(0, 0, 255), thickness=-1)

        # Top: original with cross at click
        orig_draw = self.orig_bgr0.copy()
        if self.top_xy is not None:
            draw_cross(orig_draw, self.top_xy, half=6, color=(0, 0, 255), thickness=5)

        # --- Global uniform scaling and vertical stacking ---
        hT, wT = orig_draw.shape[:2]
        hM, wM = bev_draw.shape[:2]
        hB, wB = rbev_draw.shape[:2]

        total_w = max(wT, wM, wB)
        total_h = hT + self.gap + hM + self.gap + hB

        s1 = min(1.0, self.max_width / max(1, total_w))
        s2 = min(1.0, self.max_height / max(1, total_h))
        self.scale = min(s1, s2)

        disp_top = cv2.resize(orig_draw, (int(round(wT * self.scale)), int(round(hT * self.scale))),
                              interpolation=cv2.INTER_NEAREST)
        disp_mid = cv2.resize(bev_draw, (int(round(wM * self.scale)), int(round(hM * self.scale))),
                              interpolation=cv2.INTER_NEAREST)
        disp_bot = cv2.resize(rbev_draw, (int(round(wB * self.scale)), int(round(hB * self.scale))),
                              interpolation=cv2.INTER_NEAREST)

        Wc = max(disp_top.shape[1], disp_mid.shape[1], disp_bot.shape[1])
        Hc = disp_top.shape[0] + self.gap + disp_mid.shape[0] + self.gap + disp_bot.shape[0]
        canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)

        # top (original)
        xT, yT = 0, 0
        canvas[yT:yT + disp_top.shape[0], xT:xT + disp_top.shape[1]] = disp_top
        self.roi_top = (xT, yT, disp_top.shape[1], disp_top.shape[0])

        # middle (BEV)
        xM, yM = 0, disp_top.shape[0] + self.gap
        canvas[yM:yM + disp_mid.shape[0], xM:xM + disp_mid.shape[1]] = disp_mid
        self.roi_middle = (xM, yM, disp_mid.shape[1], disp_mid.shape[0])

        # bottom (RBEV)
        xB, yB = 0, yM + disp_mid.shape[0] + self.gap
        canvas[yB:yB + disp_bot.shape[0], xB:xB + disp_bot.shape[1]] = disp_bot
        self.roi_bottom = (xB, yB, disp_bot.shape[1], disp_bot.shape[0])

        self.canvas_bgr = canvas

    def _on_mouse(self, event, x, y, flags, userdata=None):
        """Handle clicks: only respond to clicks inside the TOP (original) image region."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        xT, yT, wT, hT = self.roi_top
        inside_top = (xT <= x < xT + wT) and (yT <= y < yT + hT)
        if not inside_top:
            return

        # Convert canvas coordinates to top-image pixel coordinates
        x_local = (x - xT) / max(1e-6, self.scale)
        y_local = (y - yT) / max(1e-6, self.scale)
        self.top_xy = (x_local, y_local)

        # Homography: original (x,y,1) -> BEV (u,v,1)
        p = np.array([x_local, y_local, 1.0], dtype=np.float64)
        q = self.H @ p
        if abs(q[2]) < 1e-9:
            print("[warn] homography returned wH0; ignoring click.")
            self.marker_uv = None
            self.marker_rbev = None
            self._build_canvas()
            cv2.imshow(self.window, self.canvas_bgr)
            return

        u = q[0] / q[2]
        v = q[1] / q[2]
        u_i = int(round(u))
        v_i = int(round(v))

        hB, wB = self.bev_mask.shape[:2]
        if not (0 <= u_i < wB and 0 <= v_i < hB):
            print(f"[info] click ({x_local:.1f},{y_local:.1f}) -> BEV ({u:.1f},{v:.1f}) is OUTSIDE [{wB}x{hB}]")
            self.marker_uv = None
            self.marker_rbev = None
            self._build_canvas()
            cv2.imshow(self.window, self.canvas_bgr)
            return

        self.marker_uv = (u_i, v_i)

        # Project BEV marker to RBEV
        r_pt = self._project_bev_to_rbev(self.marker_uv)
        if r_pt is None:
            print("[warn] H_RBEV not available or invalid; skipping RBEV marker.")
            self.marker_rbev = None
        else:
            xr, yr = r_pt
            hR, wR = self.rbev_bgr0.shape[:2]
            if 0 <= xr < wR and 0 <= yr < hR:
                self.marker_rbev = (xr, yr)
                print(f"[ok] click ({x_local:.1f},{y_local:.1f}) -> BEV ({u_i},{v_i}) -> RBEV ({xr},{yr})")
            else:
                print(f"[info] RBEV point out of bounds: ({xr},{yr}) for size {wR}x{hR}")
                self.marker_rbev = None

        self._build_canvas()
        cv2.imshow(self.window, self.canvas_bgr)

    def run(self):
        print("[help] Left-click on TOP image to set BEV & RBEV markers; 'r' reset; 's' save; 'q'/Esc quit.")
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord('q')):  # Esc or 'q'
                break
            elif key == ord('r'):
                self.marker_uv = None
                self.marker_rbev = None
                self.top_xy = None
                self._build_canvas()
                cv2.imshow(self.window, self.canvas_bgr)
            elif key == ord('s'):
                out = "triple_view_vertical.png"
                cv2.imwrite(out, self.canvas_bgr)
                print(f"[saved] {out}")
        cv2.destroyAllWindows()


def load_mask_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read mask from: {path}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    return img


def load_h_3x3(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        H = np.load(path)
    else:
        H = np.loadtxt(path)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    return H


def parse_args():
    ap = argparse.ArgumentParser(description="Interactive Original�BEV�RBEV (Vertical Stack)")
    ap.add_argument("--orig", required=True, help="path to original single-channel mask (uint8)")
    ap.add_argument("--bev", required=True, help="path to BEV single-channel mask (uint8)")
    ap.add_argument("--H", required=True, help="path to 3x3 homography (txt with 9 numbers or .npy)")
    ap.add_argument("--gap", type=int, default=16, help="vertical gap (pixels) between views")
    ap.add_argument("--max-width", type=int, default=1400, help="max canvas width")
    ap.add_argument("--max-height", type=int, default=2400, help="max canvas height")
    ap.add_argument("--marker-size", type=int, default=8, help="down-triangle half-size in pixels")
    ap.add_argument("--window", default="Original�BEV�RBEV (Vertical)", help="window title")
    ap.add_argument("--rbev", default="RBEV_1400.png", help="RBEV image path")
    return ap.parse_args()


def main():
    args = parse_args()
    orig = load_mask_gray(args.orig)
    bev = load_mask_gray(args.bev)
    H = load_h_3x3(args.H)

    viewer = MaskBEVViewer(
        original_mask=orig,
        bev_mask=bev,
        H_src2bev=H,
        dst_pts_bev_px=((0, 0), (0, 0), (0, 0), (0, 0)),  # will be overridden by your caller if provided
        window_name=args.window,
        gap=args.gap,
        max_width=args.max_width,
        max_height=args.max_height,
        marker_size=args.marker_size,
        rbev_path=args.rbev
    )
    viewer.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
