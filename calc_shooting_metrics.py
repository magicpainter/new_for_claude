from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional
import numpy as np
import math


@dataclass
class SideViewShotFlags:
    ok: bool
    reason: str = ""

    # flat-shot metrics
    entry_angle_deg: float = float("nan")     # downward angle magnitude near hoop
    apex_over_rim_px: float = float("nan")    # peak height above rim (Y-up, px)
    time_in_air_frames: int = 0

    # rim-height crossing metrics (descending)
    x_along_at_rim_px: float = float("nan")   # along approach axis; <0 short, >0 long (proxy)
    lat_at_rim_px: float = float("nan")       # perpendicular-to-approach offset (proxy)
    img_dx_at_rim_px: float = float("nan")    # (x - hoop_x) at rim height (most intuitive L/R)

    # flags (single-shot heuristics)
    flat_shot: bool = False
    left_right: str = "unknown"               # "left"|"right"|"center"|"unknown"
    short_long: str = "unknown"               # "short"|"long"|"center"|"unknown"


def _extract_xy(ball_trajectory: Iterable[Any]) -> np.ndarray:
    pts = []
    for it in ball_trajectory:
        if not it:  # [] / None
            continue
        if isinstance(it, (list, tuple, np.ndarray)) and len(it) >= 2:
            try:
                x = float(it[0]); y = float(it[1])
                if np.isfinite(x) and np.isfinite(y):
                    pts.append((x, y))
            except Exception:
                pass
    return np.asarray(pts, dtype=np.float32)


def analyze_sideview_shot(
    ball_trajectory: Iterable[Any],  # points like [x,y] in image coords, can include []
    hoop_ring_x: float,
    hoop_ring_y: float,
    *,
    min_points: int = 10,
    # flat-shot thresholds (tune per distance bin)
    flat_entry_angle_deg_max: float = 35.0,
    flat_apex_over_rim_min_px: float = 60.0,
    # rim-height crossing selection
    rim_y_tolerance_px: float = 8.0,
    # left/right thresholds (in image-x at rim height)
    lr_threshold_px: float = 30.0,
    # short/long thresholds (along approach axis at rim height)
    sl_threshold_px: float = 20.0,
) -> SideViewShotFlags:
    """
    Mid-court sideline camera:
      - left/right bias is visible as image-x offset near rim height.
      - short/long can be approximated along the approach direction in the image.

    Steps:
      1) Extract valid points
      2) Define approach unit vector u from release -> hoop in image
         and lateral unit vector v perpendicular to u
      3) Convert points into:
           X_along = dot((p - hoop), u)
           L_lat   = dot((p - hoop), v)
           Y_up    = hoop_y - y
      4) Fit parabola Y_up = a X_along^2 + b X_along + c
         entry angle at hoop: dY/dX at X=0 => b
      5) Find descending point closest to rim height (Y_up ~ 0)
         measure left/right and short/long proxies there
    """
    pts = _extract_xy(ball_trajectory)
    if pts.shape[0] < min_points:
        return SideViewShotFlags(False, reason=f"not enough points: {pts.shape[0]}")

    hoop = np.array([float(hoop_ring_x), float(hoop_ring_y)], dtype=np.float32)

    # "release" = first valid point in this trajectory window
    rel = pts[0]
    d = hoop - rel
    norm = float(np.hypot(d[0], d[1]))
    if norm < 1e-6:
        return SideViewShotFlags(False, reason="release point too close to hoop point")

    # approach axis (image-plane)
    u = d / norm                      # towards hoop
    v = np.array([-u[1], u[0]], dtype=np.float32)  # perpendicular

    # center points at hoop
    P = pts - hoop[None, :]

    X_along = P @ u                   # along approach direction
    L_lat = P @ v                     # lateral (perp) offset
    Y_up = float(hoop_ring_y) - pts[:, 1]  # Y-up (positive above rim)

    # Fit Y_up = aX^2 + bX + c
    A = np.stack([X_along**2, X_along, np.ones_like(X_along)], axis=1)
    try:
        (a, b, c), *_ = np.linalg.lstsq(A, Y_up, rcond=None)
        a = float(a); b = float(b); c = float(c)
    except Exception as e:
        return SideViewShotFlags(False, reason=f"fit failed: {e}")

    # Entry angle at hoop center X=0 (use magnitude)
    entry_angle_deg = math.degrees(math.atan(abs(b)))

    # Apex above rim (rim is Y_up=0)
    if abs(a) > 1e-9:
        Xv = -b / (2.0 * a)
        apex = a * Xv * Xv + b * Xv + c
    else:
        apex = float(np.nanmax(Y_up))
    apex_over_rim_px = float(apex)

    # Find descending rim-height crossing (closest Y_up to 0, while descending)
    dY = np.diff(Y_up)
    descending = np.concatenate([[False], dY < 0])  # align to frames
    near_rim = np.abs(Y_up) <= rim_y_tolerance_px
    candidates = np.where(descending & near_rim)[0]

    x_along_at_rim = float("nan")
    lat_at_rim = float("nan")
    img_dx_at_rim = float("nan")
    left_right = "unknown"
    short_long = "unknown"

    if candidates.size > 0:
        idx = int(candidates[np.argmin(np.abs(Y_up[candidates]))])

        x_along_at_rim = float(X_along[idx])
        lat_at_rim = float(L_lat[idx])
        img_dx_at_rim = float(pts[idx, 0] - float(hoop_ring_x))  # simplest L/R measure

        # Left/right label from image-x offset at rim height
        if img_dx_at_rim > lr_threshold_px:
            left_right = "right"
        elif img_dx_at_rim < -lr_threshold_px:
            left_right = "left"
        else:
            left_right = "center"

        # Short/long proxy from along-axis offset at rim height
        # negative => still before reaching hoop along u => "short"
        if x_along_at_rim < -sl_threshold_px:
            short_long = "short"
        elif x_along_at_rim > sl_threshold_px:
            short_long = "long"
        else:
            short_long = "center"

    # Flat-shot heuristic (single shot) - require BOTH conditions to avoid false positives
    flat_shot = (entry_angle_deg < flat_entry_angle_deg_max) and (apex_over_rim_px < flat_apex_over_rim_min_px)

    return SideViewShotFlags(
        ok=True,
        entry_angle_deg=float(entry_angle_deg),
        apex_over_rim_px=float(apex_over_rim_px),
        time_in_air_frames=int(len(pts)),
        x_along_at_rim_px=float(x_along_at_rim),
        lat_at_rim_px=float(lat_at_rim),
        img_dx_at_rim_px=float(img_dx_at_rim),
        flat_shot=bool(flat_shot),
        left_right=left_right,
        short_long=short_long,
    )
