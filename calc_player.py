import numpy as np

def analyze_player_features(
    player_q,
    ball_q,
    *,
    N: int = 60,
    alpha_close: float = 0.4,     # "ball is close" radius = alpha_close * max(w,h)
    alpha_far: float = 0.8,       # "ball is clearly separated" radius = alpha_far * max(w,h)
    confirm_frames: int = 3,       # after candidate release, require this many frames staying "far"
    smooth_win: int = 3,           # 1=off, 3/5=light smoothing on distance
) -> dict | None:
    """
    player_q element: [cx, cy, w, h] or [cx, cy, w, h, foot_x, foot_y]  (CENTER FORMAT)
    ball_q element:   [x, y]

    Returns:
      {"release_idx": int, "release_height_ratio": float, "release_lateral_ratio": float}
    Returns None only if there is no index where both player and ball are valid.
    """

    P = list(player_q)[-N:]
    B = list(ball_q)[-N:]
    N = min(len(P), len(B))
    if N == 0:
        return None

    # validity - accept both 4-element and 6-element (with foot_loc) formats
    def vp(p):
        return (isinstance(p, (list, tuple)) and len(p) >= 4 and
                np.isfinite(p[:4]).all() and p[2] > 1e-6 and p[3] > 1e-6)

    def vb(b):
        return isinstance(b, (list, tuple)) and len(b) == 2 and np.isfinite(b).all()

    valid = np.array([vp(P[i]) and vb(B[i]) for i in range(N)], dtype=bool)
    if not valid.any():
        return None

    # arrays (NaN where invalid)
    cx = np.full(N, np.nan, np.float32)
    cy = np.full(N, np.nan, np.float32)
    w  = np.full(N, np.nan, np.float32)
    h  = np.full(N, np.nan, np.float32)
    bx = np.full(N, np.nan, np.float32)
    by = np.full(N, np.nan, np.float32)

    for i in range(N):
        if not valid[i]:
            continue
        cx[i], cy[i], w[i], h[i] = map(float, P[i][:4])  # only use first 4 elements
        bx[i], by[i] = map(float, B[i])

    size = np.maximum(w, h)  # per-frame player scale
    d = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)  # ball-center distance

    # Light smoothing on distance to reduce jitter (only on valid indices)
    if smooth_win >= 3 and smooth_win % 2 == 1:
        half = smooth_win // 2
        ds = d.copy()
        for i in range(N):
            if not valid[i]:
                continue
            lo = max(0, i - half)
            hi = min(N, i + half + 1)
            m = valid[lo:hi]
            if m.any():
                ds[i] = float(np.nanmedian(d[lo:hi][m]))
        d = ds

    # thresholds (NaN-safe)
    r_close = alpha_close * size
    r_far   = alpha_far * size

    # We care about the last continuous valid segment (shot confirmation window)
    idx_valid = np.where(valid)[0]
    end = int(idx_valid[-1])
    start = end
    while start - 1 >= 0 and valid[start - 1]:
        start -= 1

    # Detect: last time we were "close", followed by sustained "far"
    release_idx = None
    K = max(1, int(confirm_frames))

    # Scan from start->end-K to find separation; pick the LAST such event (closest to shot end)
    for t in range(start, end - K + 1):
        if not valid[t]:
            continue

        # need to have been close recently (otherwise it's already separated)
        if not (d[t] <= r_close[t]):
            continue

        # candidate separation point is when it becomes far and stays far for K frames
        # check next frames t+1..t+K
        ok = True
        for k in range(1, K + 1):
            j = t + k
            if not valid[j]:
                ok = False
                break
            if not (d[j] >= r_far[j]):
                ok = False
                break
        if ok:
            release_idx = t  # last close moment before confirmed separation

    # Fallback: if no "close->far" found, use the closest-approach frame in the segment
    if release_idx is None:
        seg = np.arange(start, end + 1)
        seg = seg[valid[seg]]
        if seg.size == 0:
            return None
        release_idx = int(seg[np.nanargmin(d[seg])])

    # Compute ratios at release_idx using bbox edges from center format
    top  = cy[release_idx] - h[release_idx] / 2.0
    left = cx[release_idx] - w[release_idx] / 2.0

    release_height_ratio  = float((by[release_idx] - top) / h[release_idx])
    release_lateral_ratio = float((bx[release_idx] - left) / w[release_idx])

    return {
        "release_idx": int(release_idx),
        "release_height_ratio": release_height_ratio,
        "release_lateral_ratio": release_lateral_ratio,
    }
