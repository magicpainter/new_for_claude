#!/usr/bin/env python3
import cv2, numpy as np, math, sys
from pathlib import Path

# ---- spec ----
LONG_LEN_TARGET, LONG_TOL = 380, 40    # 420 ± 40
SHORT_LEN_TARGET, SHORT_TOL = 320, 35  # 235 ± 30
LONG_SLOPE_MAX_DIFF = 4                # degrees
SHORT_SLOPE_MAX_DIFF = 8              # degrees

# ---- math helpers ----
def seg_len(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.hypot(*(b - a)))

def seg_angle_deg(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ang = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
    return (ang + 180) % 180  # normalize to [0,180)

def ang_diff(a, b):
    d = abs(a - b)
    return min(d, 180 - d)

def order_quad_clockwise(pts):
    # pts: (4,2) possibly unordered. Sort around centroid by angle for stability.
    c = np.mean(pts, axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    idx = np.argsort(ang)  # CCW
    return pts[idx].astype(np.float32)

# ---- polygon extraction ----
def get_green_quadrilateral(mask):
    # exact (0,255,0) in BGR
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None

    # Try to approximate to 4 points (quadrilateral). Adapt epsilon if needed.
    peri = cv2.arcLength(cnt, True)
    for scale in (0.01, 0.015, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(cnt, epsilon=scale*peri, closed=True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            return order_quad_clockwise(quad)

    # Fallback: take convex hull and try again
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for scale in (0.01, 0.015, 0.02, 0.03, 0.05):
        approx = cv2.approxPolyDP(hull, epsilon=scale*peri, closed=True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            return order_quad_clockwise(quad)

    # Last resort: use minAreaRect corners (still 4 pts, but rectangle)
    rect = cv2.minAreaRect(cnt)
    quad = cv2.boxPoints(rect).astype(np.float32)
    return order_quad_clockwise(quad)

# ---- analysis on a 4-pt polygon ----
def analyze_quad(quad):
    # edges in order: (0-1),(1-2),(2-3),(3-0)
    edges = [(quad[i], quad[(i+1) % 4]) for i in range(4)]
    lengths = [seg_len(a, b) for a, b in edges]
    angles  = [seg_angle_deg(a, b) for a, b in edges]

    # Opposite pairs: (0,2) and (1,3). Decide which is "long" by total length.
    sum02 = lengths[0] + lengths[2]
    sum13 = lengths[1] + lengths[3]
    if sum02 >= sum13:
        long_pair, short_pair = (0, 2), (1, 3)
    else:
        long_pair, short_pair = (1, 3), (0, 2)

    long_lens  = [lengths[i] for i in long_pair]
    short_lens = [lengths[i] for i in short_pair]
    long_diff  = ang_diff(angles[long_pair[0]],  angles[long_pair[1]])
    short_diff = ang_diff(angles[short_pair[0]], angles[short_pair[1]])

    # length checks use individual edges within their pair
    ok_long_len  = all(abs(L - LONG_LEN_TARGET)  <= LONG_TOL  for L in long_lens)
    ok_short_len = all(abs(L - SHORT_LEN_TARGET) <= SHORT_TOL for L in short_lens)
    ok_long_ang  = long_diff  < LONG_SLOPE_MAX_DIFF
    ok_short_ang = short_diff < SHORT_SLOPE_MAX_DIFF
    passed = ok_long_len and ok_short_len and ok_long_ang and ok_short_ang

    return {
        "edges": edges,
        "lengths": lengths,
        "angles": angles,
        "long_pair": long_pair,
        "short_pair": short_pair,
        "long_diff": long_diff,
        "short_diff": short_diff,
        "passed": passed,
    }

# ---- drawing ----
def draw_overlay(bgr, quad, res):
    out = bgr.copy()
    poly = quad.astype(int).reshape(-1,1,2)
    cv2.polylines(out, [poly], True, (255,255,255), 3)

    # corners
    for (x,y) in quad:
        p = (int(x), int(y))
        cv2.circle(out, p, 7, (0,0,0), -1)
        cv2.circle(out, p, 5, (0,255,255), -1)

    # edge labels
    for i, (a,b) in enumerate(res["edges"]):
        mid = (int((a[0]+b[0])/2), int((a[1]+b[1])/2))
        txt = f"E{i}:{res['lengths'][i]:.1f}px  θ:{res['angles'][i]:.1f}°"
        cv2.putText(out, txt, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

    # pass/fail banner + slope diffs
    status = "PASS" if res["passed"] else "FAIL"
    color = (0,200,0) if res["passed"] else (0,0,255)
    cv2.rectangle(out, (10,10), (360,100), (0,0,0), -1)
    cv2.putText(out, status, (25,75), cv2.FONT_HERSHEY_DUPLEX, 2.0, color, 3, cv2.LINE_AA)

    sub = (f"long pair: {res['long_pair']}  Δθ={res['long_diff']:.1f}°   "
           f"short pair: {res['short_pair']} Δθ={res['short_diff']:.1f}°")
    cv2.putText(out, sub, (370,75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(out, sub, (370,75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return out

# ---- batch ----
def process_folder(in_dir, out_dir=None):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir) if out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    paths = sorted([p for ext in exts for p in in_dir.glob(ext)])
    if not paths:
        print(f"[WARN] no images in {in_dir}")
        return

    ok = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERR] cannot read {p}")
            continue
        mask = cv2.inRange(img, (0, 255, 0), (0, 255, 0))
        quad = get_green_quadrilateral(mask)
        if quad is None:
            print(f"[FAIL] {p.name}: no quadrilateral found")
            out = img.copy()
            cv2.putText(out, "FAIL: no green quad", (20,60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
        else:
            res = analyze_quad(quad)
            out = draw_overlay(img, quad, res)
            lbl = "PASS" if res["passed"] else "FAIL"
            print(f"[{lbl}] {p.name}  "
                  f"lens={','.join(f'{L:.1f}' for L in res['lengths'])}  "
                  f"angs={','.join(f'{a:.1f}' for a in res['angles'])}  "
                  f"Δθ_long={res['long_diff']:.2f}  Δθ_short={res['short_diff']:.2f}")
            if res["passed"]: ok += 1

        cv2.imwrite(str(out_dir / f"{p.stem}_analysis.png"), out)

    print(f"[SUMMARY] {ok}/{len(paths)} PASS")

# # ---- main ----
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python quad_analysis.py <in_dir> [out_dir]")
#         sys.exit(1)
#     process_folder(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
