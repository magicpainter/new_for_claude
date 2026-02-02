import os
import cv2
import numpy as np
from BEV.pass_fail_critera import get_green_quadrilateral, analyze_quad

PALETTE = np.array([
    [255,   0,   0],  # class 0 - red
    [  0, 255,   0],  # class 1 - green
    [  0,   0, 255],  # class 2 - blue
    [255, 255,   0],  # class 3 - yellow
    [255,   0, 255],  # class 4 - magenta
], dtype=np.uint8)

def project_mask_to_bev(
    mask_hw_int: np.ndarray,
    out_path: str = "BEV.png",
    dst_points=((500,700),(100,100),(100,700),(500,100)),  # (x_max→, x_min→, y_max→, y_min→) 现在可当“世界坐标”
    world_bbox=(-2000, 2000, 0, 2000),               # <- 新：例如 (-2000, 2000, 0, 2000)，不传则走旧逻辑
    px_per_unit: float = 1.0,      # <- 新：1 世界单位=1像素
    y_up: bool = False             # <- 新：True=图像y向上增长；False=向下（OpenCV惯例）
):
    """
    返回:
      bev_mask_hw_uint8:  变换后的单通道 BEV mask（uint8）
      src_pts_xy:         源图上 4 个 (x,y) 点，顺序与 dst_points 对应
      H_3x3:              最终用于 warp 的 3x3 单应矩阵（float64）
    """
    if mask_hw_int.ndim != 2:
        raise ValueError(f"Expect 2D mask, got shape={mask_hw_int.shape}")
    H, W = mask_hw_int.shape
    if (H, W) != (1200, 1920):
        print(f"[warn] mask size is {W}x{H}, expected 1920x1200 per your spec.")

    # —— 1) 在 class==1 中找 4 个极值点 ——
    # ys, xs = np.where(mask_hw_int == 1)
    mask_rgb = PALETTE[np.clip(mask_hw_int, 0, len(PALETTE) - 1)]
    mask = cv2.inRange(mask_rgb, (0, 255, 0), (0, 255, 0))
    quad = get_green_quadrilateral(mask)  # get the four corners
    res = analyze_quad(quad)   # get the pass-fail result
    f_passed_4_corners_check = res["passed"] # pass the criteria check
    print(f"Passed : {f_passed_4_corners_check}")


    xs = quad[:, 0]
    ys = quad[:, 1]
    if xs.size == 0:
        raise ValueError("No pixels with class ID == 1; cannot define 4 anchor points.")

    idx_x_max = np.lexsort((ys, -xs))  # x最大，y最小优先
    idx_x_min = np.lexsort((-ys, xs))  # x最小，y最大优先
    idx_y_max = np.lexsort((xs, -ys))  # y最大，x最小优先
    idx_y_min = np.lexsort((-xs, ys))  # y最小，x最大优先

    x_max_pt = (int(xs[idx_x_max[0]]), int(ys[idx_x_max[0]]))
    x_min_pt = (int(xs[idx_x_min[0]]), int(ys[idx_x_min[0]]))
    y_max_pt = (int(xs[idx_y_max[0]]), int(ys[idx_y_max[0]]))
    y_min_pt = (int(xs[idx_y_min[0]]), int(ys[idx_y_min[0]]))

    def ensure_unique(points_xy, candidates_order):
        seen = set(points_xy)
        if len(seen) == 4:
            return points_xy
        out = list(points_xy)
        for k in range(4):
            for cand_idx in candidates_order[k][1:]:
                cand = (int(xs[cand_idx]), int(ys[cand_idx]))
                if cand not in out:
                    out[k] = cand
                    break
        if len(set(out)) < 4:
            raise ValueError("Failed to find 4 distinct anchor points inside class-1 region.")
        return tuple(out)

    cand_lists = (idx_x_max, idx_x_min, idx_y_max, idx_y_min)
    # src_pts_xy = ensure_unique(
    #     (x_max_pt, x_min_pt, y_max_pt, y_min_pt),
    #     cand_lists
    # )

    if y_max_pt[0]>y_min_pt[0]: # camera at left side of BEV
        src_pts_xy = ensure_unique(
            (x_max_pt, x_min_pt, y_max_pt, y_min_pt),
            cand_lists
        )
        print('camera at left side of BEV')
    elif y_max_pt[0]<y_min_pt[0]:
        src_pts_xy = ensure_unique(
            (y_max_pt, y_min_pt, x_min_pt, x_max_pt),
            cand_lists
        )
        print('camera at right side of BEV')
    else:
        src_pts_xy = ensure_unique(
            (x_max_pt, x_min_pt, y_max_pt, y_min_pt),
            cand_lists
        )
        print('cannot tell which side camera is at!!!')
        print('assumed a side!!!')


    src = np.array(src_pts_xy, dtype=np.float32)

    # —— 2) 两种模式：
    #  a) 兼容旧逻辑（dst_points 是像素，直接求 src->dst 并输出到自适应画布）
    #  b) 世界坐标逻辑（dst_points 是世界坐标；再用 world_bbox & px_per_unit 生成画布并合成仿射）
    if world_bbox is None:
        dst = np.array(tuple((float(x), float(y)) for (x, y) in dst_points), dtype=np.float32)
        H_final = cv2.getPerspectiveTransform(src, dst)
        bev_w = int(max(p[0] for p in dst)) + 1
        bev_h = int(max(p[1] for p in dst)) + 1
    else:
        # —— 2b-1) 源→世界 单应性
        dst_world = np.array(tuple((float(x), float(y)) for (x, y) in dst_points), dtype=np.float32)
        H_src2world = cv2.getPerspectiveTransform(src, dst_world)

        # —— 2b-2) 世界→像素 仿射，把 world_bbox 装进画布
        x_min, x_max, y_min, y_max = world_bbox
        # 注意：若你要“端点也占一列/行像素”，可加 +1
        bev_w = int(round((x_max - x_min) * px_per_unit))
        bev_h = int(round((y_max - y_min) * px_per_unit))
        if bev_w <= 0 or bev_h <= 0:
            raise ValueError("Invalid world_bbox or px_per_unit → non-positive canvas size.")

        if not y_up:
            # 图像y向下： u=(X - x_min)*s, v=(Y - y_min)*s
            A_world2pix = np.array([
                [ px_per_unit, 0.0,           -x_min*px_per_unit],
                [ 0.0,         px_per_unit,   -y_min*px_per_unit],
                [ 0.0,         0.0,            1.0              ]
            ], dtype=np.float64)
        else:
            # 图像y向上： u=(X - x_min)*s, v=(y_max - Y)*s
            A_world2pix = np.array([
                [ px_per_unit,  0.0,            -x_min*px_per_unit],
                [ 0.0,         -px_per_unit,     y_max*px_per_unit],
                [ 0.0,          0.0,              1.0              ]
            ], dtype=np.float64)

        H_final = A_world2pix @ H_src2world  # 源像素 → 世界 → 像素

    # —— 3) 透视变换（最近邻，保持类别整数） ——
    mask_f = mask_hw_int.astype(np.float32)
    bev_mask_f = cv2.warpPerspective(
        mask_f, H_final, dsize=(bev_w, bev_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    bev_mask = bev_mask_f.astype(np.uint8)

    # —— 4) 上色 & 保存 ——
    bev_rgb = PALETTE[np.clip(bev_mask, 0, len(PALETTE)-1)]
    bev_bgr = cv2.cvtColor(bev_rgb, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, bev_bgr)

    print(f"[done] BEV saved to: {out_path}")
    print(f"      src_pts (x,y_px):             {tuple(map(tuple, src.astype(int)))}")
      # == 新增：输出 dst_points 在“最终 BEV 图像”坐标系下的像素坐标 ==
    def _clip_xy(x, y, w, h):
        return (int(max(0, min(w - 1, round(x)))), int(max(0, min(h - 1, round(y)))))


    if world_bbox is None:
        # 无 world_bbox：dst 已经是最终 BEV 像素坐标
        dst_pts_bev_px = [tuple(map(int, p)) for p in dst.reshape(-1, 2)]
        print(f"      dst_pts (x,y_px):             {tuple(dst_pts_bev_px)}")
    else:
        # 有 world_bbox：dst_world 为世界坐标，需要映射到 BEV 像素并考虑 y_up
        def world_to_bev_px(xw, yw):
            xpx = (xw - x_min) * px_per_unit
            ypx = (y_max - yw) * px_per_unit if y_up else (yw - y_min) * px_per_unit
            return _clip_xy(xpx, ypx, bev_w, bev_h)
        dst_pts_bev_px = [world_to_bev_px(float(xw), float(yw)) for xw, yw in dst_world.reshape(-1, 2)]
        print(f"      dst_world (x,y):              {tuple(map(tuple, dst_world.astype(float)))}")
        print(f"      dst_pts_in_final_BEV_px:      {tuple(dst_pts_bev_px)}")
        print(f"      world_bbox: x[{x_min},{x_max}], y[{y_min},{y_max}], px_per_unit={px_per_unit}, y_up={y_up}")
    print(f"      canvas (WxH):                  {bev_w}x{bev_h}")


    return bev_mask, src_pts_xy, H_final, np.array(dst_pts_bev_px, dtype=np.int32), f_passed_4_corners_check
