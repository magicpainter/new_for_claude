import numpy as np

def find_triple_points_0_2_3(
    mask_hw_int: np.ndarray,
    A_xy: tuple,
    B_xy: tuple,
    step_x: int = 100,
    step_y: int = 100,
    max_vert_iters: int = 5,
    max_outer_iters: int = 12,
    refine_window: int = 6,
):
    mask = np.asarray(mask_hw_int)
    if mask.ndim != 2:
        raise ValueError(f"expect 2D mask, got shape={mask.shape}")
    H, W = mask.shape

    def _clamp_y(y): return max(0, min(H - 1, int(y)))
    def _clamp_x(x): return max(0, min(W - 1, int(x)))

    # 从 (x, y_seed) 出发，踏步 + 小区间二分，返回 (y0, below_cls)
    def _bottom_boundary_from_seed(x: int, y_seed: int):
        x = _clamp_x(x)
        y = _clamp_y(y_seed)
        cls_prev = int(mask[y, x])
        dy = step_y if cls_prev == 0 else -step_y

        max_steps = (H // max(1, step_y)) + 4
        for _ in range(max_steps):
            y_next = _clamp_y(y + dy)
            if y_next == y:
                break
            cls_next = int(mask[y_next, x])
            if cls_next != cls_prev:
                lo, hi = (y, y_next) if y < y_next else (y_next, y)

                if int(mask[lo, x]) == int(mask[hi, x]):
                    found = False
                    for k in range(1, refine_window + 1):
                        yl = _clamp_y(lo - k)
                        yh = _clamp_y(hi + k)
                        if int(mask[yl, x]) != int(mask[lo, x]): lo, found = yl, True; break
                        if int(mask[yh, x]) != int(mask[hi, x]): hi, found = yh, True; break
                    if not found:
                        return None, None

                for _ in range(max_vert_iters):
                    if hi - lo <= 1:
                        break
                    mid = (lo + hi) // 2
                    if mask[mid, x] == 0:
                        lo = mid
                    else:
                        hi = mid

                y0 = lo
                adjusted = True
                if mask[y0, x] != 0:
                    adjusted = False
                    for d in range(1, refine_window + 1):
                        yy = y0 - d
                        if yy >= 0 and mask[yy, x] == 0:
                            y0 = yy; adjusted = True; break
                elif y0 + 1 < H and mask[y0 + 1, x] == 0:
                    adjusted = False
                    for d in range(1, refine_window + 1):
                        yy = y0 + d
                        if yy + 1 < H and (mask[yy, x] == 0 and mask[yy + 1, x] != 0):
                            y0 = yy; adjusted = True; break
                if not adjusted or y0 >= H - 1:
                    return None, None

                below = int(mask[y0 + 1, x])
                return int(y0), int(below)

            y, cls_prev = y_next, cls_next

        # 兜底：围绕 y_seed 小窗直接找 “0 段末端”
        window = max(24, refine_window * 3)
        for dir_sign in (+1, -1):
            y = _clamp_y(y_seed)
            for _ in range(window):
                yn = _clamp_y(y + dir_sign)
                if yn == y: break
                if mask[y, x] == 0 and (y + 1 < H and mask[y + 1, x] != 0):
                    return int(y), int(mask[y + 1, x])
                y = yn
        return None, None

    # —— 关键修正：保持“语义端点”，不做大小排序 ——
    # 输入：x02/y02（该列是 0/2），x03/y03（该列是 0/3）。它们可以 x02 < x03，也可以 x02 > x03。
    def _horizontal_bisect_between(x02: int, y02: int, x03: int, y03: int):
        # 端点宽容校验（不抛错，尽力自救）
        def _soft_ensure(x_target, want_below, y_seed, x_a, y_a, x_b, y_b):
            yb, bb = _bottom_boundary_from_seed(x_target, y_seed)
            if yb is not None and bb == want_below:
                return x_target, yb
            # 邻域扩张搜索
            for max_dx in (refine_window, max(refine_window*4, 48)):
                for dx in range(1, max_dx + 1):
                    for xx in (x_target - dx, x_target + dx):
                        if 0 <= xx < W:
                            # 两种 y 种子都试：上一列的 y_seed 与线性插值
                            y_try_list = [y_seed]
                            if (x_b - x_a) != 0:
                                y_lin = int(round(np.interp(xx, [x_a, x_b], [y_a, y_b])))
                                y_try_list.append(y_lin)
                            for yy in y_try_list:
                                yb2, bb2 = _bottom_boundary_from_seed(xx, yy)
                                if yb2 is not None and bb2 == want_below:
                                    return xx, yb2
            return None, None

        # 软校验两端：拿不到也不报错，后续二分时再修复
        x02_fix, y02_fix = _soft_ensure(x02, 2, y02, x02, y02, x03, y03)
        if x02_fix is not None:
            x02, y02 = x02_fix, y02_fix
        x03_fix, y03_fix = _soft_ensure(x03, 3, y03, x02, y02, x03, y03)
        if x03_fix is not None:
            x03, y03 = x03_fix, y03_fix

        for _ in range(max_outer_iters):
            if abs(x02 - x03) <= 1:
                break
            xm = (x02 + x03) // 2  # 无论谁大都没关系
            # 用端点的 y 线性插值做 xm 的种子（端点可能同 x，interp 也能处理）
            if x02 != x03:
                y_seed = int(round(np.interp(xm, [x02, x03], [y02, y03])))
            else:
                y_seed = y02

            ym, bm = _bottom_boundary_from_seed(xm, y_seed)
            if ym is None:
                # 就近 +/- refine_window 内找一个能解出边界的列
                moved = False
                for dx in range(1, refine_window + 1):
                    for xx in (xm - dx, xm + dx):
                        if 0 <= xx < W:
                            y_guess = y_seed
                            if x02 != x03:
                                y_lin = int(round(np.interp(xx, [x02, x03], [y02, y03])))
                                # 试两次
                                for yy in (y_guess, y_lin):
                                    yx, bx = _bottom_boundary_from_seed(xx, yy)
                                    if yx is not None:
                                        xm, ym, bm = xx, yx, bx
                                        moved = True; break
                            else:
                                yx, bx = _bottom_boundary_from_seed(xx, y_guess)
                                if yx is not None:
                                    xm, ym, bm = xx, yx, bx
                                    moved = True
                            if moved: break
                    if moved: break
                if not moved:
                    break

            if bm == 2:
                x02, y02 = xm, ym
            elif bm == 3:
                x03, y03 = xm, ym
            else:
                # 其它类别，选离它更近的一侧替换（保住括号宽度单调收缩）
                if abs(xm - x02) <= abs(xm - x03):
                    x02, y02 = xm, ym if ym is not None else y02
                else:
                    x03, y03 = xm, ym if ym is not None else y03

        # 输出：x 取 “0/2 侧”的列；y 取两侧边界 y 的平均（缺失则用现存）
        x_star = int(x02)
        y_star = int(round((y02 if y02 is not None else y03) if (y02 is None or y03 is None)
                           else 0.5 * (y02 + y03)))
        return x_star, y_star

    # 从起点 P，沿 dir_sign（-1: 左，+1: 右）粗扫，括住 0/2 与 0/3，然后做左右二分
    def _solve_one_side(P_xy: tuple, dir_sign: int):
        x = _clamp_x(P_xy[0]); y_seed = _clamp_y(P_xy[1])
        last_x_02, last_y_02 = None, None
        first_x_03, first_y_03 = None, None

        yb0, bb0 = _bottom_boundary_from_seed(x, y_seed)
        if yb0 is not None:
            y_seed = yb0
            if bb0 == 2: last_x_02, last_y_02 = x, yb0
            elif bb0 == 3: first_x_03, first_y_03 = x, yb0

        tried = 0
        while (first_x_03 is None) and (0 <= x < W):
            x_next = _clamp_x(x + dir_sign * step_x)
            if x_next == x: break
            yb, bb = _bottom_boundary_from_seed(x_next, y_seed)
            if yb is not None:
                y_seed = yb
                if bb == 2:
                    last_x_02, last_y_02 = x_next, yb
                elif bb == 3:
                    first_x_03, first_y_03 = x_next, yb
                    break
            x = x_next
            tried += 1
            if tried > (W // max(1, step_x) + 4):
                break

        if last_x_02 is None or first_x_03 is None:
            raise ValueError(f"Cannot bracket 0/2→0/3 transition from start {P_xy}, dir={dir_sign}.")

        return _horizontal_bisect_between(last_x_02, last_y_02, first_x_03, first_y_03)

    Cx, Cy = _solve_one_side(A_xy, dir_sign=-1)
    Dx, Dy = _solve_one_side(B_xy, dir_sign=+1)
    return (int(Cx), int(Cy)), (int(Dx), int(Dy))
