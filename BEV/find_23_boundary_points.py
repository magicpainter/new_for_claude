import numpy as np

def find_23_boundary_points(bev_mask: np.ndarray,
                            A_xy: tuple,
                            B_xy: tuple,
                            step: int = 100):
    """
    在 BEV mask 上，从点 A、B 出发，分别向左右两侧（-x与+x）搜索位于“class 2 与 class 3 边界”的4个点。
    - 粗搜索：以 step（默认100像素）为间隔做定步进试探，直到从 {1,2} 跨到 {3,0}（认为“越界”）形成区间
    - 细搜索：在“内(in)←→外(out)”之间做二分，最终锁定到 1 像素范围
    - 边界定义：返回满足
        右侧边界：mask[y, x*] == 2 且 mask[y, x*+1] == 3
        左侧边界：mask[y, x*] == 2 且 mask[y, x*-1] == 3
      的点 (x*, y)，即“最后一个2、紧邻方向上是3”的像素坐标
    - 若没法找到 2/3 边界（只出现 2→0 或没有 3），将抛出 ValueError

    参数:
      bev_mask: (H,W) 单通道整型语义图（uint8/int32等），类ID用 0/1/2/3/...
      A_xy, B_xy: 输入点的 (x, y) 像素坐标（x=列, y=行）
      step: 粗搜索步长（像素）

    返回:
      dict:
        {
          'A_left':  (xL_A, yA),
          'A_right': (xR_A, yA),
          'B_left':  (xL_B, yB),
          'B_right': (xR_B, yB),
        }
    """
    if bev_mask.ndim != 2:
        raise ValueError(f"bev_mask must be 2D single-channel, got shape={bev_mask.shape}")
    H, W = bev_mask.shape

    IN_SET  = {1, 2}   # 认为在“场内/允许区域”
    OUT_SET = {0, 3}   # 认为“越界/场外”，用于粗搜索截断
    TARGET_IN, TARGET_OUT = 2, 3  # 最终要定位到 2/3 边界

    def _sample(x: int, y: int) -> int:
        """越界视作 0（背景），保证稳健性。"""
        if 0 <= x < W and 0 <= y < H:
            return int(bev_mask[y, x])
        return 0

    def _ensure_inside_start(x0: int, y: int):
        """要求起点在 {1,2} 内，否则报错（按需求也可改成就近微调）。"""
        v0 = _sample(x0, y)
        if v0 not in IN_SET:
            raise ValueError(f"Start ({x0},{y}) not in classes {IN_SET}, got {v0}.")
        return v0

    def _bracket_with_stepping(x0: int, y: int, dir_sign: int, step_pix: int):
        """
        从 x0 沿 dir_sign（+1 右，-1 左）粗步进，直到从 IN_SET 跨到 OUT_SET，返回一个 (x_in, x_out) 括区。
        x_in 处于 IN_SET，x_out 处于 OUT_SET，且两者按数值并不保证 x_in < x_out。
        """
        prev_x = x0
        prev_v = _sample(prev_x, y)
        if prev_v not in IN_SET:
            raise ValueError(f"Row {y}: start x={x0} not in {IN_SET}, v={prev_v}")

        # 最多尝试 W//step 次，避免死循环
        for _ in range(max(1, (W + step_pix - 1) // step_pix + 2)):
            # 试探下一个点
            x = prev_x + dir_sign * step_pix
            # 防止跨出很远：夹到边界，仍然继续判断
            x = max(0, min(W - 1, x))
            v = _sample(x, y)

            # 出现 IN→OUT（3或0）即形成括区
            if v in OUT_SET:
                # prev_x 属于 IN，x 属于 OUT
                return prev_x, x

            # 如果已经到边界仍未越界，说明这一侧找不到
            if (dir_sign > 0 and x == W - 1) or (dir_sign < 0 and x == 0):
                break

            prev_x, prev_v = x, v

        raise ValueError(f"Row {y}: cannot find IN→OUT transition from x={x0} dir={dir_sign} with step={step_pix}.")

    def _bisect_to_edge(x_in: int, x_out: int, y: int):
        """
        在 x_in(∈IN_SET) 与 x_out(∈OUT_SET) 之间二分，直到两者相邻 (|x_out - x_in| == 1)。
        返回 (x_in_final, x_out_final)。
        """
        # 为了二分，使用左右有序的下标，但同时保留哪边是IN、哪边是OUT的信息。
        a, b = x_in, x_out
        # 二分次数上限：log2(W) ~ 12 for 4096，留冗余
        for _ in range(32):
            if abs(b - a) <= 1:
                break
            mid = (a + b) // 2 if a < b else (b + a) // 2
            vmid = _sample(mid, y)
            if vmid in IN_SET:
                # mid 属于 IN，应向 OUT 方向靠拢
                # 让 a 表示“IN侧”的点，b 表示“OUT侧”的点
                # 维持 a 是 IN、b 是 OUT 的语义
                if a < b:
                    a = mid
                else:
                    a = mid
            else:
                # mid 属于 OUT
                if a < b:
                    b = mid
                else:
                    b = mid

        # 确保返回顺序保持 “in, out”
        if _sample(a, y) in IN_SET and _sample(b, y) in OUT_SET:
            return a, b
        if _sample(b, y) in IN_SET and _sample(a, y) in OUT_SET:
            return b, a
        # 理论上不应发生：若发生说明该行并非单调边界
        raise ValueError(f"Row {y}: bisection failed to maintain IN/OUT near ({x_in},{x_out}).")

    def _refine_to_23(x_in: int, x_out: int, y: int, dir_sign: int):
        """
        已有相邻括区 (x_in ∈ IN_SET, x_out ∈ OUT_SET, |x_out - x_in|=1)，
        进一步确保是 2/3 的组合，并返回边界点（最后一个2）。
        - 右侧边界(dir=+1)：寻找 xi 使 v[xi]==2 且 v[xi+1]==3，优先从靠近 x_out 一侧向内查找
        - 左侧边界(dir=-1)：寻找 xi 使 v[xi]==2 且 v[xi-1]==3，优先从靠近 x_out 一侧向内查找
        若周边不是 2/3（而是 2/0、1/3 等），在一个很小窗口内再扫几步；仍找不到则报错。
        """
        # 将搜索窗口控制在很小范围（防止奇异情况）；但通常 |x_out - x_in|==1，下面循环几乎 O(1)
        window = 8
        if dir_sign > 0:
            # 从右界往左找第一个 2/3
            start = min(x_out, x_in + window)
            end   = max(x_in,   x_out - window)
            # 右→左
            for xi in range(start, end - 1, -1):
                v0 = _sample(xi, y)
                v1 = _sample(min(xi + 1, W - 1), y)
                if v0 == TARGET_IN and v1 == TARGET_OUT:
                    return xi, y
        else:
            # 从左界往右找第一个 2/3
            start = max(x_out, x_in - window)
            end   = min(x_in,  x_out + window)
            # 左→右
            for xi in range(start, end + 1):
                v0 = _sample(xi, y)
                v1 = _sample(max(xi - 1, 0), y)
                if v0 == TARGET_IN and v1 == TARGET_OUT:
                    return xi, y

        raise ValueError(
            f"Row {y}: could not refine to 2/3 boundary near IN={x_in}, OUT={x_out}, dir={dir_sign}."
        )

    def _solve_for_point(P_xy: tuple):
        """对单个点 P，返回(P_left, P_right) 两个边界点。"""
        x0 = int(round(P_xy[0]))
        y0 = int(round(P_xy[1]))
        if not (0 <= x0 < W and 0 <= y0 < H):
            raise ValueError(f"Point {P_xy} out of image bounds ({W}x{H}).")
        _ensure_inside_start(x0, y0)

        # 右侧：+x
        xin, xout = _bracket_with_stepping(x0, y0, dir_sign=+1, step_pix=max(1, int(step)))
        xin, xout = _bisect_to_edge(xin, xout, y0)
        P_right   = _refine_to_23(xin, xout, y0, dir_sign=+1)

        # 左侧：-x
        xin, xout = _bracket_with_stepping(x0, y0, dir_sign=-1, step_pix=max(1, int(step)))
        xin, xout = _bisect_to_edge(xin, xout, y0)
        P_left    = _refine_to_23(xin, xout, y0, dir_sign=-1)

        return P_left, P_right

    A_left, A_right = _solve_for_point(A_xy)
    B_left, B_right = _solve_for_point(B_xy)

    return {
        "A_left":  A_left,
        "A_right": A_right,
        "B_left":  B_left,
        "B_right": B_right,
    }
