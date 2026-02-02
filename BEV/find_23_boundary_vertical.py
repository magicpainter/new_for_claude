# find_23_boundary_vertical.py
import numpy as np

def find_23_boundary_vertical(bev_mask: np.ndarray,
                              A_xy: tuple,
                              B_xy: tuple,
                              step: int = 100):
    """
    从 A、B 的中点 C 出发，沿“垂直向下（+y）”在 BEV mask 上搜索 2/3 边界并返回 D 点。
    搜索逻辑复用 find_23_boundary_points.py 的思路：粗步进 → 二分 → 2/3 精修。

    设定：
      IN_SET  = {1,2}     认为是“场内/允许区域”
      OUT_SET = {0,3}     认为是“越界/场外”
      目标边界：mask[y*, x] == 2 且 mask[y*+1, x] == 3
               即“最后一个 2 的像素”，它的正下方是 3

    参数:
      bev_mask: (H,W) 单通道整型语义图（uint8/int32等）
      A_xy, B_xy: 两个输入点 (x, y)
      step: 粗搜索的步长（像素），默认 100

    返回:
      { 'A': (xA, yA), 'B': (xB, yB), 'D': (xD, yD) }
    """
    if bev_mask.ndim != 2:
        raise ValueError(f"bev_mask must be 2D single-channel, got shape={bev_mask.shape}")
    H, W = bev_mask.shape

    # 统一坐标为整数像素
    Ax, Ay = int(round(A_xy[0])), int(round(A_xy[1]))
    Bx, By = int(round(B_xy[0])), int(round(B_xy[1]))

    # 计算 C（取整用于采样）
    Cx = int(round((Ax + Bx) * 0.5))
    Cy = int(round((Ay + By) * 0.5))

    # 越界检查
    def _in_bounds(x, y): return (0 <= x < W and 0 <= y < H)
    if not _in_bounds(Ax, Ay):
        raise ValueError(f"Point A out of bounds: A={A_xy} for size W={W}, H={H}")
    if not _in_bounds(Bx, By):
        raise ValueError(f"Point B out of bounds: B={B_xy} for size W={W}, H={H}")
    if not _in_bounds(Cx, Cy):
        raise ValueError(f"Point C out of bounds: C=({Cx},{Cy}) derived from A,B")

    IN_SET  = {1, 2}
    OUT_SET = {0, 3}
    TARGET_IN, TARGET_OUT = 2, 3  # 需要最终锁定 2/3 的邻接

    def _sample(x: int, y: int) -> int:
        """越界一律视作 0，保证稳健。"""
        if 0 <= x < W and 0 <= y < H:
            return int(bev_mask[y, x])
        return 0

    # 要求起点 C 在 {1,2} 内，若不在则直接报错（如需更智能可改成向上回退若干像素寻找 in 区）
    vC = _sample(Cx, Cy)
    if vC not in IN_SET:
        raise ValueError(f"Start C=({Cx},{Cy}) not in {IN_SET}, got {vC}.")

    def _bracket_with_stepping_vertical(x: int, y0: int, step_pix: int):
        """
        垂直向下粗步进，直到首次出现 IN→OUT，返回 (y_in, y_out)。
        其中 y_in ∈ IN_SET，y_out ∈ OUT_SET，且通常 y_out > y_in。
        """
        prev_y = y0
        prev_v = _sample(x, prev_y)
        if prev_v not in IN_SET:
            raise ValueError(f"Col x={x}: start y={y0} not in {IN_SET}, v={prev_v}")

        max_trials = max(1, (H + step_pix - 1) // step_pix + 2)
        for _ in range(max_trials):
            y = prev_y + step_pix
            y = min(H - 1, y)  # 向下不越过底部
            v = _sample(x, y)
            if v in OUT_SET:
                return prev_y, y  # prev_y ∈ IN, y ∈ OUT → 形成括区
            if y == H - 1:
                break
            prev_y, prev_v = y, v

        raise ValueError(f"Col x={x}: cannot find IN→OUT transition from y={y0} with step={step_pix}.")

    def _bisect_to_edge_vertical(x: int, y_in: int, y_out: int):
        """
        在 y_in(∈IN_SET) 与 y_out(∈OUT_SET) 之间二分，直到相邻 (|y_out - y_in| == 1)，
        返回 (y_in_final, y_out_final)。
        """
        a, b = y_in, y_out
        for _ in range(32):  # log2(4096)≈12，留足余量
            if abs(b - a) <= 1:
                break
            mid = (a + b) // 2 if a < b else (b + a) // 2
            vmid = _sample(x, mid)
            if vmid in IN_SET:
                a = mid
            else:
                b = mid

        if _sample(x, a) in IN_SET and _sample(x, b) in OUT_SET:
            return a, b
        if _sample(x, b) in IN_SET and _sample(x, a) in OUT_SET:
            return b, a
        raise ValueError(f"Col x={x}: bisection failed near (y_in={y_in}, y_out={y_out}).")

    def _refine_to_23_vertical(x: int, y_in: int, y_out: int):
        """
        已有相邻括区 (y_in ∈ IN_SET, y_out ∈ OUT_SET, |y_out - y_in|=1)，
        进一步确保是 2/3 的组合，并返回边界点 D=(x, y*):
          mask[y*, x] == 2 且 mask[y*+1, x] == 3
        若局部是 2/0 或 1/3 等，允许在很小窗口内就近扫描，仍无则报错。
        """
        window = 8
        # 优先从接近 OUT 的一侧向内检查
        y_start = min(y_out, y_in + window)
        y_end   = max(y_in,  y_out - window)
        for yi in range(y_start, y_end - 1, -1):
            v0 = _sample(x, yi)
            v1 = _sample(x, min(yi + 1, H - 1))
            if v0 == TARGET_IN and v1 == TARGET_OUT:
                return (x, yi)
        # 如果上面的从 OUT 往回找没命中，再做一遍向下遍历的保险
        for yi in range(max(y_out - window, 0), min(y_in + window, H - 2) + 1):
            v0 = _sample(x, yi)
            v1 = _sample(x, yi + 1)
            if v0 == TARGET_IN and v1 == TARGET_OUT:
                return (x, yi)

        raise ValueError(f"Col x={x}: could not refine to 2/3 boundary near (y_in={y_in}, y_out={y_out}).")

    # —— 主流程：只在 C 的列 x=Cx 上，向“下”找 2/3 垂直边界 —— #
    y_in, y_out = _bracket_with_stepping_vertical(Cx, Cy, step_pix=max(1, int(step)))
    y_in, y_out = _bisect_to_edge_vertical(Cx, y_in, y_out)
    Dx, Dy = _refine_to_23_vertical(Cx, y_in, y_out)

    return {
        "A": (Ax, Ay),
        "B": (Bx, By),
        "D": (Dx, Dy),
    }
