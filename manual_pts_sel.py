import cv2
import os
import numpy as np  # If this errors, change to: import numpy as np
from calibrations import f_pinyin 
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

def _load_cjk_font(size: int):
    for p in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    raise RuntimeError("CJK font not found. Install: sudo apt-get install fonts-noto-cjk")

def cv2_putText_cn(img_bgr, text, org_xy, font_size=24, bgr=(0,255,255)):
    # OpenCV BGR -> PIL RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = _load_cjk_font(font_size)
    # PIL uses RGB color
    draw.text(org_xy, text, font=font, fill=(bgr[2], bgr[1], bgr[0]))

    # Back to OpenCV BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def select_critical_zone_corners(
    img_bgr,
    DEFAULT_CORNERS,
    init_points=None,
    circle_radius=20,
    dot_radius=4,
    mag_scale=3,
    inset_size=180,
    inset_margin=14,
    inset_bg_alpha=0.25,
    btn_margin=14,            # distance from window edges
    btn_size=(150, 46),       # (width, height) of the ENTER button
    image_path=None           # path to image file for refresh capability
):
    """
    Interactively place 4 corners on `img_bgr`.

    Controls:
      • Drag a circle to move it.
      • Click ENTER button (top-right) to finish.
      • Keys: [1-5] select point • r reset • s/Enter save • q/Esc quit • f refresh image
    """
    img = img_bgr.copy()
    h, w = img.shape[:2]
    pts = np.array(init_points if init_points is not None else DEFAULT_CORNERS, dtype=np.int32)

    dragging_idx = None
    active_idx = 0
    finished = False
    hover_enter_btn = False
    hover_refresh_btn = False
    win_main = "Select 4 Corners and Hoop location"

    # inset placement (top-left)
    ix0 = inset_margin
    iy0 = inset_margin
    ix1 = ix0 + inset_size
    iy1 = iy0 + inset_size

    # ENTER button placement (top-right)
    bw, bh = btn_size
    bx1 = w - btn_margin
    by0 = inset_margin  # align with zoom inset top
    bx0 = max(0, bx1 - bw)
    by1 = by0 + bh
    btn_rect = (bx0, by0, bx1, by1)

    # REFRESH button placement (below ENTER button, only shown if image_path provided)
    rbx0 = bx0
    rby0 = by1 + 10  # 10px gap below ENTER button
    rbx1 = bx1
    rby1 = rby0 + bh
    refresh_btn_rect = (rbx0, rby0, rbx1, rby1)

    def clamp_point(p):
        x = int(np.clip(p[0], 0, w - 1))
        y = int(np.clip(p[1], 0, h - 1))
        return (x, y)

    def nearest_point_idx(x, y):
        d2 = np.sum((pts - np.array([x, y]))**2, axis=1)
        i = int(np.argmin(d2))
        return i, float(np.sqrt(d2[i]))

    def draw_targets(canvas):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x), int(y)), circle_radius, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), dot_radius, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{i+1}", (int(x)+circle_radius+6, int(y)+5),
                        font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    def render_inset(frame, idx):
        x, y = pts[idx]
        roi_half = max(8, inset_size // (2 * mag_scale))
        x0, y0 = max(0, x - roi_half), max(0, y - roi_half)
        x1, y1 = min(w, x + roi_half), min(h, y + roi_half)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            return

        zoom = cv2.resize(roi, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST)

        cz = inset_size // 2
        cv2.circle(zoom, (cz, cz), circle_radius * mag_scale, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(zoom, (cz, cz), dot_radius * mag_scale, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(zoom, (cz, 0), (cz, inset_size), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(zoom, (0, cz), (inset_size, cz), (200, 200, 200), 1, cv2.LINE_AA)

        overlay = frame.copy()
        cv2.rectangle(overlay, (ix0 - 6, iy0 - 6), (ix1 + 6, iy1 + 6), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, inset_bg_alpha, frame, 1 - inset_bg_alpha, 0, frame)

        frame[iy0:iy1, ix0:ix1] = zoom
        cv2.rectangle(frame, (ix0 - 1, iy0 - 1), (ix1 + 1, iy1 + 1), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Zoom: point {idx+1}", (ix0, iy1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def render_button(frame, hovering=False):
        (x0, y0, x1, y1) = btn_rect
        # background
        color_bg = (40, 40, 40) if not hovering else (70, 70, 70)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color_bg, -1, cv2.LINE_AA)
        # border
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2, cv2.LINE_AA)
        # text - use Chinese if f_pinyin
        label = "确认" if f_pinyin else "ENTER"
        if f_pinyin:
            # Use PIL for Chinese text
            frame_temp = cv2_putText_cn(frame, label, (x0 + 45, y0 + 8), font_size=28, bgr=(255, 255, 255))
            frame[:] = frame_temp
        else:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            tx = x0 + (x1 - x0 - tw) // 2
            ty = y0 + (y1 - y0 + th) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # check icon
        cv2.line(frame, (x1 - 28, y0 + 14), (x1 - 18, y0 + 26), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (x1 - 18, y0 + 26), (x1 - 8, y0 + 8), (0, 255, 255), 2, cv2.LINE_AA)

    def render_refresh_button(frame, hovering=False):
        """Render the refresh button (only if image_path is provided)."""
        if not image_path:
            return
        (x0, y0, x1, y1) = refresh_btn_rect
        # background - blue tint for refresh
        color_bg = (60, 40, 40) if not hovering else (90, 60, 60)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color_bg, -1, cv2.LINE_AA)
        # border - cyan
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 2, cv2.LINE_AA)
        # text - use Chinese if f_pinyin
        label = "刷新画面" if f_pinyin else "REFRESH"
        if f_pinyin:
            frame_temp = cv2_putText_cn(frame, label, (x0 + 25, y0 + 8), font_size=28, bgr=(255, 255, 255))
            frame[:] = frame_temp
        else:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx = x0 + (x1 - x0 - tw) // 2
            ty = y0 + (y1 - y0 + th) // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def point_in_button(x, y):
        x0, y0, x1, y1 = btn_rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def point_in_refresh_button(x, y):
        if not image_path:
            return False
        x0, y0, x1, y1 = refresh_btn_rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def do_refresh():
        """Trigger image refresh from GUI."""
        nonlocal img, h, w
        if not image_path:
            return False
        refresh_signal = image_path + ".refresh"
        done_signal = image_path + ".done"
        try:
            # Remove any existing done signal
            if os.path.exists(done_signal):
                os.remove(done_signal)
            # Create refresh request
            with open(refresh_signal, 'w') as f:
                f.write("refresh")
            # Wait for GUI to create the done signal
            import time
            for i in range(100):  # Wait up to 5 seconds
                time.sleep(0.05)
                if os.path.exists(done_signal):
                    os.remove(done_signal)
                    # Wait for file to be fully written and try loading with retries
                    for retry in range(5):
                        time.sleep(0.15)
                        if os.path.exists(image_path):
                            new_img = cv2.imread(image_path)
                            if new_img is not None and new_img.size > 0:
                                img = new_img.copy()
                                h, w = img.shape[:2]
                                return True
                    break
            # Clean up signal file
            if os.path.exists(refresh_signal):
                os.remove(refresh_signal)
        except Exception:
            pass
        return False

    def on_mouse(event, x, y, flags, userdata):
        nonlocal dragging_idx, active_idx, finished, hover_enter_btn, hover_refresh_btn
        hover_enter_btn = point_in_button(x, y)
        hover_refresh_btn = point_in_refresh_button(x, y)

        if event == cv2.EVENT_MOUSEMOVE:
            if dragging_idx is not None:
                pts[dragging_idx] = clamp_point((x, y))
                active_idx = dragging_idx

        elif event == cv2.EVENT_LBUTTONDOWN:
            if point_in_button(x, y):
                finished = True
                return
            if point_in_refresh_button(x, y):
                do_refresh()
                return
            i, d = nearest_point_idx(x, y)
            if d <= circle_radius * 2.0:
                dragging_idx = i
                active_idx = i

        elif event == cv2.EVENT_LBUTTONUP:
            if dragging_idx is not None:
                pts[dragging_idx] = clamp_point((x, y))
            dragging_idx = None

    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_main, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(win_main, on_mouse)

    while True:
        frame = img.copy()
        draw_targets(frame)
        y_offset = 400
        if f_pinyin:
            hud = "拖动 1–2 到三分线—底线交点（左/右底角)"
            frame = cv2_putText_cn(frame, hud, (950, 500+y_offset), font_size=32, bgr=(0,255,255))
        else:
            hud = "Drag 1 - 2 to 3PT-baseline intersection (left/right corner)"
            cv2.putText(frame, hud, (950, 500+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hud, (950, 500+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 1, cv2.LINE_AA)

        if f_pinyin:
            hud = "拖动 3–4 到罚球线与禁区线交点（右/左肘区)"
            frame = cv2_putText_cn(frame, hud, (950, 540+y_offset), font_size=32, bgr=(0,255,255))
        else:
            hud = "Drag 3 - 4 to FT-lane intersection (right/left elbow)"
            cv2.putText(frame, hud, (950, 540+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hud, (950, 540+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 1, cv2.LINE_AA)

        if f_pinyin:
            hud = "拖动 5 到篮筐中心, 点击 [确认] 完成"
            frame = cv2_putText_cn(frame, hud, (950, 580+y_offset), font_size=32, bgr=(0,255,255))
        else:
            hud = "Drag 5 to rim center, click [ENTER] to finish"
            cv2.putText(frame, hud, (950, 580+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, hud, (950, 580+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 1, cv2.LINE_AA)

        # zoom inset + buttons
        render_inset(frame, active_idx)
        render_button(frame, hovering=hover_enter_btn)
        render_refresh_button(frame, hovering=hover_refresh_btn)

        cv2.imshow(win_main, frame)

        if finished:
            cv2.destroyWindow(win_main)
            return [(int(x), int(y)) for (x, y) in pts.tolist()]

        key = cv2.waitKey(16) & 0xFF
        if key in (ord('q'), 27):
            cv2.destroyWindow(win_main)
            return [(int(x), int(y)) for (x, y) in pts.tolist()]
        elif key in (ord('s'), 13):
            cv2.destroyWindow(win_main)
            return [(int(x), int(y)) for (x, y) in pts.tolist()]
        elif key == ord('r'):
            pts = np.array(DEFAULT_CORNERS, dtype=np.int32)
            active_idx = 0
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            active_idx = int(chr(key)) - 1
        elif key in (ord('f'), ord('F')) and image_path:
            # Use the same refresh logic as the button
            do_refresh()

# --- Example usage ---
if __name__ == "__main__":
    # replace with your own image path as needed
    path = "2025-11-07_09-28-1000000009.png"
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Failed to load image: {path}")

    DEFAULT_CORNERS = np.array([[986, 382], [1685, 506], [1104, 484], [839, 432], [1176, 179]]) # left
    # DEFAULT_CORNERS = [(960, 448), (584, 497), (909, 541), (1283, 472)]  # right
    corners = select_critical_zone_corners(img, DEFAULT_CORNERS)
    print("Selected corners:", corners)
