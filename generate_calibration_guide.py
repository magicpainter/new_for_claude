#!/usr/bin/env python3
"""Generate calibration guide image showing points 1-5 on a basketball half-court."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_calibration_guide(width=800, height=600, save_path="calibration_guide.png"):
    """Generate a reference image showing calibration point locations."""

    # Create image with transparent background
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Semi-transparent dark background
    img[:, :, 3] = 180  # Alpha
    img[:, :, 0:3] = 40  # Dark gray

    # Colors (BGRA)
    white = (255, 255, 255, 255)
    yellow = (0, 255, 255, 255)
    green = (0, 255, 0, 255)
    cyan = (255, 255, 0, 255)

    # Court dimensions (scaled to fit image)
    margin = 50
    court_bottom = height - margin - 30  # Leave room for labels
    court_top = margin + 80
    court_left = margin + 100
    court_right = width - margin - 100
    court_center_x = width // 2

    # Paint/Lane dimensions
    paint_width = 180
    paint_height = 200
    paint_left = court_center_x - paint_width // 2
    paint_right = court_center_x + paint_width // 2
    paint_top = court_top + 60
    paint_bottom = paint_top + paint_height

    # Free throw line Y position
    ft_line_y = paint_bottom

    # 3-point line corners
    corner_y = court_bottom - 20
    corner_left_x = court_left + 50
    corner_right_x = court_right - 50

    # Rim position
    rim_y = court_top + 30
    rim_x = court_center_x

    # Draw court lines (white, semi-transparent)
    line_color = (255, 255, 255, 200)

    # Baseline
    cv2.line(img, (court_left, court_bottom), (court_right, court_bottom), line_color, 2)

    # Paint/Lane rectangle
    cv2.rectangle(img, (paint_left, paint_top), (paint_right, paint_bottom), line_color, 2)

    # Free throw line (already part of paint bottom)

    # 3-point line (simplified as lines from corners curving to top)
    # Left side
    pts_left = np.array([
        [corner_left_x, corner_y],
        [corner_left_x - 20, corner_y - 100],
        [paint_left - 80, ft_line_y],
        [paint_left - 60, paint_top + 50],
    ], np.int32)
    cv2.polylines(img, [pts_left], False, line_color, 2)

    # Right side
    pts_right = np.array([
        [corner_right_x, corner_y],
        [corner_right_x + 20, corner_y - 100],
        [paint_right + 80, ft_line_y],
        [paint_right + 60, paint_top + 50],
    ], np.int32)
    cv2.polylines(img, [pts_right], False, line_color, 2)

    # Rim circle
    cv2.circle(img, (rim_x, rim_y), 25, line_color, 2)

    # Backboard
    cv2.line(img, (rim_x - 50, rim_y - 30), (rim_x + 50, rim_y - 30), line_color, 3)

    # Define calibration points
    points = {
        1: (corner_left_x, corner_y, "1", "Left Corner\n左底角"),
        2: (corner_right_x, corner_y, "2", "Right Corner\n右底角"),
        3: (paint_right, ft_line_y, "3", "Right Elbow\n右肘区"),
        4: (paint_left, ft_line_y, "4", "Left Elbow\n左肘区"),
        5: (rim_x, rim_y, "5", "Rim Center\n篮筐中心"),
    }

    # Draw calibration points
    point_radius = 20
    for num, (x, y, label, desc) in points.items():
        # Outer circle (yellow)
        cv2.circle(img, (x, y), point_radius, yellow, 3)
        # Inner filled circle
        cv2.circle(img, (x, y), point_radius - 5, (0, 100, 100, 200), -1)

    # Convert to PIL for text rendering (better font support)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    draw = ImageDraw.Draw(img_pil)

    # Try to load CJK font that supports both English and Chinese
    cjk_font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]

    cjk_font_path = None
    for path in cjk_font_paths:
        try:
            ImageFont.truetype(path, 12)
            cjk_font_path = path
            break
        except:
            pass

    if cjk_font_path:
        font_large = ImageFont.truetype(cjk_font_path, 24)
        font_small = ImageFont.truetype(cjk_font_path, 14)
        font_title = ImageFont.truetype(cjk_font_path, 28)
        font_cjk = ImageFont.truetype(cjk_font_path, 12)
    else:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_cjk = font_small

    # Draw point numbers
    for num, (x, y, label, desc) in points.items():
        # Center the number in the circle
        bbox = draw.textbbox((0, 0), label, font=font_large)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x - tw//2, y - th//2 - 2), label, fill=(255, 255, 255, 255), font=font_large)

    # Draw labels near points
    label_offsets = {
        1: (-80, 20),
        2: (30, 20),
        3: (30, -10),
        4: (-120, -10),
        5: (35, -15),
    }

    for num, (x, y, label, desc) in points.items():
        ox, oy = label_offsets[num]
        lines = desc.split('\n')
        for i, line in enumerate(lines):
            # Use CJK font for all labels (supports both English and Chinese)
            draw.text((x + ox, y + oy + i * 18), line, fill=(255, 255, 0, 255), font=font_small)

    # Title
    title = "Calibration Points / 标定点位置"
    bbox = draw.textbbox((0, 0), title, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text((width//2 - tw//2, 15), title, fill=(255, 255, 255, 255), font=font_title)

    # Instructions at bottom
    inst = "Drag points to match court lines"
    bbox = draw.textbbox((0, 0), inst, font=font_small)
    tw = bbox[2] - bbox[0]
    draw.text((width//2 - tw//2, height - 25), inst, fill=(200, 200, 200, 255), font=font_small)

    # Convert back to OpenCV format and save
    img_final = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(save_path, img_final)
    print(f"Saved calibration guide to: {save_path}")

    return save_path


if __name__ == "__main__":
    generate_calibration_guide(save_path="UI_Button/calibration_guide.png")
