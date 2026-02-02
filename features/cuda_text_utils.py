"""Render Unicode text (including CJK / Chinese) onto jetson_utils cudaImages.

cudaFont.OverlayText() only supports ASCII.  This module uses Pillow to
rasterise arbitrary Unicode text into an RGBA numpy array, converts it to a
cudaImage via cudaAllocMapped, and (optionally) overlays it onto a destination
cudaImage.  A per-string cache avoids re-rendering identical text every frame.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from jetson_utils import cudaAllocMapped, cudaOverlay, cudaToNumpy

# ── font discovery ──────────────────────────────────────────────────────────
_FONT_SEARCH_PATHS = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
]

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for path in _FONT_SEARCH_PATHS:
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                return _font_cache[size]
            except (OSError, IOError):
                continue
        _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


# ── rendering ───────────────────────────────────────────────────────────────
_cuda_cache: dict[tuple, object] = {}


def render_text_cuda(text, font_size=40,
                     text_color=(255, 255, 0, 255),
                     bg_color=(0, 0, 0, 160),
                     padding=6):
    """Render *text* into an RGBA cudaImage (no caching)."""
    font = _get_font(font_size)
    left, top, right, bottom = font.getbbox(text)
    w = right - left + 2 * padding
    h = bottom - top + 2 * padding

    bg = tuple(bg_color) if bg_color else (0, 0, 0, 0)
    img = Image.new('RGBA', (w, h), bg)
    draw = ImageDraw.Draw(img)

    if len(text_color) == 3:
        text_color = (*text_color, 255)
    draw.text((padding - left, padding - top), text, font=font, fill=text_color)

    rgba = np.ascontiguousarray(np.array(img, dtype=np.uint8))
    # Use cudaAllocMapped with explicit format to ensure compatibility with cudaOverlay
    cuda_img = cudaAllocMapped(width=w, height=h, format='rgba8')
    # Get numpy view of mapped memory and copy data into it
    cuda_np = cudaToNumpy(cuda_img)
    np.copyto(cuda_np, rgba)
    return cuda_img


def get_text_cuda(text, font_size=40,
                  text_color=(255, 255, 0, 255),
                  bg_color=(0, 0, 0, 160)):
    """Cached version of :func:`render_text_cuda`."""
    key = (text, font_size, text_color,
           tuple(bg_color) if bg_color else None)
    if key not in _cuda_cache:
        _cuda_cache[key] = render_text_cuda(text, font_size, text_color,
                                            bg_color)
    return _cuda_cache[key]


# ── convenience overlay helpers ─────────────────────────────────────────────

def overlay_text(cuda_img, text, x, y, font_size=40,
                 text_color=(255, 255, 0, 255),
                 bg_color=(0, 0, 0, 160)):
    """Render *text* and composite it onto *cuda_img* at *(x, y)*.

    Returns the rendered cudaImage so callers can inspect ``.width`` /
    ``.height`` if needed.
    """
    text_img = get_text_cuda(text, font_size, text_color, bg_color)
    cudaOverlay(text_img, cuda_img, int(x), int(y))
    return text_img


def overlay_text_centered(cuda_img, text, y, screen_w, font_size=40,
                          text_color=(255, 255, 0, 255),
                          bg_color=(0, 0, 0, 160)):
    """Like :func:`overlay_text` but horizontally centred on *screen_w*."""
    text_img = get_text_cuda(text, font_size, text_color, bg_color)
    x = (screen_w - text_img.width) // 2
    cudaOverlay(text_img, cuda_img, int(x), int(y))
    return text_img
