# calibrations.py
# Language and calibration settings

import json
from pathlib import Path

# Settings file path
_SETTINGS_FILE = Path(__file__).parent / "user_settings.json"

# Default language setting (False = English, True = Chinese/Pinyin)
f_pinyin = False

# OD overlay toggle (False = off, True = show detection/track overlay)
f_od_overlay = False


def _load_settings():
    """Load settings from file."""
    global f_pinyin, f_od_overlay
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                f_pinyin = settings.get('f_pinyin', False)
                f_od_overlay = settings.get('f_od_overlay', False)
        except Exception as e:
            print(f"[calibrations] Warning: Failed to load settings: {e}")


def _save_settings():
    """Save settings to file."""
    try:
        settings = {
            'f_pinyin': f_pinyin,
            'f_od_overlay': f_od_overlay
        }
        with open(_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"[calibrations] Warning: Failed to save settings: {e}")


def get_language() -> bool:
    """Get current language setting.

    Returns:
        True for Chinese, False for English
    """
    return f_pinyin


def set_language(use_chinese: bool):
    """Set language preference and save to file.

    Args:
        use_chinese: True for Chinese, False for English
    """
    global f_pinyin
    f_pinyin = use_chinese
    _save_settings()


def toggle_language() -> bool:
    """Toggle language between English and Chinese.

    Returns:
        New language setting (True for Chinese, False for English)
    """
    global f_pinyin
    f_pinyin = not f_pinyin
    _save_settings()
    return f_pinyin


def get_od_overlay() -> bool:
    """Get current OD overlay setting.

    Returns:
        True if OD overlay is enabled, False otherwise
    """
    return f_od_overlay


def set_od_overlay(enabled: bool):
    """Set OD overlay preference and save to file.

    Args:
        enabled: True to enable OD overlay, False to disable
    """
    global f_od_overlay
    f_od_overlay = enabled
    _save_settings()


def toggle_od_overlay() -> bool:
    """Toggle OD overlay on/off.

    Returns:
        New OD overlay setting (True = enabled, False = disabled)
    """
    global f_od_overlay
    f_od_overlay = not f_od_overlay
    _save_settings()
    return f_od_overlay


# Load settings on module import
_load_settings()
