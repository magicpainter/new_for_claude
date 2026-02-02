# ai_advice_popup.py
# Popup dialog to display output from compute_AI_advice.compute_shooting_ai_advice()

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from PyQt5 import QtCore, QtWidgets


class AiAdviceBridge(QtCore.QObject):
    """
    Thread-safe bridge: emit `show_advice` from ANY thread.
    The connected slot will run on the GUI thread if you connect with Qt.QueuedConnection.
    """
    # (result_dict, accuracy_percent, f_pinyin, side)
    # side: 'left', 'right', or 'single' for non-both mode
    show_advice = QtCore.pyqtSignal(dict, float, bool, str)


def _t(f_pinyin: bool, en: str, zh: str) -> str:
    """Tiny translator: if f_pinyin=True, use Chinese string."""
    return zh if f_pinyin else en


def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _escape(s: Any) -> str:
    # QTextBrowser supports basic HTML; escape user strings.
    if s is None:
        return ""
    t = str(s)
    return (
        t.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def _compact_metrics(metrics: Dict[str, Any], max_items: int = 6) -> List[Tuple[str, str]]:
    """
    Make metrics readable; avoid dumping huge dicts.
    """
    if not isinstance(metrics, dict) or not metrics:
        return []
    items: List[Tuple[str, str]] = []
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            items.append((k, f"{v:.4g}"))
        else:
            items.append((k, _escape(v)))
        if len(items) >= max_items:
            break
    return items


def _build_html(result: Dict[str, Any], accuracy_pct: float, f_pinyin: bool = False) -> str:
    ok = bool((result or {}).get("ok", False))
    n = int((result or {}).get("n", 0) or 0)
    summary = (result or {}).get("summary", {}) or {}
    advice = (result or {}).get("advice", []) or []
    is_layup = bool((result or {}).get("is_layup", False))

    reason = ""
    if not ok:
        reason = _escape((summary or {}).get("reason", ""))

    # Quick summary nuggets (optional)
    flat_rate = _as_float((summary or {}).get("flat_rate", float("nan")))
    moving_rate = _as_float((summary or {}).get("moving_rate", float("nan")))

    nuggets = []
    if flat_rate == flat_rate:   # not NaN
        nuggets.append(f"flat_rate={flat_rate:.0%}")
    if moving_rate == moving_rate:
        nuggets.append(f"moving_rate={moving_rate:.0%}")
    nugget_txt = (" | " + " | ".join(nuggets)) if nuggets else ""

    # Labels
    if is_layup:
        lbl_hdr = _t(f_pinyin, "Layup AI Advice", "上篮AI建议")
    else:
        lbl_hdr = _t(f_pinyin, "Shooting AI Advice", "投篮AI建议")
    lbl_acc = _t(f_pinyin, "Accuracy", "命中率")
    lbl_shots = _t(f_pinyin, "Shots", "出手")
    lbl_ok = _t(f_pinyin, "OK", "通过")
    lbl_not_enough = _t(f_pinyin, "Not enough samples", "样本不足")
    lbl_no_advice = _t(f_pinyin, "No advice generated", "暂无建议")
    lbl_collect_more = _t(f_pinyin, "Try collecting more shots, then rerun the session.", "请多采集一些投篮数据后再运行。")
    lbl_what_to_do = _t(f_pinyin, "What to do:", "怎么做：")

    html = []
    html.append(f"""
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial,
                     "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei", "Microsoft YaHei",
                     sans-serif;
        font-size: 24px;
      }}
      .hdr {{ font-size: 36px; font-weight: 700; margin-bottom: 12px; }}
      .sub {{ font-size: 32px; font-weight: 700; color: #00ff00; margin-bottom: 20px; text-shadow: 0 2px 4px #000; }}
      .pill {{ display: inline-block; padding: 4px 16px; border-radius: 999px; font-size: 24px; margin-left: 16px; }}
      .pill-ok {{ background: #e8f5e9; color: #1b5e20; border: 1px solid #c8e6c9; }}
      .pill-warn {{ background: #fff3e0; color: #e65100; border: 1px solid #ffe0b2; }}
      .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 20px 24px; margin: 20px 0; }}
      .title {{ font-size: 32px; font-weight: 700; margin: 0 0 12px 0; }}
      .why {{ margin: 0 0 16px 0; font-size: 24px; }}
      .cue {{ margin: 0; padding: 16px 20px; background: #f7f7f7; border-radius: 8px; font-size: 24px; }}
      .meta {{ color: #666; font-size: 20px; margin-top: 16px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
    </style>
    """)

    pill = (
        f'<span class="pill pill-ok">{lbl_ok}</span>'
        if ok else
        f'<span class="pill pill-warn">{lbl_not_enough}</span>'
    )

    html.append(f'<div class="hdr">{lbl_hdr} {pill}</div>')
    if ok:
        if is_layup:
            # Show layup-specific FG% breakdown
            left_fg = _as_float((summary or {}).get("left_fg_pct", 0))
            right_fg = _as_float((summary or {}).get("right_fg_pct", 0))
            left_att = int((summary or {}).get("left_attempts", 0) or 0)
            right_att = int((summary or {}).get("right_attempts", 0) or 0)
            peak_speed = _as_float((summary or {}).get("peak_speed_mps", 0))
            avg_speed = _as_float((summary or {}).get("avg_speed_mps", 0))
            fatigue_onset = (summary or {}).get("fatigue_onset")
            best_zone = (summary or {}).get("best_zone")

            lbl_overall = _t(f_pinyin, "Overall", "整体")
            lbl_left = _t(f_pinyin, "Left-hand", "左手")
            lbl_right = _t(f_pinyin, "Right-hand", "右手")
            lbl_peak_speed = _t(f_pinyin, "Peak Speed", "最高速度")
            lbl_avg_speed = _t(f_pinyin, "Avg Speed", "平均速度")
            lbl_fatigue = _t(f_pinyin, "Fatigue from", "疲劳开始于")
            lbl_best_zone = _t(f_pinyin, "Best Zone", "最佳区域")
            lbl_attempt = _t(f_pinyin, "attempt", "次")
            lbl_mps = _t(f_pinyin, "m/s", "米/秒")

            # Zone name translations
            zone_names = {
                'center': _t(f_pinyin, "Center", "中路"),
                'left_wing': _t(f_pinyin, "Left Wing", "左侧45度"),
                'right_wing': _t(f_pinyin, "Right Wing", "右侧45度"),
                'left_corner': _t(f_pinyin, "Left Corner", "左底角"),
                'right_corner': _t(f_pinyin, "Right Corner", "右底角"),
            }

            # Main FG% line
            html.append(
                f'<div class="sub">{lbl_overall}: <b>{accuracy_pct:.1f}%</b> &nbsp;|&nbsp; '
                f'{lbl_left}: <b>{left_fg:.1f}%</b> ({left_att}) &nbsp;|&nbsp; '
                f'{lbl_right}: <b>{right_fg:.1f}%</b> ({right_att})</div>'
            )

            # Secondary metrics line (speed, fatigue, best zone)
            metrics_parts = []
            if peak_speed > 0:
                metrics_parts.append(f'{lbl_peak_speed}: <b>{peak_speed:.1f}</b> {lbl_mps}')
            if avg_speed > 0:
                metrics_parts.append(f'{lbl_avg_speed}: <b>{avg_speed:.1f}</b> {lbl_mps}')
            if fatigue_onset is not None:
                metrics_parts.append(f'{lbl_fatigue}: #{fatigue_onset} {lbl_attempt}')
            if best_zone:
                zone_display = zone_names.get(best_zone, best_zone)
                metrics_parts.append(f'{lbl_best_zone}: <b>{zone_display}</b>')

            if metrics_parts:
                html.append(
                    f'<div style="font-size: 24px; color: #666; margin-bottom: 16px;">'
                    f'{" &nbsp;|&nbsp; ".join(metrics_parts)}</div>'
                )
        else:
            html.append(
                f'<div class="sub">{lbl_acc}: <b>{accuracy_pct:.1f}%</b> &nbsp;|&nbsp; {lbl_shots}: <b>{n}</b>{nugget_txt}</div>'
            )
    else:
        html.append(
            f'<div class="sub">{lbl_acc}: <b>{accuracy_pct:.1f}%</b> &nbsp;|&nbsp; {lbl_shots}: <b>{n}</b>'
            + (f' &nbsp;|&nbsp; <span style="color:#b71c1c;"><b>{reason}</b></span>' if reason else '')
            + '</div>'
        )

    if not advice:
        html.append(
            f'<div class="card"><div class="title">{lbl_no_advice}</div>'
            f'<div class="why">{lbl_collect_more}</div></div>'
        )
        return "\n".join(html)

    for i, a in enumerate(advice, 1):
        if not isinstance(a, dict):
            continue
        title = _escape(a.get("title", ""))
        why = _escape(a.get("why", ""))
        cue = _escape(a.get("cue", ""))          # <-- from compute_AI_advice.py
        key = _escape(a.get("key", ""))          # <-- from compute_AI_advice.py
        priority = a.get("priority", "")         # <-- from compute_AI_advice.py
        metrics = a.get("metrics", {}) or {}     # <-- from compute_AI_advice.py

        pri_txt = _escape(priority)
        html.append('<div class="card">')
        html.append(f'<div class="title">{i}. {title} <span class="pill pill-warn">P{pri_txt}</span></div>')
        if why:
            html.append(f'<div class="why">{why}</div>')
        if cue:
            html.append(f'<div class="cue"><b>{lbl_what_to_do}</b> {cue}</div>')

        met = _compact_metrics(metrics)
        meta_bits = []
        if key:
            meta_bits.append(f'key=<span class="mono">{key}</span>')
        if met:
            meta_bits.append('metrics=' + ", ".join([f'{_escape(k)}:{_escape(v)}' for k, v in met]))
        if meta_bits:
            html.append('<div class="meta">' + " &nbsp;|&nbsp; ".join(meta_bits) + '</div>')

        html.append('</div>')

    # Add replay video hint
    replay_hint = _t(
        f_pinyin,
        "Tip: Click Home → Setting → Replay video to review your training session.",
        "提示：点击 主页 → 设置 → 回放视频 可以查看您的训练录像。"
    )
    html.append(
        f'<div style="margin-top: 24px; padding: 16px 20px; background: #e3f2fd; '
        f'border-radius: 8px; font-size: 22px; color: #1565c0;">'
        f'💡 {replay_hint}</div>'
    )

    return "\n".join(html)


def _build_plain_text(result: Dict[str, Any], accuracy_pct: float, f_pinyin: bool = False) -> str:
    ok = bool((result or {}).get("ok", False))
    n = int((result or {}).get("n", 0) or 0)
    summary = (result or {}).get("summary", {}) or {}
    advice = (result or {}).get("advice", []) or []

    lbl_acc = _t(f_pinyin, "Accuracy", "命中率")
    lbl_shots = _t(f_pinyin, "Shots", "出手")
    lbl_ok = _t(f_pinyin, "OK", "通过")
    lbl_reason = _t(f_pinyin, "Reason", "原因")
    lbl_why = _t(f_pinyin, "Why", "原因")
    lbl_do = _t(f_pinyin, "Do", "建议")

    lines = []
    lines.append(f"{lbl_acc}: {accuracy_pct:.1f}%   {lbl_shots}: {n}   {lbl_ok}: {ok}")
    if not ok:
        reason = (summary or {}).get("reason", "")
        if reason:
            lines.append(f"{lbl_reason}: {reason}")
    lines.append("")

    for i, a in enumerate(advice, 1):
        if not isinstance(a, dict):
            continue
        title = a.get("title", "")
        why = a.get("why", "")
        cue = a.get("cue", "")
        pr = a.get("priority", "")
        lines.append(f"{i}. [P{pr}] {title}")
        if why:
            lines.append(f"   {lbl_why}: {why}")
        if cue:
            lines.append(f"   {lbl_do}:  {cue}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def show_ai_advice_popup(
    parent: QtWidgets.QWidget,
    result: Dict[str, Any],
    accuracy_pct: float,
    f_pinyin: bool = False,
    side: str = 'single',
) -> QtWidgets.QDialog:
    """
    Create and show a non-modal popup dialog with AI advice.
    IMPORTANT: keep a reference to the returned dialog (e.g., self._ai_popup = ...),
    otherwise Python may GC it.

    Args:
        parent: Parent widget
        result: AI advice result dict
        accuracy_pct: Accuracy percentage
        f_pinyin: Use Chinese if True
        side: 'left', 'right', or 'single'. Controls window positioning for "both" mode.
    """
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(_t(f_pinyin, "Shooting AI Advice", "投篮AI建议"))
    dlg.setModal(False)

    # Size and position based on side
    if side in ('left', 'right'):
        # Smaller windows for dual display in "both" mode
        dlg.resize(900, 700)

        # Get screen geometry
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            screen_geom = screen.availableGeometry()
            screen_w = screen_geom.width()
            screen_h = screen_geom.height()

            if side == 'left':
                # Position on left side of screen
                dlg.move(50, (screen_h - 700) // 2)
            else:
                # Position on right side of screen
                dlg.move(screen_w - 950, (screen_h - 700) // 2)
    else:
        # Single mode - centered, larger window
        dlg.resize(1290, 840)

    layout = QtWidgets.QVBoxLayout(dlg)

    viewer = QtWidgets.QTextBrowser()
    viewer.setOpenExternalLinks(False)
    viewer.setHtml(_build_html(result, accuracy_pct, f_pinyin=f_pinyin))
    layout.addWidget(viewer)

    btn_row = QtWidgets.QHBoxLayout()
    close_btn = QtWidgets.QPushButton(_t(f_pinyin, "Close", "关闭"))
    btn_row.addStretch(1)
    btn_row.addWidget(close_btn)
    layout.addLayout(btn_row)

    close_btn.clicked.connect(dlg.close)

    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
