from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from collections import Counter


@dataclass
class AdviceItem:
    key: str
    priority: int            # higher = more important
    title: str
    why: str
    metrics: Dict[str, Any]
    cue: str                 # what to do


def _clean_str_arr(arr: Sequence[Any]) -> List[str]:
    out = []
    for v in arr:
        if v is None:
            continue
        s = str(v).strip().lower()
        if s == "":
            continue
        out.append(s)
    return out


def _clean_float_arr(arr: Sequence[Any]) -> np.ndarray:
    vals = []
    for v in arr:
        try:
            if v is None:
                continue
            f = float(v)
            if np.isfinite(f):
                vals.append(f)
        except Exception:
            pass
    return np.asarray(vals, dtype=np.float32)


def _clean_bool_arr(arr: Sequence[Any]) -> np.ndarray:
    vals = []
    for v in arr:
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            vals.append(bool(v))
        else:
            s = str(v).strip().lower()
            if s in ("true", "1", "yes", "y"):
                vals.append(True)
            elif s in ("false", "0", "no", "n"):
                vals.append(False)
    return np.asarray(vals, dtype=bool)


def _dominant_direction(counts: Counter, valid_keys: Tuple[str, ...]) -> Tuple[str, float, int]:
    """Return (dominant_key, dominant_ratio_over_valid, valid_n)."""
    valid_n = sum(counts[k] for k in valid_keys)
    if valid_n <= 0:
        return ("none", 0.0, 0)
    dom_key = max(valid_keys, key=lambda k: counts[k])
    dom_ratio = counts[dom_key] / valid_n
    return (dom_key, float(dom_ratio), int(valid_n))


# -------------------------------
# Chinese text option (f_pinyin)
# -------------------------------


def _translate_advice_item_cn(a: AdviceItem) -> AdviceItem:
    """Translate AdviceItem title/why/cue to Chinese (metrics/priority/key unchanged)."""
    k = a.key
    m = a.metrics or {}

    # Defaults: keep original if we don't recognize the key
    title = a.title
    why = a.why
    cue = a.cue

    if k == "flat_shot":
        flat_rate = float(m.get("flat_rate", 0.0) or 0.0)
        title = "出手弧线偏平（弧度不足）"
        why = f"{flat_rate:.0%} 的投篮被检测为平射（低弧线 / 低入射角）。"
        cue = "把弧线抬起来：肘在球下方，先向上再向前送；随手更高、更久。"

    elif k == "release_height_inconsistent":
        h_std = float(m.get("release_height_std", 0.0) or 0.0)
        title = "出手高度不稳定"
        why = f"出手高度比例的标准差为 {h_std:.3f}（按球员框归一化）。"
        cue = "固定出手‘窗口’：每次都在眉毛上方同一高度出手。先慢练到位置一致，再加速度。"

    elif k == "release_lateral_inconsistent":
        lat_std = float(m.get("release_lateral_std", 0.0) or 0.0)
        title = "出手左右位置不稳定"
        why = f"出手横向比例的标准差为 {lat_std:.3f}（按球员框宽度归一化）。"
        cue = "出手走直线：辅助手早点离球；出手后手臂伸直、手腕下压，中指指篮筐正中。"

    elif k == "systematic_lateral_push":
        lat_mean = float(m.get("release_lateral_mean", 0.0) or 0.0)
        direction = "右" if lat_mean > 0 else "左"
        title = f"出手存在系统性向{direction}推"
        why = f"平均出手横向比例为 {lat_mean:+.3f}（中心=0）。"
        cue = f"你的出手整体偏{direction}。别往{direction}侧‘推’：手腕保持正，随手让中指指向篮筐正中；辅助手早点离球。"

    elif k == "release_too_low":
        h_mean = float(m.get("release_height_mean", 0.0) or 0.0)
        title = "出手点偏低"
        why = f"平均出手高度比例为 {h_mean:.3f}（归一化）。"
        cue = "出手点抬高：充分伸展，在起跳最高点（或抬到最高处）就把球送出去；随手向上停住一拍。"

    elif k == "release_too_high":
        h_mean = float(m.get("release_height_mean", 0.0) or 0.0)
        title = "出手点非常高（可能是出手偏晚/发力被迫）"
        why = f"平均出手高度比例为 {h_mean:.3f}（归一化）。"
        cue = "别等到最高点才出手：动作更干净、更快的一动式上举，减少停顿，让出手更自然。"

    elif k == "short_long_bias":
        short_rate = float(m.get("short_rate", 0.0) or 0.0)
        long_rate = float(m.get("long_rate", 0.0) or 0.0)
        dominant = "偏短" if short_rate > long_rate else "偏长"
        title = f"距离控制问题：多数{dominant}"
        why = f"偏短={short_rate:.0%}，偏长={long_rate:.0%}（不含未知）。"
        cue = "偏短：用腿多一点、抬升多一点。偏长：随手更柔和，少往前推，落地时腿部把力量‘收住’。"

    elif k == "left_right_bias":
        left_rate = float(m.get("left_rate", 0.0) or 0.0)
        right_rate = float(m.get("right_rate", 0.0) or 0.0)
        dominant = "左" if left_rate > right_rate else "右"
        title = f"瞄准偏差：命中点集中偏{dominant}"
        why = f"偏左={left_rate:.0%}，偏右={right_rate:.0%}（不含未知）。"
        cue = f"你经常偏{dominant}。先保证肩膀对正、出手走直线；在动作不变的前提下，把瞄准点轻微往相反方向微调。"

    return AdviceItem(
        key=a.key,
        priority=a.priority,
        title=title,
        why=why,
        metrics=a.metrics,
        cue=cue,
    )


def _translate_result_cn(res: Dict[str, Any], *, min_samples: int) -> Dict[str, Any]:
    """Translate top-level reason and each advice item to Chinese (numbers/metrics unchanged)."""
    if not res:
        return res
    out = dict(res)
    # Translate "not enough samples" reason if present
    summary = out.get("summary", {}) or {}
    if isinstance(summary, dict):
        reason = summary.get("reason", "")
        if isinstance(reason, str) and "not enough samples" in reason:
            summary = dict(summary)
            summary["reason"] = f"样本不足（需要 >= {min_samples}）"
            out["summary"] = summary

    adv = out.get("advice", []) or []
    new_adv = []
    for a in adv:
        if isinstance(a, dict):
            try:
                item = AdviceItem(
                    key=str(a.get("key", "")),
                    priority=int(a.get("priority", 0) or 0),
                    title=str(a.get("title", "")),
                    why=str(a.get("why", "")),
                    metrics=a.get("metrics", {}) or {},
                    cue=str(a.get("cue", "")),
                )
                item_cn = _translate_advice_item_cn(item)
                new_adv.append(asdict(item_cn))
            except Exception:
                new_adv.append(a)
        else:
            new_adv.append(a)
    out["advice"] = new_adv
    return out


def _calc_fatigue_onset(speed_history: List[Tuple[float, int, bool, str]]) -> Tuple[Optional[int], float]:
    """
    Detect from which attempt the player shows fatigue based on speed decline.
    Returns: (fatigue_onset_attempt, avg_speed_drop_pct) or (None, 0) if no fatigue detected.
    """
    if len(speed_history) < 4:
        return None, 0.0

    speeds = [s[0] for s in speed_history]
    # Use first 3 attempts as baseline
    baseline_speed = np.mean(speeds[:3])
    if baseline_speed < 0.5:  # Too slow to analyze
        return None, 0.0

    # Look for sustained speed drop (2+ consecutive attempts below 80% of baseline)
    threshold = baseline_speed * 0.80
    fatigue_onset = None
    consecutive_slow = 0

    for i, speed in enumerate(speeds[3:], start=4):
        if speed < threshold:
            consecutive_slow += 1
            if consecutive_slow >= 2 and fatigue_onset is None:
                fatigue_onset = i - 1  # Mark first attempt of fatigue
        else:
            consecutive_slow = 0

    if fatigue_onset is not None:
        # Calculate average speed drop from fatigue onset
        post_fatigue_speeds = speeds[fatigue_onset - 1:]
        avg_post = np.mean(post_fatigue_speeds)
        speed_drop_pct = (baseline_speed - avg_post) / baseline_speed * 100
        return fatigue_onset, speed_drop_pct

    return None, 0.0


def _calc_best_zones(zone_history: List[Tuple[str, bool, str]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate success rate per zone and hand.
    Returns: {zone_name: {"attempts": int, "makes": int, "fg_pct": float, "left_fg": float, "right_fg": float}}
    """
    zone_stats = {}
    for zone, is_make, hand in zone_history:
        if zone not in zone_stats:
            zone_stats[zone] = {
                "attempts": 0, "makes": 0,
                "left_attempts": 0, "left_makes": 0,
                "right_attempts": 0, "right_makes": 0
            }
        zone_stats[zone]["attempts"] += 1
        if is_make:
            zone_stats[zone]["makes"] += 1
        if hand == 'left':
            zone_stats[zone]["left_attempts"] += 1
            if is_make:
                zone_stats[zone]["left_makes"] += 1
        else:
            zone_stats[zone]["right_attempts"] += 1
            if is_make:
                zone_stats[zone]["right_makes"] += 1

    # Calculate FG percentages
    for zone, stats in zone_stats.items():
        stats["fg_pct"] = (stats["makes"] / stats["attempts"] * 100) if stats["attempts"] > 0 else 0
        stats["left_fg"] = (stats["left_makes"] / stats["left_attempts"] * 100) if stats["left_attempts"] > 0 else 0
        stats["right_fg"] = (stats["right_makes"] / stats["right_attempts"] * 100) if stats["right_attempts"] > 0 else 0

    return zone_stats


def compute_layup_ai_advice(
    left_attempts: int,
    left_makes: int,
    right_attempts: int,
    right_makes: int,
    *,
    speed_history: Optional[List[Tuple[float, int, bool, str]]] = None,
    zone_history: Optional[List[Tuple[str, bool, str]]] = None,
    f_pinyin: bool = False,
) -> Dict[str, Any]:
    """
    Compute AI advice for layup training with left/right hand FG% breakdown,
    peak speed analysis, fatigue detection, and best zones.

    Args:
        left_attempts: Number of left-hand layup attempts
        left_makes: Number of left-hand layup makes
        right_attempts: Number of right-hand layup attempts
        right_makes: Number of right-hand layup makes
        speed_history: List of (peak_speed_m_per_s, attempt_number, is_make, hand)
        zone_history: List of (zone_name, is_make, hand)
        f_pinyin: True for Chinese output

    Returns:
      {
        "ok": bool,
        "n": int,  # total attempts
        "is_layup": True,  # marker for layup-specific display
        "summary": {
            "overall_fg_pct": float,
            "left_fg_pct": float,
            "right_fg_pct": float,
            "left_attempts": int,
            "left_makes": int,
            "right_attempts": int,
            "right_makes": int,
            "peak_speed_mps": float,  # maximum peak speed
            "avg_speed_mps": float,   # average approach speed
            "fatigue_onset": int or None,  # attempt number when fatigue detected
            "best_zone": str,  # zone with highest FG%
            "zone_stats": dict,  # per-zone statistics
        },
        "advice": [AdviceItem as dict, ...]
      }
    """
    speed_history = speed_history or []
    zone_history = zone_history or []

    total_att = left_attempts + right_attempts
    total_makes = left_makes + right_makes

    overall_fg = (total_makes / total_att * 100) if total_att > 0 else 0.0
    left_fg = (left_makes / left_attempts * 100) if left_attempts > 0 else 0.0
    right_fg = (right_makes / right_attempts * 100) if right_attempts > 0 else 0.0

    # Calculate speed metrics
    peak_speed = max([s[0] for s in speed_history], default=0.0)
    avg_speed = np.mean([s[0] for s in speed_history]) if speed_history else 0.0

    # Detect fatigue onset
    fatigue_onset, fatigue_speed_drop = _calc_fatigue_onset(speed_history)

    # Calculate zone statistics
    zone_stats = _calc_best_zones(zone_history)
    # Find best zone (highest FG% with at least 2 attempts)
    best_zone = None
    best_zone_fg = -1
    for zone, stats in zone_stats.items():
        if stats["attempts"] >= 2 and stats["fg_pct"] > best_zone_fg:
            best_zone_fg = stats["fg_pct"]
            best_zone = zone

    advice: List[AdviceItem] = []

    # Generate advice based on performance
    if total_att >= 3:
        # Check for weak hand
        if left_attempts > 0 and right_attempts > 0:
            if left_fg < right_fg - 20:
                if f_pinyin:
                    advice.append(AdviceItem(
                        key="weak_left_hand",
                        priority=85,
                        title="左手上篮命中率较低",
                        why=f"左手命中率 {left_fg:.0f}% 比右手 {right_fg:.0f}% 低 {right_fg - left_fg:.0f}%。",
                        metrics={"left_fg": left_fg, "right_fg": right_fg},
                        cue="多练习左手上篮：先从静止出发，单独用左手练习护球、挑球，逐渐增加速度。"
                    ))
                else:
                    advice.append(AdviceItem(
                        key="weak_left_hand",
                        priority=85,
                        title="Left-hand layup needs improvement",
                        why=f"Left-hand FG% ({left_fg:.0f}%) is {right_fg - left_fg:.0f}% lower than right-hand ({right_fg:.0f}%).",
                        metrics={"left_fg": left_fg, "right_fg": right_fg},
                        cue="Practice left-hand layups: start stationary, focus on ball control and touch, then add speed."
                    ))
            elif right_fg < left_fg - 20:
                if f_pinyin:
                    advice.append(AdviceItem(
                        key="weak_right_hand",
                        priority=85,
                        title="右手上篮命中率较低",
                        why=f"右手命中率 {right_fg:.0f}% 比左手 {left_fg:.0f}% 低 {left_fg - right_fg:.0f}%。",
                        metrics={"left_fg": left_fg, "right_fg": right_fg},
                        cue="多练习右手上篮：先从静止出发，单独用右手练习护球、挑球，逐渐增加速度。"
                    ))
                else:
                    advice.append(AdviceItem(
                        key="weak_right_hand",
                        priority=85,
                        title="Right-hand layup needs improvement",
                        why=f"Right-hand FG% ({right_fg:.0f}%) is {left_fg - right_fg:.0f}% lower than left-hand ({left_fg:.0f}%).",
                        metrics={"left_fg": left_fg, "right_fg": right_fg},
                        cue="Practice right-hand layups: start stationary, focus on ball control and touch, then add speed."
                    ))

        # Check for overall low FG%
        if overall_fg < 50:
            if f_pinyin:
                advice.append(AdviceItem(
                    key="low_overall_fg",
                    priority=80,
                    title="整体上篮命中率偏低",
                    why=f"整体命中率为 {overall_fg:.0f}%，低于50%的基准线。",
                    metrics={"overall_fg": overall_fg},
                    cue="专注基本功：眼睛看篮板方框、轻柔放球、用篮板。练习时先慢后快。"
                ))
            else:
                advice.append(AdviceItem(
                    key="low_overall_fg",
                    priority=80,
                    title="Overall layup FG% is low",
                    why=f"Overall FG% is {overall_fg:.0f}%, below 50% benchmark.",
                    metrics={"overall_fg": overall_fg},
                    cue="Focus on fundamentals: eyes on the backboard square, soft touch, use the glass. Practice slow before fast."
                ))

        # Check for one-sided practice
        if left_attempts == 0 and right_attempts > 0:
            if f_pinyin:
                advice.append(AdviceItem(
                    key="no_left_practice",
                    priority=75,
                    title="没有练习左手上篮",
                    why="本次训练只练习了右手上篮。",
                    metrics={"left_attempts": left_attempts, "right_attempts": right_attempts},
                    cue="下次训练时记得练习左手上篮，两只手都要熟练才能在比赛中更有威胁。"
                ))
            else:
                advice.append(AdviceItem(
                    key="no_left_practice",
                    priority=75,
                    title="No left-hand layup practice",
                    why="This session only included right-hand layups.",
                    metrics={"left_attempts": left_attempts, "right_attempts": right_attempts},
                    cue="Remember to practice left-hand layups next time. Being proficient with both hands makes you a more versatile player."
                ))
        elif right_attempts == 0 and left_attempts > 0:
            if f_pinyin:
                advice.append(AdviceItem(
                    key="no_right_practice",
                    priority=75,
                    title="没有练习右手上篮",
                    why="本次训练只练习了左手上篮。",
                    metrics={"left_attempts": left_attempts, "right_attempts": right_attempts},
                    cue="下次训练时记得练习右手上篮，两只手都要熟练才能在比赛中更有威胁。"
                ))
            else:
                advice.append(AdviceItem(
                    key="no_right_practice",
                    priority=75,
                    title="No right-hand layup practice",
                    why="This session only included left-hand layups.",
                    metrics={"left_attempts": left_attempts, "right_attempts": right_attempts},
                    cue="Remember to practice right-hand layups next time. Being proficient with both hands makes you a more versatile player."
                ))

        # Fatigue detection advice
        if fatigue_onset is not None:
            if f_pinyin:
                advice.append(AdviceItem(
                    key="fatigue_detected",
                    priority=88,
                    title=f"从第{fatigue_onset}次开始出现疲劳迹象",
                    why=f"从第{fatigue_onset}次上篮开始，你的冲刺速度下降了{fatigue_speed_drop:.0f}%。",
                    metrics={"fatigue_onset": fatigue_onset, "speed_drop_pct": fatigue_speed_drop},
                    cue="注意体能分配。疲劳时动作容易变形，命中率下降。可以在感到疲劳前短暂休息。"
                ))
            else:
                advice.append(AdviceItem(
                    key="fatigue_detected",
                    priority=88,
                    title=f"Fatigue detected starting at attempt #{fatigue_onset}",
                    why=f"Your approach speed dropped by {fatigue_speed_drop:.0f}% from attempt #{fatigue_onset} onward.",
                    metrics={"fatigue_onset": fatigue_onset, "speed_drop_pct": fatigue_speed_drop},
                    cue="Pace yourself. Fatigue leads to poor form and lower accuracy. Consider short rest breaks before you feel tired."
                ))

        # Peak speed advice
        if peak_speed > 0:
            # Speed in m/s: walking ~1.4, jogging ~2.5, running ~4, sprinting ~6-8
            if peak_speed < 2.0:
                if f_pinyin:
                    advice.append(AdviceItem(
                        key="low_approach_speed",
                        priority=70,
                        title="冲刺速度偏慢",
                        why=f"最高速度仅为{peak_speed:.1f}米/秒，接近步行速度。",
                        metrics={"peak_speed": peak_speed, "avg_speed": avg_speed},
                        cue="尝试加快冲刺速度。上篮需要一定的动量才能在防守下完成。"
                    ))
                else:
                    advice.append(AdviceItem(
                        key="low_approach_speed",
                        priority=70,
                        title="Approach speed is slow",
                        why=f"Peak speed was only {peak_speed:.1f} m/s, close to walking pace.",
                        metrics={"peak_speed": peak_speed, "avg_speed": avg_speed},
                        cue="Try increasing your approach speed. Layups require momentum to finish through contact."
                    ))

        # Best/worst zone advice
        if zone_stats and len(zone_stats) >= 2:
            # Find worst zone with at least 2 attempts
            worst_zone = None
            worst_zone_fg = 101
            for zone, stats in zone_stats.items():
                if stats["attempts"] >= 2 and stats["fg_pct"] < worst_zone_fg:
                    worst_zone_fg = stats["fg_pct"]
                    worst_zone = zone

            zone_name_cn = {
                'center': '中路', 'left_wing': '左侧45度', 'right_wing': '右侧45度',
                'left_corner': '左底角', 'right_corner': '右底角'
            }
            zone_name_en = {
                'center': 'Center', 'left_wing': 'Left Wing', 'right_wing': 'Right Wing',
                'left_corner': 'Left Corner', 'right_corner': 'Right Corner'
            }

            if worst_zone and best_zone and worst_zone != best_zone:
                if best_zone_fg - worst_zone_fg >= 30:  # Significant difference
                    if f_pinyin:
                        advice.append(AdviceItem(
                            key="zone_weakness",
                            priority=72,
                            title=f"弱势区域：{zone_name_cn.get(worst_zone, worst_zone)}",
                            why=f"{zone_name_cn.get(worst_zone, worst_zone)}命中率{worst_zone_fg:.0f}%，"
                                f"而{zone_name_cn.get(best_zone, best_zone)}达到{best_zone_fg:.0f}%。",
                            metrics={"best_zone": best_zone, "best_fg": best_zone_fg,
                                     "worst_zone": worst_zone, "worst_fg": worst_zone_fg},
                            cue=f"多练习从{zone_name_cn.get(worst_zone, worst_zone)}起步的上篮，提高薄弱环节。"
                        ))
                    else:
                        advice.append(AdviceItem(
                            key="zone_weakness",
                            priority=72,
                            title=f"Weak zone: {zone_name_en.get(worst_zone, worst_zone)}",
                            why=f"{zone_name_en.get(worst_zone, worst_zone)} FG% is {worst_zone_fg:.0f}%, "
                                f"vs {best_zone_fg:.0f}% from {zone_name_en.get(best_zone, best_zone)}.",
                            metrics={"best_zone": best_zone, "best_fg": best_zone_fg,
                                     "worst_zone": worst_zone, "worst_fg": worst_zone_fg},
                            cue=f"Practice more layups starting from the {zone_name_en.get(worst_zone, worst_zone)} to improve your weak spot."
                        ))

    advice_sorted = sorted(advice, key=lambda x: x.priority, reverse=True)

    summary = {
        "overall_fg_pct": overall_fg,
        "left_fg_pct": left_fg,
        "right_fg_pct": right_fg,
        "left_attempts": left_attempts,
        "left_makes": left_makes,
        "right_attempts": right_attempts,
        "right_makes": right_makes,
        # New metrics
        "peak_speed_mps": float(peak_speed),
        "avg_speed_mps": float(avg_speed),
        "fatigue_onset": fatigue_onset,
        "fatigue_speed_drop_pct": float(fatigue_speed_drop) if fatigue_onset else None,
        "best_zone": best_zone,
        "zone_stats": zone_stats,
    }

    return {
        "ok": True,
        "n": total_att,
        "is_layup": True,
        "summary": summary,
        "advice": [asdict(a) for a in advice_sorted],
    }


def compute_shooting_ai_advice(
    release_height_ratio_arr: Sequence[Any],
    release_lateral_ratio_arr: Sequence[Any],
    flat_shot_arr: Sequence[Any],
    bias_short_long_arr: Sequence[Any],
    bias_left_right_arr: Sequence[Any],
    *,
    # thresholds (tune to your system)
    min_samples: int = 12,
    flat_rate_warn: float = 0.45,
    shortlong_rate_warn: float = 0.55,
    leftright_rate_warn: float = 0.55,
    release_height_low: float = 0.45,     # normalized: 0=top of player, 1=bottom; >0.45 means low release
    release_height_high: float = 0.15,    # <0.15 means very high release (above head)
    release_lat_mean_warn: float = 0.08,  # normalized by bbox width (0=center)
    release_lat_std_warn: float = 0.10,
    release_h_std_warn: float = 0.10,
    # language option (keep logic unchanged; only changes displayed text)
    f_pinyin: bool = False,
) -> Dict[str, Any]:
    """
    Same output format as before:
      {
        "ok": bool,
        "n": int,
        "summary": {...},
        "advice": [AdviceItem as dict, ...]  # sorted by priority desc
      }

    Difference vs old version:
      - No drift_dir_* inputs
      - No "body_drift" advice
    """

    # --- clean inputs ---
    h = _clean_float_arr(release_height_ratio_arr)
    lat = _clean_float_arr(release_lateral_ratio_arr)
    flat = _clean_bool_arr(flat_shot_arr)
    sl = _clean_str_arr(bias_short_long_arr)
    br = _clean_str_arr(bias_left_right_arr)

    # define N as number of shots represented (use min to avoid overestimating when arrays differ)
    raw_lens = [
        len(release_height_ratio_arr),
        len(release_lateral_ratio_arr),
        len(flat_shot_arr),
        len(bias_short_long_arr),
        len(bias_left_right_arr),
    ]
    n = int(min(raw_lens)) if raw_lens else 0

    if n < min_samples:
        res = {
            "ok": False,
            "n": n,
            "summary": {"reason": f"not enough samples (need >= {min_samples})"},
            "advice": []
        }
        return _translate_result_cn(res, min_samples=min_samples) if f_pinyin else res

    advice: List[AdviceItem] = []

    # --- 1) Flat shot (low arc) ---
    flat_rate = float(np.mean(flat)) if flat.size else 0.0
    if flat_rate >= flat_rate_warn:
        advice.append(AdviceItem(
            key="flat_shot",
            priority=85,
            title="Your arc is often too low (flat shot)",
            why=f"{flat_rate:.0%} of shots were detected as flat (low arc / low entry).",
            metrics={"flat_rate": flat_rate},
            cue="Add more lift: finish higher, keep elbow under the ball, and let the ball travel up before forward."
        ))

    # --- 2) Release-point consistency + systematic side push (release ratios) ---
    h_mean = float(np.mean(h)) if h.size else float("nan")
    h_std = float(np.std(h)) if h.size else float("nan")
    lat_mean = float(np.mean(lat)) if lat.size else float("nan")
    lat_std = float(np.std(lat)) if lat.size else float("nan")

    if np.isfinite(h_std) and h_std >= release_h_std_warn:
        advice.append(AdviceItem(
            key="release_height_inconsistent",
            priority=80,
            title="Your release height varies too much",
            why=f"Release height ratio std is {h_std:.3f} (normalized by player bbox).",
            metrics={"release_height_mean": h_mean, "release_height_std": h_std},
            cue="Pick one release: same 'window' above the eyebrow every time. Film cue: pause at set-point, then shoot."
        ))

    if np.isfinite(lat_std) and lat_std >= release_lat_std_warn:
        advice.append(AdviceItem(
            key="release_lateral_inconsistent",
            priority=78,
            title="Your release is not laterally consistent",
            why=f"Release lateral ratio std is {lat_std:.3f} (normalized by bbox width).",
            metrics={"release_lateral_mean": lat_mean, "release_lateral_std": lat_std},
            cue="Keep the ball path straight: guide hand quiet, shoot through the middle finger, avoid side-spin."
        ))

    if np.isfinite(lat_mean) and abs(lat_mean) >= release_lat_mean_warn:
        direction = "right" if lat_mean > 0 else "left"
        advice.append(AdviceItem(
            key="systematic_lateral_push",
            priority=75,
            title=f"Systematic {direction} push at release",
            why=f"Average release lateral ratio is {lat_mean:+.3f} (center=0).",
            metrics={"release_lateral_mean": lat_mean, "release_lateral_std": lat_std},
            cue=f"Your ball is leaving {direction} of center. Cue: keep wrist straight, finish fingers to the rim, guide hand off earlier."
        ))

    if np.isfinite(h_mean):
        # Note: ratio 0 = top of player (high release), ratio 1 = bottom (low release)
        if h_mean > release_height_low:
            advice.append(AdviceItem(
                key="release_too_low",
                priority=70,
                title="Release tends to be low",
                why=f"Average release height ratio is {h_mean:.3f} (normalized; lower=higher release point).",
                metrics={"release_height_mean": h_mean, "release_height_std": h_std},
                cue="Raise the finish: extend fully, keep eyes on target, and release at the top of your jump (or stable set-shot top)."
            ))
        elif h_mean < release_height_high:
            advice.append(AdviceItem(
                key="release_too_high",
                priority=60,
                title="Release tends to be very high (could be late/forced)",
                why=f"Average release height ratio is {h_mean:.3f} (normalized; lower=higher release point).",
                metrics={"release_height_mean": h_mean, "release_height_std": h_std},
                cue="If you feel you're 'waiting' to shoot, simplify: quicker one-motion up, don't over-hold at the top."
            ))

    # --- 3) Short/long pattern ---
    sl_counts = Counter([x for x in sl if x in ("short", "long", "center")])
    sl_valid = sl_counts["short"] + sl_counts["long"] + sl_counts["center"]
    if sl_valid > 0:
        short_rate = sl_counts["short"] / sl_valid
        long_rate = sl_counts["long"] / sl_valid
        if max(short_rate, long_rate) >= shortlong_rate_warn:
            dominant = "short" if short_rate > long_rate else "long"
            advice.append(AdviceItem(
                key="short_long_bias",
                priority=82,
                title=f"Distance control issue: mostly {dominant}",
                why=f"Short={short_rate:.0%}, Long={long_rate:.0%} (excluding unknowns).",
                metrics={"short_rate": short_rate, "long_rate": long_rate, "valid": sl_valid},
                cue=("If short: use more legs, slightly more lift. "
                     "If long: soften the finish, reduce forward push, absorb with legs.")
            ))

    # --- 4) Left/right bias ---
    br_counts = Counter([x for x in br if x in ("left", "right", "center")])
    br_valid = br_counts["left"] + br_counts["right"] + br_counts["center"]
    if br_valid > 0:
        left_rate = br_counts["left"] / br_valid
        right_rate = br_counts["right"] / br_valid
        if max(left_rate, right_rate) >= leftright_rate_warn:
            dominant = "left" if left_rate > right_rate else "right"
            advice.append(AdviceItem(
                key="left_right_bias",
                priority=77,
                title=f"Aim bias: misses cluster to the {dominant}",
                why=f"Left={left_rate:.0%}, Right={right_rate:.0%} (excluding unknowns).",
                metrics={"left_rate": left_rate, "right_rate": right_rate, "valid": br_valid},
                cue=f"You’re consistently {dominant}. Cue: square shoulders, release straight; try aiming slightly opposite while keeping form unchanged."
            ))

    advice_sorted = sorted(advice, key=lambda x: x.priority, reverse=True)

    summary = {
        "n_raw": n,
        "flat_rate": float(flat_rate),
        "release_height_mean": h_mean, "release_height_std": h_std,
        "release_lateral_mean": lat_mean, "release_lateral_std": lat_std,
        "short_long_counts": dict(sl_counts),
        "left_right_counts": dict(br_counts),
    }

    res = {
        "ok": True,
        "n": n,
        "summary": summary,
        "advice": [asdict(a) for a in advice_sorted],
    }
    return _translate_result_cn(res, min_samples=min_samples) if f_pinyin else res

