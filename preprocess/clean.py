"""数据清洗：检测并裁掉死亡/拔线/菜单片段。

基于手柄数据做启发式检测：
- 菜单段：START/BACK 后出现十字键操作 + 无摇杆
- 闲置尾段：session 末尾长时间无操作
- 闲置中段：中间超过 N 秒完全无操作（加载/死亡）

用法：
    python -m preprocess.clean

内存安全：基于间隔检测，不展开为 set，兼容 Python 3.13。
"""

import json
from pathlib import Path

from preprocess.dataset import _normalize

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
PREPROCESSED_ROOT = Path(__file__).resolve().parent.parent / "preprocessed"

IDLE_TAIL_SEC = 5.0
IDLE_GAP_SEC = 8.0
MENU_WINDOW_FRAMES = 30
MENU_MIN_DPAD = 3
FPS = 15


def _is_active(buttons, axes):
    if any(b > 0.5 for b in buttons):
        return True
    return abs(axes[0]) > 0.2 or abs(axes[1]) > 0.2


def _is_menu_frame(buttons, axes):
    dpad = buttons[13] > 0.5 or buttons[14] > 0.5 or buttons[15] > 0.5 or buttons[16] > 0.5
    no_stick = abs(axes[0]) < 0.2 and abs(axes[1]) < 0.2
    return dpad and no_stick


def _in_any_interval(idx, intervals):
    """二分检查 idx 是否在任一 [start, end] 闭区间内。intervals 按 start 排序。"""
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e = intervals[mid]
        if idx < s:
            hi = mid - 1
        elif idx > e:
            lo = mid + 1
        else:
            return True
    return False


def find_bad_intervals(active_flags, menu_flags, start_buttons):
    """扫描一次，返回所有坏区间的 (start, end) 列表 + tail_cutoff。

    active_flags[i] = _is_active(frame[i])
    menu_flags[i] = _is_menu_frame(frame[i])
    start_buttons[i] = BACK or START pressed
    """
    n = len(active_flags)
    bad = []

    # ── 闲置间隔 ─────────────────────────────────────
    min_idle = int(IDLE_GAP_SEC * FPS)
    gap_start = None
    for i in range(n):
        if active_flags[i]:
            if gap_start is not None and (i - gap_start) >= min_idle:
                bad.append((gap_start, i - 1))
            gap_start = None
        else:
            if gap_start is None:
                gap_start = i
    # 末尾闲置单独处理 → tail_cutoff
    tail_cutoff = n
    if gap_start is not None:
        tail_idle = n - gap_start
        if tail_idle >= int(IDLE_TAIL_SEC * FPS):
            # 从末尾往前找最后活跃帧
            last_active = None
            for i in range(n - 1, -1, -1):
                if active_flags[i]:
                    last_active = i
                    break
            if last_active is not None:
                tail_cutoff = last_active + 1
            else:
                tail_cutoff = 0  # 整局无操作
        # 如果 gap_start 到末尾不是整段闲置（只是短尾），不计入 bad
        # 但如果超过了 IDLE_GAP_SEC 且不是末尾，则已在上面的循环中处理

    # ── 菜单间隔 ─────────────────────────────────────
    i = 0
    while i < n:
        if not start_buttons[i]:
            i += 1
            continue
        window_end = min(i + MENU_WINDOW_FRAMES, n)
        dpad_count = sum(1 for j in range(i + 1, window_end) if menu_flags[j])
        if dpad_count >= MENU_MIN_DPAD:
            # 确认菜单，找结束点（摇杆恢复 >0.3 或到末尾）
            menu_start = i
            menu_end = i
            for j in range(i + 1, n):
                # 需要重新读 active 信息... 这里我们只能找摇杆恢复
                menu_end = j
            # 简化：菜单从 START 按下持续到末尾或找到下一段活跃
            for j in range(i + 1, n):
                if active_flags[j]:
                    menu_end = j - 1
                    break
                menu_end = j
            bad.append((menu_start, menu_end))
            i = menu_end
        i += 1

    # 合并排序
    bad.sort()
    merged = []
    for s, e in bad:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged, tail_cutoff


def clean_session(sess_path: Path):
    """清洗一个 session，返回统计信息。输出到 preprocessed/<session>/labels_clean.json"""
    inputs_path = sess_path / "inputs.jsonl"
    if not inputs_path.exists():
        return {"error": "no inputs.jsonl"}

    # ── 第一遍：逐行读取，只保存轻量 flags ──────────────
    active_flags = []
    menu_flags = []
    start_flags = []
    frame_ids = []

    with open(inputs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            btns, axes = _normalize(d["buttons"], d["axes"])
            active_flags.append(_is_active(btns, axes))
            menu_flags.append(_is_menu_frame(btns, axes))
            start_flags.append(btns[6] > 0.5 or btns[7] > 0.5)
            frame_ids.append(d["frame"])

    n_total = len(frame_ids)
    if n_total == 0:
        return {"error": "empty session"}

    bad_intervals, tail_cutoff = find_bad_intervals(active_flags, menu_flags, start_flags)

    # ── 第二遍：读取完整数据，写入清洗后标签 ────────────
    out_dir = PREPROCESSED_ROOT / sess_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_fids = []
    clean_btns = []
    clean_axes = []
    cleaned_count = 0

    with open(inputs_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i >= tail_cutoff:
                cleaned_count += 1
                continue
            if _in_any_interval(i, bad_intervals):
                cleaned_count += 1
                continue

            d = json.loads(line)
            btns, axes = _normalize(d["buttons"], d["axes"])
            clean_fids.append(d["frame"])
            clean_btns.append(btns)
            clean_axes.append(axes)

    n_clean = len(clean_fids)

    # 写入
    with open(out_dir / "labels_clean.json", "w", encoding="utf-8") as f:
        json.dump({
            "frame_ids": clean_fids,
            "buttons": clean_btns,
            "axes": clean_axes,
        }, f, ensure_ascii=False)

    return {
        "session": sess_path.name,
        "total_frames": n_total,
        "cleaned_frames": cleaned_count,
        "cleaned_pct": round(cleaned_count / max(n_total, 1) * 100, 1),
        "clean_frames": n_clean,
        "idle_menu_segments": len(bad_intervals),
        "tail_cutoff": tail_cutoff < n_total,
    }


def clean_all():
    sessions = sorted(DATA_ROOT.glob("session_*"))
    if not sessions:
        print("No sessions found.")
        return

    total_total = 0
    total_clean = 0
    for sess in sessions:
        stats = clean_session(sess)
        if "error" in stats:
            print(f"  {sess.name}: SKIP ({stats['error']})")
            continue
        print(f"  {sess.name}: {stats['total_frames']} → {stats['clean_frames']} "
              f"(-{stats['cleaned_pct']}%) | bad_segs={stats['idle_menu_segments']}")
        total_total += stats["total_frames"]
        total_clean += stats["clean_frames"]

    pct = round((total_total - total_clean) / max(total_total, 1) * 100, 1)
    print(f"\nTotal: {total_total} → {total_clean} frames (-{pct}%)")


if __name__ == "__main__":
    clean_all()
