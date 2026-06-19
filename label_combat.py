"""基于按键特征标记战斗帧：LS(锁定)按下 → 扩展窗口 → 输出标签。

用法：
    python label_combat.py

输出每个 session 的 combat_labels.npy：0=探索, 1=战斗
"""

import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess.dataset import BUTTON_NAMES

# === 参数 ===
RS_IDX = BUTTON_NAMES.index("RS")      # 右摇杆按下 = 锁定敌人

LOCK_BEFORE = 2.0   # 锁定前 N 秒 = 接近敌人
LOCK_AFTER  = 5.0   # 锁定后 N 秒 = 持续战斗
FPS = 15.0

BEFORE_FRAMES = int(LOCK_BEFORE * FPS)
AFTER_FRAMES  = int(LOCK_AFTER * FPS)

# 攻击类按钮（在战斗窗口内高频出现可延长窗口）
COMBAT_BTNS = ["RB", "RT", "A", "B"]
COMBAT_IDX = [BUTTON_NAMES.index(b) for b in COMBAT_BTNS]

PREPROCESSED = Path("preprocessed")


def load_labels(session_dir: Path):
    """加载标签，返回 (frame_ids, buttons_array, axes_array)。"""
    for name in ["labels_clean.json", "labels.json"]:
        p = session_dir / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            fids = data["frame_ids"]
            btns = np.array(data["buttons"], dtype=np.float32)
            axes = np.array(data["axes"], dtype=np.float32)
            return fids, btns, axes
    return None, None, None


def label_session(session_dir: Path) -> tuple[int, np.ndarray]:
    """返回 (总帧数, combat_labels)。"""
    fids, btns, axes = load_labels(session_dir)
    if fids is None or len(fids) == 0:
        return 0, np.array([])

    n = len(fids)
    print(f"    {n} 帧, ", end="")

    # 找 LS 按下沿
    ls = btns[:, RS_IDX]
    ls_binary = (ls > 0.5).astype(np.int32)
    edges = np.zeros(n, dtype=np.int32)
    edges[1:] = (ls_binary[1:] == 1) & (ls_binary[:-1] == 0)
    if ls_binary[0] == 1:
        edges[0] = 1

    lock_positions = np.where(edges)[0]
    print(f"{len(lock_positions)} 次锁定, ", end="")

    if len(lock_positions) == 0:
        mask = np.zeros(n, dtype=np.uint8)
        print("战斗帧=0 (0.0%)")
        return n, mask

    # 扩展窗口
    mask = np.zeros(n, dtype=np.uint8)
    for pos in lock_positions:
        lo = max(0, pos - BEFORE_FRAMES)
        hi = min(n, pos + AFTER_FRAMES)
        mask[lo:hi] = 1

    # 攻击密度扩展：如果战斗窗口结束时攻击频率还很高，再延长
    extend_by_density(mask, btns)

    combat_n = mask.sum()
    print(f"战斗帧={combat_n} ({combat_n/n*100:.1f}%)")
    return n, mask


def extend_by_density(mask: np.ndarray, btns: np.ndarray):
    """在已标记区域末端，若攻击密度仍高则继续延长。"""
    combat_activity = (btns[:, COMBAT_IDX] > 0.5).any(axis=1).astype(np.int32)

    # 从每个战斗窗口末尾往后检查
    in_combat = False
    extend_counter = 0
    for i in range(len(mask)):
        if mask[i] == 1:
            in_combat = True
            extend_counter = 0
        elif in_combat:
            # 看前 1 秒的攻击密度
            check_start = max(0, i - int(FPS))
            density = combat_activity[check_start:i].mean()
            if density > 0.05:  # 每秒 5% 的帧有攻击 → 还在打
                mask[i] = 1
                extend_counter += 1
            else:
                if extend_counter > int(FPS * 2):  # 沉默超过 2 秒
                    in_combat = False


def main():
    print("=" * 50)
    print(f"  战斗帧标注：RS右摇杆锁定 ±{LOCK_BEFORE}s/{LOCK_AFTER}s")
    print("=" * 50)

    sessions = sorted(PREPROCESSED.glob("session_*"))
    total_all = 0
    total_combat = 0

    for sess_dir in sessions:
        print(f"\n[{sess_dir.name}]")
        n, mask = label_session(sess_dir)
        if n == 0:
            continue
        np.save(sess_dir / "combat_labels.npy", mask)
        total_all += n
        total_combat += mask.sum()

    print(f"\n{'=' * 50}")
    print(f"  总计: {total_all:,} 帧")
    print(f"  战斗: {total_combat:,} ({total_combat/total_all*100:.1f}%)")
    print(f"  探索: {total_all - total_combat:,} ({100 - total_combat/total_all*100:.1f}%)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
