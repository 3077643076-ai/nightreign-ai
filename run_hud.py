"""实时血条悬浮窗 — 最简版。

python run_hud.py
按 Q 退出
"""

import sys, io, time
sys.path.insert(0, '.')
sys.dont_write_bytecode = True

import cv2, numpy as np
from capture import capture_game
from perception import GameState


def capture_quiet():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return capture_game()
    finally:
        sys.stdout = old


def main():
    print("初始化...")
    frame, (gw, gh) = capture_quiet()
    gs = GameState((gw, gh))
    print(f"游戏: {gw}x{gh}")

    # 校准 15 帧
    for _ in range(15):
        frame, _ = capture_quiet()
        gs.detect(frame)
        time.sleep(0.03)

    print("运行中... 按 Q 退出")
    print()

    ov_w, ov_h = 210, 90
    cv2.namedWindow("HUD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HUD", ov_w, ov_h)
    cv2.moveWindow("HUD", 5, gh - ov_h - 80)

    fps_n = 0
    fps_t = time.time()

    try:
        while True:
            frame, _ = capture_quiet()
            state = gs.detect(frame)
            fps_n += 1

            # 画悬浮窗
            panel = np.zeros((ov_h, ov_w, 3), dtype=np.uint8) + 18
            cv2.rectangle(panel, (0, 0), (ov_w - 1, ov_h - 1), (50, 50, 50), 1)

            y = 16
            for tag, label, clr in [
                ("hp", "HP", (50, 50, 255)),
                ("fp", "FP", (255, 160, 50)),
                ("stamina", "STM", (50, 210, 100)),
            ]:
                v = state.get(tag, -1)
                if v >= 0:
                    txt = f"{label}: {int(v * 100):3d}%"
                else:
                    txt = f"{label}:  --"
                cv2.putText(panel, txt, (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)
                # 小条
                bx, bw = 68, 132
                cv2.rectangle(panel, (bx, y - 7), (bx + bw, y + 2), (40, 40, 40), -1)
                if v >= 0:
                    fx = bx + int(bw * max(0.0, min(1.0, v)))
                    cv2.rectangle(panel, (bx, y - 7), (fx, y + 2), clr, -1)
                y += 22

            # FPS
            if fps_n >= 20:
                fps = fps_n / (time.time() - fps_t)
                cv2.putText(panel, f"{fps:.0f}fps", (ov_w - 45, ov_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
                fps_n = 0
                fps_t = time.time()

            cv2.imshow("HUD", panel)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("退出")


if __name__ == "__main__":
    main()
