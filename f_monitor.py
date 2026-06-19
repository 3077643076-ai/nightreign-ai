"""F 按键监测器：实时显示 F 键按下间隔 + 死亡文字检测结果。"""

import time
import ctypes
import numpy as np
import cv2
import mss
from pynput.keyboard import Controller as KB, Listener

kb = KB()
sct = mss.MSS()
monitor = sct.monitors[1]

last_f = 0
f_count = 0
frame = 0

# 简单的死亡检测（画面中下方变暗+白字）
def detect_death(img):
    h, w = img.shape[:2]
    roi = img[h*2//5:h*3//4, w//4:3*w//4]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bright_px = (gray > 200).sum()
    return bright_px > 500  # 画面中下方有很多亮像素（死亡对话框文字）

def on_press(key):
    global f_count
    try:
        if hasattr(key, 'char') and key.char == 'f':
            global last_f
            now = time.perf_counter()
            dt = now - last_f if last_f else 0
            last_f = now
            f_count += 1
            print(f"\n>>> F 按下! 距上次 {dt:.1f}s 总计 #{f_count} <<<")
    except: pass

listener = Listener(on_press=on_press)
listener.daemon = True
listener.start()

print("F 按键监测中 - 每秒打印状态 - Q 退出")
print("=" * 50)
t0 = time.perf_counter()

while True:
    time.sleep(1.0)
    frame += 1
    img = np.array(sct.grab(monitor))
    img_bgr = img[:, :, :3].copy()
    dead = detect_death(img_bgr)
    elapsed = time.perf_counter() - t0
    last_f_ago = elapsed - last_f if last_f else 999
    status = "[DEAD?]" if dead else "[alive]"
    print(f"#{frame} {status} 距上次F: {last_f_ago:.1f}s 总F: {f_count} 运行: {elapsed:.0f}s", flush=True)

    if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
        break

print("退出")
