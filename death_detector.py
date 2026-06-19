"""死亡画面检测器：OCR 识别"重试"/"返回圆桌"文字，检测到就按 F。"""

import cv2
import numpy as np
import pytesseract
from pynput.keyboard import Controller as KB
import time

# 检测区域（画面中下方，死亡对话框位置）
Y_START = 0.45  # 从画面 45% 高度开始
Y_END = 0.70    # 到 70% 高度
X_START = 0.20  # 左 20%
X_END = 0.80    # 右 80%

KEYWORDS = ['重试', '圆桌', '返回', 'retry', 'roundtable']


def detect(frame):
    """检测死亡画面，返回 True 表示找到。"""
    h, w = frame.shape[:2]
    y1, y2 = int(h * Y_START), int(h * Y_END)
    x1, x2 = int(w * X_START), int(w * X_END)
    roi = frame[y1:y2, x1:x2]

    # 转灰度 + 二值化（白字黑底）
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # OCR
    try:
        text = pytesseract.image_to_string(thresh, lang='chi_sim+eng')
        for kw in KEYWORDS:
            if kw.lower() in text.lower():
                return True
    except Exception:
        pass
    return False


def test():
    """测试：截屏检测死亡画面。"""
    import mss, ctypes
    sct = mss.MSS()
    kb = KB()
    print("监测死亡画面，检测到就按 F，Q 退出...")
    i = 0
    while True:
        img = np.array(sct.grab(sct.monitors[1]))
        frame = img[:, :, :3].copy()
        i += 1
        if i % 45 == 0:  # ~3秒检测一次
            if detect(frame):
                print(f"  #{i} [DEATH!] 按 F 重试！", flush=True)
                kb.press('f'); time.sleep(0.3); kb.release('f')
            else:
                print(f"  #{i} no death", flush=True)
        if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
            break
        time.sleep(0.05)


if __name__ == "__main__":
    test()
