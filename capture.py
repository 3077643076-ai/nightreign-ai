"""游戏窗口截取 — 只截艾尔登法环黑夜君临的客户区，不截全屏。

用法：
    from capture import capture_game

    frame, (w, h) = capture_game()   # 返回 BGR 图像 + 分辨率
"""

import numpy as np
import win32gui
import win32con
import win32ui
import mss


# 窗口标题关键词（大小写不敏感）
WINDOW_KEYWORDS = [
    "ELDEN RING NIGHTREIGN",
    "ELDEN RING™",
    "ELDEN RING",
]


def find_game_window():
    """查找游戏窗口句柄。返回 (hwnd, title) 或 (None, None)。"""
    # 先用 FindWindow 精确匹配（快且稳定）
    for kw in WINDOW_KEYWORDS:
        hwnd = win32gui.FindWindow(None, kw)
        if hwnd:
            title = win32gui.GetWindowText(hwnd)
            return hwnd, title

    # 退而求其次：枚举窗口模糊匹配
    result = [None, None]
    def callback(hwnd, _):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            upper = title.upper()
            for kw in WINDOW_KEYWORDS:
                if kw.upper() in upper:
                    result[0] = hwnd
                    result[1] = title
                    return False
        except Exception:
            pass
        return True

    try:
        win32gui.EnumWindows(callback, None)
    except Exception:
        pass
    return result[0], result[1]


def get_client_rect(hwnd):
    """获取窗口客户区尺寸（游戏实际渲染分辨率）和屏幕位置。"""
    rect = win32gui.GetClientRect(hwnd)
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    # 客户区左上角 → 屏幕坐标
    pt = win32gui.ClientToScreen(hwnd, (0, 0))
    return w, h, pt[0], pt[1]


def capture_game():
    """截取游戏窗口客户区，返回 (BGR图像, (宽, 高))。

    失败则退回到全屏截取。
    """
    hwnd, title = find_game_window()
    if hwnd:
        w, h, left, top = get_client_rect(hwnd)
        print(f"[capture] 游戏窗口: \"{title}\"  {w}x{h} @ ({left},{top})")
        with mss.MSS() as sct:
            monitor = {"left": left, "top": top, "width": w, "height": h}
            img = np.array(sct.grab(monitor))
            frame = img[:, :, :3].copy()
            return frame, (w, h)
    else:
        print("[capture] 未找到游戏窗口，回退到全屏截取")
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            w, h = monitor["width"], monitor["height"]
            img = np.array(sct.grab(monitor))
            frame = img[:, :, :3].copy()
            return frame, (w, h)


if __name__ == "__main__":
    frame, res = capture_game()
    print(f"分辨率: {res}")
    import cv2
    cv2.imwrite("capture_window.png", frame)
    print("已保存 capture_window.png")
