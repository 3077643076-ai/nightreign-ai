"""血条位置标注工具 (tkinter版) - 鼠标拖框标注 HP/FP/体力条

用法：先打开游戏确保HUD可见，然后运行
    python calibrate_hud.py

操作：在图上鼠标拖框，依次标注 FP蓝条 → HP红条 → 体力绿条
      标完按 ENTER 输出坐标
"""

import sys
from pathlib import Path
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture import capture_game


class CalibrateApp:
    NAMES = ["FP蓝条", "HP红条", "体力绿条"]
    COLORS = ["#ff8800", "#ff0000", "#00ff00"]

    def __init__(self, img_bgr):
        # BGR -> RGB -> PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._pil_img = Image.fromarray(img_rgb)
        self._w, self._h = self._pil_img.size
        self._boxes = []  # [(x1,y1,x2,y2)]
        self._drag_start = None
        self._current_box = None
        self._idx = 0

        # tkinter 窗口
        self._root = tk.Tk()
        self._root.title("标注血条 - 拖框: FP → HP → 体力 → ENTER完成")

        # 缩放以适应屏幕
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        scale = min(1.0, (sw - 100) / self._w, (sh - 200) / self._h)
        self._scale = scale
        dw, dh = int(self._w * scale), int(self._h * scale)

        self._canvas = tk.Canvas(self._root, width=dw, height=dh, cursor="cross")
        self._canvas.pack()

        # 显示图片
        self._photo = ImageTk.PhotoImage(
            self._pil_img.resize((dw, dh), Image.LANCZOS))
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # 提示文字
        self._hint = self._canvas.create_text(
            dw // 2, dh - 20, text=f">>> 拖框标注: {self.NAMES[0]} <<<",
            fill="yellow", font=("", 14, "bold"))

        # 鼠标事件
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._root.bind("<BackSpace>", self._undo)
        self._root.bind("<Return>", self._finish)
        self._root.bind("<Escape>", lambda e: self._root.quit())

        print(f"\n屏幕: {self._w}x{self._h}  |  窗口已打开")
        print("=" * 50)
        print("  拖框标注: FP蓝条 → HP红条 → 体力绿条")
        print("  BACKSPACE = 撤销  |  ENTER = 完成  |  ESC = 退出")
        print("=" * 50)

    def _to_real(self, x, y):
        """缩放坐标 → 原始坐标"""
        return int(x / self._scale), int(y / self._scale)

    def _on_press(self, event):
        if self._idx >= 3:
            return
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        if self._current_box:
            self._canvas.delete(self._current_box)
        self._current_box = self._canvas.create_rectangle(
            self._drag_start[0], self._drag_start[1],
            event.x, event.y, outline=self.COLORS[self._idx], width=3)

    def _on_release(self, event):
        if self._drag_start is None:
            return
        x1, y1 = self._drag_start
        x2, y2 = event.x, event.y
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        if x2 - x1 < 20 or y2 - y1 < 8:
            self._drag_start = None
            if self._current_box:
                self._canvas.delete(self._current_box)
                self._current_box = None
            return

        # 转回原始坐标
        rx1, ry1 = self._to_real(x1, y1)
        rx2, ry2 = self._to_real(x2, y2)
        self._boxes.append((rx1, ry1, rx2, ry2))
        self._drag_start = None
        self._current_box = None

        name = self.NAMES[self._idx]
        pct_y1 = ry1 / self._h
        pct_y2 = ry2 / self._h
        pct_x1 = rx1 / self._w
        pct_x2 = rx2 / self._w
        print(f"  [{name}] ({rx1},{ry1})-({rx2},{ry2}) "
              f"{rx2-rx1}x{ry2-ry1}  "
              f"比例: y={pct_y1:.4f}~{pct_y2:.4f} x={pct_x1:.4f}~{pct_x2:.4f}")

        self._idx += 1
        if self._idx < 3:
            self._canvas.itemconfig(
                self._hint, text=f">>> 拖框标注: {self.NAMES[self._idx]} <<<")
        else:
            self._canvas.itemconfig(
                self._hint, text=">>> 标完了! 按 ENTER 输出坐标 <<<")

    def _undo(self, event=None):
        if self._idx > 0:
            self._idx -= 1
            self._boxes.pop()
            # 清除画布上画的所有矩形，重绘
            self._canvas.delete("box")
            for i, (x1, y1, x2, y2) in enumerate(self._boxes):
                sx1, sy1 = x1 * self._scale, y1 * self._scale
                sx2, sy2 = x2 * self._scale, y2 * self._scale
                self._canvas.create_rectangle(
                    sx1, sy1, sx2, sy2,
                    outline=self.COLORS[i], width=3, tags="box")
            self._canvas.itemconfig(
                self._hint, text=f">>> 拖框标注: {self.NAMES[self._idx]} <<<")
            print(f"  撤销 → 重新标注 {self.NAMES[self._idx]}")

    def _finish(self, event=None):
        if self._idx < 3:
            print(f"\n只标了 {self._idx} 条，请标完 FP/HP/体力!")
            return

        print("\n\n======= 复制到 game_state.py 的 REGIONS 字典 =======")
        for i, (name, key) in enumerate(
            [("FP", "fp"), ("HP", "hp"), ("体力", "stamina")]
        ):
            x1, y1, x2, y2 = self._boxes[i]
            r = (y1/self._h, y2/self._h, x1/self._w, x2/self._w)
            print(f'        "{key}": ({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f}),  # {name}')
        print("=====================================================\n")
        self._root.quit()

    def run(self):
        self._root.mainloop()


def main():
    print("正在截取游戏窗口...")
    frame, (w, h) = capture_game()
    cv2.imwrite("calibrate.png", frame)

    app = CalibrateApp(frame)
    app.run()
    print("\n退出标注工具")


if __name__ == "__main__":
    main()
