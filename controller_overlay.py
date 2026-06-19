"""游戏状态悬浮窗：血量/蓝量/体力/锁定/敌人HP。

透明背景 + 置顶，叠在游戏画面左下角。
不开 AI 也能用，纯监视。

用法：
    from controller_overlay import ControllerOverlay
    overlay = ControllerOverlay()
    overlay.update(ai_enabled, game_state)
"""

import tkinter as tk


class ControllerOverlay:
    """悬浮窗：AI 状态 + 游戏数值。"""

    _CLR = {
        "hp":    "#ff3333",
        "fp":    "#3388ff",
        "sta":   "#33cc66",
        "boss":  "#ff6666",
        "enemy": "#ff9966",
        "on":    "#00ff00",
        "off":   "#555555",
    }

    def __init__(self):
        self._root = tk.Tk()
        self._root.title("AI Monitor")
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.85)
        self._root.configure(bg="#010101")
        self._root.wm_attributes("-transparentcolor", "#010101")

        self._w = 180
        self._h = 195
        sh = self._root.winfo_screenheight()
        self._root.geometry(f"{self._w}x{self._h}+{10}+{sh - self._h - 60}")

        self._c = tk.Canvas(self._root, width=self._w, height=self._h,
                           bg="#010101", highlightthickness=0)
        self._c.pack()

        self._text_ids = {}
        self._build()
        self._root.update()

    def _build(self):
        c = self._c
        w, h = self._w, self._h

        # 底板
        c.create_rectangle(3, 3, w - 3, h - 3,
                          fill="#0a0a0a", outline="#333333", width=1)

        x = 12
        y = 10

        # AI 状态
        c.create_text(x, y, text="AI:", fill="#888888",
                     font=("Consolas", 10, "bold"), anchor="w")
        self._text_ids["ai"] = c.create_text(x + 35, y, text="OFF",
                     fill=self._CLR["off"], font=("Consolas", 10, "bold"), anchor="w")
        y += 18

        # 分割线
        c.create_line(x, y, w - 12, y, fill="#333333")
        y += 6

        # HP / FP / 体力 — 文字 + 小血条
        for label, tag, clr in [
            ("HP",   "hp",   "hp"),
            ("FP",   "fp",   "fp"),
            ("体力", "stamina", "sta"),
        ]:
            c.create_text(x, y, text=f"{label}:", fill=self._CLR[clr],
                         font=("Consolas", 10, "bold"), anchor="w")
            self._text_ids[tag] = c.create_text(x + 60, y, text="--%",
                         fill="#888888", font=("Consolas", 10), anchor="w")
            # 血条
            bar_x1 = x + 85
            bar_x2 = w - 12
            c.create_rectangle(bar_x1, y - 5, bar_x2, y + 5,
                              fill="#1a1a1a", outline="")
            bar = c.create_rectangle(bar_x1, y - 5, bar_x1, y + 5,
                                    fill=self._CLR[clr], outline="")
            self._text_ids[f"{tag}_bar"] = bar
            self._text_ids[f"{tag}_bar_rng"] = (bar_x1, bar_x2, y - 5, y + 5)
            y += 18

        # 分割线
        c.create_line(x, y, w - 12, y, fill="#333333")
        y += 6

        # 锁定
        c.create_text(x, y, text="锁定:", fill="#ffcc00",
                     font=("Consolas", 10, "bold"), anchor="w")
        self._text_ids["lock"] = c.create_text(x + 60, y, text="-",
                     fill="#888888", font=("Consolas", 10), anchor="w")
        y += 18

        # Boss
        c.create_text(x, y, text="Boss:", fill=self._CLR["boss"],
                     font=("Consolas", 10, "bold"), anchor="w")
        self._text_ids["boss_hp"] = c.create_text(x + 60, y, text="--",
                     fill="#888888", font=("Consolas", 10), anchor="w")
        bar_x1, bar_x2 = x + 85, w - 12
        c.create_rectangle(bar_x1, y - 5, bar_x2, y + 5,
                          fill="#1a1a1a", outline="")
        bar = c.create_rectangle(bar_x1, y - 5, bar_x1, y + 5,
                                fill=self._CLR["boss"], outline="")
        self._text_ids["boss_bar"] = bar
        self._text_ids["boss_bar_rng"] = (bar_x1, bar_x2, y - 5, y + 5)
        y += 18

        # 小怪
        c.create_text(x, y, text="小怪:", fill=self._CLR["enemy"],
                     font=("Consolas", 10, "bold"), anchor="w")
        self._text_ids["enemy_hp"] = c.create_text(x + 60, y, text="--",
                     fill="#888888", font=("Consolas", 10), anchor="w")
        c.create_rectangle(bar_x1, y - 5, bar_x2, y + 5,
                          fill="#1a1a1a", outline="")
        bar = c.create_rectangle(bar_x1, y - 5, bar_x1, y + 5,
                                fill=self._CLR["enemy"], outline="")
        self._text_ids["enemy_bar"] = bar
        self._text_ids["enemy_bar_rng"] = (bar_x1, bar_x2, y - 5, y + 5)

    # ================================================================
    # 公开接口
    # ================================================================

    def update(self, btn_state=None, axis_state=None,
               ai_enabled=False, game_state=None):
        """更新显示。btn_state/axis_state 保留接口兼容，实际不使用。"""
        self._update_game_state(game_state)
        self._update_ai(ai_enabled)
        self._root.update()

    def _update_game_state(self, gs):
        if gs is None:
            return
        c = self._c
        for tag in ["hp", "fp", "stamina", "boss", "enemy"]:
            self._set_bar(tag, gs.get(tag if tag != "stamina" else "stamina", -1))
        lock_val = gs.get("locked", False)
        c.itemconfig(self._text_ids["lock"],
                     text="YES" if lock_val else "NO",
                     fill="#00ff00" if lock_val else "#ff4444")

    def _set_bar(self, tag, value):
        """设置百分比血条。value: 0~1 或 -1(无数据)。"""
        c = self._c
        text_id = self._text_ids.get(f"{tag}_hp" if tag in ("boss","enemy") else tag)
        bar_id = self._text_ids.get(f"{tag}_bar")
        rng = self._text_ids.get(f"{tag}_bar_rng")

        if text_id is None or bar_id is None or rng is None:
            return

        if value >= 0:
            pct = int(value * 100)
            c.itemconfig(text_id, text=f"{pct}%")
            x1, x2, by1, by2 = rng
            fill_x = x1 + int((x2 - x1) * value)
            c.coords(bar_id, x1, by1, fill_x, by2)
        else:
            c.itemconfig(text_id, text="--")
            x1, x2, by1, by2 = rng
            c.coords(bar_id, x1, by1, x1, by2)

    def _update_ai(self, enabled):
        c = self._c
        c.itemconfig(self._text_ids["ai"],
                     text="ON" if enabled else "OFF",
                     fill=self._CLR["on"] if enabled else self._CLR["off"])

    def destroy(self):
        try:
            self._root.destroy()
        except Exception:
            pass
