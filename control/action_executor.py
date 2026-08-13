from dataclasses import dataclass
import time


@dataclass(frozen=True)
class ActionResult:
    action: str
    executed: bool
    summary_cn: str


class ActionExecutor:
    def __init__(self, dry_run: bool, confirm_key: str, lock_on_key: str, press_duration: float = 0.05):
        self.dry_run = dry_run
        self.confirm_key = confirm_key
        self.lock_on_key = lock_on_key
        self.press_duration = press_duration
        self._keyboard = None if dry_run else self._new_keyboard_controller()

    def press_confirm(self) -> ActionResult:
        return self._press("confirm", self.confirm_key, "确认/再战")

    def lock_on(self) -> ActionResult:
        return self._press("lock_on", self.lock_on_key, "锁定目标")

    def release_all(self) -> ActionResult:
        if self.dry_run:
            return ActionResult("release_all", False, "[动作执行] dry-run：释放所有按键，防止卡键")
        for key in (self.confirm_key, self.lock_on_key):
            self._keyboard.release(self._to_key(key))
        return ActionResult("release_all", True, "[动作执行] 已释放确认键和锁定键，防止卡键")

    def _press(self, action: str, key: str, label_cn: str) -> ActionResult:
        if self.dry_run:
            return ActionResult(action, False, f"[动作执行] dry-run：按下 {key}（{label_cn}）")
        normalized = self._to_key(key)
        if self.press_duration < 0:
            raise ValueError("press_duration must be >= 0")
        self._keyboard.press(normalized)
        try:
            time.sleep(self.press_duration)
        finally:
            self._keyboard.release(normalized)
        return ActionResult(action, True, f"[动作执行] 已按下 {key}（{label_cn}）")

    def _new_keyboard_controller(self):
        from pynput.keyboard import Controller

        return Controller()

    def _to_key(self, key: str):
        from pynput.keyboard import Key

        special = {
            "space": Key.space,
            "enter": Key.enter,
            "esc": Key.esc,
            "tab": Key.tab,
        }
        return special.get(key.lower(), key)
