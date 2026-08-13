import pytest

from control.action_executor import ActionExecutor


class FakeKeyboard:
    def __init__(self):
        self.events = []

    def press(self, key):
        self.events.append(("press", key))

    def release(self, key):
        self.events.append(("release", key))


class RealInputTestExecutor(ActionExecutor):
    def _new_keyboard_controller(self):
        self.fake_keyboard = FakeKeyboard()
        return self.fake_keyboard

    def _to_key(self, key: str):
        return key


def test_dry_run_confirm_returns_chinese_summary():
    executor = ActionExecutor(dry_run=True, confirm_key="f", lock_on_key="q")
    result = executor.press_confirm()
    assert result.action == "confirm"
    assert result.executed is False
    assert "确认" in result.summary_cn
    assert "dry-run" in result.summary_cn


def test_dry_run_lock_on_returns_chinese_summary():
    executor = ActionExecutor(dry_run=True, confirm_key="f", lock_on_key="q")
    result = executor.lock_on()
    assert result.action == "lock_on"
    assert result.executed is False
    assert "锁定" in result.summary_cn


def test_release_all_is_safe_in_dry_run():
    executor = ActionExecutor(dry_run=True, confirm_key="f", lock_on_key="q")
    result = executor.release_all()
    assert result.action == "release_all"
    assert result.executed is False
    assert "释放" in result.summary_cn


def test_real_input_releases_key_if_sleep_fails(monkeypatch):
    executor = RealInputTestExecutor(dry_run=False, confirm_key="f", lock_on_key="q")

    def fail_sleep(_duration):
        raise RuntimeError("sleep interrupted")

    monkeypatch.setattr("control.action_executor.time.sleep", fail_sleep)

    with pytest.raises(RuntimeError, match="sleep interrupted"):
        executor.press_confirm()

    assert executor.fake_keyboard.events == [("press", "f"), ("release", "f")]


def test_real_input_rejects_negative_press_duration_before_pressing():
    executor = RealInputTestExecutor(dry_run=False, confirm_key="f", lock_on_key="q", press_duration=-0.01)

    with pytest.raises(ValueError, match="press_duration must be >= 0"):
        executor.press_confirm()

    assert executor.fake_keyboard.events == []
