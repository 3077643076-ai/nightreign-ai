from control.action_executor import ActionResult
from recorder.retry_controller import RetryController


class FakeExecutor:
    def __init__(self):
        self.calls = 0

    def press_confirm(self):
        self.calls += 1
        return ActionResult("confirm", True, "pressed")


def test_retry_controller_does_nothing_when_disabled():
    executor = FakeExecutor()
    result = RetryController(executor, enabled=False).retry()

    assert result.attempted is False
    assert result.action_result is None
    assert executor.calls == 0


def test_retry_controller_presses_confirm_when_enabled():
    executor = FakeExecutor()
    result = RetryController(executor, enabled=True).retry()

    assert result.attempted is True
    assert result.action_result.action == "confirm"
    assert executor.calls == 1
