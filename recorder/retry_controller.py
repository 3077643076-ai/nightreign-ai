from dataclasses import dataclass

from control.action_executor import ActionExecutor, ActionResult


@dataclass(frozen=True)
class RetryResult:
    attempted: bool
    action_result: ActionResult | None


class RetryController:
    def __init__(self, executor: ActionExecutor, enabled: bool = False):
        self.executor = executor
        self.enabled = enabled

    def retry(self) -> RetryResult:
        if not self.enabled:
            return RetryResult(False, None)
        return RetryResult(True, self.executor.press_confirm())
