from control.action_executor import ActionExecutor


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
