"""Background task manager behavior: start, incremental read, kill, notify."""

from __future__ import annotations

import time

import pytest

from agent.background.manager import BackgroundTaskManager, UnknownTaskError, format_task_line


def _poll_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")


@pytest.fixture
def manager(tmp_path) -> BackgroundTaskManager:
    return BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)


def test_start_returns_immediately_with_task_id_and_output_path(manager):
    record = manager.start("echo hello-bg", description="say hello")

    assert record["task_id"] == "bash-1"
    assert record["status"] == "running"
    assert record["output_path"].endswith("bash-1.output")
    assert record["pid"] is not None

    final = manager.wait(record["task_id"], timeout=10)
    assert final["status"] == "completed"
    assert final["exit_code"] == 0
    assert final["finished_at"] is not None
    assert final["pid"] is None


def test_output_is_readable_and_read_is_incremental(manager):
    record = manager.start("echo first-line; sleep 2; echo second-line", description="two lines")

    # First read: only what has been produced so far (the second line is 2s away).
    _poll_until(lambda: "first-line" in manager.read(record["task_id"]).new_output)
    mid = manager.read(record["task_id"])
    assert "second-line" not in mid.new_output
    assert mid.status == "running"

    manager.wait(record["task_id"], timeout=10)
    final = manager.read(record["task_id"])
    assert "second-line" in final.new_output
    assert final.status == "completed"
    assert final.exit_code == 0


def test_read_returns_empty_after_everything_was_consumed(manager):
    record = manager.start("echo only-once", description="once")
    manager.wait(record["task_id"], timeout=10)

    first = manager.read(record["task_id"])
    assert "only-once" in first.new_output
    assert manager.read(record["task_id"]).new_output == ""


def test_read_filter_consumes_non_matching_lines(manager):
    record = manager.start("printf 'keep\\nskip\\nkeep2\\n'", description="filter me")
    manager.wait(record["task_id"], timeout=10)

    snapshot = manager.read(record["task_id"], filter="keep")
    assert snapshot.filtered is True
    assert "skip" not in snapshot.new_output
    assert "keep2" in snapshot.new_output

    # Everything was consumed by the first (filtered) read.
    assert manager.read(record["task_id"]).new_output == ""


def test_kill_stops_a_long_running_task(manager):
    record = manager.start("sleep 300", description="sleep forever")

    killed = manager.kill(record["task_id"])

    assert killed["status"] == "killed"
    assert killed["pid"] is None
    assert manager.get(record["task_id"])["status"] == "killed"
    # Killing is agent-initiated, so it must not queue a wake-up notification.
    assert manager.take_notifications() == []
    # Killing twice is a no-op rather than an error.
    assert manager.kill(record["task_id"])["status"] == "killed"


def test_timeout_marks_task_failed_and_notifies(manager):
    record = manager.start("sleep 30", description="sleep past timeout", timeout_seconds=0.3)

    final = manager.wait(record["task_id"], timeout=10)

    assert final["status"] == "failed"
    assert final["timed_out"] is True
    notifications = manager.take_notifications()
    assert [n["task_id"] for n in notifications] == [record["task_id"]]


def test_completion_queues_exactly_one_notification(manager):
    record = manager.start("echo done", description="finish")
    manager.wait(record["task_id"], timeout=10)

    first = manager.take_notifications()
    assert len(first) == 1
    assert first[0]["status"] == "completed"
    assert manager.take_notifications() == []


def test_reading_a_finished_task_drops_its_duplicate_notification(manager):
    """A read that saw the terminal status makes the notification redundant.

    eval (webapp 20260926-235404): notifications for two tasks the model had
    already read arrived after the turn and sent it into a ~10-turn re-audit
    storm that blew the run's time limit.
    """
    record = manager.start("echo done", description="finish")
    manager.wait(record["task_id"], timeout=10)

    snapshot = manager.read(record["task_id"])
    assert snapshot.status == "completed"

    assert manager.take_notifications() == [], "the model saw the finish itself"


def test_notification_survives_when_output_was_read_while_running(manager):
    record = manager.start("sleep 1 && echo late", description="slow")
    early = manager.read(record["task_id"])
    assert early.status == "running", "a mid-flight read proves nothing about completion"

    manager.wait(record["task_id"], timeout=10)

    notes = manager.take_notifications()
    assert [n["task_id"] for n in notes] == [record["task_id"]]


def test_failed_command_is_reported_as_failed(manager):
    record = manager.start("exit 3", description="fail")
    final = manager.wait(record["task_id"], timeout=10)

    assert final["status"] == "failed"
    assert final["exit_code"] == 3
    assert [n["task_id"] for n in manager.take_notifications()] == [record["task_id"]]


def test_list_filters_by_status_and_orders_by_start(manager):
    first = manager.start("echo one", description="one")
    second = manager.start("sleep 300", description="two")
    manager.wait(first["task_id"], timeout=10)

    running = [r["task_id"] for r in manager.list("running")]
    completed = [r["task_id"] for r in manager.list("completed")]
    assert running == [second["task_id"]]
    assert completed == [first["task_id"]]
    assert [r["task_id"] for r in manager.list()] == [first["task_id"], second["task_id"]]
    manager.kill(second["task_id"])


def test_unknown_task_raises(manager):
    with pytest.raises(UnknownTaskError):
        manager.read("bash-999")
    with pytest.raises(UnknownTaskError):
        manager.kill("bash-999")
    assert manager.get("bash-999") is None


def test_shutdown_kills_running_tasks(manager):
    record = manager.start("sleep 300", description="leftover")
    manager.shutdown()

    assert manager.get(record["task_id"])["status"] == "killed"
    with pytest.raises(RuntimeError):
        manager.start("echo nope", description="after shutdown")


def test_format_task_line_contains_status(manager):
    record = manager.start("echo fmt", description="format")
    manager.wait(record["task_id"], timeout=10)
    line = format_task_line(manager.get(record["task_id"]))
    assert record["task_id"] in line
    assert "status=completed" in line


def test_requeue_notifications_preserves_order(manager):
    first = manager.start("echo one", description="one")
    second = manager.start("echo two", description="two")
    manager.wait(first["task_id"], timeout=10)
    manager.wait(second["task_id"], timeout=10)

    drained: list = []
    _poll_until(lambda: drained.extend(manager.take_notifications()) or len(drained) >= 2)

    manager.requeue_notifications(drained)
    back = manager.take_notifications()

    assert [r["task_id"] for r in back] == [r["task_id"] for r in drained]
    assert manager.take_notifications() == []


def test_pending_notifications_peek_and_wait(manager):
    assert manager.pending_notifications() == 0
    assert manager.wait_for_notifications(0.05) is False

    record = manager.start("echo peek", description="peek")
    manager.wait(record["task_id"], timeout=10)

    assert manager.wait_for_notifications(2.0) is True
    assert manager.pending_notifications() == 1  # peeking must not drain
    assert len(manager.take_notifications()) == 1
    assert manager.pending_notifications() == 0


def test_records_survive_restart(tmp_path):
    """A manager pointed at the same directory restores the task history."""
    kwargs = {"output_dir": tmp_path / "tasks", "session_id": "s1", "cwd": tmp_path}
    first = BackgroundTaskManager(**kwargs)
    record = first.start("echo persisted", description="persisted")
    first.wait(record["task_id"], timeout=10)  # finalize() writes tasks.json

    # Simulate a process restart: a brand-new manager, same store.
    second = BackgroundTaskManager(**kwargs)
    restored = second.list()

    assert [r["task_id"] for r in restored] == ["bash-1"]
    assert restored[0]["status"] == "completed"
    assert restored[0]["command"] == "echo persisted"
    assert restored[0]["exit_code"] == 0
    assert "persisted" in second.read("bash-1").new_output

    # The id counter continues, so new ids cannot collide with history.
    nxt = second.start("echo next", description="next")
    assert nxt["task_id"] == "bash-2"
    second.shutdown()
    first.shutdown()


def test_running_tasks_restore_as_interrupted(tmp_path):
    kwargs = {"output_dir": tmp_path / "tasks", "session_id": "s1", "cwd": tmp_path}
    first = BackgroundTaskManager(**kwargs)
    first.start("echo partial-output; sleep 30", description="long runner")

    # Wait for the output file to have content (the restart must still see it).
    deadline = time.time() + 5
    while time.time() < deadline:
        if "partial-output" in first.read("bash-1").new_output:
            break
        time.sleep(0.1)

    second = BackgroundTaskManager(**kwargs)
    restored = second.get("bash-1")

    assert restored is not None
    assert restored["status"] == "interrupted"
    assert restored["finished_at"] and restored["pid"] is None
    assert second.pending_notifications() == 0, "history must not wake the model"
    assert "partial-output" in second.read("bash-1").new_output
    assert second.kill("bash-1")["status"] == "interrupted"  # nothing to signal

    first.shutdown()  # cleans up the real process
    second.shutdown()


def test_corrupt_store_does_not_break_startup(tmp_path):
    kwargs = {"output_dir": tmp_path / "tasks", "session_id": "s1", "cwd": tmp_path}
    manager = BackgroundTaskManager(**kwargs)
    (manager.output_dir / "tasks.json").write_text("{ this is not json", encoding="utf-8")

    fresh = BackgroundTaskManager(**kwargs)

    assert fresh.list() == []
    fresh.start("echo ok", description="ok").keys()  # manager is fully usable
