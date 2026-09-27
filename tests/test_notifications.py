"""Notification rendering (the message the harness injects into a turn)."""

from __future__ import annotations

from agent.background.manager import BackgroundTask
from agent.background.notifications import render_notification, render_notifications


def _task(**overrides) -> BackgroundTask:
    base: BackgroundTask = {
        "task_id": "bash-1",
        "command": "npm run build",
        "description": "build the project",
        "output_path": "/tmp/deepai-bg-tasks/s/bash-1.output",
        "status": "completed",
        "exit_code": 0,
        "pid": None,
        "started_at": "2026-09-26T10:00:00Z",
        "finished_at": "2026-09-26T10:00:31Z",
        "timed_out": False,
    }
    base.update(overrides)
    return base


def test_render_notification_contains_headline_and_machine_block():
    text = render_notification(_task())

    assert 'Background command "build the project" completed (exit code 0).' in text
    assert "<task-notification>" in text
    assert "<task_id>bash-1</task_id>" in text
    assert "<command>npm run build</command>" in text
    assert "<status>completed</status>" in text
    assert "<exit_code>0</exit_code>" in text
    assert "</task-notification>" in text


def test_render_notification_reports_failures_and_timeouts():
    failed = render_notification(_task(status="failed", exit_code=2))
    assert 'failed (exit code 2)' in failed

    timed_out = render_notification(_task(status="failed", timed_out=True, exit_code=None))
    assert "timed out" in timed_out
    assert "<exit_code>null</exit_code>" in timed_out


def test_render_notifications_collapses_successes_but_keeps_failures():
    tasks = [_task(task_id="bash-1"), _task(task_id="bash-2"), _task(task_id="bash-3", status="failed", exit_code=1)]
    text = render_notifications(tasks)

    assert "3 background commands completed" not in text  # only successes collapse
    assert "2 background commands completed (bash-1, bash-2)" in text
    assert text.count("<task-notification>") == 3  # every id stays actionable
    assert 'failed (exit code 1)' in text


def test_render_notifications_points_at_the_output_tool():
    text = render_notifications([_task()])

    assert "get_task_output" in text
    assert "bash-1" in text


def test_render_notifications_empty_input():
    assert render_notifications([]) == ""
