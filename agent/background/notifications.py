"""Rendering of background-task completion notifications.

Claude Code injects a user-level notification into the conversation when a
background command finishes, e.g.::

    Background command "build" completed (exit code 0).

That single message is what wakes the model up after it stopped polling.
:func:`render_notifications` produces the equivalent for this harness: a short
human-readable line, the machine-readable ``<task-notification>`` block, and a
pointer to the tool that fetches the output.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent.background.manager import BackgroundTask

__all__ = ["render_notification", "render_notifications"]

_STATUS_WORDS: dict[str, str] = {
    "completed": "completed",
    "failed": "failed",
    "killed": "was killed",
    "interrupted": "was interrupted when the session ended",
}


def _headline(task: BackgroundTask) -> str:
    status = task["status"]
    if task.get("timed_out"):
        return f'Background command "{task["description"]}" timed out'
    if status == "completed":
        return f'Background command "{task["description"]}" completed (exit code {task["exit_code"]})'
    if status == "failed":
        return f'Background command "{task["description"]}" failed (exit code {task["exit_code"]})'
    return f'Background command "{task["description"]}" {_STATUS_WORDS.get(status, status)}'


def render_notification(task: BackgroundTask) -> str:
    """Render a single task's notification message body."""
    exit_code = "null" if task["exit_code"] is None else str(task["exit_code"])
    return (
        f"{_headline(task)}.\n\n"
        "<task-notification>\n"
        f"<task_id>{task['task_id']}</task_id>\n"
        f"<command>{task['command']}</command>\n"
        f"<status>{task['status']}</status>\n"
        f"<exit_code>{exit_code}</exit_code>\n"
        f"<output_path>{task['output_path']}</output_path>\n"
        "</task-notification>"
    )


def render_notifications(tasks: Sequence[BackgroundTask]) -> str:
    """Render one or more notifications as a single user message.

    Successful completions are collapsed into a leading summary line (Claude
    Code does the same to avoid flooding the context), while failures and
    timeouts always keep their own headline so they cannot be missed.
    """
    if not tasks:
        return ""

    parts: list[str] = []
    succeeded = [t for t in tasks if t["status"] == "completed"]
    if len(succeeded) > 1:
        ids = ", ".join(t["task_id"] for t in succeeded)
        parts.append(f"{len(succeeded)} background commands completed ({ids}).")
    parts.extend(render_notification(task) for task in tasks)

    ids = ", ".join(t["task_id"] for t in tasks)
    parts.append(
        f"Fetch the output with get_task_output(task_id=\"...\") for: {ids}. "
        "If a notification is not relevant to the current work, acknowledge it "
        "in one line and continue — do not re-derive results you have not read."
    )
    return "\n\n".join(parts)
