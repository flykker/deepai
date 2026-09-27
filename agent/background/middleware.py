"""Middleware exposing Claude Code-style background task tools.

Adds four tools to the agent:

====================== =========================================================
Tool                   Purpose
====================== =========================================================
``start_background_task``  Run a shell command in the background, return a task
                       id immediately (Claude Code's ``run_in_background``).
``get_task_output``    Read output produced since the last read, plus the live
                       status (Claude Code's ``BashOutput``).
``kill_task``          Stop a background task (Claude Code's ``KillShell``).
``list_tasks``         List tracked tasks (Claude Code's ``/tasks``).
====================== =========================================================

Task snapshots are kept in graph state under ``background_tasks`` so they
survive checkpointing, and completion notifications are produced by the
harness (see :mod:`agent.harness`) rather than by the model polling.
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.background.manager import (
    BackgroundTask,
    BackgroundTaskManager,
    UnknownTaskError,
    format_task_line,
)

__all__ = ["BackgroundTasksMiddleware", "BackgroundTasksState", "BackgroundTask"]

DISABLE_ENV_VAR = "DEEPAI_DISABLE_BACKGROUND_TASKS"
"""Set to a truthy value to drop the background tools from the schema."""


def _tasks_reducer(
    existing: dict[str, BackgroundTask] | None,
    update: dict[str, BackgroundTask],
) -> dict[str, BackgroundTask]:
    """Merge task updates into the existing ``background_tasks`` dict."""
    merged = dict(existing or {})
    merged.update(update)
    return merged


class BackgroundTasksState(AgentState):
    """Agent state extension that tracks background task snapshots."""

    background_tasks: Annotated[NotRequired[dict[str, BackgroundTask]], _tasks_reducer]


class StartBackgroundTaskSchema(BaseModel):
    """Input schema for ``start_background_task``."""

    command: str = Field(description="The shell command to run in the background.")
    description: str = Field(
        description=(
            "A short, active-voice summary of what the command does "
            "(5-10 words for simple commands). Shown in completion notifications."
        )
    )
    timeout_ms: int | None = Field(
        default=None,
        description=(
            "Optional wall-clock limit in milliseconds. On expiry the process "
            "group is killed and the task is marked failed. Omit for tasks that "
            "are expected to run until stopped (servers, watchers)."
        ),
    )


class GetTaskOutputSchema(BaseModel):
    """Input schema for ``get_task_output``."""

    task_id: str = Field(description='The task id returned by start_background_task, e.g. "bash-1".')
    filter: str | None = Field(  # noqa: A003
        default=None,
        description=(
            "Optional regular expression. Only matching lines are returned. "
            "Lines that do not match are consumed and will not be available "
            "in later reads."
        ),
    )


class KillTaskSchema(BaseModel):
    """Input schema for ``kill_task``."""

    task_id: str = Field(description='The task id returned by start_background_task, e.g. "bash-1".')


class ListTasksSchema(BaseModel):
    """Input schema for ``list_tasks``."""

    status_filter: Literal["running", "completed", "failed", "killed", "interrupted", "all"] | None = Field(
        default=None,
        description="Filter tasks by status. Defaults to 'all'.",
    )


START_TOOL_DESCRIPTION = """Start a long-running shell command in the background and return immediately.

Use this instead of `execute` whenever the command would block for more than ~10
seconds: dev servers, watch modes, builds, test suites, downloads, data
pipelines. For short commands prefer `execute`.

Returns the new task id (e.g. `bash-1`) and the path of the output file.

## Usage notes
1. Do not append `&` — backgrounding is handled internally.
2. Never wait for a task yourself: no polling loops with `get_task_output`,
   no `execute sleep ...`. Finish your reply and stop — when the task ends,
   a `<task-notification>` message is delivered to you automatically, even
   with no new user message. If you genuinely need the output sooner than
   that, a single `get_task_output` call is fine.
3. Several background tasks run at once — they are independent OS processes.
   When a job needs several long commands that do not depend on each other,
   emit several `start_background_task` calls **in the same response**
   (parallel tool calls) so they all start immediately, instead of starting
   them one after another. Report every returned id.
4. Only serialize when a later command truly needs the output of an earlier
   one; then start the next task from the completion notification.
5. Clean up: kill servers and watchers you started with `kill_task` once you
   no longer need them, otherwise they keep occupying ports and resources.
6. Report the task id to the user and end the turn; do not block waiting for
   output, and do not keep the turn alive with sleeps or repeated reads."""

GET_OUTPUT_TOOL_DESCRIPTION = (
    "Read output from a background task. Returns only the output produced since "
    "the previous call (plus the current status and exit code), so repeated calls "
    "are cheap. Use this when you need a running command's output before its "
    "completion notification arrives, or to fetch the final output of a task you "
    "already saw complete."
)

KILL_TOOL_DESCRIPTION = (
    "Stop a running background task. Sends SIGTERM to the process group and "
    "escalates to SIGKILL if the task is still alive after a grace period. "
    "Returns the task's final status."
)

LIST_TOOL_DESCRIPTION = (
    "List tracked background tasks with their current statuses. Statuses shown "
    "earlier in the conversation may be stale, so call this (or "
    "`get_task_output`) to read the current state before reporting it."
)


def _err(message: str) -> str:
    return f"Error: {message}"


def _resolve(task_id: str, runtime: ToolRuntime, manager: BackgroundTaskManager) -> BackgroundTask | str:
    """Look up a task snapshot from state, then from the (persistent) manager.

    State only knows tasks started in this conversation — records restored
    from a previous session never go through state, so the manager is the
    fallback that keeps them readable.
    """
    tasks: dict[str, BackgroundTask] = runtime.state.get("background_tasks") or {}
    tracked = tasks.get(task_id.strip())
    if tracked is None:
        tracked = manager.get(task_id.strip())
    if tracked is None:
        return f"No tracked background task with task_id {task_id!r}. Use list_tasks to see known ids."
    return tracked


def _build_start_tool(manager: BackgroundTaskManager) -> StructuredTool:
    def start_background_task(
        command: str,
        description: str,
        runtime: ToolRuntime,
        timeout_ms: int | None = None,
    ) -> str | Command:
        timeout_seconds = None if timeout_ms is None else max(timeout_ms, 1) / 1000
        try:
            record = manager.start(
                command,
                description=description,
                timeout_seconds=timeout_seconds,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return _err(f"failed to start background command: {exc}")

        msg = (
            f'Command running in background with ID: {record["task_id"]}. '
            f'Output is being written to: {record["output_path"]}\n'
            f'Command: {record["command"]}\n'
            "A notification will be injected into the conversation when it "
            'finishes. Use get_task_output(task_id="'
            f'{record["task_id"]}") if you need output sooner.'
        )
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "background_tasks": {record["task_id"]: record},
            }
        )

    return StructuredTool.from_function(
        name="start_background_task",
        func=start_background_task,
        description=START_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=StartBackgroundTaskSchema,
    )


def _build_get_output_tool(manager: BackgroundTaskManager) -> StructuredTool:
    def get_task_output(
        task_id: str,
        runtime: ToolRuntime,
        filter: str | None = None,  # noqa: A003
    ) -> str | Command:
        tracked = _resolve(task_id, runtime, manager)
        if isinstance(tracked, str):
            return tracked
        try:
            snapshot = manager.read(tracked["task_id"], filter=filter)
        except UnknownTaskError as exc:
            return _err(str(exc))

        lines = [
            f"task_id: {snapshot.task_id}",
            f"status: {snapshot.status}",
            f"exit_code: {snapshot.exit_code if snapshot.exit_code is not None else 'n/a'}",
        ]
        if snapshot.truncated:
            lines.append("output: (truncated — narrow with the filter argument if you need specific lines)")
        output = snapshot.new_output if snapshot.new_output.strip() else "<no new output>"
        lines.append(f"output:\n{output}")
        if snapshot.status in {"completed", "failed"} and not snapshot.new_output.strip():
            lines.append("(task has finished; this was its final output)")

        current = manager.get(snapshot.task_id) or tracked
        return Command(
            update={
                "messages": [ToolMessage("\n".join(lines), tool_call_id=runtime.tool_call_id)],
                "background_tasks": {snapshot.task_id: current},
            }
        )

    return StructuredTool.from_function(
        name="get_task_output",
        func=get_task_output,
        description=GET_OUTPUT_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=GetTaskOutputSchema,
    )


def _build_kill_tool(manager: BackgroundTaskManager) -> StructuredTool:
    def kill_task(task_id: str, runtime: ToolRuntime) -> str | Command:
        tracked = _resolve(task_id, runtime, manager)
        if isinstance(tracked, str):
            return tracked
        try:
            record = manager.kill(tracked["task_id"])
        except UnknownTaskError as exc:
            return _err(str(exc))

        if record["status"] == "killed":
            msg = f'Stopped task {record["task_id"]} ({record["command"]})'
        else:
            msg = (
                f'Task {record["task_id"]} was already {record["status"]} '
                f'(exit code {record["exit_code"]}); nothing to stop.'
            )
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "background_tasks": {record["task_id"]: record},
            }
        )

    return StructuredTool.from_function(
        name="kill_task",
        func=kill_task,
        description=KILL_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=KillTaskSchema,
    )


def _build_list_tool(manager: BackgroundTaskManager) -> StructuredTool:
    def list_tasks(
        runtime: ToolRuntime,
        status_filter: Literal["running", "completed", "failed", "killed", "interrupted", "all"] | None = None,
    ) -> str | Command:
        records = manager.list(status_filter)
        if not records:
            return "No background tasks tracked."

        msg = f"{len(records)} background task(s):\n" + "\n".join(format_task_line(r) for r in records)
        updates = {r["task_id"]: r for r in records}
        return Command(update={"messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)], "background_tasks": updates})

    return StructuredTool.from_function(
        name="list_tasks",
        func=list_tasks,
        description=LIST_TOOL_DESCRIPTION,
        infer_schema=False,
        args_schema=ListTasksSchema,
    )


BACKGROUND_SYSTEM_PROMPT = """## Background tasks

Long-running shell commands belong in the background:

- `start_background_task(command=..., description=...)` starts a command and
  returns a task id immediately. Use it for dev servers, watch modes, builds,
  test suites, downloads — anything that would block `execute` for more than
  ~10 seconds. Do not append `&`.
- Deciding how to run: a command whose result you need before continuing stays
  in the foreground with `execute`; a command you only need to *finish* (or be
  told about) goes to the background.
- Independent long commands run **in parallel**: when a task needs several
  commands that do not depend on each other (e.g. build + tests + download),
  put several `start_background_task` calls into a single response so they all
  start at once, then report every returned id. Serialize only when a command
  consumes the output of the previous one.
- You are notified when a background task finishes; a `<task-notification>`
  message is injected into the conversation with the task id and exit code —
  it is delivered **on its own, without any new user message**, so ending your
  turn is how you "wait": report what you started, stop, and you will be woken
  up. Do not poll `get_task_output` in a row and never sleep the turn away —
  not with `execute sleep ...` and not inside a compound command either
  (`sleep 30; cat ...` was caught doing exactly this): every poll or sleep is
  wasted turns, tokens and wall-clock time.
- Notifications for output you have already read to the end are dropped by
  the harness — a notification that does arrive is always news. If one still
  refers to a task you just checked, acknowledge it in a line and move on:
  do not re-read it and do not re-audit completed work to "double-check"
  your own earlier summary — re-verification storms cost more than any
  single mistake they could find.
- `get_task_output(task_id=...)` returns only output produced since your last
  call, together with the current status. Read it after a completion
  notification when you need the result, or when you need progress sooner.
  Put that read and the next independent action (starting the next task,
  running the next check) into the **same response** — a read alone in a
  response costs an extra turn for nothing.
- `list_tasks()` shows every tracked task; statuses seen earlier in the
  conversation may be stale, so re-read them before reporting.
- `kill_task(task_id=...)` stops a task. Always kill servers/watchers you
  started once they are no longer needed."""


class BackgroundTasksMiddleware(AgentMiddleware):
    """Register background task tools, state and prompt guidance.

    Args:
        manager: Shared :class:`BackgroundTaskManager` for this session.
        enabled: When ``False`` (or the ``DEEPAI_DISABLE_BACKGROUND_TASKS``
            environment variable is truthy) no tools or prompt fragment are
            registered, mirroring Claude Code's ``CLAUDE_CODE_DISABLE_BACKGROUND_TASKS``.
        system_prompt: Override the default background-tasks prompt fragment.
    """

    state_schema = BackgroundTasksState

    def __init__(
        self,
        manager: BackgroundTaskManager,
        *,
        enabled: bool | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Build the background task tools."""
        super().__init__()
        if enabled is None:
            enabled = os.environ.get(DISABLE_ENV_VAR, "").strip().lower() not in {"1", "true", "yes", "on"}
        self.manager = manager
        self.enabled = enabled

        tools: list[BaseTool] = []
        if enabled:
            tools = [
                _build_start_tool(manager),
                _build_get_output_tool(manager),
                _build_kill_tool(manager),
                _build_list_tool(manager),
            ]
        self.tools = tools
        if not enabled:
            self.system_prompt: str | None = None
        else:
            self.system_prompt = BACKGROUND_SYSTEM_PROMPT if system_prompt is None else system_prompt
