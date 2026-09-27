"""CLI harness: turns, streaming output and background-task notifications.

The harness owns the part of Claude Code's background-task design that cannot
live inside the graph: once a turn ends, queued completion notifications are
injected as user messages and the agent is invoked again, so the model wakes up
with the result instead of polling. Notifications are drained both *before* a
new user message (so tasks that finished while the session was idle are not
lost) and *after* every turn.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Sequence
from typing import Any, Callable

from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage

from agent.background import BackgroundTaskManager, render_notifications
from agent.background.manager import BackgroundTask, format_task_line

__all__ = ["AgentHarness"]

_PREVIEW_CHARS = 300

_TRANSIENT_MODEL_MARKS = (
    "ConnectionError",
    "TimeoutError",
    "RateLimit",
    "InternalServerError",
    "ServiceUnavailable",
)
"""OpenAI-style error names that mean «the endpoint flaked, try again»."""


def _is_transient_model_error(exc: BaseException) -> bool:
    """True when the failure is the endpoint flaking, not the task itself.

    The proxy drops connections and times out mid-run — an eval turn once
    died 372 s in with everything else done. The turn's progress is already
    checkpointed by LangGraph, so the harness can resume from it instead of
    failing the whole turn.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__
    return any(mark in name for mark in _TRANSIENT_MODEL_MARKS)


class AgentHarness:
    """Run conversation turns against a deep agent with background tasks."""

    def __init__(
        self,
        agent: Any,
        manager: BackgroundTaskManager,
        *,
        config: dict[str, Any] | None = None,
        echo: bool = True,
        printer: Callable[[str], None] = print,
        max_notification_turns: int = 20,
        notify_when_idle: bool = True,
        model_turn_retries: int = 2,
        retry_delay: float = 2.0,
    ) -> None:
        """Wrap a compiled agent.

        Args:
            agent: Compiled LangGraph agent built by :func:`agent.graph.build_agent`.
            manager: The session's background task manager.
            config: Run config; a thread-id config is generated when omitted.
            echo: Stream text/tool progress to ``printer`` as it arrives.
            printer: Sink for echoed lines (``print`` by default).
            max_notification_turns: Safety cap on chained notification turns.
            notify_when_idle: Deliver completions in the background while no
                turn is running (Claude Code wakes the model as soon as a task
                finishes, instead of waiting for the next user message).
            model_turn_retries: How many times to resume a turn whose model
                call died on a transient error (connection drop, timeout, 5xx).
                Progress is checkpointed, so the retry continues where the
                model stopped without re-sending the input message.
            retry_delay: Base seconds between those retries (grows linearly).
        """
        self.agent = agent
        self.manager = manager
        self.echo = echo
        self.printer = printer
        self.max_notification_turns = max_notification_turns
        self.notify_when_idle = notify_when_idle
        self.model_turn_retries = max(model_turn_retries, 0)
        self.retry_delay = retry_delay
        self._text_buffer = ""
        # One graph run at a time: the idle notifier must not race a turn.
        self._lock = threading.RLock()
        self._busy = 0
        self._notifier: threading.Thread | None = None
        self._notifier_stop = threading.Event()
        self.config: dict[str, Any] = config or {
            "configurable": {"thread_id": uuid.uuid4().hex[:12]},
            "recursion_limit": 200,
        }

    # ------------------------------------------------------------------ turns

    def run_turn(self, user_text: str) -> dict[str, Any]:
        """Run one user message plus any background-task notifications it triggers.

        Args:
            user_text: Raw user input for this turn.

        Returns:
            The final graph state values.
        """
        self._ensure_notifier()
        self.flush_notifications()
        state = self._invoke([HumanMessage(user_text)])
        state = self.flush_notifications() or state
        return state

    def flush_notifications(self) -> dict[str, Any] | None:
        """Deliver queued completion notifications as their own turns.

        Called after every turn, and by the idle notifier as soon as a task
        finishes while no turn is running.

        Returns:
            The last state produced, or ``None`` if nothing was pending.
        """
        last_state: dict[str, Any] | None = None
        with self._lock:
            for _ in range(self.max_notification_turns):
                pending = self.manager.take_notifications()
                if not pending:
                    break
                if self.echo:
                    self.printer(self._headline(pending))
                try:
                    last_state = self._invoke([HumanMessage(render_notifications(pending))])
                except Exception:
                    # Keep the completions queued — the next turn retries them.
                    self.manager.requeue_notifications(pending)
                    raise
        return last_state

    @property
    def busy(self) -> bool:
        """True while a turn or a notification delivery is in flight."""
        return bool(self._busy)

    # ----------------------------------------------------------------- output

    def tasks_report(self, status_filter: str | None = None) -> str:
        """Render the tracked background tasks (the ``/tasks`` command)."""
        records = self.manager.list(status_filter)
        if not records:
            return "No background tasks tracked."
        header = f"{len(records)} background task(s) (output dir: {self.manager.output_dir}):"
        return "\n".join([header, *(format_task_line(r) for r in records)])

    # -------------------------------------------------------------- internals

    def close(self) -> None:
        """Stop the idle notifier and kill leftover background tasks."""
        self._notifier_stop.set()
        thread = self._notifier
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._notifier = None
        self.manager.shutdown()

    def _invoke(self, messages: Sequence[AnyMessage]) -> dict[str, Any]:
        # The lock makes the idle notifier and a user turn mutually exclusive:
        # two graph runs on one thread-id would fight over the checkpointer.
        with self._lock:
            self._busy += 1
            try:
                # First attempt carries the input; once it has been applied,
                # LangGraph has checkpointed it, so retries resume with `None`
                # — re-sending the messages would append them a second time.
                pending: Sequence[AnyMessage] | None = messages
                attempt = 0
                while True:
                    try:
                        if not self.echo:
                            payload = None if pending is None else {"messages": list(pending)}
                            state = self.agent.invoke(payload, config=self.config)
                            return dict(state)
                        return self._stream(pending)
                    except Exception as exc:  # noqa: BLE001 - classified below
                        if attempt >= self.model_turn_retries or not _is_transient_model_error(exc):
                            raise
                        attempt += 1
                        delay = self.retry_delay * attempt
                        self.printer(
                            f"(связь с моделью оборвалась: {type(exc).__name__}; "
                            f"повтор {attempt}/{self.model_turn_retries} через {delay:g} с — продолжаю с чекпоинта)"
                        )
                        if delay:
                            time.sleep(delay)
                        pending = None
            finally:
                self._busy -= 1

    # ------------------------------------------------------- idle notifier

    def _ensure_notifier(self) -> None:
        """Start the background thread that wakes the model on completions."""
        if not self.notify_when_idle or self._notifier is not None:
            return
        self._notifier_stop.clear()
        self._notifier = threading.Thread(
            target=self._notify_loop,
            name="task-notifier",
            daemon=True,
        )
        self._notifier.start()

    def _notify_loop(self) -> None:
        """Wait for a task to finish and deliver its notification immediately.

        Without this thread a completion would sit in the queue until the user
        happens to type again — and a model that said "I'll wait for the
        notification" would never hear back (exactly what the eval caught).
        """
        failed_attempts = 0
        while not self._notifier_stop.is_set():
            if not self.manager.wait_for_notifications(0.5):
                continue
            if not self._lock.acquire(blocking=False):
                # A turn is running; its own post-turn flush will deliver.
                time.sleep(0.2)
                continue
            try:
                self.flush_notifications()
                failed_attempts = 0
            except Exception as exc:  # noqa: BLE001 - never kill the notifier
                failed_attempts += 1
                self.printer(f"(уведомление доставить не удалось: {exc})")
                time.sleep(min(failed_attempts, 5))
            finally:
                self._lock.release()

    def _stream(self, messages: Sequence[AnyMessage] | None) -> dict[str, Any]:
        # Tool calls are accumulated from raw `tool_call_chunks` and only
        # printed once their arguments are final (a ToolMessage proves the
        # model finished emitting them): some OpenAI-compatible backends emit
        # the function name in a first delta with empty arguments, which the
        # chunk parser would otherwise show as `tool({})`.
        pending: dict[Any, dict[str, str]] = {}
        printed_ids: set[str] = set()
        self._text_buffer = ""
        payload = None if messages is None else {"messages": list(messages)}
        stream = self.agent.stream(
            payload,
            config=self.config,
            stream_mode="messages",
        )
        for msg, _metadata in stream:
            if isinstance(msg, AIMessage) or isinstance(msg, AIMessageChunk):
                if getattr(msg, "tool_call_chunks", None):
                    _merge_tool_chunks(msg, pending)
                else:  # a complete (non-streamed) message: args are already final
                    for call in getattr(msg, "tool_calls", None) or []:
                        call_id = str(call.get("id") or "")
                        key: Any = call_id or call.get("name") or f"#{len(pending)}"
                        entry = pending.setdefault(key, {"name": "", "id": "", "args": ""})
                        entry["name"] = str(call.get("name") or entry["name"])
                        if call_id:
                            entry["id"] = call_id
                        if not entry["args"]:
                            entry["args"] = _compact(call.get("args") or {})
                self._emit_text(_safe_text(msg))
            elif isinstance(msg, ToolMessage):
                self._flush_tool_calls(pending, printed_ids)
                preview = " ".join(_safe_text(msg).split())
                if len(preview) > _PREVIEW_CHARS:
                    preview = preview[:_PREVIEW_CHARS] + "…"
                name = msg.name or "tool"
                self.printer(f"  ⎿ [{name}] {preview}")
        self._flush_text()
        self._flush_tool_calls(pending, printed_ids)
        state = self.agent.get_state(self.config)
        return dict(state.values)

    def _flush_tool_calls(
        self, pending: dict[Any, dict[str, str]], printed_ids: set[str]
    ) -> None:
        """Print every accumulated tool call whose arguments are final."""
        if not pending:
            return
        # Flush any half-written prose so the tool line stays intact.
        self._flush_text()
        for key in list(pending):
            entry = pending.pop(key)
            call_id = entry.get("id", "")
            if call_id and call_id in printed_ids:
                continue
            if call_id:
                printed_ids.add(call_id)
            raw = entry.get("args", "")
            if raw.strip():
                try:
                    args: Any = json.loads(raw)
                except ValueError:  # unparseable stream: show it verbatim
                    args = raw
            else:
                args = {}
            name = entry.get("name") or "tool"
            self.printer(f"  ● {name}({_compact(args)})")

    def _emit_text(self, text: str) -> None:
        """Buffer streamed text and print only complete lines."""
        if not text:
            return
        self._text_buffer += text
        while "\n" in self._text_buffer:
            line, _, rest = self._text_buffer.partition("\n")
            self.printer(line)
            self._text_buffer = rest

    def _flush_text(self) -> None:
        """Print whatever buffered text has not ended with a newline yet."""
        if self._text_buffer:
            self.printer(self._text_buffer)
            self._text_buffer = ""

    @staticmethod
    def _headline(pending: Sequence[BackgroundTask]) -> str:
        if len(pending) == 1:
            task = pending[0]
            exit_code = task["exit_code"]
            return f'\n● Background command "{task["description"]}" finished: {task["status"]} (exit code {exit_code})'
        return f"\n● {len(pending)} background commands finished"


def _safe_text(message: Any) -> str:
    """Best-effort text extraction from a streamed message."""
    try:
        text = message.text
    except Exception:  # noqa: BLE001 - malformed content must not break streaming
        text = ""
    if not text and isinstance(getattr(message, "content", None), str):
        text = message.content
    return text if isinstance(text, str) else str(text)


def _compact(value: Any) -> str:
    """One-line JSON for tool call arguments."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


def _merge_tool_chunks(message: Any, pending: dict[Any, dict[str, str]]) -> None:
    """Merge a streamed message's ``tool_call_chunks`` into ``pending``.

    Chunk deltas carry partial name/id/arguments; they are keyed by the
    backend's chunk ``index`` (falling back to the call id, then the name) so a
    call is tracked as one entry no matter how many deltas it arrives in.
    """
    chunks = getattr(message, "tool_call_chunks", None) or []
    for chunk in chunks:
        key = chunk.get("index")
        if key is None:
            key = chunk.get("id") or chunk.get("name") or f"#{len(pending)}"
        entry = pending.setdefault(key, {"name": "", "id": "", "args": ""})
        if chunk.get("name"):
            entry["name"] = str(chunk["name"])
        if chunk.get("id"):
            entry["id"] = str(chunk["id"])
        entry["args"] += chunk.get("args") or ""
