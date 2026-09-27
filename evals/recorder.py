"""JSONL event recorder: what the model and tools actually did during a run.

Attached to a run through LangChain callbacks (``{"callbacks": [recorder]}``),
so the agent itself stays untouched. Events are appended to ``events.jsonl``
as they happen, which makes them inspectable even if a run crashes or is
killed by the time limit.

Event kinds:

=============== =========================================================
``run_start``   scenario/prompt issued by the runner
``turn_start``  a harness turn (user message or notification) began
``turn_end``    that turn finished
``llm_start``   a model request went out
``first_token`` first streamed token arrived (``wait_ms`` = stall detector)
``llm_end``     model answered — includes its tool calls and usage
``llm_error``   model request failed (a retry follows unless exhausted)
``tool_start``  a tool call began (name + arguments)
``tool_end``    the tool returned (duration + output preview)
``check``       an eval check produced its verdict
``timeout``     the scenario time limit was hit
``run_end``     the runner finished scoring
=============== =========================================================
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

__all__ = ["RunRecorder", "read_events"]

_PREVIEW = 400
"""Characters of tool output kept in the event log."""


class RunRecorder(BaseCallbackHandler):
    """Append-only JSONL log of model/tool activity for one eval run."""

    raise_error = False  # a recording problem must never fail the run

    def __init__(self, path: str | Path) -> None:
        """Open ``path`` for appending and seed the timeline at t=0."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0
        self._t0 = time.monotonic()
        self._llm_started: dict[str, float] = {}
        self._first_token: dict[str, float] = {}
        self._tool_started: dict[str, tuple[float, str]] = {}

    # ------------------------------------------------------------- emitting

    def emit(self, kind: str, **data: Any) -> dict[str, Any]:
        """Write one event now (thread-safe, flushed immediately)."""
        with self._lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "t": round(time.monotonic() - self._t0, 3),
                "kind": kind,
                **data,
            }
            self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        return event

    def close(self) -> None:
        """Flush and close the events file."""
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    # ------------------------------------------------------------ callbacks

    def on_llm_start(self, serialized: dict, prompts: list[str], run_id: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self._llm_started[str(run_id)] = time.monotonic()
        self.emit("llm_start", run=str(run_id), prompts=len(prompts))

    def on_llm_new_token(self, token: str, run_id: Any, **kwargs: Any) -> None:  # noqa: ANN401
        rid = str(run_id)
        started = self._llm_started.get(rid)
        if started is None or rid in self._first_token:
            return
        now = time.monotonic()
        self._first_token[rid] = now
        self.emit("first_token", run=rid, wait_ms=round((now - started) * 1000))

    def on_llm_end(self, result: LLMResult, run_id: Any, **kwargs: Any) -> None:  # noqa: ANN401
        rid = str(run_id)
        started = self._llm_started.pop(rid, None)
        first = self._first_token.pop(rid, None)
        now = time.monotonic()
        # `generations` is list[list[Generation]] — one inner list per prompt.
        inner = result.generations[0] if result.generations else []
        message = inner[0].message if inner and hasattr(inner[0], "message") else None
        tool_calls = [
            {"id": call.get("id"), "name": call.get("name"), "args": call.get("args")}
            for call in (getattr(message, "tool_calls", None) or [])
        ]
        self.emit(
            "llm_end",
            run=rid,
            duration_ms=round((now - started) * 1000) if started else None,
            first_token_ms=round((first - started) * 1000) if started and first else None,
            tool_calls=tool_calls or None,
            usage=_usage(result, message),
            text_len=len(_text_of(message)),
        )

    def on_llm_error(self, error: BaseException, run_id: Any, **kwargs: Any) -> None:  # noqa: ANN401
        rid = str(run_id)
        self._llm_started.pop(rid, None)
        self._first_token.pop(rid, None)
        self.emit("llm_error", run=rid, error=f"{type(error).__name__}: {error}")

    def on_tool_start(
        self,
        serialized: dict | None,
        input_str: str,
        run_id: Any,
        *,
        name: str | None = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:  # noqa: ANN401
        tool_name = name or (serialized or {}).get("name") or "tool"
        args = inputs if inputs is not None else input_str
        self._tool_started[str(run_id)] = (time.monotonic(), tool_name)
        self.emit("tool_start", run=str(run_id), name=tool_name, args=_preview(args))

    def on_tool_end(self, output: Any, run_id: Any, **kwargs: Any) -> None:  # noqa: ANN401
        started, name = self._tool_started.pop(str(run_id), (None, "tool"))
        self.emit(
            "tool_end",
            run=str(run_id),
            name=name,
            duration_ms=round((time.monotonic() - started) * 1000) if started else None,
            output=_preview(output),
        )

    def on_tool_error(self, error: BaseException, run_id: Any, **kwargs: Any) -> None:  # noqa: ANN401
        started, name = self._tool_started.pop(str(run_id), (None, "tool"))
        self.emit(
            "tool_error",
            run=str(run_id),
            name=name,
            error=f"{type(error).__name__}: {error}",
        )


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read an ``events.jsonl`` file back into a list of dicts."""
    events: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _text_of(message: Any) -> str:
    """Best-effort text of a message (content may be a list of blocks)."""
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    try:
        return " ".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    except Exception:  # noqa: BLE001 - exotic content shapes are not eval-critical
        return str(content)


def _usage(result: LLMResult, message: Any) -> dict[str, int] | None:
    """Token usage when the provider reports it (OpenAI-compatible do)."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
        }
    token_usage = (result.llm_output or {}).get("token_usage") if result.llm_output else None
    if isinstance(token_usage, dict):
        return {
            "input": token_usage.get("prompt_tokens", 0),
            "output": token_usage.get("completion_tokens", 0),
        }
    return None


def _preview(value: Any, limit: int = _PREVIEW) -> str:
    """Short single-line-ish rendering of an argument or tool output."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = str(value)
    text = str(text)
    if len(text) > limit:
        return text[:limit] + "…"
    return text
