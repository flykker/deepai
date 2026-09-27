"""Agent assembly: deep agent + background task middleware."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from agent.background import BackgroundTaskManager, BackgroundTasksMiddleware

__all__ = ["BATCHING_PROMPT", "DEFAULT_MODEL", "SANDBOX_PATHS_PROMPT", "build_agent", "resolve_model_spec"]

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"
"""Model used when neither an explicit model nor ``AGENT_MODEL`` is given."""

MODEL_ENV_VAR = "AGENT_MODEL"
"""Environment variable holding a ``provider:model`` string, e.g. ``openai:gpt-5.1``."""

TIMEOUT_ENV_VAR = "AGENT_MODEL_TIMEOUT"
"""Seconds a single model request may stall before it is aborted and retried."""

RETRIES_ENV_VAR = "AGENT_MODEL_MAX_RETRIES"
"""How many times a stalled/failed model request is retried."""

DEFAULT_MODEL_TIMEOUT = 30.0
"""Silence budget for one attempt, in seconds.

Measured across eval runs: a healthy response produces its first token in
2–4 s (worst legit case seen: 16 s), while a stalled attempt is given up on
exactly at the timeout. Worst-case wait is ``(retries + 1) × (timeout +
retry overhead)``. History: 60 s → 125–190 s of dead air but no failures;
25 s → the proxy's SSL handshake exhausted all attempts and turns failed;
40 s × 2 retries still failed once against a ~2 min proxy outage. Now:
30 s per attempt (2× the worst healthy first token) with more retries, so a
call survives an outage of ~2 min while a single dead attempt is cheap.
"""
DEFAULT_MODEL_MAX_RETRIES = 4
"""Extra attempts per model call (total = this + 1).

Higher than the client default because this proxy black-holes whole
requests for 1–2 minutes at a time: with three attempts a 124 s outage
killed the turn outright (`OpenAITimeoutError`), five ride it out.
"""

OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {
        "openai",
        "deepseek",
        "openrouter",
        "together",
        "groq",
        "xai",
        "mistralai",
        "azure_openai",
        "fireworks",
        "perplexity",
        "llama",
    }
)
"""Providers served by the ``langchain_openai`` client, which accepts timeouts."""

SANDBOX_PATHS_PROMPT = """## Sandbox paths

The filesystem tools (`write_file`, `read_file`, `edit_file`, `delete`, `ls`,
`glob`) resolve an **absolute path against the sandbox root**: `/` *is* the
project root. Host paths are not accepted.

- The shell prints the real host directory (`pwd` → e.g.
  `/public/deepai/runs/x/workspace`), but passing that to `write_file` makes
  the file land at `/public/deepai/runs/x/workspace/public/deepai/runs/x/...`
  — files end up in the wrong place and tests or `read_file` cannot find them.
- Project files are always written under the root: top-level `app.py` →
  `/app.py`, nested → `/src/app.py`. To convert a `pwd` result, drop
  everything up to the workspace and keep the remainder under `/`.
- Shell commands (`execute`, background tasks) run in the real working
  directory: there, paths relative to the project and normal absolute host
  paths both work — the root rule applies only to filesystem tools."""


BATCHING_PROMPT = """## Batch independent calls into one response

A response carrying several tool calls costs one turn; the same calls sent one
per response cost one turn each — and every extra turn is latency, tokens and
wall-clock time. Before answering, collect every *independent* tool call of
the current step into that single response:

- several `write_file` calls (app + tests + config) or `read_file`/`ls` calls
  over artifacts that do not depend on each other;
- the `get_task_output` read after a completion notification **together with**
  the next action that only needed that notification (starting the next task,
  running the next check);
- independent shell checks that read different resources.

Only real dependencies force serialization: a command consuming the previous
command's output, or a check that must observe the previous action's result.
Everything else batches."""


def _env_number(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def resolve_model_spec(model: str | BaseChatModel | None = None) -> str | BaseChatModel:
    """Resolve the model from the argument, ``AGENT_MODEL``, or the default.

    OpenAI-compatible specs are turned into a configured ``ChatOpenAI`` so the
    request carries a finite timeout: the default client waits up to ten
    minutes for a stalled proxy response, which looks like a hung session.
    Total worst-case wait is ``(AGENT_MODEL_MAX_RETRIES + 1) × (timeout +
    ~3.5 s retry overhead)``.

    Args:
        model: Explicit model spec or a pre-built ``BaseChatModel``.

    Returns:
        A model instance (timeout applied) or the original spec string for
        providers this module does not configure itself.
    """
    spec = model if model is not None else os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL
    if not isinstance(spec, str):
        return spec

    provider, separator, name = spec.partition(":")
    model_id = name if separator else provider
    if not (separator and provider) or provider.lower() in OPENAI_COMPATIBLE_PROVIDERS:
        from langchain_openai import ChatOpenAI  # local import keeps startup light

        timeout = _env_number(TIMEOUT_ENV_VAR, DEFAULT_MODEL_TIMEOUT)
        retries = int(_env_number(RETRIES_ENV_VAR, DEFAULT_MODEL_MAX_RETRIES))
        return ChatOpenAI(
            model=model_id,
            timeout=timeout if timeout > 0 else None,
            max_retries=max(retries, 0),
        )
    return spec


def build_agent(
    model: str | BaseChatModel | None = None,
    *,
    manager: BackgroundTaskManager,
    root_dir: str | Path | None = None,
    system_prompt: str | None = None,
    background_enabled: bool | None = None,
    foreground_timeout: int = 120,
    checkpointer: Checkpointer | None = None,
    **agent_kwargs: Any,
) -> CompiledStateGraph:
    """Build the deep agent with background task support.

    Args:
        model: ``provider:model`` string or a ``BaseChatModel``. Falls back to
            ``AGENT_MODEL`` and then ``anthropic:claude-sonnet-4-6``.
        manager: Session-wide :class:`BackgroundTaskManager` shared by every
            tool call — background tasks outlive individual graph runs.
        root_dir: Working directory for the filesystem tools and for shell
            commands started by the agent. Defaults to the current directory.
        system_prompt: Caller-authored instructions placed first in the system
            prompt. The sandbox-paths guidance (:data:`SANDBOX_PATHS_PROMPT`)
            and the tool-batching guidance (:data:`BATCHING_PROMPT`) are
            always appended — filesystem tools treat `/` as the project root,
            which the model otherwise gets wrong by copying `pwd`, and
            unbatched calls waste one model turn each. The
            background-task guidance itself is appended by the middleware.
        background_enabled: Force the background tools on/off. ``None`` reads
            ``DEEPAI_DISABLE_BACKGROUND_TASKS``.
        foreground_timeout: Timeout (seconds) for foreground ``execute`` calls.
        checkpointer: Checkpointer used to keep conversation state and
            ``background_tasks`` across turns. Defaults to an in-memory
            checkpointer (state lives for the process lifetime).
        **agent_kwargs: Forwarded to :func:`deepagents.create_deep_agent`
            (``subagents``, ``permissions``, ``name``, ...).

    Returns:
        A compiled LangGraph agent.
    """
    backend = LocalShellBackend(
        root_dir=root_dir if root_dir is not None else Path.cwd(),
        inherit_env=True,
        timeout=foreground_timeout,
    )
    middleware = [BackgroundTasksMiddleware(manager, enabled=background_enabled)]
    effective_prompt = f"{SANDBOX_PATHS_PROMPT}\n\n{BATCHING_PROMPT}"
    if system_prompt:
        effective_prompt = f"{system_prompt.rstrip()}\n\n{effective_prompt}"
    return create_deep_agent(
        resolve_model_spec(model),
        backend=backend,
        system_prompt=effective_prompt,
        middleware=middleware,
        checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
        name=agent_kwargs.pop("name", "deepai"),
        **agent_kwargs,
    )
