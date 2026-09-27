"""Interactive CLI for the agent.

Run with::

    python -m agent.main --root /path/to/project

Slash commands mirror Claude Code: ``/tasks`` (a.k.a. ``/bashes``) lists
background tasks, ``/exit`` quits and kills them.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from agent.background import BackgroundTaskManager
from agent.env import load_env
from agent.graph import MODEL_ENV_VAR, build_agent, resolve_model_spec
from agent.harness import AgentHarness

__all__ = ["main", "repl"]

HELP = """Commands:
  /tasks [status]   list background tasks (aliases: /bashes); status filter:
                    running | completed | failed | killed | all
  /help             show this help
  /exit             quit (background tasks are killed on exit)

Anything else is sent to the agent. Long-running commands are backgrounded by
the agent itself via start_background_task."""


def repl(harness: AgentHarness) -> None:
    """Read-eval-print loop with slash commands."""
    print("Type /help for commands, /exit to quit.")
    while True:
        try:
            line = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        text = line.strip()
        if not text:
            continue
        if text in {"/exit", "/quit", "/q"}:
            return
        if text in {"/help", "/h"}:
            print(HELP)
            continue
        if text == "/tasks" or text.startswith("/tasks ") or text in {"/bashes"} or text.startswith("/bashes "):
            _, _, status = text.partition(" ")
            print(harness.tasks_report(status.strip() or None))
            continue
        try:
            harness.run_turn(text)
        except KeyboardInterrupt:
            print("\n(turn interrupted — background tasks keep running; use /tasks to inspect them)")
        except Exception as exc:  # noqa: BLE001 - a failed turn must not kill the session
            print(f"error: turn failed: {exc}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m agent.main``."""
    parser = argparse.ArgumentParser(prog="agent", description="Deep agent with Claude Code-style background tasks")
    parser.add_argument("--root", default=".", help="working directory for filesystem tools and shell commands")
    parser.add_argument("--model", default=None, help=f"provider:model string (default: ${MODEL_ENV_VAR} or the built-in default)")
    parser.add_argument("--prompt", default=None, help="run a single turn and exit (non-interactive)")
    parser.add_argument("--thread-id", default=None, help="resume a specific conversation thread")
    parser.add_argument("--env", default=None, help="path to a .env file (default: ./.env)")
    parser.add_argument("--quiet", action="store_true", help="do not stream tool/text output")
    args = parser.parse_args(argv)

    # Read .env (OPENAI_API_KEY / OPENAI_BASE_URL / AGENT_MODEL) before the
    # model is constructed; exported variables take precedence.
    load_env(args.env)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"error: root directory does not exist: {root}", file=sys.stderr)
        return 2

    # Stable session id per project: the manager persists task records to
    # tasks.json in that directory, so a restarted CLI keeps its task history
    # (finished tasks stay readable, dead ones come back as "interrupted").
    # DEEPAI_SESSION_ID overrides it — use it to run parallel instances.
    session_id = os.environ.get("DEEPAI_SESSION_ID") or f"root-{hashlib.sha1(str(root).encode()).hexdigest()[:10]}"
    manager = BackgroundTaskManager(cwd=root, session_id=session_id)
    config = {"configurable": {"thread_id": args.thread_id or manager.session_id}, "recursion_limit": 200}
    try:
        agent = build_agent(model=args.model, manager=manager, root_dir=root)
    except Exception as exc:  # noqa: BLE001 - surface configuration errors cleanly
        print(f"error: failed to build agent: {exc}", file=sys.stderr)
        print(
            "hint: set an API key (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY) "
            f"and choose a model with --model or ${MODEL_ENV_VAR}",
            file=sys.stderr,
        )
        return 2

    harness = AgentHarness(agent, manager, config=config, echo=not args.quiet)
    if not args.quiet:
        spec = resolve_model_spec(args.model)
        shown = getattr(spec, "model_name", None) or getattr(spec, "model", None) or spec
        print(f"model: {shown}")
        print(f"root:  {root}")
        print(f"tasks: {manager.output_dir}")

    try:
        if args.prompt is not None:
            harness.run_turn(args.prompt)
            print(harness.tasks_report())
        else:
            repl(harness)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - model/provider errors surface as many types
        print(f"error: turn failed: {exc}", file=sys.stderr)
        print(
            "hint: check the API key for your provider (ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY / GOOGLE_API_KEY) and the --model value",
            file=sys.stderr,
        )
        return 1
    finally:
        harness.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
