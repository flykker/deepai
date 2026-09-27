"""Run one scenario end to end: sandbox → agent turn → checks → reports.

The runner never trusts the agent's own claims: after the turn it reads the
workspace, executes the produced tests, starts the produced server, and
inspects the recorded event log before scoring. Everything lands in a
self-contained ``runs/<timestamp>_<scenario>/`` directory.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.background import BackgroundTaskManager
from agent.graph import build_agent, resolve_model_spec
from agent.harness import AgentHarness
from evals.checks import Check, CheckResult, Context, budget_results, run_all, summarize
from evals.recorder import RunRecorder, read_events
from evals.report import build_metrics, write_reports

__all__ = ["RunResult", "Scenario", "run_scenario"]


@dataclass
class Scenario:
    """A named task for the agent plus the checks its result must satisfy."""

    name: str
    prompt: str
    checks: tuple[Check, ...] = ()
    time_limit: float = 300.0
    """Whole-run budget: the turn is joined with it, and whatever is left is
    the settling (quiesce) budget — notifications continue the work after the
    first turn, and scoring mid-turn would judge a half-finished run."""
    description: str = ""
    max_llm_calls: int | None = None
    """Hard budget on model turns: «correct but twice as expensive» must not
    pass silently — efficiency regressions flip the verdict (see
    ``evals.checks.budget_results``)."""
    max_polls: int | None = None
    """Hard budget on ``get_task_output`` calls (polling instead of
    waiting for notifications)."""
    max_duration_s: float | None = None
    """Soft budget on total run time: stalls come from proxy weather, so an
    overrun is reported as a warning, not a FAIL."""
    setup: Callable[[Path], None] | None = None
    """Plant the workspace's starting state before the agent's turn — the
    ``debug`` scenario seeds a pre-broken app plus failing tests here."""


@dataclass
class RunResult:
    """Outcome of a single scenario run (also mirrored in report.json)."""

    scenario: str
    run_dir: Path
    verdict: str
    results: list[CheckResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    report_md: Path | None = None
    report_json: Path | None = None
    timed_out: bool = False
    error: str | None = None

    @property
    def passed(self) -> bool:
        """True when the run met every hard check."""
        return self.verdict == "PASS"


def _transcript(messages: list[Any], limit: int = 4_000) -> list[dict[str, Any]]:
    """Flatten graph messages into a JSON-able conversation record."""
    out: list[dict[str, Any]] = []
    for message in messages:
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        entry: dict[str, Any] = {
            "type": getattr(message, "type", "?"),
            "name": getattr(message, "name", None),
            "content": content[:limit],
        }
        calls = getattr(message, "tool_calls", None)
        if calls:
            entry["tool_calls"] = [
                {"name": c.get("name"), "args": c.get("args")} for c in calls
            ]
        out.append(entry)
    return out


def _quiesce(
    harness: AgentHarness,
    manager: BackgroundTaskManager,
    recorder: RunRecorder,
    *,
    timeout: float = 60.0,
    quiet_s: float = 2.0,
) -> bool:
    """Wait until the session settles before scoring the run.

    The model legitimately stops after starting background work, and its
    notifications are delivered asynchronously (idle notifier). Scoring the
    instant ``run_turn`` returns would judge a half-finished conversation, so
    wait for: no running tasks, nothing queued, no delivery in flight — stable
    for ``quiet_s`` seconds, at most ``timeout``.
    """
    started = time.monotonic()
    deadline = started + timeout
    stable = started
    ok = False
    while time.monotonic() < deadline:
        running = any(t["status"] == "running" for t in manager.list())
        idle = not running and manager.pending_notifications() == 0 and not harness.busy
        now = time.monotonic()
        if idle:
            if now - stable >= quiet_s:
                ok = True
                break
        else:
            stable = now
        time.sleep(0.2)
    recorder.emit("quiesce", waited_s=round(time.monotonic() - started, 1), ok=ok)
    return ok


def run_scenario(
    scenario: Scenario,
    *,
    runs_dir: str | Path = "runs",
    model: str | Any | None = None,
    echo: bool = True,
) -> RunResult:
    """Execute ``scenario`` in a fresh sandbox and score the outcome.

    Args:
        scenario: The scenario to run.
        runs_dir: Root directory for run artifacts.
        model: Model spec forwarded to :func:`agent.graph.build_agent`.
        echo: Stream the agent's progress to stdout while it runs.

    Returns:
        A :class:`RunResult` with the verdict, metrics and report paths.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Absolute: every artifact path is used from subprocesses whose cwd is the
    # workspace, so a relative run dir would resolve twice.
    run_dir = Path(runs_dir).resolve() / f"{stamp}_{scenario.name}"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    recorder = RunRecorder(run_dir / "events.jsonl")
    recorder.emit("run_start", scenario=scenario.name, prompt=scenario.prompt)
    if scenario.setup is not None:
        scenario.setup(workspace)
        recorder.emit("setup", files=sorted(p.name for p in workspace.iterdir()))

    def printer(line: str) -> None:
        if echo:
            print(line, flush=True)

    manager = BackgroundTaskManager(output_dir=run_dir / "tasks", cwd=workspace)
    agent = build_agent(model, manager=manager, root_dir=workspace)
    config: dict[str, Any] = {
        "configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:8]}"},
        "recursion_limit": 300,
        "callbacks": [recorder],
    }
    harness = AgentHarness(agent, manager, config=config, echo=echo, printer=printer)

    error: BaseException | None = None
    timed_out = False
    started = time.monotonic()

    def _turn() -> None:
        nonlocal error
        try:
            harness.run_turn(scenario.prompt)
        except BaseException as exc:  # noqa: BLE001 - reported, not raised
            error = exc

    recorder.emit("turn_start", limit=scenario.time_limit)
    worker = threading.Thread(target=_turn, name="eval-turn", daemon=True)
    worker.start()
    worker.join(scenario.time_limit)
    timed_out = worker.is_alive()
    if timed_out:
        recorder.emit("timeout", limit_s=scenario.time_limit)
        printer(f"[eval] лимит {scenario.time_limit:.0f}s исчерпан — продолжаю оценивать")
    elif error is not None:
        recorder.emit("turn_error", error=f"{type(error).__name__}: {error}")
    recorder.emit("turn_end", duration_s=round(time.monotonic() - started, 1))
    settled = True
    if not timed_out:
        # The settling budget is whatever is left of the scenario's time
        # limit, not an arbitrary fixed window: notifications are delivered
        # asynchronously and their turns legitimately continue the work.
        # Scoring while a turn is still in flight truncates the transcript
        # and reports tasks as "leaked" before the agent had a chance to
        # stop them (caught on webapp run 20260926-221939: quiesce gave up
        # at 60 s with the model mid-call, server still running).
        remaining = scenario.time_limit - (time.monotonic() - started)
        settled = _quiesce(harness, manager, recorder, timeout=max(remaining, 4.0))

    # ---- collect evidence (before cleanup, so leaks are still visible) ----
    try:
        state = agent.get_state(harness.config)
        transcript = _transcript(list(state.values.get("messages", [])))
    except Exception:  # noqa: BLE001 - a hung turn may still hold the checkpointer
        transcript = []
    (run_dir / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    events = read_events(recorder.path)
    tasks = manager.list()

    results: list[CheckResult] = [
        CheckResult(
            name="ход выполнен вовремя и без ошибок",
            ok=not timed_out and error is None and settled,
            detail=(
                "лимит исчерпан"
                if timed_out
                else (
                    f"{type(error).__name__}: {error}"
                    if error
                    else ("не успел осесть за лимит сценария" if not settled else "ok")
                )
            ),
        )
    ]
    ctx = Context(workspace=workspace, manager=manager, events=events, transcript=transcript)
    results += run_all(scenario.checks, ctx)
    metrics = build_metrics(events, duration_s=time.monotonic() - started)
    # Budgets come last: they compare the run's own numbers against the
    # scenario's caps, so a correct-but-expensive run fails instead of
    # quietly regressing the cost of the agent.
    results += budget_results(
        {
            "llm_calls": scenario.max_llm_calls,
            "output_polls": scenario.max_polls,
            "duration_s": scenario.max_duration_s,
        },
        metrics,
    )
    for result in results:
        recorder.emit("check", name=result.name, ok=result.ok, soft=result.soft, detail=result.detail)

    verdict = "PASS" if summarize(results) else "FAIL"

    model_name = resolve_model_spec(model)
    model_name = getattr(model_name, "model_name", None) or str(model_name)
    md_path, json_path = write_reports(
        run_dir,
        scenario=scenario.name,
        prompt=scenario.prompt,
        results=results,
        metrics=metrics,
        events=events,
        workspace=workspace,
        tasks=tasks,
        model=model_name,
    )
    recorder.emit("run_end", verdict=verdict, run_dir=str(run_dir))
    recorder.close()

    try:
        harness.close()  # kills anything the agent left running
    except Exception as exc:  # noqa: BLE001 - cleanup must not lose the report
        printer(f"[eval] cleanup failed: {exc}")

    return RunResult(
        scenario=scenario.name,
        run_dir=run_dir,
        verdict=verdict,
        results=results,
        metrics=metrics,
        report_md=md_path,
        report_json=json_path,
        timed_out=timed_out,
        error=f"{type(error).__name__}: {error}" if error else None,
    )
