"""``python -m evals`` — run scenarios, print verdicts, diff against a baseline.

Examples::

    python -m evals --list
    python -m evals parallel
    python -m evals webapp
    python -m evals --all --baseline runs/20260926-120000_webapp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agent.env import load_env
from evals.report import diff_metrics, load_report
from evals.runner import RunResult, run_scenario
from evals.scenarios import SCENARIOS, get_scenario, scenario_names

__all__ = ["main"]


def _load_baseline(path: str) -> dict[str, Any]:
    """Accept either a run directory or a direct path to report.json."""
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "report.json"
    if not candidate.exists():
        raise SystemExit(f"baseline not found: {candidate}")
    return load_report(candidate)


def _print_result(result: RunResult) -> None:
    """Console summary for one run."""
    print(f"\n{result.verdict}: {result.scenario}")
    m = result.metrics
    print(
        "  "
        f"ходов LLM: {m.get('llm_calls')} | tool-вызовов: {m.get('tool_calls')} | "
        f"фон: {m.get('background_starts')} (батчей: {m.get('parallel_batches')}, "
        f"поллингов: {m.get('output_polls')}) | столлы: {m.get('stalls')} | "
        f"{m.get('duration_s')}s"
    )
    for check in result.results:
        if not check.ok:
            mark = "warn" if check.soft else "FAIL"
            print(f"  [{mark}] {check.name}: {check.detail}")
    if result.error:
        print(f"  error: {result.error}")
    if result.report_md:
        print(f"  отчёт: {result.report_md}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m evals``."""
    parser = argparse.ArgumentParser(prog="evals", description="Run agent evaluation scenarios")
    parser.add_argument("scenario", nargs="?", default=None, help="scenario name (see --list)")
    parser.add_argument("--all", action="store_true", help="run every scenario")
    parser.add_argument("--list", action="store_true", help="list available scenarios")
    parser.add_argument("--trend", action="store_true", help="print the metrics history of all past runs and exit")
    parser.add_argument("--runs-dir", default="runs", help="where run artifacts go (default: runs/)")
    parser.add_argument("--baseline", default=None, help="previous run dir or report.json to diff against")
    parser.add_argument("--model", default=None, help="model spec override (provider:model)")
    parser.add_argument("--quiet", action="store_true", help="do not stream the agent's progress")
    args = parser.parse_args(argv)

    load_env()  # .env: OPENAI_* / AGENT_MODEL / AGENT_MODEL_TIMEOUT

    if args.list:
        for name in scenario_names():
            scenario = SCENARIOS[name]
            print(f"{name:12} {scenario.description or scenario.prompt.splitlines()[0]}")
        return 0

    if args.trend:
        from evals.report import load_trends, render_trends

        print(render_trends(load_trends(args.runs_dir)))
        return 0

    if args.all:
        names = scenario_names()
    elif args.scenario:
        names = [args.scenario]
    else:
        parser.error("specify a scenario name or --all")

    baseline = _load_baseline(args.baseline) if args.baseline else None
    exit_code = 0
    for name in names:
        scenario = get_scenario(name)
        print(f"=== eval: {name} ===")
        result = run_scenario(
            scenario,
            runs_dir=args.runs_dir,
            model=args.model,
            echo=not args.quiet,
        )
        _print_result(result)
        if baseline and len(names) == 1:
            print("\nдельта к baseline:")
            rows = diff_metrics(baseline, result.metrics)
            print("\n".join(rows) if rows else "  (метрики не изменились)")
        if not result.passed:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
