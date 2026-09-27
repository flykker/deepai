"""Metrics, ``report.md``/``report.json`` rendering and baseline diffs.

The report is the artefact a human actually reads after a run: verdict,
metrics, auto-generated hints about weak spots, the artifact list and a
compact timeline. ``report.json`` carries the same numbers in machine form so
``--baseline`` can show what a prompt/tool change did to them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.checks import CheckResult, summarize

__all__ = ["build_metrics", "diff_metrics", "hints", "render_markdown", "write_reports"]

STALL_MS = 10_000
"""A model request whose first token took longer than this counts as a stall."""

REPORT_JSON = "report.json"
REPORT_MD = "report.md"


def wait_ms(event: dict[str, Any]) -> int:
    """How long the model kept us waiting before it started answering.

    Uses the first streamed token when the turn was streamed (harness with
    ``echo``) and falls back to the full call duration otherwise: in quiet
    mode the harness invokes the graph without streaming, no token callbacks
    fire, and a stalled proxy would otherwise be invisible to the metrics
    (a 58 s call recorded as «0 stalls»).
    """
    return int(event.get("first_token_ms") or event.get("duration_ms") or 0)


def build_metrics(events: list[dict[str, Any]], duration_s: float) -> dict[str, Any]:
    """Reduce the event log to the numbers we iterate on."""
    llm_ends = [e for e in events if e["kind"] == "llm_end"]
    tool_starts = [e for e in events if e["kind"] == "tool_start"]
    by_name = Counter(e.get("name", "?") for e in tool_starts)
    batches = [len(e.get("tool_calls") or []) for e in llm_ends]
    waits = [wait_ms(e) for e in llm_ends]
    stalls = [w for w in waits if w >= STALL_MS]
    usage_in = sum((e.get("usage") or {}).get("input", 0) for e in llm_ends)
    usage_out = sum((e.get("usage") or {}).get("output", 0) for e in llm_ends)

    metrics: dict[str, Any] = {
        "duration_s": round(duration_s, 1),
        "llm_calls": sum(1 for e in events if e["kind"] == "llm_start"),
        "tool_calls": len(tool_starts),
        "background_starts": by_name.get("start_background_task", 0),
        "output_polls": by_name.get("get_task_output", 0),
        "kills": by_name.get("kill_task", 0),
        "parallel_batches": sum(1 for b in batches if b >= 2),
        "max_batch": max(batches, default=0),
        "stalls": len(stalls),
        "worst_first_token_ms": max(waits, default=0),
        # How much of the run was spent waiting on the proxy: a run whose
        # waits dominate its duration measures the weather, not the agent.
        # `wait_streamed` marks whether the waits are honest first-token
        # timings (streamed) or full-call fallbacks (quiet mode).
        "total_wait_s": round(sum(waits) / 1000, 1),
        "wait_streamed": any(e.get("first_token_ms") for e in llm_ends),
        "model_errors": sum(1 for e in events if e["kind"] == "llm_error"),
        "tool_errors": sum(1 for e in events if e["kind"] == "tool_error"),
        "tokens_in": usage_in,
        "tokens_out": usage_out,
    }
    metrics["tool_calls_by_name"] = dict(sorted(by_name.items()))
    return metrics


def hints(metrics: dict[str, Any], results: list[CheckResult]) -> list[str]:
    """Plain-language suggestions for the next iteration of the agent."""
    ideas: list[str] = []
    if metrics["output_polls"] > 3:
        ideas.append(
            f"модель {metrics['output_polls']} раз вызвала `get_task_output` — похоже на поллинг вместо "
            "ожидания уведомления; усилить пункт промпта про «не опрашивать»"
        )
    if metrics["background_starts"] == 0:
        ideas.append("ни одной фоновой задачи — правило «>10s → в фон» не сработало, усилить описание инструмента")
    if metrics["parallel_batches"] == 0 and metrics["background_starts"] >= 2:
        ideas.append("задачи запускались по одной — стоит напомнить про батчи в одном ответе")
    if metrics["stalls"]:
        ideas.append(
            f"{metrics['stalls']} столл(ов) модели, худшее ожидание {metrics['worst_first_token_ms']} мс — "
            "либо прокси «залипает», либо стоит подстроить AGENT_MODEL_TIMEOUT "
            "(худший случай ≈ (ретраи+1) × таймаут)"
        )
    if metrics["stalls"] >= 3:
        ideas.append(
            "столлов несколько подряд — похоже на деградацию прокси, а не на поведение агента: "
            "повторить прогон позже, прежде чем что-то менять"
        )
    if (
        metrics.get("wait_streamed")
        and metrics.get("duration_s")
        and metrics["total_wait_s"] >= 0.5 * metrics["duration_s"]
    ):
        share = round(100 * metrics["total_wait_s"] / metrics["duration_s"])
        ideas.append(
            f"{share}% прогона ушло на ожидание первых токенов "
            f"({metrics['total_wait_s']} с из {metrics['duration_s']} с) — такой прогон меряет "
            "погоду прокси, а не агента: повторить позже, ничего не меняя"
        )
    if metrics["model_errors"]:
        ideas.append(f"ошибок модели: {metrics['model_errors']} — посмотреть ретраи в events.jsonl")
    failed = [r.name for r in results if not r.ok and not r.soft]
    if failed:
        ideas.append("не прошли жёсткие проверки: " + "; ".join(failed))
    return ideas


def render_markdown(
    *,
    scenario: str,
    prompt: str,
    results: list[CheckResult],
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    workspace: Path,
    tasks: list[dict[str, Any]],
    model: str,
) -> str:
    """Render the human-readable report for one run."""
    passed = [r for r in results if r.ok]
    hard = [r for r in results if not r.soft]
    verdict = "PASS" if summarize(results) else "FAIL"

    lines = [
        f"# Eval: {scenario} — {verdict}",
        "",
        f"- модель: `{model}`",
        f"- длительность: {metrics['duration_s']} s",
        f"- проверки: {len(passed)}/{len(results)} (жёстких: {sum(1 for r in hard if r.ok)}/{len(hard)})",
        "",
        "## Задача",
        "",
        "```",
        prompt.strip(),
        "```",
        "",
        "## Метрики",
        "",
        "| метрика | значение |",
        "|---|---|",
    ]
    for key in (
        "llm_calls",
        "tool_calls",
        "background_starts",
        "output_polls",
        "parallel_batches",
        "max_batch",
        "kills",
        "stalls",
        "worst_first_token_ms",
        "total_wait_s",
        "model_errors",
        "tool_errors",
        "tokens_in",
        "tokens_out",
        "duration_s",
    ):
        lines.append(f"| {key} | {metrics[key]} |")
    lines += ["", "Вызовы инструментов: " + ", ".join(f"`{k}`×{v}" for k, v in metrics["tool_calls_by_name"].items())]

    lines += ["", "## Проверки", "", "| чек | результат | деталь |", "|---|---|---|"]
    for r in results:
        mark = "✅" if r.ok else ("⚠️" if r.soft else "❌")
        detail = r.detail.replace("|", "\\|").replace("\n", " ⏎ ")
        lines.append(f"| {mark} {r.name} | {'ok' if r.ok else 'FAIL'} | {detail} |")

    ideas = hints(metrics, results)
    if ideas:
        lines += ["", "## Слабые места / идеи для улучшения", ""]
        lines += [f"- {idea}" for idea in ideas]

    lines += ["", "## Артефакты", "", "### workspace", ""]
    files = sorted(p for p in workspace.rglob("*") if p.is_file() and ".pytest_cache" not in p.parts and "__pycache__" not in p.parts)
    if not files:
        lines.append("_пусто_")
    for path in files:
        size = path.stat().st_size
        lines.append(f"- `{path.relative_to(workspace)}` ({size} B)")

    lines += ["", "### фоновые задачи", ""]
    if not tasks:
        lines.append("_задач не было_")
    for task in tasks:
        lines.append(
            f"- `{task['task_id']}` {task['status']} (exit={task['exit_code']}) "
            f"`{task['command']}` → {Path(task['output_path']).name}"
        )

    slow = sorted(
        (e for e in events if e["kind"] == "llm_end"),
        key=wait_ms,
        reverse=True,
    )[:5]
    slow = [e for e in slow if wait_ms(e) >= 1_000]
    if slow:
        lines += ["", "## Самые медленные запросы модели", "", "| t (s) | ожидание (ms) | всего (ms) | tool-вызовы |", "|---|---|---|---|"]
        for e in slow:
            calls = ", ".join(c.get("name") or "?" for c in (e.get("tool_calls") or [])) or "—"
            lines.append(
                f"| {e['t']} | {e.get('first_token_ms')} | {e.get('duration_ms')} | {calls} |"
            )

    tool_seq = [e for e in events if e["kind"] == "tool_start"]
    if tool_seq:
        lines += ["", "## Последовательность инструментов", ""]
        for e in tool_seq:
            args = (e.get("args") or "").replace("\n", " ")
            lines.append(f"- [{e['t']}] `{e.get('name')}` {args}")

    lines.append("")
    return "\n".join(lines)


def write_reports(
    run_dir: Path,
    *,
    scenario: str,
    prompt: str,
    results: list[CheckResult],
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    workspace: Path,
    tasks: list[dict[str, Any]],
    model: str,
) -> tuple[Path, Path]:
    """Write ``report.md`` and ``report.json``; return their paths."""
    md = render_markdown(
        scenario=scenario,
        prompt=prompt,
        results=results,
        metrics=metrics,
        events=events,
        workspace=workspace,
        tasks=tasks,
        model=model,
    )
    md_path = run_dir / REPORT_MD
    md_path.write_text(md, encoding="utf-8")

    payload = {
        "scenario": scenario,
        "verdict": "PASS" if summarize(results) else "FAIL",
        "model": model,
        "metrics": metrics,
        "checks": [
            {"name": r.name, "ok": r.ok, "soft": r.soft, "detail": r.detail} for r in results
        ],
        "hints": hints(metrics, results),
        "artifacts": {
            "workspace": [str(p.relative_to(workspace)) for p in sorted(workspace.rglob("*")) if p.is_file()],
            "tasks": [
                {
                    "task_id": t["task_id"],
                    "status": t["status"],
                    "command": t["command"],
                    "output": t["output_path"],
                }
                for t in tasks
            ],
        },
    }
    json_path = run_dir / REPORT_JSON
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return md_path, json_path


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a ``report.json`` produced by a previous run."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diff_metrics(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Human-readable deltas for every numeric metric that changed."""
    base = baseline.get("metrics", baseline)
    cur = current.get("metrics", current)
    rows: list[str] = []
    for key in sorted(set(base) | set(cur)):
        old, new = base.get(key), cur.get(key)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old != new:
            delta = round(new - old, 1)
            arrow = "↑" if delta > 0 else "↓"
            rows.append(f"- {key}: {old} → {new} ({arrow} {abs(delta)})")
    return rows


# -------------------------------------------------------------------- trends

TREND_COLUMNS: tuple[tuple[str, str], ...] = (
    ("run", "прогон"),
    ("scenario", "сценарий"),
    ("verdict", "вердикт"),
    ("llm_calls", "ходов"),
    ("tool_calls", "tool"),
    ("background_starts", "фон"),
    ("output_polls", "поллинги"),
    ("stalls", "столлы"),
    ("duration_s", "сек"),
    ("tokens_in", "токенов-in"),
)
"""Column key → header for the history table."""


def load_trends(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Read every ``runs/*/report.json`` (oldest first) into trend rows.

    A run without a readable report is skipped silently: a crashed eval must
    not hide the history of the good ones.
    """
    rows: list[dict[str, Any]] = []
    for report_path in sorted(Path(runs_dir).glob("*/report.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "run": report_path.parent.name,
                "scenario": payload.get("scenario", "?"),
                "verdict": payload.get("verdict", "?"),
                "llm_calls": metrics.get("llm_calls"),
                "tool_calls": metrics.get("tool_calls"),
                "background_starts": metrics.get("background_starts"),
                "output_polls": metrics.get("output_polls"),
                "stalls": metrics.get("stalls"),
                "duration_s": metrics.get("duration_s"),
                "tokens_in": metrics.get("tokens_in"),
            }
        )
    return rows


def render_trends(rows: list[dict[str, Any]]) -> str:
    """Markdown table of the trend rows (``load_trends`` output)."""
    if not rows:
        return "_нет прогонов — запустите `python -m evals <scenario>`_"
    headers = [header for _, header in TREND_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in TREND_COLUMNS) + " |")
    return "\n".join(lines)
