"""Independent checks that score an eval run.

Everything here is verified *outside* the agent: the runner inspects the
sandbox, runs the produced tests itself, starts the produced server and talks
to it, and reads the recorded event log. The agent's own claims ("tests pass!")
are never taken at face value.

Checks come in two flavours: hard checks decide the verdict, soft checks are
reported as warnings (e.g. "the model never used a background task") so an
otherwise good run is not failed by policy preferences.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.background.manager import BackgroundTaskManager

__all__ = [
    "Check",
    "CheckResult",
    "Context",
    "background_starts",
    "budget_results",
    "file_content_is",
    "files_exist",
    "no_running_tasks",
    "parallel_batch",
    "port_closed",
    "pytest_runs",
    "server_responds",
    "tasks_overlap",
    "tool_call_count",
    "transcript_contains",
]


@dataclass
class Context:
    """Everything a check may look at."""

    workspace: Path
    manager: BackgroundTaskManager
    events: list[dict[str, Any]]
    transcript: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CheckResult:
    """Verdict of a single check."""

    name: str
    ok: bool
    detail: str
    soft: bool = False


Check = Callable[[Context], CheckResult]


# --------------------------------------------------------------------- files


def file_content_is(relative_path: str, expected: str, soft: bool = False) -> Check:
    """The file must still match ``expected`` byte for byte (anti-tamper).

    In the ``debug`` scenario the planted test file is the independent judge:
    "fixing" it instead of the code would make the whole run meaningless, so
    the scenario pins it to the exact source it was seeded with.
    """

    def check(ctx: Context) -> CheckResult:
        path = ctx.workspace / relative_path
        if not path.exists():
            return CheckResult(f"{relative_path} не изменён", False, "файл пропал", soft=soft)
        actual = path.read_text(encoding="utf-8")
        return CheckResult(
            f"{relative_path} не изменён",
            actual == expected,
            "как и был" if actual == expected else "файл правился — тесты тронуты",
            soft=soft,
        )

    return check


def files_exist(*relative_paths: str, soft: bool = False) -> Check:
    """Require the given files to exist inside the workspace."""

    def check(ctx: Context) -> CheckResult:
        missing = [p for p in relative_paths if not (ctx.workspace / p).exists()]
        if missing:
            return CheckResult(
                name=f"есть файлы: {', '.join(relative_paths)}",
                ok=False,
                detail=f"отсутствуют: {', '.join(missing)}",
                soft=soft,
            )
        return CheckResult(
            name=f"есть файлы: {', '.join(relative_paths)}",
            ok=True,
            detail="все на месте",
            soft=soft,
        )

    return check


# ------------------------------------------------------------------- pytest


def _interpreters(workspace: Path) -> list[str]:
    """Python candidates: the agent's own venv first, then ours, then system.

    The agent picks the venv name itself (``.venv``, ``venv``, …), so glob for
    anything venv-shaped inside the workspace instead of hardcoding one.
    """
    candidates: list[Path] = []
    for pattern in ("*venv*/bin/python", "*venv*/bin/python3"):
        candidates.extend(sorted(workspace.glob(pattern)))
    candidates.extend([Path(sys.executable), Path("/usr/bin/python3")])
    seen: set[str] = set()
    ordered: list[str] = []
    for path in candidates:
        if not path.exists() or not os.access(path, os.X_OK):
            continue
        # Absolute but symlink-preserving: the subprocess runs with
        # cwd=workspace (a relative path would resolve twice), while
        # Path.resolve() would collapse `.venv/bin/python` to the system
        # interpreter and lose the venv's site-packages.
        resolved = os.path.abspath(path)
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def _tail(text: str, lines: int = 15) -> str:
    chunk = text.strip().splitlines()[-lines:]
    return "\n".join(chunk)


def pytest_runs(soft: bool = False, timeout: float = 180.0, *markers: str) -> Check:
    """Run the workspace's tests ourselves and require a green result.

    Tries every available interpreter (workspace venv → eval interpreter →
    system python), because the agent may have installed dependencies into
    any of them.
    """

    def check(ctx: Context) -> CheckResult:
        if not any(ctx.workspace.glob("test*.py")) and not any((ctx.workspace / "tests").glob("test*.py")):
            return CheckResult("тесты написаны и проходят", False, "в workspace нет test*.py")
        attempts: list[str] = []
        for python in _interpreters(ctx.workspace):
            cmd = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *markers]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=ctx.workspace,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
            except subprocess.TimeoutExpired:
                attempts.append(f"{python}: pytest не уложился в {timeout:.0f}s")
                continue
            output = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                return CheckResult(
                    "тесты написаны и проходят",
                    True,
                    f"{Path(python).name}: {_tail(output, 3)}",
                )
            attempts.append(f"{python} (rc={proc.returncode}):\n{_tail(output)}")
        return CheckResult("тесты написаны и проходят", False, " | ".join(attempts) or "нет интерпретаторов")

    return check


# ------------------------------------------------------------------ server


def server_responds(
    port: int = 8765,
    path: str = "/hello",
    entrypoint: str = "app.py",
    soft: bool = False,
    per_candidate_timeout: float = 20.0,
) -> Check:
    """Start ``entrypoint`` ourselves and require ``GET path`` to return 200."""

    def check(ctx: Context) -> CheckResult:
        script = ctx.workspace / entrypoint
        if not script.exists():
            return CheckResult(f"сервер отвечает на {path}", False, f"нет {entrypoint}")
        last_error = "нет интерпретаторов"
        for python in _interpreters(ctx.workspace):
            proc = subprocess.Popen(  # noqa: S603 - eval sandbox command
                [python, entrypoint],
                cwd=ctx.workspace,
                env={**os.environ, "PORT": str(port), "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            status, body, error = _poll(port, path, proc, per_candidate_timeout)
            _stop(proc)
            if status is not None:
                if status == 200:
                    return CheckResult(
                        f"сервер отвечает на {path}",
                        True,
                        f"HTTP 200, тело: {body[:80]!r} (порт {port})",
                    )
                return CheckResult(
                    f"сервер отвечает на {path}",
                    False,
                    f"HTTP {status} вместо 200 (порт {port})",
                )
            last_error = f"{Path(python).name}: {error}"
        return CheckResult(f"сервер отвечает на {path}", False, last_error, soft=soft)

    return check


def _poll(port: int, path: str, proc: subprocess.Popen[str], timeout: float) -> tuple[int | None, str, str]:
    """Wait for an HTTP answer; returns (status, body, error)."""
    deadline = time.monotonic() + timeout
    last_error = "connection refused"
    url = f"http://127.0.0.1:{port}{path}"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            return None, "", f"процесс завершился (rc={proc.returncode}): {_tail(output, 5)}"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - fixed local URL
                return response.status, response.read(4096).decode("utf-8", "replace"), ""
        except urllib.error.HTTPError as exc:  # server answered with 4xx/5xx
            return exc.code, "", ""
        except Exception as exc:  # noqa: BLE001 - not up yet, retry
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.4)
    return None, "", f"нет ответа за {timeout:.0f}s ({last_error})"


def _stop(proc: subprocess.Popen[str]) -> None:
    """Terminate the server's whole process group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover - defensive
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()


# ------------------------------------------------------------------ events


def tool_call_count(name: str, min_count: int = 1, soft: bool = False) -> Check:
    """The agent must have called tool ``name`` at least ``min_count`` times."""

    def check(ctx: Context) -> CheckResult:
        count = sum(1 for e in ctx.events if e["kind"] == "tool_start" and e.get("name") == name)
        return CheckResult(
            f"вызовов `{name}` ≥ {min_count}",
            count >= min_count,
            f"вызвано: {count} (нужно ≥ {min_count})",
            soft=soft,
        )

    return check


def port_closed(port: int, soft: bool = False) -> Check:
    """Nothing may still be listening on ``port`` when the run is scored."""

    def check(ctx: Context) -> CheckResult:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.5)
            open_ = sock.connect_ex(("127.0.0.1", port)) == 0
        if open_:
            return CheckResult(f"порт {port} закрыт", False, "на порту ещё что-то слушает", soft=soft)
        return CheckResult(f"порт {port} закрыт", True, "соединение отклонено", soft=soft)

    return check


def background_starts(min_count: int = 1, soft: bool = True) -> Check:
    """The agent must have used ``start_background_task`` at least N times."""

    def check(ctx: Context) -> CheckResult:
        count = sum(1 for e in ctx.events if e["kind"] == "tool_start" and e.get("name") == "start_background_task")
        verdict = count >= min_count
        detail = f"стартов: {count} (нужно ≥ {min_count})"
        if verdict and count == 0:
            detail = "стартов: 0"
        return CheckResult(f"фоновые задачи использованы (≥ {min_count})", verdict, detail, soft=soft)

    return check


def parallel_batch(min_size: int = 2, soft: bool = False) -> Check:
    """At least one model response must carry ≥ N tool calls at once."""

    def check(ctx: Context) -> CheckResult:
        batches = [
            len(e.get("tool_calls") or [])
            for e in ctx.events
            if e["kind"] == "llm_end" and len(e.get("tool_calls") or []) >= min_size
        ]
        best = max(batches, default=0)
        return CheckResult(
            f"параллельный батч вызовов (≥ {min_size})",
            best >= min_size,
            f"крупнейший батч: {best}",
            soft=soft,
        )

    return check


def transcript_contains(*markers: str, soft: bool = False) -> Check:
    """Required substrings must appear somewhere in the conversation."""

    def check(ctx: Context) -> CheckResult:
        blob = json.dumps(ctx.transcript, ensure_ascii=False, default=str)
        missing = [m for m in markers if m not in blob]
        return CheckResult(
            f"вывод содержит: {', '.join(markers)}",
            not missing,
            "все маркеры найдены" if not missing else f"не найдено: {', '.join(missing)}",
            soft=soft,
        )

    return check


def no_running_tasks() -> Check:
    """No background task may be left running when the run ends."""

    def check(ctx: Context) -> CheckResult:
        leaked = [r["task_id"] for r in ctx.manager.list() if r["status"] == "running"]
        return CheckResult(
            "фоновые задачи не остались висеть",
            not leaked,
            "утечек нет" if not leaked else f"ещё работают: {', '.join(leaked)}",
        )

    return check


def tasks_overlap(min_concurrency: int = 2, soft: bool = False) -> Check:
    """N tasks must have been alive at the same time (real parallelism)."""

    def check(ctx: Context) -> CheckResult:
        intervals: list[tuple[float, float]] = []
        for record in ctx.manager.list():
            started = _epoch(record.get("started_at"))
            finished = _epoch(record.get("finished_at"))
            if started is None:
                continue
            intervals.append((started, finished if finished is not None else time.time()))
        peak = 0
        for point in {p for span in intervals for p in span}:
            active = sum(1 for start, end in intervals if start <= point < end)
            peak = max(peak, active)
        return CheckResult(
            f"задачи бежали одновременно (≥ {min_concurrency})",
            peak >= min_concurrency,
            f"максимальная параллельность: {peak}",
            soft=soft,
        )

    return check


def _epoch(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except ValueError:  # pragma: no cover - defensive
        return None


# ------------------------------------------------------------------ budgets

_BUDGET_LABELS = {
    "llm_calls": "ходов LLM",
    "output_polls": "поллингов get_task_output",
    "duration_s": "секунд",
}
# Wall-clock time is dominated by proxy stalls, so an overrun warns instead
# of failing the run; turns and polling are fully agent-controlled.
_SOFT_BUDGETS = frozenset({"duration_s"})


def budget_results(limits: dict[str, Any], metrics: dict[str, Any]) -> list[CheckResult]:
    """Turn scenario budgets into check results.

    A run may satisfy every correctness check and still have quietly become
    twice as expensive — eval caught exactly that after a model swap (webapp:
    10 → 18 LLM turns, parallel batches gone) while staying PASS. Budgets are
    declared per scenario (``max_llm_calls``/``max_polls``/``max_duration_s``)
    and map onto metrics produced by :func:`evals.report.build_metrics`.

    Args:
        limits: Metric key → cap; ``None`` means "no budget for this metric".
        metrics: The run's metrics, as computed by the runner.

    Returns:
        One result per configured budget; hard ones decide the verdict,
        soft ones (duration) surface as warnings in the report.
    """
    results: list[CheckResult] = []
    for key, limit in limits.items():
        if limit is None or key not in metrics:
            continue
        value = metrics[key]
        label = _BUDGET_LABELS.get(key, key)
        results.append(
            CheckResult(
                name=f"бюджет: {label} ≤ {limit:g}",
                ok=float(value) <= float(limit),
                detail=f"факт {value}, лимит {limit:g}",
                soft=key in _SOFT_BUDGETS,
            )
        )
    return results


def run_all(checks: Iterable[Check], ctx: Context) -> list[CheckResult]:
    """Run every check, turning a raising check into a failed result."""
    results: list[CheckResult] = []
    for check in checks:
        name = getattr(check, "__name__", "check")
        try:
            results.append(check(ctx))
        except Exception as exc:  # noqa: BLE001 - a broken check must not kill the report
            results.append(CheckResult(name=name, ok=False, detail=f"check упал: {type(exc).__name__}: {exc}"))
    return results


def summarize(results: Sequence[CheckResult]) -> bool:
    """True when every hard (non-soft) check passed."""
    return all(r.ok for r in results if not r.soft)
