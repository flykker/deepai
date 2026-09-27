"""Offline tests for the eval suite itself (no network, fake model).

The eval tooling must be trustworthy before it is used to judge the agent:
events are recorded, checks really fail on bad input, reports and diffs are
produced, and a full ``run_scenario`` pipeline writes its artifacts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agent.background.manager import BackgroundTaskManager
from agent.graph import build_agent
from evals.checks import (
    CheckResult,
    Context,
    background_starts,
    budget_results,
    file_content_is,
    files_exist,
    no_running_tasks,
    parallel_batch,
    port_closed,
    pytest_runs,
    run_all,
    server_responds,
    summarize,
    tasks_overlap,
    tool_call_count,
    transcript_contains,
)
from evals.recorder import RunRecorder, read_events
from evals.report import (
    build_metrics,
    diff_metrics,
    hints,
    load_trends,
    render_markdown,
    render_trends,
    write_reports,
)
from evals.runner import Scenario, run_scenario
from evals.scenarios import SCENARIOS, get_scenario, scenario_names
from tests.conftest import ai, tool_call


def _ctx(tmp_path: Path, *, events=None, transcript=None) -> Context:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=workspace)
    return Context(
        workspace=workspace,
        manager=manager,
        events=events or [],
        transcript=transcript or [],
    )


# ------------------------------------------------------------------ recorder


def test_recorder_captures_llm_and_tool_events(tmp_path, fake_model):
    recorder = RunRecorder(tmp_path / "events.jsonl")
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(
        fake_model(
            [
                tool_call("start_background_task", {"command": "echo x", "description": "x"}, "c1"),
                ai("done"),
            ]
        ),
        manager=manager,
        root_dir=tmp_path,
    )

    agent.invoke(
        {"messages": [{"role": "user", "content": "go"}]},
        config={"configurable": {"thread_id": "rec"}, "callbacks": [recorder]},
    )
    recorder.close()

    events = read_events(recorder.path)
    kinds = {e["kind"] for e in events}
    assert {"llm_start", "llm_end", "tool_start", "tool_end"} <= kinds

    tool_starts = [e for e in events if e["kind"] == "tool_start"]
    assert any(e.get("name") == "start_background_task" for e in tool_starts)
    llm_ends = [e for e in events if e["kind"] == "llm_end"]
    assert any(e.get("tool_calls") for e in llm_ends), "tool calls must be visible in llm_end"
    # Events are ordered and carry the elapsed timeline.
    assert events[0]["seq"] == 1
    assert all(events[i]["seq"] < events[i + 1]["seq"] for i in range(len(events) - 1))


def test_recorder_sees_parallel_batches(tmp_path, fake_model):
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "start_background_task", "args": {"command": "echo a", "description": "a"}, "id": "c1"},
            {"name": "start_background_task", "args": {"command": "echo b", "description": "b"}, "id": "c2"},
        ],
    )
    recorder = RunRecorder(tmp_path / "events.jsonl")
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(fake_model([both, ai("done")]), manager=manager, root_dir=tmp_path)

    agent.invoke(
        {"messages": [{"role": "user", "content": "go"}]},
        config={"configurable": {"thread_id": "par"}, "callbacks": [recorder]},
    )
    recorder.close()

    ctx = Context(workspace=tmp_path, manager=manager, events=read_events(recorder.path))
    result = parallel_batch(2)(ctx)
    assert result.ok, result.detail
    assert background_starts(2, soft=False)(ctx).ok


# -------------------------------------------------------------------- checks


def test_files_exist_check(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.workspace / "app.py").write_text("print('hi')\n")

    assert files_exist("app.py")(ctx).ok
    missing = files_exist("app.py", "nope.py")(ctx)
    assert not missing.ok
    assert "nope.py" in missing.detail


def test_transcript_contains_check(tmp_path):
    ctx = _ctx(tmp_path, transcript=[{"type": "ai", "content": "готово: STEP-A"}])
    assert transcript_contains("STEP-A")(ctx).ok
    assert not transcript_contains("STEP-B")(ctx).ok


def test_no_running_tasks_check(tmp_path):
    ctx = _ctx(tmp_path)
    assert no_running_tasks()(ctx).ok

    record = ctx.manager.start("sleep 30", description="leak")
    leaked = no_running_tasks()(ctx)
    assert not leaked.ok
    assert record["task_id"] in leaked.detail
    ctx.manager.shutdown()


def test_tasks_overlap_detects_concurrent_tasks(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.manager.start("sleep 5", description="first")
    ctx.manager.start("sleep 5", description="second")

    assert tasks_overlap(2)(ctx).ok
    assert not tasks_overlap(3)(ctx).ok
    ctx.manager.shutdown()


def test_pytest_runs_check_green_and_red(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.workspace / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    green = pytest_runs()(ctx)
    assert green.ok, green.detail

    (ctx.workspace / "test_ok.py").write_text("def test_bad():\n    assert False\n")
    red = pytest_runs()(ctx)
    assert not red.ok


def test_pytest_runs_check_requires_tests(tmp_path):
    ctx = _ctx(tmp_path)
    result = pytest_runs()(ctx)
    assert not result.ok
    assert "test*.py" in result.detail


def test_server_responds_check_starts_real_server(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.workspace / "app.py").write_text(
        "import http.server, os, socketserver\n"
        "socketserver.TCPServer.allow_reuse_address = 1  # back-to-back runs hit TIME_WAIT otherwise\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        body = b'{\"status\": \"ok\"}'\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Length', str(len(body)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(body)\n"
        "    def log_message(self, *args): pass\n"
        "with socketserver.TCPServer(('127.0.0.1', int(os.environ['PORT'])), H) as s:\n"
        "    s.serve_forever()\n"
    )
    ok = server_responds(port=8877, path="/hello")(ctx)
    assert ok.ok, ok.detail

    missing = server_responds(port=8878, path="/hello", entrypoint="server.py")(ctx)
    assert not missing.ok
    assert "server.py" in missing.detail


def test_run_all_survives_a_raising_check(tmp_path):
    def broken(_ctx):  # noqa: ANN001
        raise RuntimeError("boom")

    results = run_all([broken], _ctx(tmp_path))
    assert len(results) == 1 and not results[0].ok
    assert "boom" in results[0].detail
    assert not summarize(results)


# -------------------------------------------------------------------- report


def _events() -> list[dict]:
    return [
        {"kind": "llm_start", "t": 0.1},
        {"kind": "llm_end", "t": 1.0, "first_token_ms": 300, "duration_ms": 900, "tool_calls": None, "usage": {"input": 10, "output": 5}},
        {"kind": "tool_start", "t": 1.1, "name": "start_background_task", "args": "{}"},
        {"kind": "tool_end", "t": 1.2, "name": "start_background_task"},
        {"kind": "tool_start", "t": 1.3, "name": "get_task_output", "args": "{}"},
        {"kind": "tool_start", "t": 1.4, "name": "get_task_output", "args": "{}"},
        {"kind": "tool_start", "t": 1.5, "name": "get_task_output", "args": "{}"},
        {"kind": "tool_start", "t": 1.6, "name": "get_task_output", "args": "{}"},
        {"kind": "llm_end", "t": 3.0, "first_token_ms": 20_000, "duration_ms": 21_000, "tool_calls": [{"name": "a"}, {"name": "b"}]},
    ]


def test_build_metrics_counts_work():
    metrics = build_metrics(_events(), duration_s=5.0)

    assert metrics["tool_calls"] == 5
    assert metrics["background_starts"] == 1
    assert metrics["output_polls"] == 4
    assert metrics["parallel_batches"] == 1
    assert metrics["max_batch"] == 2
    assert metrics["stalls"] == 1  # 20s first token
    assert metrics["tokens_in"] == 10
    assert metrics["tool_calls_by_name"]["get_task_output"] == 4


def test_hints_flag_polling_stalls_and_failures():
    metrics = build_metrics(_events(), duration_s=5.0)
    ideas = hints(
        metrics,
        [CheckResult(name="чек", ok=False, detail="всё плохо")],
    )
    blob = " ".join(ideas)
    assert "поллинг" in blob
    assert "столл" in blob
    assert "не прошли" in blob


def test_reports_are_written_and_readable(tmp_path):
    ctx = _ctx(tmp_path, events=_events())
    results = run_all([files_exist("app.py")], ctx)
    metrics = build_metrics(_events(), duration_s=5.0)

    md_path, json_path = write_reports(
        tmp_path,
        scenario="demo",
        prompt="сделай",
        results=results,
        metrics=metrics,
        events=_events(),
        workspace=ctx.workspace,
        tasks=ctx.manager.list(),
        model="openai:test",
    )

    md = md_path.read_text()
    assert "# Eval: demo" in md
    assert "## Метрики" in md and "## Проверки" in md and "## Артефакты" in md
    payload = json.loads(json_path.read_text())
    assert payload["verdict"] == "FAIL"
    assert payload["metrics"]["output_polls"] == 4
    assert payload["checks"] and payload["checks"][0]["ok"] is False


def test_render_markdown_includes_prompt_and_timeline(tmp_path):
    ctx = _ctx(tmp_path, events=_events())
    md = render_markdown(
        scenario="demo",
        prompt="напиши приложение",
        results=run_all([], ctx),
        metrics=build_metrics(_events(), duration_s=5.0),
        events=_events(),
        workspace=ctx.workspace,
        tasks=[],
        model="m",
    )
    assert "напиши приложение" in md
    assert "Последовательность инструментов" in md


def test_diff_metrics_reports_changes():
    baseline = {"metrics": {"output_polls": 4, "duration_s": 10.0, "stalls": 1}}
    current = {"metrics": {"output_polls": 1, "duration_s": 12.5, "stalls": 1}}
    rows = diff_metrics(baseline, current)
    blob = "\n".join(rows)
    assert "output_polls: 4 → 1" in blob
    assert "duration_s: 10.0 → 12.5" in blob
    assert "stalls" not in blob  # unchanged metrics are omitted


# ---------------------------------------------------------------- scenarios


def test_scenario_registry():
    assert set(scenario_names()) == set(SCENARIOS) == {"debug", "devserver", "parallel", "webapp"}

    webapp = get_scenario("webapp")
    assert "app.py" in webapp.prompt
    assert "pip" in webapp.prompt
    assert webapp.checks

    parallel = get_scenario("parallel")
    assert "start_background_task" in parallel.prompt
    assert parallel.time_limit > 0

    devserver = get_scenario("devserver")
    assert "kill_task" in devserver.prompt
    assert devserver.time_limit > 0

    debug = get_scenario("debug")
    assert debug.setup is not None, "debug must seed its broken app before the turn"
    assert "test_app.py" in debug.prompt and "НЕ меняй" in debug.prompt

    # every scenario declares its efficiency budgets
    for name in scenario_names():
        scenario = get_scenario(name)
        assert scenario.max_llm_calls and scenario.max_polls and scenario.max_duration_s

    with pytest.raises(SystemExit):
        get_scenario("nope")


# ------------------------------------------------------- full pipeline (fake)


def test_run_scenario_writes_artifacts_and_fails_unmet_checks(tmp_path, fake_model):
    scenario = Scenario(
        name="demo",
        prompt="go",
        checks=(files_exist("app.py"),),
        time_limit=30.0,
    )

    result = run_scenario(scenario, runs_dir=tmp_path / "runs", model=fake_model([ai("я не писал код")]), echo=False)

    assert result.verdict == "FAIL" and not result.passed
    assert (result.run_dir / "events.jsonl").exists()
    assert (result.run_dir / "transcript.json").exists()
    assert (result.run_dir / "workspace").exists()
    assert result.report_md and result.report_md.exists()
    events = read_events(result.run_dir / "events.jsonl")
    assert events[0]["kind"] == "run_start"
    assert events[-1]["kind"] == "run_end"
    assert any(e["kind"] == "check" for e in events)


def test_run_scenario_passes_when_checks_are_met(tmp_path, fake_model):
    scenario = Scenario(
        name="demo-pass",
        prompt="go",
        checks=(transcript_contains("привет"), no_running_tasks()),
        time_limit=30.0,
    )

    result = run_scenario(scenario, runs_dir=tmp_path / "runs", model=fake_model([ai("привет")]), echo=False)

    assert result.verdict == "PASS" and result.passed
    assert all(check.ok for check in result.results)
    assert result.metrics["llm_calls"] >= 1


def test_interpreters_discover_any_workspace_venv(tmp_path):
    import sys

    from evals.checks import _interpreters

    workspace = tmp_path / "workspace"
    fake_python = workspace / "venv" / "bin" / "python"  # no leading dot!
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/sh\nexit 0\n")
    fake_python.chmod(0o755)

    found = _interpreters(workspace)

    assert found[0] == os.path.abspath(fake_python), "the agent's own venv must win"
    assert str(sys.executable) in found or os.path.abspath(sys.executable) in found


def test_checks_with_relative_workspace_do_not_double_resolve(tmp_path, monkeypatch):
    """A relative workspace must not break subprocess interpreter lookup.

    The runner used to pass `runs/<id>/workspace` (relative) as cwd while the
    discovered venv path was relative too, so Popen resolved it from the
    workspace and raised FileNotFoundError. Regression guard for that.
    """
    monkeypatch.chdir(tmp_path)
    workspace = Path("runs/demo/workspace")
    (workspace).mkdir(parents=True)
    (workspace / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    fake_python = workspace / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("#!/bin/sh\nexit 3\n")  # a venv that runs, but fails
    fake_python.chmod(0o755)

    manager = BackgroundTaskManager(output_dir=Path("runs/demo/tasks"), cwd=workspace)
    ctx = Context(workspace=workspace, manager=manager, events=[], transcript=[])

    result = pytest_runs()(ctx)

    assert result.ok, f"must fall through to a real interpreter: {result.detail}"
    server = server_responds(port=8879, entrypoint="nope.py")(ctx)
    assert not server.ok


def test_tool_call_count_check(tmp_path):
    events = [
        {"kind": "tool_start", "name": "kill_task"},
        {"kind": "tool_end", "name": "kill_task"},
        {"kind": "tool_start", "name": "kill_task"},
    ]
    ctx = _ctx(tmp_path, events=events)

    assert tool_call_count("kill_task", 2)(ctx).ok
    assert not tool_call_count("kill_task", 3)(ctx).ok
    assert not tool_call_count("start_background_task")(ctx).ok


def test_port_closed_check(tmp_path):
    import socket

    ctx = _ctx(tmp_path)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    assert not port_closed(port)(ctx).ok, "a listening server must fail the check"

    server.close()
    assert port_closed(port)(ctx).ok, "a closed port must pass"


def test_load_trends_and_render(tmp_path):
    for name, verdict, llm in (
        ("20260101-000000_webapp", "FAIL", 29),
        ("20260102-000000_webapp", "PASS", 10),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    "scenario": "webapp",
                    "verdict": verdict,
                    "metrics": {"llm_calls": llm, "duration_s": 100.0},
                }
            ),
            encoding="utf-8",
        )
    (tmp_path / "20260103-000000_broken").mkdir()
    (tmp_path / "20260103-000000_broken" / "report.json").write_text("not json", encoding="utf-8")

    rows = load_trends(tmp_path)

    assert [r["verdict"] for r in rows] == ["FAIL", "PASS"], "unreadable reports are skipped"
    table = render_trends(rows)
    assert "webapp" in table and "FAIL" in table and "PASS" in table
    assert render_trends([]) == "_нет прогонов — запустите `python -m evals <scenario>`_"


def test_stall_metrics_fall_back_to_duration_when_not_streaming(tmp_path):
    """Quiet runs invoke the graph without streaming — no token callbacks."""
    events = [
        {"kind": "llm_end", "first_token_ms": None, "duration_ms": 58_000},
        {"kind": "llm_end", "first_token_ms": None, "duration_ms": 3_000},
        {"kind": "llm_end", "first_token_ms": 12_000, "duration_ms": 58_000},
    ]

    metrics = build_metrics(events, duration_s=60.0)

    assert metrics["stalls"] == 2, "a 58 s silent call is a stall even without tokens"
    assert metrics["worst_first_token_ms"] == 58_000


def test_hints_recommend_rerun_when_stalls_dominate(tmp_path):
    """Several stalls in a row are proxy weather, not agent behavior."""
    events = [{"kind": "llm_end", "first_token_ms": 61_000, "duration_ms": 62_000}] * 3
    metrics = build_metrics(events, duration_s=300.0)

    text = " ".join(hints(metrics, []))

    assert "повторить прогон позже" in text


def test_quiesce_waits_for_a_busy_turn_then_settles(tmp_path):
    import threading

    from agent.background import BackgroundTaskManager
    from evals.recorder import RunRecorder
    from evals.runner import _quiesce

    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    rec = RunRecorder(tmp_path / "events.jsonl")

    class FakeHarness:
        def __init__(self) -> None:
            self._busy = True

        @property
        def busy(self) -> bool:
            return self._busy

    harness = FakeHarness()
    threading.Timer(1.0, setattr, (harness, "_busy", False)).start()

    # A turn in flight must keep quiesce waiting within its budget instead of
    # scoring a half-finished run (regression: 60 s window gave up mid-turn).
    assert _quiesce(harness, manager, rec, timeout=10.0, quiet_s=0.3) is True

    rec.close()
    manager.shutdown()


def test_quiesce_times_out_when_the_agent_never_settles(tmp_path):
    from agent.background import BackgroundTaskManager
    from evals.recorder import RunRecorder
    from evals.runner import _quiesce

    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    rec = RunRecorder(tmp_path / "events.jsonl")

    class BusyHarness:
        busy = True

    assert _quiesce(BusyHarness(), manager, rec, timeout=0.4, quiet_s=0.2) is False

    rec.close()
    manager.shutdown()


# ------------------------------------------------------------------ budgets


def test_budget_results_hard_caps_and_soft_time():
    metrics = {"llm_calls": 18, "output_polls": 2, "duration_s": 500.0}
    results = budget_results(
        {"llm_calls": 16, "output_polls": 5, "duration_s": 420.0, "kills": None},
        metrics,
    )

    # «kills» has no budget (None) → dropped, the rest become checks
    assert len(results) == 3
    hard = {r.name: r for r in results if not r.soft}
    assert set(hard) == {"бюджет: ходов LLM ≤ 16", "бюджет: поллингов get_task_output ≤ 5"}
    assert not hard["бюджет: ходов LLM ≤ 16"].ok
    assert "18" in hard["бюджет: ходов LLM ≤ 16"].detail
    assert hard["бюджет: поллингов get_task_output ≤ 5"].ok

    # an overrun must not fail the run on its own — stalls are proxy weather
    soft = [r for r in results if r.soft]
    assert len(soft) == 1 and not soft[0].ok
    assert soft[0].name == "бюджет: секунд ≤ 420"
    assert summarize([r for r in results if not r.soft]) is False
    # but a run whose only problem is time still passes
    assert summarize(budget_results({"duration_s": 1}, metrics)) is True


def test_budget_results_ignores_missing_metrics():
    results = budget_results({"llm_calls": 10, "model_errors": 3}, {"llm_calls": 5})

    assert [r.name for r in results] == ["бюджет: ходов LLM ≤ 10"]
    assert results[0].ok


def test_run_scenario_fails_when_llm_budget_is_exceeded(tmp_path, fake_model):
    scenario = Scenario(
        name="demo-budget",
        prompt="go",
        time_limit=30.0,
        max_llm_calls=1,     # the model needs two turns: tool call + answer
        max_duration_s=0,    # soft by design: warns, never flips the verdict
    )
    model = fake_model([tool_call("execute", {"command": "echo hi"}, "c1"), ai("готово")])

    result = run_scenario(scenario, runs_dir=tmp_path / "runs", model=model, echo=False)

    budgets = [r for r in result.results if r.name.startswith("бюджет")]
    hard = [r for r in budgets if not r.soft]
    soft = [r for r in budgets if r.soft]
    assert result.verdict == "FAIL" and not result.passed
    assert hard and not hard[0].ok and "лимит 1" in hard[0].detail
    assert soft and not soft[0].ok, "0 s budget is always exceeded → warning only"
    events = read_events(result.run_dir / "events.jsonl")
    assert any(e["kind"] == "check" and "бюджет" in e["name"] for e in events)


def test_run_scenario_passes_within_budget(tmp_path, fake_model):
    scenario = Scenario(
        name="demo-budget-ok",
        prompt="go",
        time_limit=30.0,
        max_llm_calls=5,
        max_duration_s=0,    # exceeded but soft → verdict stays PASS
    )

    result = run_scenario(scenario, runs_dir=tmp_path / "runs", model=fake_model([ai("привет")]), echo=False)

    hard = [r for r in result.results if r.name.startswith("бюджет") and not r.soft]
    soft = [r for r in result.results if r.name.startswith("бюджет") and r.soft]
    assert result.verdict == "PASS" and result.passed
    assert hard and hard[0].ok
    assert soft and not soft[0].ok


def test_build_metrics_counts_wait_time():
    metrics = build_metrics(_events(), duration_s=5.0)

    assert metrics["total_wait_s"] == 20.3  # 0.3 + 20.0
    assert metrics["wait_streamed"] is True


def test_weather_hint_only_when_tokens_were_streamed():
    """Quiet mode falls back to full call durations — never blame the weather on those."""
    quiet = [
        {"kind": "llm_start", "t": 0.1},
        {"kind": "llm_end", "t": 60.0, "duration_ms": 55_000, "tool_calls": None},
    ]
    quiet_metrics = build_metrics(quiet, duration_s=60.0)
    assert quiet_metrics["wait_streamed"] is False
    assert quiet_metrics["total_wait_s"] == 55.0
    ideas = hints(quiet_metrics, [CheckResult(name="чек", ok=True, detail="ok")])
    assert not any("погоду" in idea for idea in ideas)

    # streamed waits dominating the run → the run measured the proxy, not the agent
    ideas = hints(build_metrics(_events(), duration_s=5.0), [CheckResult(name="чек", ok=True, detail="ok")])
    assert any("погоду" in idea for idea in ideas)


# ------------------------------------------------------------- debug scenario


def test_debug_scenario_plants_reproducible_bugs_and_accepts_a_minimal_fix(tmp_path):
    """The planted tests must fail on the planted app and pass after a fix.

    Otherwise the scenario would be unsatisfiable (permanent FAIL) or trivial
    (no bug to find).
    """
    import subprocess
    import sys

    from evals.scenarios.debug import APP_SOURCE, TEST_SOURCE, SCENARIO as debug

    debug.setup(tmp_path)

    def pytest_run() -> tuple[int, str]:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.returncode, proc.stdout

    rc, out = pytest_run()
    assert rc != 0 and "2 failed" in out, f"planted bugs must reproduce: {out}"

    # the minimal fix the agent is expected to find
    fixed = APP_SOURCE.replace(
        'return max(task["id"] for task in TASKS)',
        'return max(task["id"] for task in TASKS) + 1',
    ).replace(
        "tags: list[str] = []",
        "tags: list[str] | None = None",
    ).replace(
        '"tags": tags, "done": False',
        '"tags": tags or [], "done": False',
    )
    (tmp_path / "app.py").write_text(fixed, encoding="utf-8")
    rc, out = pytest_run()
    assert rc == 0 and "6 passed" in out, f"the fix must turn the suite green: {out}"


def test_file_content_is_check(tmp_path):
    ctx = _ctx(tmp_path)
    (ctx.workspace / "test_app.py").write_text("SOURCE\n", encoding="utf-8")

    ok = file_content_is("test_app.py", "SOURCE\n")(ctx)
    assert ok.ok

    (ctx.workspace / "test_app.py").write_text("TAMPERED\n", encoding="utf-8")
    tampered = file_content_is("test_app.py", "SOURCE\n")(ctx)
    assert not tampered.ok and "правился" in tampered.detail

    (ctx.workspace / "test_app.py").unlink()
    missing = file_content_is("test_app.py", "SOURCE\n")(ctx)
    assert not missing.ok and "пропал" in missing.detail


def test_run_scenario_plants_setup_files(tmp_path, fake_model):
    def plant(workspace: Path) -> None:
        (workspace / "seeded.txt").write_text("seed", encoding="utf-8")

    scenario = Scenario(
        name="demo-setup",
        prompt="go",
        setup=plant,
        checks=(files_exist("seeded.txt"),),
        time_limit=30.0,
    )

    result = run_scenario(scenario, runs_dir=tmp_path / "runs", model=fake_model([ai("ok")]), echo=False)

    assert result.verdict == "PASS"
    events = read_events(result.run_dir / "events.jsonl")
    setup_events = [e for e in events if e["kind"] == "setup"]
    assert setup_events and setup_events[0]["files"] == ["seeded.txt"]
