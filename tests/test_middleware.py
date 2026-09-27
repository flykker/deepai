"""Background task tools registered on the agent graph."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.background.middleware import DISABLE_ENV_VAR, BackgroundTasksMiddleware
from agent.background.manager import BackgroundTaskManager
from agent.graph import build_agent
from tests.conftest import ai, tool_call

BACKGROUND_TOOLS = {"start_background_task", "get_task_output", "kill_task", "list_tasks"}


def _config() -> dict:
    return {"configurable": {"thread_id": "test"}, "recursion_limit": 50}


@pytest.fixture
def manager(tmp_path) -> BackgroundTaskManager:
    return BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)


def _tool_names(agent) -> set[str]:
    return set(agent.nodes["tools"].bound.tools_by_name)


def test_middleware_registers_background_tools(manager):
    middleware = BackgroundTasksMiddleware(manager)
    assert {tool.name for tool in middleware.tools} == BACKGROUND_TOOLS
    assert middleware.system_prompt and "start_background_task" in middleware.system_prompt


def test_middleware_can_be_disabled(manager):
    middleware = BackgroundTasksMiddleware(manager, enabled=False)
    assert middleware.tools == []
    assert middleware.system_prompt is None


def test_disable_env_var_drops_the_tools(manager, monkeypatch):
    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    middleware = BackgroundTasksMiddleware(manager)
    assert middleware.tools == []
    assert middleware.system_prompt is None


def test_agent_exposes_background_tools(manager, fake_model, tmp_path):
    agent = build_agent(fake_model([ai("hi")]), manager=manager, root_dir=tmp_path)

    assert BACKGROUND_TOOLS <= _tool_names(agent)
    # Foreground execution stays available alongside the background tools.
    assert "execute" in _tool_names(agent)


def test_start_then_read_background_task(manager, fake_model, tmp_path):
    agent = build_agent(
        fake_model(
            [
                tool_call("start_background_task", {"command": "echo bg-hello", "description": "echo hello"}, "c1"),
                tool_call("get_task_output", {"task_id": "bash-1"}, "c2"),
                ai("done"),
            ]
        ),
        manager=manager,
        root_dir=tmp_path,
    )

    state = agent.invoke({"messages": [{"role": "user", "content": "run it"}]}, config=_config())

    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert any("Command running in background with ID: bash-1" in str(m.content) for m in tool_messages)
    assert any("task_id: bash-1" in str(m.content) for m in tool_messages)
    assert all("Error:" not in str(m.content) for m in tool_messages)

    tracked = state["background_tasks"]["bash-1"]
    assert tracked["command"] == "echo bg-hello"
    assert tracked["output_path"].endswith("bash-1.output")

    final = manager.wait("bash-1", timeout=10)
    assert final["status"] == "completed"
    # The model's get_task_output call consumed the output already, so assert on
    # the output file itself.
    assert "bg-hello" in Path(tracked["output_path"]).read_text()


def test_get_task_output_reports_unknown_task(manager, fake_model, tmp_path):
    agent = build_agent(
        fake_model(
            [
                tool_call("get_task_output", {"task_id": "bash-404"}, "c1"),
                ai("done"),
            ]
        ),
        manager=manager,
        root_dir=tmp_path,
    )

    state = agent.invoke({"messages": [{"role": "user", "content": "check"}]}, config=_config())
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]

    assert any("No tracked background task" in str(m.content) for m in tool_messages)


def test_foreground_execute_still_works(manager, fake_model, tmp_path):
    """Background tools are additive: the foreground `execute` tool keeps working."""
    agent = build_agent(
        fake_model(
            [
                tool_call("execute", {"command": "echo foreground-ok"}, "c1"),
                ai("done"),
            ]
        ),
        manager=manager,
        root_dir=tmp_path,
    )

    state = agent.invoke({"messages": [{"role": "user", "content": "run"}]}, config=_config())
    tool_messages = [str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]

    assert any("foreground-ok" in content for content in tool_messages)
    assert manager.list() == []  # foreground execution never touches the task manager


def test_list_and_kill_tools(manager, fake_model, tmp_path):
    agent = build_agent(
        fake_model(
            [
                tool_call("start_background_task", {"command": "sleep 60", "description": "long sleeper"}, "c1"),
                tool_call("list_tasks", {}, "c2"),
                tool_call("kill_task", {"task_id": "bash-1"}, "c3"),
                ai("done"),
            ]
        ),
        manager=manager,
        root_dir=tmp_path,
    )

    state = agent.invoke({"messages": [{"role": "user", "content": "start then kill"}]}, config=_config())
    tool_messages = [str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]

    assert any("status=running" in content for content in tool_messages)
    assert any("Stopped task bash-1" in content for content in tool_messages)
    assert state["background_tasks"]["bash-1"]["status"] == "killed"
    assert manager.take_notifications() == []


def test_parallel_tool_calls_start_concurrent_tasks(manager, fake_model, tmp_path):
    """Two start calls emitted in one message must run as overlapping processes."""
    both = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "start_background_task",
                "args": {"command": "sleep 3 && echo first", "description": "first sleeper"},
                "id": "c1",
            },
            {
                "name": "start_background_task",
                "args": {"command": "sleep 3 && echo second", "description": "second sleeper"},
                "id": "c2",
            },
        ],
    )
    agent = build_agent(fake_model([both, ai("done")]), manager=manager, root_dir=tmp_path)

    state = agent.invoke({"messages": [{"role": "user", "content": "run both"}]}, config=_config())

    assert set(state["background_tasks"]) == {"bash-1", "bash-2"}
    assert all(t["status"] == "running" for t in state["background_tasks"].values())

    started = time.monotonic()
    assert manager.wait("bash-1", timeout=10)["status"] == "completed"
    assert manager.wait("bash-2", timeout=10)["status"] == "completed"
    elapsed = time.monotonic() - started
    # Two 3-second sleeps waited on back to back would need ~6s; overlapping
    # execution finishes in about one sleep.
    assert elapsed < 5.5, f"tasks ran serially ({elapsed:.1f}s)"
