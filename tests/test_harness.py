"""End-to-end harness behavior: completion notifications wake the agent."""

from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.background.manager import BackgroundTaskManager
from agent.graph import build_agent
from agent.harness import AgentHarness
from tests.conftest import ai, tool_call


def _harness(tmp_path, model, *, echo=False, **kwargs) -> AgentHarness:
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(model, manager=manager, root_dir=tmp_path)
    return AgentHarness(agent, manager, echo=echo, **kwargs)


def test_notification_finished_while_idle_is_delivered_before_the_user_message(tmp_path, fake_model):
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    record = manager.start("echo pre-done", description="pre task")
    manager.wait(record["task_id"], timeout=10)

    agent = build_agent(fake_model([ai("acknowledged"), ai("ready")]), manager=manager, root_dir=tmp_path)
    harness = AgentHarness(agent, manager, echo=False)

    state = harness.run_turn("hello")
    messages = state["messages"]

    assert isinstance(messages[0], HumanMessage)
    assert "<task-notification>" in str(messages[0].content)
    assert "bash-1" in str(messages[0].content)
    assert messages[1].content == "acknowledged"
    assert str(messages[2].content) == "hello"
    assert messages[3].content == "ready"


def test_notification_arriving_during_a_turn_is_injected_after_it(tmp_path, fake_model):
    def script():
        yield tool_call("start_background_task", {"command": "echo during-turn", "description": "during turn"}, "c1")
        time.sleep(1.0)  # let the background command finish before the turn ends
        yield ai("started")
        yield ai("acknowledged the notification")

    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(fake_model(script()), manager=manager, root_dir=tmp_path)
    harness = AgentHarness(agent, manager, echo=False)

    state = harness.run_turn("start a build")
    messages = state["messages"]

    user_index = next(i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and m.content == "start a build")
    note_index = next(i for i, m in enumerate(messages) if isinstance(m, HumanMessage) and "<task-notification>" in str(m.content))

    assert note_index > user_index, "notification must arrive after the turn that started the task"
    assert isinstance(messages[note_index - 1], AIMessage)
    assert isinstance(messages[note_index + 1], AIMessage)
    assert "during turn" in str(messages[note_index].content)
    assert messages[-1].content == "acknowledged the notification"


def test_task_results_reach_the_model_via_state(tmp_path, fake_model):
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(fake_model([ai("ok")]), manager=manager, root_dir=tmp_path)
    harness = AgentHarness(agent, manager, echo=False)

    state = harness.run_turn("hello")

    assert state.get("background_tasks", {}) == {}
    assert manager.list() == []


def test_tasks_report_lists_running_task(tmp_path, fake_model):
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    record = manager.start("sleep 60", description="sleeper")
    agent = build_agent(fake_model([ai("ok")]), manager=manager, root_dir=tmp_path)
    harness = AgentHarness(agent, manager, echo=False)

    report = harness.tasks_report()
    assert "bash-1" in report
    assert "status=running" in report
    assert "sleeper" in report

    manager.kill(record["task_id"])


def test_close_kills_leftover_tasks(tmp_path, fake_model):
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    record = manager.start("sleep 60", description="leftover")
    agent = build_agent(fake_model([ai("ok")]), manager=manager, root_dir=tmp_path)
    harness = AgentHarness(agent, manager, echo=False)

    harness.close()

    assert manager.get(record["task_id"])["status"] == "killed"


def test_echo_streaming_produces_output_without_errors(tmp_path, fake_model, capsys):
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(
        fake_model(
            [
                tool_call("start_background_task", {"command": "echo streamed", "description": "stream test"}, "c1"),
                ai("all good"),
                ai("noted"),  # extra turn in case the task completes mid-run
            ]
        ),
        manager=manager,
        root_dir=tmp_path,
    )
    harness = AgentHarness(agent, manager, echo=True)

    state = harness.run_turn("go")

    captured = capsys.readouterr().out
    assert "start_background_task" in captured
    assert "all good" in captured
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert any("bash-1" in str(m.content) for m in tool_messages)
    manager.wait("bash-1", timeout=10)


def test_failed_notification_turn_keeps_notifications_queued(tmp_path):
    """A provider error during delivery must not swallow completions."""
    import pytest

    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    record = manager.start("echo done", description="done task")
    manager.wait(record["task_id"], timeout=10)

    class _BrokenAgent:
        def stream(self, *args, **kwargs):
            raise RuntimeError("provider timeout")

        def get_state(self, config):  # pragma: no cover - stream fails first
            raise AssertionError("stream should have failed first")

    harness = AgentHarness(_BrokenAgent(), manager, echo=True)

    with pytest.raises(RuntimeError, match="provider timeout"):
        harness.flush_notifications()

    requeued = manager.take_notifications()
    assert [r["task_id"] for r in requeued] == [record["task_id"]]


def test_idle_notification_delivery_wakes_the_model(tmp_path, fake_model):
    """A completion must reach the model with no further user message."""
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(
        fake_model([ai("ok"), ai("noted")]),
        manager=manager,
        root_dir=tmp_path,
    )
    harness = AgentHarness(agent, manager, echo=False)

    harness.run_turn("go")  # starts the idle notifier
    manager.start("echo done", description="idle task")

    deadline = time.time() + 10
    delivered = False
    while time.time() < deadline:
        state = harness.agent.get_state(harness.config)
        texts = [str(m.content) for m in state.values.get("messages", [])]
        if any("<task-notification>" in t for t in texts) and any("noted" in t for t in texts):
            delivered = True
            break
        time.sleep(0.1)

    assert delivered, "the model was never woken by the completed task"
    harness.close()
    assert harness._notifier is None  # close() stops the notifier thread


def test_idle_notifier_can_be_disabled(tmp_path, fake_model):
    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(fake_model([ai("ok")]), manager=manager, root_dir=tmp_path)
    harness = AgentHarness(agent, manager, echo=False, notify_when_idle=False)

    harness.run_turn("go")

    assert harness._notifier is None
    harness.close()


def test_transient_model_error_resumes_turn_without_duplicating_input(tmp_path, fake_model):
    """A dropped connection mid-turn must resume, not kill the turn.

    LangGraph checkpoints the input before the model node runs, so the retry
    continues from the checkpoint with `None` input — re-sending the messages
    would append the user text twice (and eval caught a turn killed 372 s in).
    """
    harness = _harness(
        tmp_path,
        fake_model([ConnectionError("proxy dropped"), ai("готово")]),
        echo=False,
        retry_delay=0,
    )

    state = harness.run_turn("сделай задачу")
    messages = state["messages"]

    humans = [m.content for m in messages if isinstance(m, HumanMessage)]
    assert humans == ["сделай задачу"], "input must not be sent twice"
    assert messages[-1].content == "готово"


def test_transient_error_is_retried_in_streaming_mode_too(tmp_path, fake_model):
    lines: list[str] = []
    harness = _harness(
        tmp_path,
        fake_model([ConnectionError("boom"), ai("done")]),
        echo=True,
        printer=lines.append,
        retry_delay=0,
    )

    state = harness.run_turn("go")

    assert state["messages"][-1].content == "done"
    assert any("повтор" in line for line in lines), "the retry must be visible in the stream"


def test_non_transient_model_error_is_not_retried(tmp_path, fake_model):
    harness = _harness(
        tmp_path,
        fake_model([ValueError("сломано")]),
        echo=False,
        retry_delay=0,
    )

    with pytest.raises(ValueError, match="сломано"):
        harness.run_turn("hi")


def test_transient_retries_are_bounded(tmp_path, fake_model):
    harness = _harness(
        tmp_path,
        fake_model([ConnectionError("1"), ConnectionError("2"), ConnectionError("3")]),
        echo=False,
        model_turn_retries=2,
        retry_delay=0,
    )

    with pytest.raises(ConnectionError):
        harness.run_turn("hi")
