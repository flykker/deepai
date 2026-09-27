"""CLI wiring: banner, slash commands, exit code and shutdown."""

from __future__ import annotations

import builtins

import pytest

from agent import main as cli


class StubAgent:
    """Minimal compiled-agent stand-in so the CLI can run without a model."""

    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def invoke(self, inputs, config=None):  # noqa: ANN001, ANN003, ARG002
        self.inputs.append(inputs)
        return {"messages": list(inputs["messages"]), "background_tasks": {}}


@pytest.fixture
def stub_agent(monkeypatch):
    holder: dict = {}

    def _build(model=None, *, manager, **kwargs):  # noqa: ANN001, ANN003
        agent = StubAgent()
        holder["agent"] = agent
        holder["manager"] = manager
        return agent

    monkeypatch.setattr(cli, "build_agent", _build)
    return holder


def test_single_prompt_run_returns_zero_and_prints_report(tmp_path, stub_agent, capsys):
    code = cli.main(["--root", str(tmp_path), "--prompt", "hello", "--quiet"])

    assert code == 0
    assert stub_agent["agent"].inputs[0]["messages"][0].content == "hello"
    out = capsys.readouterr().out
    assert "No background tasks tracked." in out
    # the session-wide manager is shut down when the CLI exits
    with pytest.raises(RuntimeError):
        stub_agent["manager"].start("echo nope", description="after close")


def test_slash_commands_and_exit(tmp_path, stub_agent, monkeypatch, capsys):
    commands = iter(["/help", "/tasks", "do something", "/exit"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(commands))

    code = cli.main(["--root", str(tmp_path), "--quiet"])

    assert code == 0
    printed = capsys.readouterr().out
    assert "/tasks [status]" in printed
    assert "No background tasks tracked." in printed
    assert stub_agent["agent"].inputs[-1]["messages"][0].content == "do something"


def test_missing_root_directory(tmp_path, capsys):
    code = cli.main(["--root", str(tmp_path / "does-not-exist"), "--prompt", "hi"])
    assert code == 2
    assert "root directory does not exist" in capsys.readouterr().err
