"""Model resolution: bounded request timeouts for OpenAI-compatible specs."""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from agent.graph import (
    MODEL_ENV_VAR,
    RETRIES_ENV_VAR,
    TIMEOUT_ENV_VAR,
    resolve_model_spec,
)


@pytest.fixture(autouse=True)
def _openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for var in (TIMEOUT_ENV_VAR, RETRIES_ENV_VAR):
        monkeypatch.delenv(var, raising=False)


def test_openai_spec_gets_a_bounded_timeout():
    resolved = resolve_model_spec("openai:deepseek-v4.1-flash")

    assert isinstance(resolved, ChatOpenAI)
    assert resolved.model_name == "deepseek-v4.1-flash"
    # Without this the client would silently wait up to ten minutes for a
    # stalled proxy response — which is what a hung session looks like.
    assert resolved.request_timeout == 30.0
    assert resolved.max_retries == 4


def test_bare_model_name_targets_the_openai_compatible_client(monkeypatch):
    resolved = resolve_model_spec("deepseek-v4.1-flash")

    assert isinstance(resolved, ChatOpenAI)
    assert resolved.model_name == "deepseek-v4.1-flash"


def test_timeout_and_retries_are_configurable_via_env(monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV_VAR, "25")
    monkeypatch.setenv(RETRIES_ENV_VAR, "5")

    resolved = resolve_model_spec("openai:deepseek-v4.1-flash")

    assert isinstance(resolved, ChatOpenAI)
    assert resolved.request_timeout == 25.0
    assert resolved.max_retries == 5


def test_zero_timeout_disables_the_limit(monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV_VAR, "0")

    resolved = resolve_model_spec("openai:some-model")

    assert isinstance(resolved, ChatOpenAI)
    assert resolved.request_timeout is None


def test_non_openai_provider_spec_is_left_for_deepagents():
    """Providers this module does not configure keep the plain string spec."""
    assert resolve_model_spec("anthropic:claude-sonnet-4-6") == "anthropic:claude-sonnet-4-6"


def test_prebuilt_model_instance_is_returned_unchanged():
    model = ChatOpenAI(model="already-built", timeout=5.0)
    assert resolve_model_spec(model) is model


def test_agent_model_env_var_is_used(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "openai:from-env")

    resolved = resolve_model_spec(None)

    assert isinstance(resolved, ChatOpenAI)
    assert resolved.model_name == "from-env"


def test_build_agent_always_includes_sandbox_paths_prompt(tmp_path, fake_model):
    """`/` is the project root for filesystem tools — the model must be told."""
    from agent.background.manager import BackgroundTaskManager
    from agent.graph import SANDBOX_PATHS_PROMPT, build_agent

    manager = BackgroundTaskManager(output_dir=tmp_path / "tasks", cwd=tmp_path)
    agent = build_agent(fake_model([]), manager=manager, root_dir=tmp_path)
    assert agent is not None  # compiled with the default (paths-only) prompt

    assert "Sandbox paths" in SANDBOX_PATHS_PROMPT
    assert "`/app.py`" in SANDBOX_PATHS_PROMPT
    assert "pwd" in SANDBOX_PATHS_PROMPT


def test_background_prompt_forbids_sleeping_the_turn_away():
    """eval showed the model polling/sleeping instead of ending the turn."""
    from agent.background.middleware import BACKGROUND_SYSTEM_PROMPT, START_TOOL_DESCRIPTION

    for text in (BACKGROUND_SYSTEM_PROMPT, START_TOOL_DESCRIPTION):
        assert "on its own" in text or "end" in text
        assert "sleep" in text, "must explicitly forbid `execute sleep ...` as a wait"


def test_prompt_batches_independent_tool_calls():
    """eval: qwen spent one turn per `write_file` (18 turns vs deepseek's 10).

    The general batching rule must be in the system prompt, not only the
    background-task-specific one.
    """
    from agent.graph import BATCHING_PROMPT

    assert "one response" in BATCHING_PROMPT
    assert "write_file" in BATCHING_PROMPT, "the exact case eval caught must be named"
    assert "get_task_output" in BATCHING_PROMPT, "notification read + next action batch together"
    assert "depend" in BATCHING_PROMPT, "must carve out real dependencies"


def test_background_prompt_batches_notification_reads():
    from agent.background.middleware import BACKGROUND_SYSTEM_PROMPT

    assert "same response" in BACKGROUND_SYSTEM_PROMPT
