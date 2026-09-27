"""Agent package: deep agent + Claude Code-style background tasks."""

from agent.background import (
    BackgroundTask,
    BackgroundTaskManager,
    BackgroundTasksMiddleware,
    BackgroundTasksState,
    render_notifications,
)
from agent.graph import build_agent
from agent.harness import AgentHarness

__all__ = [
    "AgentHarness",
    "BackgroundTask",
    "BackgroundTaskManager",
    "BackgroundTasksMiddleware",
    "BackgroundTasksState",
    "build_agent",
    "render_notifications",
]
