"""Claude Code-style background tasks for a deep agent.

Public surface:

- :class:`~agent.background.manager.BackgroundTaskManager` — spawn/track/read/
  kill background shell commands and collect completion notifications.
- :class:`~agent.background.middleware.BackgroundTasksMiddleware` — LangGraph
  middleware that exposes the task tools to the model.
- :func:`~agent.background.notifications.render_notifications` — turn finished
  tasks into the user-level message the harness injects into the conversation.
"""

from agent.background.manager import (
    BackgroundTask,
    BackgroundTaskManager,
    OutputSnapshot,
    TaskStatus,
    UnknownTaskError,
)
from agent.background.middleware import BackgroundTasksMiddleware, BackgroundTasksState
from agent.background.notifications import render_notification, render_notifications

__all__ = [
    "BackgroundTask",
    "BackgroundTaskManager",
    "BackgroundTasksMiddleware",
    "BackgroundTasksState",
    "OutputSnapshot",
    "TaskStatus",
    "UnknownTaskError",
    "render_notification",
    "render_notifications",
]
