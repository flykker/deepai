"""Evaluation suite for the agent: runnable scenarios, artifacts and reports.

An eval run puts the agent in a fresh sandbox, gives it a task, records every
model/tool event and then scores the result with independent checks (files
written, tests really pass, background tasks used and cleaned up). The output
is a self-contained run directory that can be compared against a baseline.
"""

from __future__ import annotations

from evals.checks import Check, CheckResult, Context
from evals.recorder import RunRecorder
from evals.runner import RunResult, Scenario, run_scenario
from evals.scenarios import SCENARIOS, get_scenario

__all__ = [
    "Check",
    "CheckResult",
    "Context",
    "SCENARIOS",
    "RunRecorder",
    "RunResult",
    "Scenario",
    "get_scenario",
    "run_scenario",
]
