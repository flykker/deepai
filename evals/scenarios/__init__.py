"""Scenario registry: the tasks an eval run can be given."""

from __future__ import annotations

from evals.runner import Scenario
from evals.scenarios.debug import SCENARIO as DEBUG
from evals.scenarios.devserver import SCENARIO as DEVSERVER
from evals.scenarios.parallel import SCENARIO as PARALLEL
from evals.scenarios.webapp import SCENARIO as WEBAPP

__all__ = ["SCENARIOS", "get_scenario", "scenario_names"]

SCENARIOS: dict[str, Scenario] = {s.name: s for s in (WEBAPP, PARALLEL, DEVSERVER, DEBUG)}


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name (raises with the list of known ones)."""
    try:
        return SCENARIOS[name]
    except KeyError:
        known = ", ".join(sorted(SCENARIOS))
        raise SystemExit(f"unknown scenario {name!r}; available: {known}") from None


def scenario_names() -> list[str]:
    """Sorted scenario names, for ``--list``."""
    return sorted(SCENARIOS)
