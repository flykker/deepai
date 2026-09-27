"""Instrumented e2e demo: three background tasks started in parallel."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from agent.env import load_env  # noqa: E402

load_env()

from agent.background import BackgroundTaskManager  # noqa: E402
from agent.graph import build_agent  # noqa: E402
from agent.harness import AgentHarness  # noqa: E402

START = time.monotonic()


def log(message: str) -> None:
    print(f"[{time.monotonic() - START:6.1f}s] {message}", flush=True)


def printer(line: str) -> None:
    log(line)


def main() -> None:
    manager = BackgroundTaskManager(cwd=".")
    agent = build_agent(manager=manager, root_dir=".")
    harness = AgentHarness(agent, manager, echo=True, printer=printer)

    log("turn start")
    harness.run_turn(
        "Параллельно запусти три фоновые задачи: сборка (sleep 5 && echo BUILD), "
        "тесты (sleep 5 && echo TESTS), загрузка (sleep 5 && echo DL). "
        "Запусти их все сразу одним ответом, перечисли id, не дожидаясь завершения."
    )
    log(f"turn done; tasks={[(r['task_id'], r['status']) for r in manager.list()]}")

    time.sleep(6)
    log("flush start")
    harness.flush_notifications()
    log("flush done")
    log(harness.tasks_report())
    manager.shutdown()
    log("OK")


if __name__ == "__main__":
    main()
