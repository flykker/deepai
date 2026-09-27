"""Scenario: a dev server's full lifecycle — start, verify, kill, verify.

Scores the part of the background-task design that the other scenarios only
touch in passing: a *long-lived* task that never completes on its own must be
verified while alive and then stopped with ``kill_task``, leaving nothing
listening on the port.
"""

from __future__ import annotations

from evals.checks import (
    background_starts,
    no_running_tasks,
    port_closed,
    tool_call_count,
    transcript_contains,
)
from evals.runner import Scenario

PROMPT = """\
Проверь полный жизненный цикл долгой фоновой задачи — dev-сервера.

1. В фоне через start_background_task подними статик-сервер в этом каталоге:
   `python3 -m http.server 8123 --bind 127.0.0.1`
   (description: "static server on 8123").
2. Отдельным коротким `execute` убедись, что сервер отвечает: сделай GET на
   http://127.0.0.1:8123/ и покажи код ответа (должен быть 200).
3. Не опрашивай сервер и не жди — сразу после проверки сообщи id задачи.
4. Останови сервер через kill_task(task_id=...).
5. Тем же коротким `execute` проверь, что соединение теперь отклоняется.
6. `list_tasks` подтверди, что задача не осталась висеть.

Ответь кратко: id сервера, что отвечал, и что после kill порт закрыт.
"""

SCENARIO = Scenario(
    name="devserver",
    prompt=PROMPT,
    description="дев-сервер в фоне: поднять, проверить 200, убить, проверить порт",
    time_limit=300.0,
    # History: 5–6 turns, 0–1 reads, 74–89 s; the flow is fixed, so extra
    # turns mean the model lost the thread (id → kill → re-check).
    max_llm_calls=9,
    max_polls=3,
    max_duration_s=180,
    checks=(
        background_starts(1, soft=False),
        tool_call_count("kill_task", 1),
        transcript_contains("200"),
        port_closed(8123),
        no_running_tasks(),
    ),
)
