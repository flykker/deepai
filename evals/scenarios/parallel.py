"""Scenario: three independent long tasks must start in parallel.

Scores the core of the background-task design: several `start_background_task`
calls in a single model response, real process overlap, and the outputs read
back after the notifications arrive.
"""

from __future__ import annotations

from evals.checks import (
    background_starts,
    no_running_tasks,
    parallel_batch,
    tasks_overlap,
    transcript_contains,
)
from evals.runner import Scenario

PROMPT = """\
Запусти три независимые долгие команды строго параллельно:

  sleep 6 && echo STEP-A
  sleep 6 && echo STEP-B
  sleep 6 && echo STEP-C

Правила:
- каждую — через start_background_task;
- все три вызова сделай одним своим ответом (параллельно), а не по одному;
- не дожидайся завершения, просто перечисли id;
- когда придут уведомления — прочитай вывод каждой задачи (get_task_output)
  и заверши, назвав id и что напечатала каждая задача;
- ничего не оставляй висеть.
"""

SCENARIO = Scenario(
    name="parallel",
    prompt=PROMPT,
    description="три фоновые задачи одним батчем + чтение вывода",
    time_limit=240.0,
    # History: healthy runs take 4–5 turns and read 3 outputs (17–27 s under
    # good weather); a 140 s outlier was pure proxy stall, hence the soft time.
    max_llm_calls=7,
    max_polls=6,
    max_duration_s=150,
    checks=(
        background_starts(3, soft=False),
        parallel_batch(3),
        tasks_overlap(3),
        transcript_contains("STEP-A", "STEP-B", "STEP-C"),
        no_running_tasks(),
    ),
)
