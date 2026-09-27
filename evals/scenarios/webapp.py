"""Scenario: write a small Python web app, test it, prove it works.

pip is allowed (venv inside the project or system-wide) — the checks verify
the outcome with independent interpreters, so the agent may pick any stack.
"""

from __future__ import annotations

from evals.checks import (
    background_starts,
    files_exist,
    no_running_tasks,
    pytest_runs,
    server_responds,
)
from evals.runner import Scenario

PROMPT = """\
Напиши простое web-приложение на Python (REST-список задач) и протестируй его.

Требования:
1. Код — в файле `app.py` текущего каталога:
   - порт берётся из переменной окружения PORT (по умолчанию 8000);
   - `GET /hello` отвечает 200 и JSON вида {"status": "ok"}.
2. Тесты — в файле `test_app.py` (pytest), покрывающие основные сценарии.
3. Зависимости можно ставить через pip (создавай venv в проекте, сеть доступна).
   Если зависимости есть — положи их в requirements.txt.
4. Долгие шаги (установка зависимостей, прогон тестов, подъём сервера) выполняй
   через start_background_task, а не через execute: долгие команды идут в фон.
5. Финал: тесты зелёные (прогони их сам и проверь вывод), сервер остановлен,
   фоновые задачи не остались висеть.

Ответь кратко: что сделано, где файлы, результат тестов.
"""

SCENARIO = Scenario(
    name="webapp",
    prompt=PROMPT,
    description="простое web-приложение + тесты + зелёный прогон",
    # 600 s of weather headroom: healthy runs take 290–360 s, but proxy
    # stalls (6 × up to 67 s observed) pushed two runs past the old 480 s
    # limit mid-fix — the eval must fail the agent, not the proxy's mood.
    time_limit=600.0,
    # History: the healthy deepseek run took 10 turns / 2 reads in 291 s; the
    # qwen swap regressed to 18 turns with batching gone while staying PASS —
    # 16 is the ceiling the prompt fix must get back under. Time is soft
    # because proxy stalls swing this scenario by minutes.
    max_llm_calls=16,
    max_polls=5,
    max_duration_s=420,
    checks=(
        files_exist("app.py"),
        files_exist("test_app.py"),
        pytest_runs(),
        server_responds(port=8765, path="/hello", entrypoint="app.py"),
        no_running_tasks(),
        background_starts(1, soft=True),
    ),
)
