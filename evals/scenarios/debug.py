"""Scenario: a planted bug in existing code must be found and fixed.

Every other scenario scores *generation from scratch*; this one scores the
skill real work is made of — reading unfamiliar code, reproducing a failure,
and fixing it minimally. The workspace is seeded (``setup``) with a broken
``app.py`` and a *correct* ``test_app.py``: the tests are the independent
judge, so ``file_content_is`` pins them byte for byte — editing the tests
instead of the code is a FAIL, not a shortcut.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from evals.checks import file_content_is, pytest_runs, tool_call_count
from evals.runner import Scenario

APP_SOURCE = '''"""In-memory список задач: добавление, выполнение, сводка."""

from __future__ import annotations

TASKS: list[dict] = []


def add_task(title: str, tags: list[str] = []) -> dict:
    """Добавить задачу (новые получают пустой список тегов по умолчанию)."""
    task = {"id": next_id(), "title": title, "tags": tags, "done": False}
    TASKS.append(task)
    return task


def complete_task(task_id: int) -> dict | None:
    """Отметить задачу выполненной; None, если id не найден."""
    for task in TASKS:
        if task["id"] == task_id:
            task["done"] = True
            return task
    return None


def summarize() -> dict:
    """Сколько задач всего, выполнено и в ожидании."""
    done = sum(1 for task in TASKS if task["done"])
    return {"total": len(TASKS), "done": done, "pending": len(TASKS) - done}


def next_id() -> int:
    """Свободный id: на 1 больше максимального (или 1, когда список пуст)."""
    if not TASKS:
        return 1
    return max(task["id"] for task in TASKS)
'''

TEST_SOURCE = '''"""Тесты сервиса список задач.

Они корректны и служат независимой проверкой — их нельзя менять,
чтобы «починить» падения.
"""

import pytest

import app


@pytest.fixture(autouse=True)
def clean_state():
    app.TASKS.clear()
    yield
    app.TASKS.clear()


def test_first_id_is_one():
    assert app.add_task("first")["id"] == 1


def test_ids_never_collide():
    ids = [app.add_task(f"t{i}")["id"] for i in range(3)]
    assert len(set(ids)) == 3


def test_complete_marks_done_and_missing_returns_none():
    task = app.add_task("work")
    assert app.complete_task(task["id"])["done"] is True
    assert app.complete_task(999) is None


def test_summarize_counts():
    app.add_task("a")
    app.add_task("b")
    app.complete_task(1)
    assert app.summarize() == {"total": 2, "done": 1, "pending": 1}


def test_default_tags_do_not_leak_between_tasks():
    first = app.add_task("one")
    first["tags"].append("urgent")
    second = app.add_task("two")
    assert second["tags"] == []


def test_explicit_tags_are_kept():
    task = app.add_task("three", ["x", "y"])
    assert task["tags"] == ["x", "y"]
'''


def plant_files(workspace: Path) -> None:
    """Seed the workspace with the broken app and its correct tests."""
    (workspace / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    (workspace / "test_app.py").write_text(TEST_SOURCE, encoding="utf-8")


def _seed_pytest_venv(workspace: Path) -> None:
    """Give the agent a ready interpreter with pytest, off the turn clock.

    The eval interpreter has pytest but the agent's ``execute`` sees the
    system ``python3``, which does not — first run wasted 6 of 13 turns
    hunting for one (``which pytest``, ``pip install``, ``cd /`` confusion).
    Best effort: on any failure the files are still planted and the prompt
    tells the agent to fall back to ``pip install pytest``.
    """
    venv = workspace / ".venv"
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv)],
            check=True,
            capture_output=True,
            timeout=90,
        )
        subprocess.run(
            [str(venv / "bin" / "python"), "-m", "pip", "install", "-q", "pytest"],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except Exception:  # noqa: BLE001 - non-fatal: the prompt has a fallback
        pass


def _plant(workspace: Path) -> None:
    plant_files(workspace)
    _seed_pytest_venv(workspace)


PROMPT = """\
В каталоге лежат два файла: `app.py` — сервис списка задач, и `test_app.py` —
его тесты. Сейчас `pytest` падает.

Сделай:
1. Запусти тесты из текущего каталога: `.venv/bin/python -m pytest -q`
   (venv уже создан). Если его вдруг нет — `pip install pytest` и
   `python3 -m pytest`. НЕ делай `cd /` и не ищи pytest по системе: shell
   уже работает в рабочем каталоге проекта.
2. Найди причину в `app.py` и почини её минимальными правками — не
   переписывай файл целиком и не подгоняй код под каждый тест отдельно.
3. `test_app.py` НЕ меняй: тесты корректны, они и есть независимая проверка.
4. Прогони тесты ещё раз, добейся зелёного результата и коротко сообщи,
   что было сломано и как починил (назови функции и суть бага).

Больше ничего делать не нужно: ни сервера, ни новых файлов, ни установок.
"""

SCENARIO = Scenario(
    name="debug",
    prompt=PROMPT,
    description="в готовом приложении спрятан баг — найти и починить, не трогая тесты",
    time_limit=300.0,
    # Healthy flow is ~6 turns: run tests, read, fix twice, re-run, report.
    max_llm_calls=10,
    max_polls=2,
    max_duration_s=180,
    setup=_plant,
    checks=(
        pytest_runs(),
        file_content_is("test_app.py", TEST_SOURCE),
        tool_call_count("execute", min_count=1, soft=True),
    ),
)
