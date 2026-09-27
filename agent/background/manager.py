"""Background shell task manager.

Mirrors the behavior of Claude Code's background bash tasks:

- ``start`` spawns a command in the background and returns immediately with a
  task id (``bash-1``, ``bash-2``, ...) plus a path to an output file.
- stdout/stderr are interleaved into that file while the command runs.
- ``read`` returns only the output produced since the previous read, so the
  agent can poll incrementally without re-reading megabytes.
- ``kill`` stops a task (SIGTERM, escalating to SIGKILL after a grace period).
- When a task finishes, a notification is queued for the harness, which injects
  it into the conversation so the model wakes up without polling.

Everything that must survive a checkpoint round-trip lives in
:class:`BackgroundTask` (plain JSON-able data). Live handles (``Popen``,
file offsets, locks) stay in process memory.

Records are additionally persisted to ``tasks.json`` next to the output
files, so a restarted session keeps its history: finished tasks stay
readable (status, exit code, whole output file) and tasks that died with the
previous process come back as ``interrupted``.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

__all__ = [
    "BackgroundTask",
    "BackgroundTaskManager",
    "OutputSnapshot",
    "TaskStatus",
    "TERMINAL_STATUSES",
    "UnknownTaskError",
    "format_task_line",
]

TaskStatus = Literal["running", "completed", "failed", "killed", "interrupted"]

TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "killed", "interrupted"})
"""Statuses that never change again, so live refreshes can be skipped."""


class BackgroundTask(TypedDict):
    """Serializable snapshot of one background task.

    This is the shape persisted in LangGraph state (and therefore in the
    checkpointer), so it must stay JSON-able: no handles, no locks.
    """

    task_id: str
    command: str
    description: str
    output_path: str
    status: TaskStatus
    exit_code: int | None
    pid: int | None
    started_at: str
    finished_at: str | None
    last_read_status: TaskStatus | None
    timed_out: bool


class UnknownTaskError(KeyError):
    """Raised when a task id is not known to this manager."""


@dataclass(frozen=True)
class OutputSnapshot:
    """Result of reading new output from a background task."""

    task_id: str
    status: TaskStatus
    exit_code: int | None
    new_output: str
    truncated: bool
    filtered: bool


@dataclass
class _LiveTask:
    """In-process handle for a running task (never leaves the manager)."""

    record: BackgroundTask
    process: subprocess.Popen[bytes]
    offset: int = 0
    notified: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    """Per-task completion signal (``default_factory`` keeps it from being shared)."""


def _now() -> str:
    """UTC timestamp with second precision (ISO-8601, ``Z`` suffix)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_task_line(record: BackgroundTask) -> str:
    """One-line human-readable summary of a task (used by ``/tasks``)."""
    exit_code = record["exit_code"] if record["exit_code"] is not None else "-"
    return (
        f'- {record["task_id"]}: status={record["status"]} exit_code={exit_code} '
        f'command={record["command"]!r} description={record["description"]!r}'
    )


class BackgroundTaskManager:
    """Spawn, track, read and kill background shell tasks for one session."""

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
        session_id: str | None = None,
        kill_grace_seconds: float = 5.0,
        max_output_chars: int = 20_000,
    ) -> None:
        """Set up the manager.

        Args:
            output_dir: Root directory for task output files. A per-session
                subdirectory is created inside it and holds ``tasks.json``
                (the persisted record store) plus every ``bash-N.output``.
                Defaults to the system temp directory + ``deepai-bg-tasks``.
            cwd: Default working directory for spawned commands.
            env: Extra environment variables (merged over the inherited ones).
            inherit_env: Whether to inherit ``os.environ`` for spawned commands.
            session_id: Identifier used in the output directory name. Keep it
                stable across restarts to recover the previous task history.
            kill_grace_seconds: How long to wait after SIGTERM before SIGKILL.
            max_output_chars: Cap on characters returned by a single read.
        """
        self.session_id = session_id or uuid.uuid4().hex[:8]
        root = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "deepai-bg-tasks"
        self.output_dir = root / self.session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.kill_grace_seconds = kill_grace_seconds
        self.max_output_chars = max_output_chars
        self._store = self.output_dir / "tasks.json"

        base_env = dict(os.environ) if inherit_env else {}
        if env:
            base_env.update(env)
        self._env = base_env

        self._records: dict[str, BackgroundTask] = {}  # every known task, live or restored
        self._live: dict[str, _LiveTask] = {}  # tasks with a Popen handle in this process
        self._notifications: deque[BackgroundTask] = deque()
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._counter = 0
        self._closed = False
        self._load_records()

    # ------------------------------------------------------------------ start

    def start(
        self,
        command: str,
        *,
        description: str,
        timeout_seconds: float | None = None,
        cwd: str | Path | None = None,
        notify: bool = True,
    ) -> BackgroundTask:
        """Start ``command`` in the background and return its snapshot.

        The call returns as soon as the process is spawned; nothing blocks on
        the command finishing.

        Args:
            command: Shell command line (run through ``/bin/sh -c``).
            description: Short human-readable summary shown in notifications.
            timeout_seconds: Optional wall-clock limit. On expiry the process
                group is killed and the task is marked ``failed`` with
                ``timed_out=True``.
            cwd: Override the manager's default working directory.
            notify: Queue a harness notification when the task finishes.

        Returns:
            The initial :class:`BackgroundTask` snapshot (status ``running``).

        Raises:
            RuntimeError: If the manager has been shut down.
            ValueError: If ``command`` is empty.
        """
        if self._closed:
            msg = "BackgroundTaskManager is shut down"
            raise RuntimeError(msg)
        if not command or not command.strip():
            msg = "command must be a non-empty string"
            raise ValueError(msg)

        with self._lock:
            self._counter += 1
            task_id = f"bash-{self._counter}"

        output_path = self.output_dir / f"{task_id}.output"
        workdir = Path(cwd) if cwd else self.cwd
        workdir.mkdir(parents=True, exist_ok=True)

        output_handle = open(output_path, "wb")  # noqa: SIM115 - closed right after spawn
        try:
            process = subprocess.Popen(  # noqa: S602
                command,
                shell=True,  # Intentional: the agent drives a real shell, like Claude Code's Bash tool.
                cwd=str(workdir),
                env=self._env,
                stdin=subprocess.DEVNULL,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
        finally:
            # The child owns a duplicate of the descriptor, so closing ours here
            # is safe — and it keeps the parent from pinning the file open.
            output_handle.close()

        record: BackgroundTask = {
            "task_id": task_id,
            "command": command,
            "description": description or command,
            "output_path": str(output_path),
            "status": "running",
            "exit_code": None,
            "pid": process.pid,
            "started_at": _now(),
            "finished_at": None,
            "last_read_status": None,
            "timed_out": False,
        }
        live = _LiveTask(record=record, process=process)
        with self._lock:
            self._live[task_id] = live
            self._records[task_id] = record  # same object: finalization shows up in both
            self._save()

        threading.Thread(
            target=self._watch,
            args=(task_id, timeout_seconds, notify),
            name=f"bg-task-{task_id}",
            daemon=True,
        ).start()

        return dict(record)

    # ------------------------------------------------------------------- read

    def read(
        self,
        task_id: str,
        *,
        filter: str | None = None,  # noqa: A002 - mirrors Claude Code's BashOutput `filter` argument
        max_chars: int | None = None,
    ) -> OutputSnapshot:
        """Return output produced since the previous ``read`` for this task.

        Args:
            task_id: Task id returned by :meth:`start`.
            filter: Optional regular expression. Only matching lines are
                returned — lines that do not match are consumed and will not be
                returned by later reads (same semantics as Claude Code).
            max_chars: Override the manager-wide output cap.

        Returns:
            An :class:`OutputSnapshot` with the task's live status. For a
            restored task (no process in this session) the whole output file
            is returned on every read — there is no offset across restarts.

        Raises:
            UnknownTaskError: If ``task_id`` is unknown.
        """
        record = self._record(task_id)
        live = self._live.get(task_id)
        if live is not None:
            self._refresh(live)
            record = live.record

        # What status did this read see? A read that observed a terminal
        # status means the model already knows the task finished — the
        # queued completion notification must not repeat the news (eval:
        # re-surfacing finished work sent the model into a ~10-turn re-audit).
        record["last_read_status"] = record["status"]

        path = Path(record["output_path"]) if record["output_path"] else None
        data = b""
        if path is not None and path.exists():
            with path.open("rb") as fh:
                fh.seek(live.offset if live is not None else 0)
                data = fh.read()
                if live is not None:
                    live.offset = fh.tell()

        text = data.decode("utf-8", errors="replace")

        filtered = False
        if filter:
            filtered = True
            try:
                pattern = re.compile(filter)
            except re.error as exc:
                msg = f"Invalid filter regex: {exc}"
                return OutputSnapshot(
                    task_id=task_id,
                    status=record["status"],
                    exit_code=record["exit_code"],
                    new_output=msg,
                    truncated=False,
                    filtered=True,
                )
            kept = [line for line in text.splitlines(keepends=True) if pattern.search(line)]
            text = "".join(kept)

        limit = max_chars if max_chars is not None else self.max_output_chars
        truncated = False
        if len(text) > limit:
            head = text[: limit // 2]
            tail = text[-(limit // 2) :]
            text = f"{head}\n... [output truncated, {len(text) - limit} characters omitted] ...\n{tail}"
            truncated = True

        return OutputSnapshot(
            task_id=task_id,
            status=record["status"],
            exit_code=record["exit_code"],
            new_output=text,
            truncated=truncated,
            filtered=filtered,
        )

    # -------------------------------------------------------------------- kill

    def kill(self, task_id: str) -> BackgroundTask:
        """Stop a running task and mark it ``killed``.

        SIGTERM is sent to the whole process group (so children of the command
        die too), escalating to SIGKILL after :attr:`kill_grace_seconds`.

        Args:
            task_id: Task id returned by :meth:`start`.

        Returns:
            The updated task snapshot.

        Raises:
            UnknownTaskError: If ``task_id`` is unknown.
        """
        record = self._record(task_id)
        live = self._live.get(task_id)
        if live is None:
            # Restored record: the process belonged to a previous session,
            # there is nothing left to signal.
            return dict(record)
        # Fold in an exit that already happened, so a finished task is reported
        # with its real status instead of being re-labeled "killed".
        self._refresh(live)
        with self._lock:
            if live.record["status"] in TERMINAL_STATUSES:
                return dict(live.record)
            # Claim the terminal status *before* signalling the process: the
            # watcher would otherwise record the SIGTERM exit code as a failure.
            record = live.record
            record["status"] = "killed"
            record["pid"] = None

        self._terminate(live.process)

        with self._lock:
            record["exit_code"] = live.process.poll()
            record["finished_at"] = _now()
            record["timed_out"] = False
            snapshot = dict(record)
            live.finished.set()
            self._save()
        return snapshot

    # ---------------------------------------------------------------- queries

    def get(self, task_id: str) -> BackgroundTask | None:
        """Return the snapshot for ``task_id``, or ``None`` if unknown."""
        with self._lock:
            record = self._records.get(task_id)
            live = self._live.get(task_id)
        if record is None:
            return None
        if live is not None:
            self._refresh(live)
        with self._lock:
            return dict(record)

    def list(self, status_filter: str | None = None) -> list[BackgroundTask]:  # noqa: A003
        """Return snapshots for all known tasks (including restored ones).

        Args:
            status_filter: One of ``running``/``completed``/``failed``/
                ``killed``/``interrupted``/``all`` (or ``None`` for ``all``).
        """
        with self._lock:
            lives = list(self._live.values())
        for live in lives:
            self._refresh(live)
        with self._lock:
            records = [dict(record) for record in self._records.values()]
        if status_filter and status_filter != "all":
            records = [r for r in records if r["status"] == status_filter]
        return sorted(records, key=lambda r: (r["started_at"], r["task_id"]))

    def wait(self, task_id: str, timeout: float | None = None) -> BackgroundTask:
        """Block until ``task_id`` reaches a terminal status.

        Mainly useful for tests and for callers that genuinely need to
        synchronize. Returns the final snapshot, or raises :class:`TimeoutError`.
        A restored record is already terminal and returns immediately.
        """
        record = self._record(task_id)
        live = self._live.get(task_id)
        if live is None:
            return dict(record)
        if not live.finished.wait(timeout):
            msg = f"Task {task_id} still {live.record['status']} after {timeout}s"
            raise TimeoutError(msg)
        self._refresh(live)
        with self._lock:
            return dict(live.record)

    def take_notifications(self) -> list[BackgroundTask]:
        """Drain queued completion notifications (FIFO).

        A completion the model already observed on its own — its last read of
        that task saw a terminal status — is dropped rather than delivered:
        re-surfacing work it has already seen only sends it back to re-audit
        what it knows (eval webapp 20260926-235404: stale notifications for
        two finished tasks cost ~10 model turns and the run's time limit).
        """
        with self._lock:
            drained = list(self._notifications)
            self._notifications.clear()
        fresh: list[BackgroundTask] = []
        for notification in drained:
            record = self._records.get(notification["task_id"])
            observed = (record if record is not None else notification).get("last_read_status")
            if observed in TERMINAL_STATUSES:
                continue  # the model saw the finish itself — nothing new to say
            fresh.append(notification)
        return fresh

    def pending_notifications(self) -> int:
        """How many notifications are queued (peek — does not drain)."""
        with self._lock:
            return len(self._notifications)

    def wait_for_notifications(self, timeout: float = 0.5) -> bool:
        """Block until a notification is queued or ``timeout`` elapses.

        Used by the harness's idle notifier to wake up exactly when a task
        finishes instead of polling the queue in a tight loop.
        """
        with self._cond:
            if self._notifications:
                return True
            self._cond.wait(timeout)
            return bool(self._notifications)

    def requeue_notifications(self, records: list[BackgroundTask]) -> None:
        """Put drained notifications back so a failed delivery is not lost.

        The harness calls this when the model call that should have carried the
        notifications raised: completions stay queued for the next attempt.
        """
        if not records:
            return
        with self._lock:
            self._notifications.extendleft(reversed(records))
            self._cond.notify_all()

    def shutdown(self) -> None:
        """Kill every running task and refuse further starts.

        Called on harness exit so stray dev servers do not outlive the session.
        """
        self._closed = True
        with self._lock:
            running = [live for live in self._live.values() if live.record["status"] == "running"]
        for live in running:
            self.kill(live.record["task_id"])

    # -------------------------------------------------------------- internals

    def _record(self, task_id: str) -> BackgroundTask:
        """Look up any known record (restored history included).

        Raises:
            UnknownTaskError: If ``task_id`` is unknown to this manager.
        """
        with self._lock:
            record = self._records.get(task_id)
        if record is None:
            known = ", ".join(sorted(self._records)) or "none"
            msg = f"No background task with id {task_id!r}. Known ids: {known}"
            raise UnknownTaskError(msg)
        return record

    def _save(self) -> None:
        """Persist every record to ``tasks.json`` atomically (call under lock)."""
        payload = {
            "version": 1,
            "tasks": {task_id: dict(record) for task_id, record in self._records.items()},
        }
        tmp = self._store.with_name(self._store.name + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._store)
        except OSError:  # pragma: no cover - disk full / permissions
            pass

    def _load_records(self) -> None:
        """Restore the previous session's records, if the store exists.

        Tasks still marked ``running`` belonged to a dead process — they are
        finalized as ``interrupted`` (the process is not re-attached: its pid
        means nothing in this session). Nothing is queued for notification:
        history must not wake the model.
        """
        if not self._store.exists():
            return
        try:
            payload = json.loads(self._store.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return  # a corrupt store must not stop the manager from starting
        raw = payload.get("tasks") if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            return

        valid = {"running", "completed", "failed", "killed", "interrupted"}
        changed = False
        for task_id, record in raw.items():
            if not isinstance(record, dict) or "command" not in record:
                continue
            status = record.get("status")
            restored: BackgroundTask = {
                "task_id": str(record.get("task_id") or task_id),
                "command": str(record["command"]),
                "description": str(record.get("description") or record["command"]),
                "output_path": str(record.get("output_path") or ""),
                "status": status if status in valid else "interrupted",
                "exit_code": record.get("exit_code"),
                "pid": record.get("pid"),
                "started_at": str(record.get("started_at") or _now()),
                "finished_at": record.get("finished_at"),
                "last_read_status": record.get("last_read_status"),
                "timed_out": bool(record.get("timed_out", False)),
            }
            if restored["status"] == "running":
                restored["status"] = "interrupted"
                restored["finished_at"] = _now()
                restored["pid"] = None
                changed = True
            self._records[restored["task_id"]] = restored
            match = re.fullmatch(r"bash-(\d+)", restored["task_id"])
            if match:
                self._counter = max(self._counter, int(match.group(1)))
        if changed:
            self._save()

    def _refresh(self, live: _LiveTask) -> None:
        """Fold a finished process into its record, if the watcher hasn't yet."""
        if live.record["status"] in TERMINAL_STATUSES:
            return
        code = live.process.poll()
        if code is None:
            return
        with self._lock:
            if live.record["status"] != "running":
                return
        self._finalize(live, status="completed" if code == 0 else "failed", exit_code=code, timed_out=False, notify=True)

    def _watch(self, task_id: str, timeout_seconds: float | None, notify: bool) -> None:
        """Wait for a spawned task and publish its terminal state."""
        with self._lock:
            live = self._live.get(task_id)
        if live is None:  # pragma: no cover - defensive
            return
        process = live.process
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate(process)
            exit_code = process.poll()
        except Exception:  # noqa: BLE001 - a broken wait must not kill the watcher thread
            exit_code = process.poll()
        with self._lock:
            already = live.record["status"]
        if already in TERMINAL_STATUSES:
            live.finished.set()
            return
        status: TaskStatus = "failed" if timed_out else ("completed" if exit_code == 0 else "failed")
        self._finalize(live, status=status, exit_code=exit_code, timed_out=timed_out, notify=notify)

    def _finalize(
        self,
        live: _LiveTask,
        *,
        status: TaskStatus,
        exit_code: int | None,
        timed_out: bool,
        notify: bool,
    ) -> None:
        """Move a task to a terminal status exactly once, then notify."""
        with self._lock:
            record = live.record
            if record["status"] in TERMINAL_STATUSES:
                live.finished.set()
                return
            record["status"] = status
            record["exit_code"] = exit_code
            record["timed_out"] = timed_out
            record["finished_at"] = _now()
            record["pid"] = None
            snapshot = dict(record)
            should_notify = notify and status in {"completed", "failed"} and not live.notified
            if should_notify:
                live.notified = True
                self._notifications.append(snapshot)
                self._cond.notify_all()
            live.finished.set()
            self._save()

    def _terminate(self, process: subprocess.Popen[bytes]) -> None:
        """SIGTERM the process group, escalating to SIGKILL after a grace period."""
        if process.poll() is not None:
            return
        if os.name == "nt":  # pragma: no cover - Windows path
            process.terminate()
            try:
                process.wait(timeout=self.kill_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            process.wait(timeout=self.kill_grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass
