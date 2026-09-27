"""Minimal ``.env`` loader (no third-party dependency).

Reads ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ`` without
overriding variables that are already set, so an exported real value always
wins over a checked-out file.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_env"]

ENV_FILE = ".env"
EXPORT_PREFIX = "export "


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` pairs from ``path`` into the process environment.

    Args:
        path: File to read. Defaults to ``.env`` in the current working
            directory (the project root when the CLI is started there).

    Returns:
        The key/value pairs that were read from the file (including ones that
        were skipped because the variable was already set).

    Notes:
        - Blank lines and ``#`` comments are ignored.
        - An optional leading ``export `` is stripped.
        - Surrounding single or double quotes are removed from values.
        - Values are taken verbatim (no ``${VAR}`` expansion); everything after
          the first ``=`` belongs to the value, so URLs with ``=`` are safe.
    """
    env_path = Path(path) if path is not None else Path(ENV_FILE)
    if env_path.is_dir() or not env_path.is_file():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(EXPORT_PREFIX):
            line = line[len(EXPORT_PREFIX) :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded
