"""
datatrust/logger.py

Logging setup: console + rotating log file.

Why not print()?
- Logs carry timestamps, severity and module name, can be filtered by level,
  and are written to logs/datatrust.log for later debugging.
- RotatingFileHandler caps disk usage (5 files x 2 MB).
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from datatrust.config import DEFAULT_SETTINGS, Settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_ROOT_LOGGER_NAME = "datatrust"
_configured = False


def _configure_root(settings: Settings) -> None:
    """Attach handlers to the package root logger exactly once."""
    global _configured
    if _configured:
        return

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(settings.log_level)
    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.log_dir / "datatrust.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # e.g. read-only filesystem: keep console logging only
        root.warning("File logging disabled: %s", exc)

    root.propagate = False
    _configured = True


def get_logger(name: str, settings: Settings = DEFAULT_SETTINGS) -> logging.Logger:
    """
    Return a logger under the 'datatrust' namespace.

    Usage:  logger = get_logger(__name__)
    """
    _configure_root(settings)
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
