"""Centralized logging configuration."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False
_STANDARD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root logger. Idempotent - safe to call multiple times."""
    global _CONFIGURED

    root = logging.getLogger()
    root.setLevel(level)

    if _CONFIGURED:
        for handler in root.handlers:
            handler.setLevel(level)
            handler.setFormatter(_make_formatter(json_logs))
        return

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(_make_formatter(json_logs))
    root.addHandler(handler)
    _CONFIGURED = True


def _make_formatter(json_logs: bool) -> logging.Formatter:
    if json_logs:
        return _JsonFormatter()
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
