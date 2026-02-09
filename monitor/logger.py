"""
Structured logging with dual output:
  - stderr: human-readable, ANSI-colored, column-aligned console output
  - file (optional): machine-readable single-line JSON (ndjson)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time


# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"

_LEVEL_STYLES = {
    "DEBUG": (_DIM, "DBG"),
    "INFO": (_CYAN, "INF"),
    "WARNING": (_YELLOW, "WRN"),
    "ERROR": (_RED, "ERR"),
    "CRITICAL": (_RED + _BOLD, "CRT"),
}


class ConsoleFormatter(logging.Formatter):
    """Human-readable log lines with timestamps and color-coded levels."""

    def __init__(self, use_color: bool = True):
        super().__init__()
        self._use_color = use_color and _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        color, tag = _LEVEL_STYLES.get(record.levelname, (_WHITE, "???"))
        msg = record.getMessage()

        if self._use_color:
            line = f"{_DIM}{ts}{_RESET} {color}{tag}{_RESET} {msg}"
        else:
            line = f"{ts} {tag} {msg}"

        if record.exc_info and record.exc_info[1]:
            if self._use_color:
                line += f"\n{_RED}     {record.exc_info[1]}{_RESET}"
            else:
                line += f"\n     {record.exc_info[1]}"

        return line


class JSONFormatter(logging.Formatter):
    """Single-line JSON for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "module": record.module,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])
        return json.dumps(entry, separators=(",", ":"))


def setup_logging(level: str = "INFO", json_log_file: str | None = None) -> None:
    """
    Configure root logger.
      - Always: human-readable ConsoleFormatter on stderr
      - Optionally: JSON file handler for machine logs
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console handler (human-readable)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    # Optional JSON file handler
    if json_log_file:
        fh = logging.FileHandler(json_log_file, mode="a")
        fh.setFormatter(JSONFormatter())
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("py_clob_client").setLevel(logging.WARNING)


def _supports_color() -> bool:
    """Check if stderr supports ANSI color."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
