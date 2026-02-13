"""
Structured logging with triple output:
  - stderr: human-readable, ANSI-colored, column-aligned console output
  - file (always): verbose debug log at logs/run_YYYYMMDD_HHMMSS.log
  - file (optional): machine-readable single-line JSON (ndjson)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone


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


def setup_logging(level: str = "INFO", json_log_file: str | None = None) -> str:
    """
    Configure root logger.
      - Always: human-readable ConsoleFormatter on stderr
      - Always: verbose debug log file at logs/run_YYYYMMDD_HHMMSS.log
      - Optionally: JSON file handler for machine logs

    Returns the path to the verbose log file.
    """
    root = logging.getLogger()
    # Root must be DEBUG so the file handler captures everything
    root.setLevel(logging.DEBUG)

    # Clear existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Console handler (human-readable, respects configured level)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    # Always: verbose debug log file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")

    verbose_fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-8s %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    verbose_handler = logging.FileHandler(log_path, mode="a")
    verbose_handler.setLevel(logging.DEBUG)
    verbose_handler.setFormatter(verbose_fmt)
    root.addHandler(verbose_handler)

    # Optional JSON file handler
    if json_log_file:
        fh = logging.FileHandler(json_log_file, mode="a")
        fh.setFormatter(JSONFormatter())
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("py_clob_client").setLevel(logging.WARNING)

    return log_path


def _supports_color() -> bool:
    """Check if stderr supports ANSI color."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
