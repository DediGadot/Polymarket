"""
Integration tests for monitor/logger.py -- structured logging.
"""

import json
import logging
import tempfile
import os

from monitor.logger import JSONFormatter, ConsoleFormatter, setup_logging


class TestJSONFormatter:
    def test_format_basic_message(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Hello %s", args=("world",), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["msg"] == "Hello world"
        assert "ts" in parsed

    def test_format_with_exception(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="test.py",
                lineno=1, msg="Failed", args=(), exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "ERROR"
        assert "test error" in parsed["exception"]


class TestConsoleFormatter:
    def test_format_basic_message(self):
        formatter = ConsoleFormatter(use_color=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="Hello %s", args=("world",), exc_info=None,
        )
        output = formatter.format(record)
        assert "INF" in output
        assert "Hello world" in output

    def test_format_warning(self):
        formatter = ConsoleFormatter(use_color=False)
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="test.py",
            lineno=1, msg="Watch out", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "WRN" in output
        assert "Watch out" in output


class TestSetupLogging:
    def test_sets_level(self):
        setup_logging("DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

        setup_logging("WARNING")
        assert root.level == logging.WARNING

    def test_console_handler_attached_by_default(self):
        setup_logging("INFO")
        root = logging.getLogger()
        assert any(
            isinstance(h.formatter, ConsoleFormatter)
            for h in root.handlers
        )

    def test_json_handler_attached_when_file_specified(self):
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            path = f.name
        try:
            setup_logging("INFO", json_log_file=path)
            root = logging.getLogger()
            assert any(
                isinstance(h.formatter, JSONFormatter)
                for h in root.handlers
            )
            assert any(
                isinstance(h.formatter, ConsoleFormatter)
                for h in root.handlers
            )
        finally:
            # Clean up: close file handlers and remove file
            for h in logging.getLogger().handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
            os.unlink(path)

    def test_no_json_handler_without_file(self):
        setup_logging("INFO")
        root = logging.getLogger()
        assert not any(
            isinstance(h.formatter, JSONFormatter)
            for h in root.handlers
        )
