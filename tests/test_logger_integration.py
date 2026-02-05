"""
Integration tests for monitor/logger.py -- structured logging.
"""

import json
import logging
from io import StringIO

from monitor.logger import JSONFormatter, setup_logging


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


class TestSetupLogging:
    def test_sets_level(self):
        setup_logging("DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

        setup_logging("WARNING")
        assert root.level == logging.WARNING

    def test_json_handler_attached(self):
        setup_logging("INFO")
        root = logging.getLogger()
        assert any(
            isinstance(h.formatter, JSONFormatter)
            for h in root.handlers
        )
