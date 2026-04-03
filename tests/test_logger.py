"""Tests for kvswitch.utils.logger."""

import logging
from io import StringIO
from unittest.mock import patch

from kvswitch.utils.logger import LOG_FORMAT, ColoredFormatter, setup_logging


class TestSetupLogging:
    def _reset(self) -> None:
        """Remove handlers from configured application loggers between tests."""
        for name in ("kvswitch", "__main__"):
            logger = logging.getLogger(name)
            logger.handlers.clear()
            logger.setLevel(logging.WARNING)
            logger.propagate = True

    def test_creates_handler(self) -> None:
        self._reset()
        setup_logging(logging.DEBUG)
        root = logging.getLogger("kvswitch")
        assert len(root.handlers) == 1
        assert root.level == logging.DEBUG

    def test_idempotent(self) -> None:
        self._reset()
        setup_logging(logging.INFO)
        setup_logging(logging.DEBUG)
        root = logging.getLogger("kvswitch")
        # Should not add a second handler.
        assert len(root.handlers) == 1
        # Level should be updated.
        assert root.level == logging.DEBUG
        assert root.handlers[0].level == logging.DEBUG

    def test_accepts_string_level(self) -> None:
        self._reset()
        setup_logging("warning")
        root = logging.getLogger("kvswitch")
        assert root.level == logging.WARNING

    def test_child_logger_inherits(self) -> None:
        self._reset()
        setup_logging(logging.DEBUG)
        child = logging.getLogger("kvswitch.vllm.profiling")
        assert child.getEffectiveLevel() == logging.DEBUG

    def test_main_logger_is_configured_for_module_entrypoints(self) -> None:
        self._reset()
        fake_stderr = StringIO()
        fake_stderr.isatty = lambda: False  # type: ignore[attr-defined]
        with patch("kvswitch.utils.logger.sys.stderr", fake_stderr):
            setup_logging(logging.INFO)
            logging.getLogger("__main__").info("cli message")

        main_logger = logging.getLogger("__main__")
        assert len(main_logger.handlers) == 1
        assert "cli message" in fake_stderr.getvalue()

    def test_uses_colored_formatter_when_tty(self) -> None:
        self._reset()
        fake_stderr = StringIO()
        fake_stderr.isatty = lambda: True  # type: ignore[attr-defined]
        with patch("kvswitch.utils.logger.sys.stderr", fake_stderr):
            setup_logging(logging.INFO)
        root = logging.getLogger("kvswitch")
        assert isinstance(root.handlers[0].formatter, ColoredFormatter)

    def test_uses_plain_formatter_when_not_tty(self) -> None:
        self._reset()
        fake_stderr = StringIO()
        fake_stderr.isatty = lambda: False  # type: ignore[attr-defined]
        with patch("kvswitch.utils.logger.sys.stderr", fake_stderr):
            setup_logging(logging.INFO)
        root = logging.getLogger("kvswitch")
        assert not isinstance(root.handlers[0].formatter, ColoredFormatter)
        assert LOG_FORMAT in root.handlers[0].formatter._fmt  # type: ignore[union-attr]


class TestColoredFormatter:
    def test_adds_ansi_to_levelname(self) -> None:
        fmt = ColoredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        assert "\033[32m" in output  # Green for INFO
        assert "\033[0m" in output  # Reset
        assert "hello" in output
        # Original record should be restored.
        assert record.levelname == "INFO"

    def test_grey_timestamp(self) -> None:
        fmt = ColoredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="warn",
            args=(),
            exc_info=None,
        )
        output = fmt.format(record)
        # Timestamp should be wrapped in grey.
        assert "\033[90m" in output

    def test_all_levels_colored(self) -> None:
        fmt = ColoredFormatter()
        for level_name, color in ColoredFormatter.LEVEL_COLORS.items():
            level = getattr(logging, level_name)
            record = logging.LogRecord(
                name="t",
                level=level,
                pathname="",
                lineno=0,
                msg="m",
                args=(),
                exc_info=None,
            )
            output = fmt.format(record)
            assert color in output
