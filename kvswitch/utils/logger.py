"""Centralized logging configuration for KVSwitch."""

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Adds ANSI colour codes to log output for terminal readability.

    Inspired by ``vllm.logging_utils.formatter.ColoredFormatter``.
    """

    LEVEL_COLORS = {
        "DEBUG": "\033[37m",  # White
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    GREY = "\033[90m"
    RESET = "\033[0m"

    def __init__(
        self,
        fmt: str = LOG_FORMAT,
        datefmt: str = DATE_FORMAT,
    ) -> None:
        # Wrap the timestamp and logger name in grey.
        colored_fmt = fmt.replace("%(asctime)s", f"{self.GREY}%(asctime)s{self.RESET}")
        colored_fmt = colored_fmt.replace(
            "%(name)s", f"{self.GREY}%(name)s{self.RESET}"
        )
        super().__init__(colored_fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        orig_levelname = record.levelname
        color = self.LEVEL_COLORS.get(record.levelname)
        if color is not None:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        msg = super().format(record)
        record.levelname = orig_levelname
        return msg


def _build_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
        handler.setFormatter(ColoredFormatter())
    else:
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    return handler


def _configure_named_logger(name: str, level: int) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        # Already configured (e.g. called twice) — just update handler levels.
        for handler in logger.handlers:
            handler.setLevel(level)
        return

    logger.addHandler(_build_handler(level))


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure KVSwitch application logging.

    All loggers under the ``kvswitch.*`` namespace inherit this config.

    Parameters
    ----------
    level:
        Logging level — an ``int`` (e.g. ``logging.DEBUG``) or a
        case-insensitive string (e.g. ``"debug"``).
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
        assert isinstance(level, int)

    for logger_name in ("kvswitch", "__main__"):
        _configure_named_logger(logger_name, level)
