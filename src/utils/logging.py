"""
Logging Module
===============

This module provides a structured logging system with features including:
    - Colored console output for different log levels
    - File rotation and retention policies
    - Structured JSON logging for production
    - Context-aware logging with request IDs
    - Performance timing decorators

Example Usage:
    >>> from src.utils.logging import setup_logging, get_logger
    >>> setup_logging(level="DEBUG", log_file="logs/app.log")
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing video", video_id="vid_001", frames=300)

Architecture:
    Uses Python's standard logging module with custom formatters and handlers.
    Supports both human-readable and structured JSON output formats.
"""

from __future__ import annotations

import logging
import sys
import time
import json
import functools
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union
from contextlib import contextmanager

# Try to import colorama for Windows color support
try:
    import colorama

    colorama.init()
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False


# =============================================================================
# Color Definitions
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


# Level to color mapping
LEVEL_COLORS = {
    logging.DEBUG: Colors.CYAN,
    logging.INFO: Colors.GREEN,
    logging.WARNING: Colors.YELLOW,
    logging.ERROR: Colors.RED,
    logging.CRITICAL: Colors.BG_RED + Colors.WHITE,
}


# =============================================================================
# Custom Formatters
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Logging formatter that adds color to console output.

    This formatter colorizes log levels and adds visual structure to make
    logs easier to scan in development environments.

    Attributes:
        use_colors: Whether to apply color formatting.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ) -> None:
        """
        Initialize the colored formatter.

        Args:
            fmt: Log format string.
            datefmt: Date format string.
            use_colors: Whether to use colored output.
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and COLORS_AVAILABLE

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with colors.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string with color codes.
        """
        # Save original values
        original_msg = record.msg
        original_levelname = record.levelname

        if self.use_colors:
            # Colorize level name
            color = LEVEL_COLORS.get(record.levelno, Colors.WHITE)
            record.levelname = f"{color}{record.levelname:8s}{Colors.RESET}"

            # Colorize logger name
            record.name = f"{Colors.BLUE}{record.name}{Colors.RESET}"

        result = super().format(record)

        # Restore original values for other handlers
        record.msg = original_msg
        record.levelname = original_levelname

        return result


class JSONFormatter(logging.Formatter):
    """
    Logging formatter that outputs structured JSON.

    This formatter is ideal for production environments where logs need
    to be parsed by log aggregation systems.

    Example output:
        {"timestamp": "2024-01-15T10:30:00", "level": "INFO", "message": "..."}
    """

    def __init__(
        self,
        include_extra: bool = True,
        timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%f",
    ) -> None:
        """
        Initialize the JSON formatter.

        Args:
            include_extra: Include extra fields from log record.
            timestamp_format: Format string for timestamps.
        """
        super().__init__()
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).strftime(
                self.timestamp_format
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if self.include_extra:
            # Get extra fields (those not in standard LogRecord)
            standard_fields = {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            }
            extras = {
                k: v for k, v in record.__dict__.items()
                if k not in standard_fields and not k.startswith("_")
            }
            if extras:
                log_data["extra"] = extras

        return json.dumps(log_data, default=str)


class ContextFormatter(logging.Formatter):
    """
    Formatter that includes context information like request IDs.

    This formatter is useful for tracing requests across microservices.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format record with context information."""
        # Add context from thread-local storage if available
        context = getattr(record, "context", {})
        if context:
            context_str = " ".join(f"{k}={v}" for k, v in context.items())
            record.msg = f"[{context_str}] {record.msg}"
        return super().format(record)


# =============================================================================
# Custom Handlers
# =============================================================================

class RotatingFileHandler(logging.Handler):
    """
    File handler with size-based rotation.

    This handler rotates log files when they exceed a maximum size,
    keeping a configurable number of backup files.

    Attributes:
        filename: Path to the log file.
        max_bytes: Maximum file size before rotation.
        backup_count: Number of backup files to keep.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        encoding: str = "utf-8",
    ) -> None:
        """
        Initialize the rotating file handler.

        Args:
            filename: Path to the log file.
            max_bytes: Maximum file size in bytes before rotation.
            backup_count: Number of backup files to keep.
            encoding: File encoding.
        """
        super().__init__()
        self.filename = Path(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding

        # Ensure directory exists
        self.filename.parent.mkdir(parents=True, exist_ok=True)

        # Open the file
        self._stream = open(self.filename, "a", encoding=encoding)

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record, rotating if necessary.

        Args:
            record: The log record to emit.
        """
        try:
            msg = self.format(record)

            # Check if rotation is needed
            if self._should_rotate():
                self._do_rotation()

            self._stream.write(msg + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def _should_rotate(self) -> bool:
        """Check if the log file should be rotated."""
        if self.max_bytes <= 0:
            return False
        try:
            return self.filename.stat().st_size >= self.max_bytes
        except OSError:
            return False

    def _do_rotation(self) -> None:
        """Perform log rotation."""
        self._stream.close()

        # Rotate existing backup files
        for i in range(self.backup_count - 1, 0, -1):
            src = self.filename.with_suffix(f".{i}.log")
            dst = self.filename.with_suffix(f".{i + 1}.log")
            if src.exists():
                src.rename(dst)

        # Rename current file to .1.log
        dst = self.filename.with_suffix(".1.log")
        if self.filename.exists():
            self.filename.rename(dst)

        # Remove oldest backup if exceeds count
        oldest = self.filename.with_suffix(f".{self.backup_count + 1}.log")
        if oldest.exists():
            oldest.unlink()

        # Open new file
        self._stream = open(self.filename, "a", encoding=self.encoding)

    def close(self) -> None:
        """Close the file handler."""
        self._stream.close()
        super().close()


# =============================================================================
# Logger Setup Functions
# =============================================================================

def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    json_format: bool = False,
    colorize: bool = True,
    include_timestamp: bool = True,
) -> None:
    """
    Set up the logging system.

    This function configures the root logger with appropriate handlers
    and formatters based on the provided options.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        json_format: Use JSON formatting (for production).
        colorize: Use colored console output.
        include_timestamp: Include timestamps in log messages.

    Example:
        >>> setup_logging(level="DEBUG", log_file="logs/app.log")
        >>> setup_logging(level="INFO", json_format=True)  # Production
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Define format
    if include_timestamp:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
    else:
        log_format = "%(levelname)-8s | %(name)s | %(message)s"
        date_format = None

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            ColoredFormatter(log_format, date_format, use_colors=colorize)
        )

    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            max_bytes=10 * 1024 * 1024,  # 10 MB
            backup_count=5,
        )
        file_handler.setLevel(level)

        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(log_format, date_format)
            )

        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is the primary interface for obtaining loggers throughout
    the application.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


# =============================================================================
# Logging Utilities
# =============================================================================

@contextmanager
def log_context(**context: Any):
    """
    Context manager that adds context to all log messages.

    Args:
        **context: Key-value pairs to add to log context.

    Example:
        >>> with log_context(request_id="req_123", user_id="user_456"):
        ...     logger.info("Processing request")
        # Output: [request_id=req_123 user_id=user_456] Processing request
    """
    # Store context in thread-local storage
    import threading

    local = threading.local()
    old_context = getattr(local, "log_context", {})
    local.log_context = {**old_context, **context}

    # Add context to log records
    old_factory = logging.getLogRecordFactory()

    def new_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.context = getattr(local, "log_context", {})
        return record

    logging.setLogRecordFactory(new_factory)

    try:
        yield
    finally:
        local.log_context = old_context
        logging.setLogRecordFactory(old_factory)


def log_execution_time(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    message: str = "Execution time: {elapsed:.3f}s",
) -> Callable:
    """
    Decorator that logs function execution time.

    Args:
        logger: Logger to use (defaults to function's module logger).
        level: Log level for timing messages.
        message: Message format (must include {elapsed}).

    Returns:
        Decorator function.

    Example:
        >>> @log_execution_time()
        ... def process_video(path):
        ...     # ... processing ...
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.log(
                    level,
                    f"{func.__name__}: {message.format(elapsed=elapsed)}"
                )
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.log(
                    logging.ERROR,
                    f"{func.__name__} failed after {elapsed:.3f}s: {e}"
                )
                raise

        return wrapper
    return decorator


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds structured context to all messages.

    This adapter allows adding consistent context to all log messages
    from a particular component.

    Example:
        >>> base_logger = get_logger(__name__)
        >>> logger = LoggerAdapter(base_logger, {"service": "video_processor"})
        >>> logger.info("Processing video", video_id="vid_001")
    """

    def process(
        self, msg: str, kwargs: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Process the logging message and keyword arguments."""
        # Merge extra context
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra

        return msg, kwargs


# =============================================================================
# Performance Logging
# =============================================================================

class PerformanceLogger:
    """
    Utility class for logging performance metrics.

    Provides methods for tracking and logging execution times,
    throughput, and other performance metrics.

    Example:
        >>> perf = PerformanceLogger(get_logger(__name__))
        >>> with perf.timer("video_processing"):
        ...     process_video(path)
        >>> perf.log_summary()
    """

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize the performance logger.

        Args:
            logger: Logger instance for output.
        """
        self.logger = logger
        self.timings: Dict[str, list[float]] = {}
        self.counters: Dict[str, int] = {}

    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing operations.

        Args:
            name: Name of the operation being timed.

        Example:
            >>> with perf.timer("face_detection"):
            ...     detect_faces(frame)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
            self.logger.debug(f"{name}: {elapsed:.4f}s")

    def increment(self, name: str, count: int = 1) -> None:
        """
        Increment a counter.

        Args:
            name: Counter name.
            count: Amount to increment.
        """
        self.counters[name] = self.counters.get(name, 0) + count

    def log_summary(self) -> None:
        """Log a summary of all recorded metrics."""
        self.logger.info("=== Performance Summary ===")

        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            total = sum(times)
            self.logger.info(
                f"{name}: avg={avg:.4f}s, total={total:.4f}s, count={len(times)}"
            )

        for name, count in self.counters.items():
            self.logger.info(f"{name}: {count}")

    def reset(self) -> None:
        """Reset all metrics."""
        self.timings.clear()
        self.counters.clear()
