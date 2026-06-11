"""Centralised loguru configuration for the escape-rooms pipeline.

Import `logger` from this module everywhere:

    from src.escape_rooms.utils.logging import logger

`setup_logging()` is called once at startup (main.py / FastAPI lifespan).
Subsequent calls are no-ops (idempotent guard via _configured flag).

Log levels used across the codebase:
  TRACE   — per-object detail, LLM prompt/response snippets, inner-loop steps
  DEBUG   — sub-step internals: repairs, retries, individual object mutations
  INFO    — one-line milestone per node phase (start, end, key counts)
  SUCCESS — node completed successfully (loguru built-in, green)
  WARNING — non-fatal anomalies: remaining issues, fallbacks triggered
  ERROR   — hard failures: missing world, oracle crash, pipeline exception
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_configured = False

LOG_DIR = Path("logs")

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[node]: <16}</cyan> | "
    "<level>{message}</level>"
)

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{extra[node]: <16} | "
    "{name}:{function}:{line} | "
    "{message}"
)


def setup_logging(
    log_dir: Path | None = None,
    console_level: str = "DEBUG",
    file_level: str = "TRACE",
) -> None:
    """Configure loguru sinks. Safe to call multiple times — only runs once."""
    global _configured
    if _configured:
        return
    _configured = True

    # Remove the default loguru sink before adding our own.
    logger.remove()

    # Console sink — coloured, human-readable.
    logger.add(
        sys.stderr,
        level=console_level,
        format=_CONSOLE_FORMAT,
        colorize=True,
    )

    # File sink — full TRACE detail, one rotating file per session.
    root = log_dir or LOG_DIR
    root.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(root / "escape_rooms_{time:YYYYMMDD_HHmmss}.log"),
        level=file_level,
        format=_FILE_FORMAT,
        rotation="50 MB",
        retention=5,
        encoding="utf-8",
    )

    logger.configure(extra={"node": "main"})
    logger.debug("Logging initialised — console={} file={}", console_level, file_level)


def get_node_logger(node: str):
    """Return a logger bound to a specific node name shown in every line."""
    return logger.bind(node=node)
