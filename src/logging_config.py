"""Logging setup shared by every entrypoint (API, pipeline, DAG, scripts)."""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger to write to stdout."""
    from src.config import get_settings

    resolved = level or get_settings().log_level

    # The Windows console defaults to cp1252 and breaks on accents.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=resolved.upper(),
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
