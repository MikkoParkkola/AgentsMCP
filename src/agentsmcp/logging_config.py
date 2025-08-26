"""Structured logging configuration for AgentsMCP.

Provides JSON or text logging with contextual fields. Integrates with Uvicorn
and FastAPI. Configure via environment variables handled by AppSettings.
"""

from __future__ import annotations

import logging

from pythonjsonlogger import jsonlogger


def _build_json_formatter() -> logging.Formatter:
    fields = [
        "asctime",
        "levelname",
        "name",
        "message",
        "module",
        "funcName",
        "lineno",
        "process",
        "thread",
    ]
    fmt = " ".join([f"{f}=%({f})s" for f in fields])
    return jsonlogger.JsonFormatter(fmt=fmt)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    level_upper = level.upper()
    log_level = getattr(logging, level_upper, logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove default handlers to avoid duplicate logs
    for h in list(root.handlers):
        root.removeHandler(h)

    # Also clear handlers from existing named loggers to avoid double emission
    # when other modules configured logging earlier.
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.handlers = []
            logger.propagate = True

    handler = logging.StreamHandler()
    if fmt.lower() == "json":
        handler.setFormatter(_build_json_formatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    root.addHandler(handler)

    # Align Uvicorn/uvicorn.access loggers to same handler/level
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]
        logger.setLevel(log_level)
