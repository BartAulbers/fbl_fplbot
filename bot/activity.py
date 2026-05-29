"""
Activity logger — writes a structured log line for every user interaction.

Log file: logs/activity.log  (rotated daily, kept 30 days)
Format:   TIMESTAMP | user_id=X | fpl_id=X | action=X
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config.settings import ROOT_DIR

LOGS_DIR = Path(ROOT_DIR) / "logs"
_configured = False


def configure_logging() -> None:
    """Call once at startup to add file sinks alongside the default stderr sink."""
    global _configured
    if _configured:
        return
    _configured = True

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # General bot log — everything INFO and above, rotated daily
    logger.add(
        LOGS_DIR / "bot.log",
        level="INFO",
        rotation="00:00",       # new file each day at midnight
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        encoding="utf-8",
    )

    # Activity-only log — tab-separated for easy parsing/import
    logger.add(
        LOGS_DIR / "activity.log",
        level="INFO",
        rotation="00:00",
        retention="90 days",
        format="{time:YYYY-MM-DD HH:mm:ss}\t{message}",
        encoding="utf-8",
        filter=lambda record: record["extra"].get("activity") is True,
    )


def log_activity(user_id: int, action: str, fpl_id: int | None = None) -> None:
    """Log a single user action. Reads fpl_id from user_state if not supplied."""
    if fpl_id is None:
        try:
            from bot.user_state import get_fpl_id
            fpl_id = get_fpl_id(user_id)
        except Exception:
            pass

    fpl_part = f"fpl_id={fpl_id}" if fpl_id else "fpl_id=none"
    logger.bind(activity=True).info(
        f"user_id={user_id}\t{fpl_part}\taction={action}"
    )
