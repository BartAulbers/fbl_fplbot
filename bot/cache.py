"""
Simple file-based cache for pre-computed FPL data.
Stores JSON in data/cache/ with a timestamp so handlers can skip
expensive live refreshes when data is already fresh.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import ROOT_DIR

CACHE_DIR = Path(ROOT_DIR) / "data" / "cache"
_STALE_HOURS = 20  # data older than this is considered stale


def _path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def write(key: str, data: Any) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    _path(key).write_text(json.dumps(payload, default=str), encoding="utf-8")


def read(key: str, max_age_hours: int = _STALE_HOURS) -> Any | None:
    p = _path(key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(payload["updated_at"])
        age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None
        return payload["data"]
    except Exception:
        return None


def is_fresh(key: str, max_age_hours: int = _STALE_HOURS) -> bool:
    return read(key, max_age_hours) is not None


def last_updated(key: str) -> datetime | None:
    p = _path(key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return datetime.fromisoformat(payload["updated_at"])
    except Exception:
        return None
