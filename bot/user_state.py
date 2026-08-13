from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from loguru import logger

from config.settings import ROOT_DIR
from src.database.db import get_connection

STATE_PATH = ROOT_DIR / "data" / "user_state.json"
_LOCK = Lock()


def _ensure_state_file() -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        STATE_PATH.write_text("{}", encoding="utf-8")


def _load_state() -> dict[str, dict[str, Any]]:
    _ensure_state_file()
    try:
        raw = STATE_PATH.read_text(encoding="utf-8").strip() or "{}"
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Invalid user state JSON found at {}. Resetting file.", STATE_PATH)
        STATE_PATH.write_text("{}", encoding="utf-8")
        return {}


def _save_state(state: dict[str, dict[str, Any]]) -> None:
    _ensure_state_file()
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _default_user_state(user_id: int) -> dict[str, Any]:
    return {
        "user_id": int(user_id),
        "chat_id": None,
        "fpl_id": None,
        "deadline_reminder": False,
        "deadline_reminder_2h": False,
    }


def _sync_user_to_db(user_id: int, state: dict[str, Any]) -> None:
    try:
        con = get_connection()
        con.execute("DELETE FROM telegram_users WHERE user_id = ?", [int(user_id)])
        con.execute(
            """
            INSERT INTO telegram_users (
                user_id, chat_id, fpl_manager_id, deadline_reminder, deadline_reminder_2h
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                int(user_id),
                state.get("chat_id"),
                state.get("fpl_id"),
                bool(state.get("deadline_reminder", False)),
                bool(state.get("deadline_reminder_2h", False)),
            ],
        )
        con.close()
    except Exception:
        logger.exception("Failed to sync Telegram user state for {}", user_id)


def get_user_state(user_id: int) -> dict[str, Any]:
    with _LOCK:
        state = _load_state()
        return dict(state.get(str(user_id), _default_user_state(user_id)))


def set_user_state(user_id: int, key: str, value: Any) -> None:
    with _LOCK:
        state = _load_state()
        user_key = str(user_id)
        user_state = dict(state.get(user_key, _default_user_state(user_id)))
        user_state[key] = value
        state[user_key] = user_state
        _save_state(state)
        _sync_user_to_db(user_id, user_state)


def get_fpl_id(user_id: int) -> Optional[int]:
    state = get_user_state(user_id)
    value = state.get("fpl_id")
    return int(value) if value not in (None, "") else None


def set_fpl_id(user_id: int, fpl_id: int) -> None:
    set_user_state(user_id, "fpl_id", int(fpl_id))


def set_chat_id(user_id: int, chat_id: int) -> None:
    set_user_state(user_id, "chat_id", int(chat_id))


def get_deadline_chat_ids() -> list[int]:
    return get_deadline_chat_ids_for("deadline_reminder")


def get_deadline_chat_ids_for(preference_key: str) -> list[int]:
    with _LOCK:
        state = _load_state()
        chat_ids = []
        for user_state in state.values():
            if user_state.get(preference_key) and user_state.get("chat_id") is not None:
                chat_ids.append(int(user_state["chat_id"]))
        return sorted(set(chat_ids))


def set_deadline_reminder(user_id: int, chat_id: int, enabled: bool) -> None:
    set_deadline_reminder_for(user_id, chat_id, "deadline_reminder", enabled)


def set_deadline_reminder_for(
    user_id: int,
    chat_id: int,
    preference_key: str,
    enabled: bool,
) -> None:
    with _LOCK:
        state = _load_state()
        user_key = str(user_id)
        user_state = dict(state.get(user_key, _default_user_state(user_id)))
        user_state["chat_id"] = int(chat_id)
        user_state[preference_key] = bool(enabled)
        state[user_key] = user_state
        _save_state(state)
        _sync_user_to_db(user_id, user_state)
