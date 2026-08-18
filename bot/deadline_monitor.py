"""
Deadline monitor — detects changes in gameweek deadlines (e.g., double gameweeks).
Notifies all subscribed users when deadlines shift.
"""
from __future__ import annotations

from datetime import datetime, timezone
from loguru import logger
from telegram.ext import Application

from src.api.fpl_client import FPLClient
from src.database.db import get_connection
from bot.user_state import get_deadline_chat_ids_for


async def check_deadline_changes(application: Application | None = None) -> dict:
    """
    Fetches latest gameweeks from FPL API and compares with database.
    Returns dict with 'changed_gws' (list of changed GW IDs) and 'changes' (detailed info).
    Notifies all users subscribed to deadline reminders if changes are found.
    """
    changes = []
    changed_gw_ids = []

    try:
        async with FPLClient() as client:
            bootstrap = await client.get_bootstrap()
            new_gws = {gw["id"]: gw for gw in bootstrap["events"]}
    except Exception as e:
        logger.exception("Failed to fetch gameweek data from FPL API: {}", e)
        return {"changed_gws": [], "changes": [], "error": str(e)}

    try:
        con = get_connection()
    except Exception:
        logger.exception("Deadline monitor: could not open DB connection (skipping this cycle)")
        return {"changed_gws": [], "changes": [], "error": "db_unavailable"}

    try:
        # Fetch current gameweeks from database
        old_rows = con.execute(
            "SELECT id, name, deadline_time FROM gameweeks ORDER BY id"
        ).fetchall()
        old_gws = {row[0]: {"name": row[1], "deadline_time": row[2]} for row in old_rows}

        # Compare deadlines
        for gw_id, new_gw in new_gws.items():
            old_gw = old_gws.get(gw_id)
            if not old_gw:
                continue

            old_deadline = old_gw["deadline_time"]
            new_deadline = new_gw.get("deadline_time")

            # Normalize timestamps for comparison
            if old_deadline and new_deadline:
                if getattr(old_deadline, "tzinfo", None) is None:
                    old_deadline = old_deadline.replace(tzinfo=timezone.utc)
                else:
                    old_deadline = old_deadline.astimezone(timezone.utc)

                if getattr(new_deadline, "tzinfo", None) is None:
                    new_deadline_dt = datetime.fromisoformat(new_deadline.replace("Z", "+00:00"))
                else:
                    new_deadline_dt = datetime.fromisoformat(new_deadline)

                if old_deadline != new_deadline_dt:
                    changed_gw_ids.append(gw_id)
                    changes.append({
                        "gw_id": gw_id,
                        "gw_name": new_gw["name"],
                        "old_deadline": old_deadline.isoformat() if old_deadline else None,
                        "new_deadline": new_deadline_dt.isoformat(),
                        "is_double": "DGW" in new_gw["name"],
                    })
                    logger.info(
                        "GW{} deadline changed: {} -> {}",
                        gw_id,
                        old_deadline,
                        new_deadline_dt,
                    )

        if changes:
            logger.warning("Detected {} deadline change(s)", len(changes))

            # Notify all subscribed users
            if application:
                await _notify_users_of_deadline_changes(application, changes)

    finally:
        con.close()

    return {
        "changed_gws": changed_gw_ids,
        "changes": changes,
    }


async def _notify_users_of_deadline_changes(
    application: Application, changes: list[dict]
) -> None:
    """Sends notification to all users subscribed to deadline reminders."""
    if not changes:
        return

    # Notify users who have either reminder enabled
    chat_ids = set()
    chat_ids.update(get_deadline_chat_ids_for("deadline_reminder"))
    chat_ids.update(get_deadline_chat_ids_for("deadline_reminder_2h"))

    if not chat_ids:
        logger.debug("No users subscribed to deadline notifications")
        return

    message = "⚠️ FPL Deadline Changes Detected!\n\n"
    for change in changes:
        old_time = change["old_deadline"]
        new_time = change["new_deadline"]
        dgw_label = " (Double Gameweek)" if change["is_double"] else ""

        # Parse ISO format for readable output
        try:
            old_dt = datetime.fromisoformat(old_time)
            new_dt = datetime.fromisoformat(new_time)
            old_str = old_dt.strftime("%Y-%m-%d %H:%M %Z")
            new_str = new_dt.strftime("%Y-%m-%d %H:%M %Z")
        except:
            old_str = old_time
            new_str = new_time

        message += (
            f"🕐 {change['gw_name']}{dgw_label}\n"
            f"   Old: {old_str}\n"
            f"   New: {new_str}\n\n"
        )

    message += "Make sure to check your squad for any deadlines you might have missed!"

    sent_count = 0
    for chat_id in chat_ids:
        try:
            await application.bot.send_message(chat_id=chat_id, text=message)
            sent_count += 1
        except Exception:
            logger.exception("Failed to notify chat {} of deadline changes", chat_id)

    if sent_count > 0:
        logger.info("Notified {} user(s) of deadline changes", sent_count)
