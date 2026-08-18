from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from telegram.ext import Application

from bot.user_state import _load_state, get_deadline_chat_ids_for
from src.database.db import get_connection


class DeadlineScheduler:
    def __init__(self, application: Application):
        self.application = application
        self.scheduler: AsyncIOScheduler | None = None
        self._notified_keys: set[str] = set()

    def start(self) -> None:
        if self.scheduler is not None:
            return

        loop = asyncio.get_running_loop()
        self.scheduler = AsyncIOScheduler(event_loop=loop, timezone="UTC")
        self.scheduler.add_job(
            self.check_deadlines,
            trigger=IntervalTrigger(minutes=30),
            id="deadline-check",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self.nightly_data_refresh,
            trigger=CronTrigger(hour=1, minute=0, timezone="UTC"),
            id="nightly-refresh",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self.check_and_send_gw_recap,
            trigger=CronTrigger(hour=9, minute=0, timezone="UTC"),
            id="gw-recap",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.start()
        loop.create_task(self.check_deadlines())
        logger.info("Scheduler started — data/model refresh 01:00, GW recap 09:00 UTC")

    async def nightly_data_refresh(self) -> None:
        logger.info("Nightly refresh job triggered")
        try:
            from bot.data_helpers import nightly_refresh
            await nightly_refresh()
        except Exception:
            logger.exception("Nightly refresh job failed")

        try:
            from bot.deadline_monitor import check_deadline_changes

            # Check for deadline changes (e.g., double gameweeks)
            result = await check_deadline_changes(self.application)
            if result.get("changed_gws"):
                logger.info("Deadline monitor: {} GW(s) changed", len(result["changed_gws"]))
        except Exception:
            logger.exception("Deadline monitor job failed")

    async def check_and_send_gw_recap(self) -> None:
        """Runs at 09:00 UTC daily. Sends GW recap to all users if a new GW just finished."""
        from bot import cache
        from bot.recap import fetch_gw_recap, format_gw_recap, get_last_finished_gw

        last_gw = get_last_finished_gw()
        if last_gw is None:
            return

        last_sent = cache.read("last_recap_gw", max_age_hours=24 * 365)
        if last_sent and int(last_sent) >= last_gw:
            logger.debug("GW recap: already sent for GW{}", last_gw)
            return

        logger.info("Sending GW{} recap to all users", last_gw)
        state = _load_state()
        sent_count = 0

        for user_data in state.values():
            fpl_id = user_data.get("fpl_id")
            chat_id = user_data.get("chat_id")
            if not fpl_id or not chat_id:
                continue
            try:
                data = await fetch_gw_recap(int(fpl_id), last_gw)
                if data is None:
                    continue
                text = format_gw_recap(data)
                await self.application.bot.send_message(chat_id=chat_id, text=text)
                sent_count += 1
            except Exception:
                logger.exception("Failed to send GW recap to chat {}", chat_id)

        if sent_count > 0:
            cache.write("last_recap_gw", last_gw)
            logger.info("GW{} recap sent to {} user(s)", last_gw, sent_count)

    async def check_deadlines(self) -> None:
        con = get_connection(read_only=True)
        try:
            rows = con.execute(
                """
                SELECT id, name, deadline_time
                FROM gameweeks
                WHERE deadline_time IS NOT NULL
                  AND deadline_time > CURRENT_TIMESTAMP
                ORDER BY deadline_time ASC
                """
            ).fetchall()
        finally:
            con.close()

        if not rows:
            return

        now = datetime.now(timezone.utc)
        reminder_windows = (
            ("deadline_reminder_2h", 115, 125, "2 hours"),
            ("deadline_reminder", 55, 65, "1 hour"),
        )
        for gw_id, _gw_name, deadline_time in rows:
            if deadline_time is None:
                continue
            if getattr(deadline_time, "tzinfo", None) is None:
                deadline_time = deadline_time.replace(tzinfo=timezone.utc)
            else:
                deadline_time = deadline_time.astimezone(timezone.utc)
            minutes_until_deadline = (deadline_time - now).total_seconds() / 60.0
            for preference_key, lower, upper, reminder_label in reminder_windows:
                notification_key = f"{preference_key}:{gw_id}:{deadline_time.isoformat()}"
                if notification_key in self._notified_keys:
                    continue
                if lower <= minutes_until_deadline <= upper:
                    message = (
                        f"⚽ FPL Deadline in {reminder_label}! GW{gw_id} deadline is at "
                        f"{deadline_time.astimezone().strftime('%Y-%m-%d %H:%M %Z')}. Make your transfers now!"
                    )
                    for chat_id in get_deadline_chat_ids_for(preference_key):
                        try:
                            await self.application.bot.send_message(chat_id=chat_id, text=message)
                        except Exception:
                            logger.exception("Failed to send deadline reminder to chat {}", chat_id)
                    self._notified_keys.add(notification_key)

    def stop(self) -> None:
        if self.scheduler is None:
            return
        self.scheduler.shutdown(wait=False)
        self.scheduler = None
        logger.info("Deadline scheduler stopped")
