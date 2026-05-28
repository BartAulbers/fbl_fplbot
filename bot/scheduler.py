from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
from telegram.ext import Application

from bot.user_state import get_deadline_chat_ids
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
            trigger=CronTrigger(hour=7, minute=0, timezone="UTC"),
            id="nightly-refresh",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.start()
        loop.create_task(self.check_deadlines())
        logger.info("Deadline scheduler started (morning refresh at 07:00 UTC)")

    async def nightly_data_refresh(self) -> None:
        logger.info("Nightly refresh job triggered")
        try:
            from bot.data_helpers import nightly_refresh
            await nightly_refresh()
        except Exception:
            logger.exception("Nightly refresh job failed")

    async def check_deadlines(self) -> None:
        con = get_connection(read_only=True)
        try:
            row = con.execute(
                """
                SELECT id, name, deadline_time
                FROM gameweeks
                WHERE deadline_time IS NOT NULL
                  AND deadline_time > CURRENT_TIMESTAMP
                ORDER BY deadline_time ASC
                LIMIT 1
                """
            ).fetchone()
        finally:
            con.close()

        if not row:
            return

        gw_id, _gw_name, deadline_time = row
        if deadline_time is None:
            return

        if getattr(deadline_time, "tzinfo", None) is None:
            deadline_time = deadline_time.replace(tzinfo=timezone.utc)
        else:
            deadline_time = deadline_time.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
        minutes_until_deadline = (deadline_time - now).total_seconds() / 60.0
        notification_key = f"{gw_id}:{deadline_time.isoformat()}"

        if notification_key in self._notified_keys:
            return

        if 55 <= minutes_until_deadline <= 65:
            message = (
                f"⚽ FPL Deadline in 1 hour! GW{gw_id} deadline is at "
                f"{deadline_time.astimezone().strftime('%Y-%m-%d %H:%M %Z')}. Make your transfers now!"
            )
            for chat_id in get_deadline_chat_ids():
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
