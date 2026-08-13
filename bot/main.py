from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from telegram import Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.activity import configure_logging, log_activity
from bot.handlers.fixtures import (
    get_easiest_fixtures_handler,
    get_fixture_swings_handler,
    get_fixtures_command_handler,
)
from bot.handlers.menu import button_callback, start_command
from bot.handlers.team import get_squad_news_handler, get_team_handler, get_team_of_gw_handler
from bot.handlers.transfers import get_missing_players_handler, get_transfers_handler
from bot.scheduler import DeadlineScheduler
from config.settings import settings
from src.database.db import init_db


def main() -> None:
    configure_logging()
    init_db()

    scheduler_holder: dict[str, DeadlineScheduler] = {}

    async def post_init(application: Application) -> None:
        scheduler = DeadlineScheduler(application)
        scheduler.start()
        scheduler_holder["scheduler"] = scheduler

    async def post_shutdown(application: Application) -> None:
        scheduler = scheduler_holder.get("scheduler")
        if scheduler:
            scheduler.stop()

    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.exception("Unhandled bot error")
        telegram_update = update if isinstance(update, Update) else None
        if telegram_update and telegram_update.effective_user:
            log_activity(telegram_update.effective_user.id, "error")
        chat = telegram_update.effective_chat if telegram_update else None
        if chat:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Sorry, something unexpected went wrong. Please try again.",
            )

    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not configured. Add it to your .env file.")

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # ConversationHandlers first (highest priority)
    app.add_handler(get_team_handler())
    app.add_handler(get_transfers_handler())

    # Simple callback handlers
    app.add_handler(get_squad_news_handler())
    app.add_handler(get_team_of_gw_handler())
    app.add_handler(get_missing_players_handler())
    app.add_handler(get_easiest_fixtures_handler())
    app.add_handler(get_fixture_swings_handler())
    app.add_handler(get_fixtures_command_handler())

    # Menu navigation (sub-menus + deadline)
    app.add_handler(CallbackQueryHandler(
        button_callback,
        pattern="^(nav:main|menu:team|menu:transfers|menu:fixtures|menu:deadline|menu:import|deadline:(on|off|2h:(on|off))|transfers:xpts_info)$",
    ))

    # Commands
    app.add_handler(CommandHandler("start", start_command))

    app.add_error_handler(error_handler)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

    scheduler_holder: dict[str, DeadlineScheduler] = {}

    async def post_init(application: Application) -> None:
        scheduler = DeadlineScheduler(application)
        scheduler.start()
        scheduler_holder["scheduler"] = scheduler

    async def post_shutdown(application: Application) -> None:
        scheduler = scheduler_holder.get("scheduler")
        if scheduler:
            scheduler.stop()

    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.exception("Unhandled bot error")
        telegram_update = update if isinstance(update, Update) else None
        chat = telegram_update.effective_chat if telegram_update else None
        if chat:
            await context.bot.send_message(
                chat_id=chat.id,
                text="Sorry, something unexpected went wrong. Please try again.",
            )

    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not configured. Add it to your .env file.")

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    # ConversationHandlers first (highest priority)
    app.add_handler(get_team_handler())
    app.add_handler(get_transfers_handler())

    # Simple callback handlers
    app.add_handler(get_squad_news_handler())
    app.add_handler(get_team_of_gw_handler())
    app.add_handler(get_missing_players_handler())
    app.add_handler(get_easiest_fixtures_handler())
    app.add_handler(get_fixture_swings_handler())
    app.add_handler(get_fixtures_command_handler())

    # Menu navigation (sub-menus + deadline)
    app.add_handler(CallbackQueryHandler(
        button_callback,
        pattern="^(nav:main|menu:team|menu:transfers|menu:fixtures|menu:deadline|menu:import|deadline:(on|off|2h:(on|off))|transfers:xpts_info)$",
    ))

    # Commands
    app.add_handler(CommandHandler("start", start_command))

    app.add_error_handler(error_handler)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
