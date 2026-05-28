from __future__ import annotations

from loguru import logger
from telegram import Update
from telegram.ext import CallbackQueryHandler, CommandHandler, ContextTypes

from bot.data_helpers import is_season_over, load_easiest_fixtures, load_fixture_swings, refresh_fpl_data
from bot.formatters import format_easiest_fixtures, format_fixture_swings
from bot.handlers.menu import get_main_menu_markup
from bot.utils import ensure_user_allowed, safe_delete_message, send_text_chunks


async def fixture_swings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_user_allowed(update, context):
        return
    query = update.callback_query
    if query:
        await query.answer()

    loading_message = None
    try:
        if is_season_over():
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⏸ The season has ended. Fixture swings will be available once the new season starts.",
                reply_markup=get_main_menu_markup(),
            )
            return
        from bot import cache
        if not cache.is_fresh("fpl_data_refreshed"):
            loading_message = await context.bot.send_message(
                chat_id=update.effective_chat.id, text="⏳ Fetching latest FPL data..."
            )
            await refresh_fpl_data(include_predictions=False)
            await safe_delete_message(loading_message)
            loading_message = None
        alerts = load_fixture_swings()
        await send_text_chunks(
            context, update.effective_chat.id,
            format_fixture_swings(alerts),
            reply_markup=get_main_menu_markup(),
        )
    except Exception:
        logger.exception("Fixture swings failed")
        await safe_delete_message(loading_message)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Sorry, couldn't load fixture swings right now.",
            reply_markup=get_main_menu_markup(),
        )


async def easiest_fixtures_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_user_allowed(update, context):
        return
    query = update.callback_query
    if query:
        await query.answer()

    loading_message = None
    try:
        if is_season_over():
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⏸ The season has ended. Fixture data will be available once the new season starts.",
                reply_markup=get_main_menu_markup(),
            )
            return
        from bot import cache
        if not cache.is_fresh("fpl_data_refreshed"):
            loading_message = await context.bot.send_message(
                chat_id=update.effective_chat.id, text="⏳ Loading fixture data..."
            )
            await refresh_fpl_data(include_predictions=False)
            await safe_delete_message(loading_message)
            loading_message = None
        rows = load_easiest_fixtures(top_n=5)
        await send_text_chunks(
            context, update.effective_chat.id,
            format_easiest_fixtures(rows),
            reply_markup=get_main_menu_markup(),
        )
    except Exception:
        logger.exception("Easiest fixtures failed")
        await safe_delete_message(loading_message)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Sorry, couldn't load fixture data right now.",
            reply_markup=get_main_menu_markup(),
        )


# Keep old name as alias so main.py doesn't need patching for the command handler
fixtures_callback = fixture_swings_callback


def get_fixtures_command_handler() -> CommandHandler:
    return CommandHandler("fixtures", fixture_swings_callback)


def get_easiest_fixtures_handler() -> CallbackQueryHandler:
    return CallbackQueryHandler(easiest_fixtures_callback, pattern="^fixtures:easiest$")


def get_fixture_swings_handler() -> CallbackQueryHandler:
    return CallbackQueryHandler(fixture_swings_callback, pattern="^fixtures:swings$")
