from __future__ import annotations

import asyncio

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from bot.data_helpers import database_has_player_data, get_my_squad_player_ids, is_season_over, load_news_for_my_squad, load_team_of_gw, refresh_fpl_data, squad_exists
from bot.formatters import format_player_news, format_squad, format_team_of_gw
from bot.handlers.menu import get_main_menu_markup, show_main_menu
from bot.user_state import get_fpl_id, set_chat_id, set_fpl_id
from bot.utils import ensure_user_allowed, safe_delete_message, send_text_chunks
from src.data.squad_importer import import_squad_from_fpl

WAITING_MANAGER_ID = 0


async def _ask_for_manager_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    markup = InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="nav:main")]])
    if update.callback_query:
        await update.callback_query.message.reply_text(
            "Enter your FPL Manager ID (the number from the URL on the FPL website).",
            reply_markup=markup,
        )
    else:
        await update.effective_message.reply_text(
            "Enter your FPL Manager ID (the number from the URL on the FPL website).",
            reply_markup=markup,
        )
    return WAITING_MANAGER_ID


async def team_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    set_chat_id(user_id, chat_id)

    if update.callback_query:
        await update.callback_query.answer()

    stored_id = get_fpl_id(user_id)
    if stored_id:
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(f"✅ Keep {stored_id}", callback_data="team:use_stored"),
                    InlineKeyboardButton("✏️ Change", callback_data="team:change"),
                ],
                [InlineKeyboardButton("❌ Cancel", callback_data="nav:main")],
            ]
        )
        target = update.callback_query.message if update.callback_query else update.effective_message
        await target.reply_text(
            f"Your saved FPL Manager ID is {stored_id}. Keep it or enter a new one?",
            reply_markup=markup,
        )
        return WAITING_MANAGER_ID

    return await _ask_for_manager_id(update, context)


async def team_option_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id

    if query.data == "team:change":
        return await _ask_for_manager_id(update, context)

    if query.data == "team:use_stored":
        manager_id = get_fpl_id(user_id)
        if manager_id is None:
            return await _ask_for_manager_id(update, context)
        return await _run_import(update, context, manager_id)

    return WAITING_MANAGER_ID


async def manager_id_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    raw_value = (update.effective_message.text or "").strip()
    if not raw_value.isdigit():
        await update.effective_message.reply_text(
            "Please send a numeric FPL Manager ID only.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="nav:main")]]),
        )
        return WAITING_MANAGER_ID

    return await _run_import(update, context, int(raw_value))


async def _run_import(update: Update, context: ContextTypes.DEFAULT_TYPE, manager_id: int) -> int:
    loading_message = None
    try:
        if not database_has_player_data():
            loading_message = await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⏳ Fetching latest FPL data first...",
            )
            await refresh_fpl_data(include_predictions=False)
            await safe_delete_message(loading_message)
            loading_message = None

        loading_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="⏳ Importing your FPL squad...",
        )
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, import_squad_from_fpl, int(manager_id), None, update.effective_user.id)
        await safe_delete_message(loading_message)

        if not result.success:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Could not import your squad. {result.message}",
                reply_markup=get_main_menu_markup(),
            )
            return ConversationHandler.END

        set_fpl_id(update.effective_user.id, int(manager_id))
        summary = f"✅ {result.message}\n\n{format_squad(result.players)}"
        await send_text_chunks(
            context,
            update.effective_chat.id,
            summary,
            reply_markup=get_main_menu_markup(),
        )
        return ConversationHandler.END
    except Exception:
        await safe_delete_message(loading_message)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Sorry, something went wrong while importing your team.",
            reply_markup=get_main_menu_markup(),
        )
        return ConversationHandler.END


async def cancel_team(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await show_main_menu(update, context, text="Operation cancelled.")
    return ConversationHandler.END


async def squad_news_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_user_allowed(update, context):
        return
    query = update.callback_query
    await query.answer()
    from bot.activity import log_activity
    log_activity(update.effective_user.id, "team:squad_news")

    if not squad_exists(update.effective_user.id):
        await query.message.reply_text(
            "No squad imported yet. Use Import Team first.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📋 Import Team", callback_data="team:import")]]),
        )
        return

    players = load_news_for_my_squad(update.effective_user.id)
    text = format_player_news(players)
    await send_text_chunks(context, update.effective_chat.id, text, reply_markup=get_main_menu_markup())


def get_squad_news_handler():
    from telegram.ext import CallbackQueryHandler as CQH
    return CQH(squad_news_callback, pattern="^team:squad_news$")


async def team_of_gw_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_user_allowed(update, context):
        return
    query = update.callback_query
    await query.answer()
    from bot.activity import log_activity
    log_activity(update.effective_user.id, "team:totgw")

    if is_season_over():
        await query.message.reply_text(
            "⏸ The season has ended. Team of the GW will return next season.",
            reply_markup=get_main_menu_markup(),
        )
        return

    from bot import cache
    loading_message = None
    try:
        if not cache.is_fresh("fpl_data_refreshed"):
            loading_message = await query.message.reply_text("⏳ Fetching latest FPL data...")
            await refresh_fpl_data(include_predictions=True)
            await safe_delete_message(loading_message)
            loading_message = None

        result = load_team_of_gw()
        owned_ids = get_my_squad_player_ids(update.effective_user.id)
        await send_text_chunks(
            context, update.effective_chat.id,
            format_team_of_gw(result, owned_ids=owned_ids or None),
            reply_markup=get_main_menu_markup(),
        )
    except Exception:
        from loguru import logger
        logger.exception("Team of GW failed")
        await safe_delete_message(loading_message)
        await query.message.reply_text(
            "Sorry, couldn't build the Team of the GW right now.",
            reply_markup=get_main_menu_markup(),
        )


def get_team_of_gw_handler():
    from telegram.ext import CallbackQueryHandler as CQH
    return CQH(team_of_gw_callback, pattern="^team:totgw$")


def get_team_handler() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[
            CallbackQueryHandler(team_entry, pattern="^team:import$"),
            CommandHandler("import", team_entry),
        ],
        states={
            WAITING_MANAGER_ID: [
                CallbackQueryHandler(team_option_callback, pattern="^team:(use_stored|change)$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, manager_id_received),
            ],
        },
        fallbacks=[
            CallbackQueryHandler(cancel_team, pattern="^nav:main$"),
            CommandHandler("cancel", cancel_team),
        ],
        conversation_timeout=120,
        per_message=False,
        per_chat=True,
        per_user=True,
    )
