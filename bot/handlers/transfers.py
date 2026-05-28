from __future__ import annotations

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackQueryHandler, CommandHandler, ContextTypes, ConversationHandler

from bot.data_helpers import (
    get_current_gw,
    is_season_over,
    load_all_players_with_enrichment,
    load_popular_missing_players,
    load_squad_with_enrichment,
    refresh_fpl_data,
    squad_exists,
)
from bot.formatters import format_popular_missing, format_transfer_suggestions
from bot.handlers.menu import get_main_menu_markup, show_main_menu
from bot.utils import ensure_user_allowed, safe_delete_message, send_text_chunks
from config.settings import settings
from src.strategy.transfer_engine import recommend_transfers

WAITING_FREE_TRANSFERS = 0


async def transfers_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    if update.callback_query:
        await update.callback_query.answer()

    if is_season_over():
        target = update.callback_query.message if update.callback_query else update.effective_message
        await target.reply_text(
            "⏸ The season has ended. Transfer suggestions will return when the new season starts.",
            reply_markup=get_main_menu_markup(),
        )
        return ConversationHandler.END

    if not squad_exists(update.effective_user.id):
        markup = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("📋 Import Team", callback_data="menu:import")],
                [InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")],
            ]
        )
        target = update.callback_query.message if update.callback_query else update.effective_message
        await target.reply_text("Import your squad first before requesting transfer suggestions.", reply_markup=markup)
        return ConversationHandler.END

    markup = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("0 FT", callback_data="transfers:ft:0"),
                InlineKeyboardButton("1 FT", callback_data="transfers:ft:1"),
                InlineKeyboardButton("2 FT", callback_data="transfers:ft:2"),
            ],
            [
                InlineKeyboardButton("3 FT", callback_data="transfers:ft:3"),
                InlineKeyboardButton("4 FT", callback_data="transfers:ft:4"),
                InlineKeyboardButton("5 FT", callback_data="transfers:ft:5"),
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data="nav:main")],
        ]
    )
    target = update.callback_query.message if update.callback_query else update.effective_message
    await target.reply_text("How many free transfers do you have available?", reply_markup=markup)
    return WAITING_FREE_TRANSFERS


async def free_transfers_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    query = update.callback_query
    await query.answer()

    try:
        free_transfers = int(query.data.rsplit(":", 1)[-1])
        from bot import cache
        if not cache.is_fresh("fpl_data_refreshed"):
            loading_message = await query.message.reply_text("⏳ Fetching latest FPL data...")
            await refresh_fpl_data(include_predictions=True)
            await safe_delete_message(loading_message)

        my_squad_df = load_squad_with_enrichment(update.effective_user.id)
        all_players_df = load_all_players_with_enrichment()
        if my_squad_df is None or my_squad_df.empty:
            await query.message.reply_text(
                "No squad data is available after refresh. Please import your team again.",
                reply_markup=get_main_menu_markup(),
            )
            return ConversationHandler.END

        current_gw = get_current_gw()
        plan = recommend_transfers(
            my_squad_df=my_squad_df,
            all_players_df=all_players_df,
            free_transfers=free_transfers,
            current_gw=current_gw,
            risk_appetite=settings.risk_appetite,
            max_suggestions=5,
        )
        await send_text_chunks(
            context,
            update.effective_chat.id,
            format_transfer_suggestions(plan),
            reply_markup=get_main_menu_markup(),
        )
        return ConversationHandler.END
    except Exception:
        logger.exception("Transfer suggestions failed")
        await query.message.reply_text(
            "Sorry, I couldn't generate transfer suggestions right now.",
            reply_markup=get_main_menu_markup(),
        )
        return ConversationHandler.END


async def cancel_transfers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await show_main_menu(update, context, text="Transfer suggestions cancelled.")
    return ConversationHandler.END


async def missing_players_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_user_allowed(update, context):
        return
    query = update.callback_query
    await query.answer()

    if not squad_exists(update.effective_user.id):
        await query.message.reply_text(
            "Import your squad first so I can show what popular players you're missing.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📋 Import Team", callback_data="team:import")]]),
        )
        return

    loading = await query.message.reply_text("⏳ Loading popular players...")
    players = load_popular_missing_players(update.effective_user.id)
    await safe_delete_message(loading)
    text = format_popular_missing(players)
    await send_text_chunks(context, update.effective_chat.id, text, reply_markup=get_main_menu_markup())


def get_missing_players_handler():
    from telegram.ext import CallbackQueryHandler as CQH
    return CQH(missing_players_callback, pattern="^transfers:missing$")


def get_transfers_handler() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[
            CallbackQueryHandler(transfers_entry, pattern="^transfers:suggestions$"),
            CommandHandler("transfers", transfers_entry),
        ],
        states={
            WAITING_FREE_TRANSFERS: [
                CallbackQueryHandler(free_transfers_selected, pattern=r"^transfers:ft:[0-5]$")
            ],
        },
        fallbacks=[
            CallbackQueryHandler(cancel_transfers, pattern="^nav:main$"),
            CommandHandler("cancel", cancel_transfers),
        ],
        conversation_timeout=120,
        per_message=False,
        per_chat=True,
        per_user=True,
    )
