from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from bot.data_helpers import load_news_for_my_squad, load_player_news, search_players_by_name, squad_exists
from bot.formatters import format_player_news, format_squad_for_news
from bot.handlers.menu import get_main_menu_markup, show_main_menu
from bot.utils import ensure_user_allowed, send_text_chunks

WAITING_NEWS_MODE = 0
WAITING_SEARCH_QUERY = 1
WAITING_PLAYER_SELECTION = 2


async def news_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    if update.callback_query:
        await update.callback_query.answer()

    markup = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🧾 My Squad", callback_data="news:mode:squad"),
                InlineKeyboardButton("🔎 Search Player", callback_data="news:mode:search"),
            ],
            [InlineKeyboardButton("❌ Cancel", callback_data="nav:main")],
        ]
    )
    target = update.callback_query.message if update.callback_query else update.effective_message
    await target.reply_text("Choose how you'd like to view player news.", reply_markup=markup)
    return WAITING_NEWS_MODE


async def news_mode_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    query = update.callback_query
    await query.answer()

    if query.data == "news:mode:squad":
        if not squad_exists():
            await query.message.reply_text(
                "Import your squad first to see news for your team.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("📋 Import Team", callback_data="menu:import")],
                        [InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")],
                    ]
                ),
            )
            return ConversationHandler.END

        players = load_news_for_my_squad()
        await send_text_chunks(
            context,
            update.effective_chat.id,
            format_player_news(players),
            reply_markup=get_main_menu_markup(),
        )
        return ConversationHandler.END

    if query.data == "news:mode:search":
        await query.message.reply_text(
            "Type the player name you want to search for.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="nav:main")]]),
        )
        return WAITING_SEARCH_QUERY

    return WAITING_NEWS_MODE


async def search_query_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    query_text = (update.effective_message.text or "").strip()
    matches = search_players_by_name(query_text, limit=5)
    if not matches:
        await update.effective_message.reply_text(
            "No matching players found. Try another name.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("❌ Cancel", callback_data="nav:main")]]),
        )
        return WAITING_SEARCH_QUERY

    buttons = [
        [InlineKeyboardButton(player["web_name"], callback_data=f"news:player:{player['player_id']}")]
        for player in matches
    ]
    buttons.append([InlineKeyboardButton("❌ Cancel", callback_data="nav:main")])
    text = "Top matches:\n\n" + format_squad_for_news(matches)
    await update.effective_message.reply_text(text, reply_markup=InlineKeyboardMarkup(buttons))
    return WAITING_PLAYER_SELECTION


async def player_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not await ensure_user_allowed(update, context):
        return ConversationHandler.END

    query = update.callback_query
    await query.answer()

    try:
        player_id = int(query.data.rsplit(":", 1)[-1])
        player_news = load_player_news(player_id)
        await send_text_chunks(
            context,
            update.effective_chat.id,
            format_player_news(player_news),
            reply_markup=get_main_menu_markup(),
        )
        return ConversationHandler.END
    except Exception:
        await query.message.reply_text(
            "Sorry, I couldn't load player news right now.",
            reply_markup=get_main_menu_markup(),
        )
        return ConversationHandler.END


async def cancel_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await show_main_menu(update, context, text="Player news cancelled.")
    return ConversationHandler.END


def get_news_handler() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[
            CallbackQueryHandler(news_entry, pattern="^menu:news$"),
            CommandHandler("news", news_entry),
        ],
        states={
            WAITING_NEWS_MODE: [
                CallbackQueryHandler(news_mode_selected, pattern=r"^news:mode:(squad|search)$")
            ],
            WAITING_SEARCH_QUERY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, search_query_received)
            ],
            WAITING_PLAYER_SELECTION: [
                CallbackQueryHandler(player_selected, pattern=r"^news:player:\d+$")
            ],
        },
        fallbacks=[
            CallbackQueryHandler(cancel_news, pattern="^nav:main$"),
            CommandHandler("cancel", cancel_news),
        ],
        conversation_timeout=120,
    )
