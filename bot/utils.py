from __future__ import annotations

from typing import Iterable, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from config.settings import settings

MAX_MESSAGE_LENGTH = 4000


def parse_allowed_user_ids() -> set[int]:
    raw = settings.telegram_allowed_user_ids.strip()
    if not raw:
        return set()
    return {
        int(part.strip())
        for part in raw.split(",")
        if part.strip()
    }


def is_user_allowed(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    allowed = parse_allowed_user_ids()
    return not allowed or int(user_id) in allowed


async def ensure_user_allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id if update.effective_user else None
    if is_user_allowed(user_id):
        return True

    text = "This bot is private. Ask the owner to add your Telegram user ID."
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.reply_text(text)
    elif update.effective_message:
        await update.effective_message.reply_text(text)
    return False


def main_menu_button() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Main Menu", callback_data="nav:main")]])


def cancel_button() -> list[list[InlineKeyboardButton]]:
    return [[InlineKeyboardButton("❌ Cancel", callback_data="nav:main")]]


async def safe_delete_message(message) -> None:
    if message is None:
        return
    try:
        await message.delete()
    except Exception:
        return


async def edit_or_reply(
    query,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    """
    Edit the message that triggered the callback in place.
    Falls back to reply_text if the message cannot be edited
    (e.g. it was sent by another bot, or is too old).
    """
    try:
        await query.edit_message_text(text=text, reply_markup=reply_markup)
    except Exception:
        await query.message.reply_text(text=text, reply_markup=reply_markup)


async def edit_or_send_chunks(
    query,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    """
    For content responses: if text fits in one message, edit in place.
    If it needs multiple chunks, delete the original and send fresh chunks.
    """
    if len(text) <= MAX_MESSAGE_LENGTH:
        try:
            await query.edit_message_text(text=text, reply_markup=reply_markup)
            return
        except Exception:
            pass

    # Multi-chunk: delete old message and send fresh
    await safe_delete_message(query.message)
    await send_text_chunks(context, chat_id, text, reply_markup=reply_markup)


async def send_text_chunks(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    text: str,
    reply_markup: Optional[InlineKeyboardMarkup] = None,
) -> None:
    if len(text) <= MAX_MESSAGE_LENGTH:
        await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
        return

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > MAX_MESSAGE_LENGTH:
            if current:
                chunks.append(current.rstrip())
                current = ""
        if len(line) > MAX_MESSAGE_LENGTH:
            start = 0
            while start < len(line):
                chunks.append(line[start:start + MAX_MESSAGE_LENGTH].rstrip())
                start += MAX_MESSAGE_LENGTH
            continue
        current += line
    if current:
        chunks.append(current.rstrip())

    for index, chunk in enumerate(chunks):
        await context.bot.send_message(
            chat_id=chat_id,
            text=chunk or " ",
            reply_markup=reply_markup if index == len(chunks) - 1 else None,
        )
