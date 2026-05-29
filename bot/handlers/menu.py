from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from bot.user_state import get_user_state, set_chat_id, set_deadline_reminder
from bot.utils import edit_or_reply, ensure_user_allowed


def get_main_menu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("⚽ Team", callback_data="menu:team"),
                InlineKeyboardButton("🔄 Transfers", callback_data="menu:transfers"),
                InlineKeyboardButton("📅 Fixtures", callback_data="menu:fixtures"),
            ],
            [InlineKeyboardButton("🔔 Deadline Reminder", callback_data="menu:deadline")],
        ]
    )


def get_team_submenu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📋 Import Team", callback_data="team:import")],
            [InlineKeyboardButton("👥 My Squad & News", callback_data="team:squad_news")],
            [InlineKeyboardButton("🏆 Team of the GW", callback_data="team:totgw")],
            [InlineKeyboardButton("❌ Back", callback_data="nav:main")],
        ]
    )


def get_transfers_submenu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔄 Transfer Suggestions", callback_data="transfers:suggestions")],
            [InlineKeyboardButton("👑 Popular Players I'm Missing", callback_data="transfers:missing")],
            [InlineKeyboardButton("ℹ️ How xPts works", callback_data="transfers:xpts_info")],
            [InlineKeyboardButton("❌ Back", callback_data="nav:main")],
        ]
    )


def get_fixtures_submenu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Top 5 Easiest Fixtures", callback_data="fixtures:easiest")],
            [InlineKeyboardButton("📈 Fixture Swings", callback_data="fixtures:swings")],
            [InlineKeyboardButton("❌ Back", callback_data="nav:main")],
        ]
    )


def build_welcome_text(user_id: int) -> str:
    state = get_user_state(user_id)
    lines = [
        "⚽ Welcome to the FBL Telegram Bot.",
        "Choose an option below to import your squad, review transfers, check news, or track fixture swings.",
    ]
    if state.get("fpl_id"):
        lines.append(f"Stored FPL Manager ID: {state['fpl_id']}")
    lines.append(
        f"Deadline reminders: {'ON' if state.get('deadline_reminder') else 'OFF'}"
    )
    return "\n\n".join(lines)


async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str | None = None) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    set_chat_id(user_id, chat_id)
    await context.bot.send_message(
        chat_id=chat_id,
        text=text or build_welcome_text(user_id),
        reply_markup=get_main_menu_markup(),
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_user_allowed(update, context):
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    set_chat_id(user_id, chat_id)
    from bot.activity import log_activity
    log_activity(user_id, "start")
    await update.effective_message.reply_text(
        build_welcome_text(user_id),
        reply_markup=get_main_menu_markup(),
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await ensure_user_allowed(update, context):
        return

    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    set_chat_id(user_id, chat_id)
    data = query.data
    from bot.activity import log_activity
    log_activity(user_id, data)

    if data == "nav:main":
        await edit_or_reply(query, build_welcome_text(user_id), reply_markup=get_main_menu_markup())
        return

    if data in ("menu:team", "menu:import"):
        await edit_or_reply(query, "⚽ Team", reply_markup=get_team_submenu_markup())
        return

    if data == "menu:transfers":
        await edit_or_reply(query, "🔄 Transfers", reply_markup=get_transfers_submenu_markup())
        return

    if data == "menu:fixtures":
        await edit_or_reply(query, "📅 Fixtures", reply_markup=get_fixtures_submenu_markup())
        return

    if data == "menu:deadline":
        state = get_user_state(user_id)
        status = "enabled" if state.get("deadline_reminder") else "disabled"
        markup = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Enable", callback_data="deadline:on"),
                    InlineKeyboardButton("🚫 Disable", callback_data="deadline:off"),
                ],
                [InlineKeyboardButton("❌ Back", callback_data="nav:main")],
            ]
        )
        await edit_or_reply(query, f"Deadline reminders are currently {status}. Choose an option:", reply_markup=markup)
        return

    if data == "deadline:on":
        set_deadline_reminder(user_id, chat_id, True)
        await edit_or_reply(query, "🔔 Deadline reminders enabled. You'll get a message about 1 hour before each FPL deadline.", reply_markup=get_main_menu_markup())
        return

    if data == "deadline:off":
        set_deadline_reminder(user_id, chat_id, False)
        await edit_or_reply(query, "🔕 Deadline reminders disabled.", reply_markup=get_main_menu_markup())
        return

    if data == "transfers:xpts_info":
        text = (
            "🤖 HOW EXPECTED POINTS (xPts) WORKS\n\n"
            "An XGBoost model trained on all 38 gameweeks of historical data. "
            "It predicts how many points a player will score using 68 features:\n\n"
            "📈 Recent form (exp. weighted — last GWs count more)\n"
            "  · Points, minutes, goals, assists, bonus, BPS\n"
            "  · Expected goal involvements (xG + xA)\n\n"
            "🛡 Defensive form (position-adjusted)\n"
            "  · Clean sheets × FPL multiplier (GK/DEF=4, MID=1, FWD=0)\n"
            "  · Goals conceded penalty (GK/DEF only)\n"
            "  · Saves, penalty saves, xGC\n\n"
            "⚠️ Deduction risk\n"
            "  · Yellow/red cards, own goals, penalties missed\n\n"
            "📊 Season-level stats\n"
            "  · ICT index, xG, xA, xGI, xGC, value, ownership %\n\n"
            "📐 Per-90 derived stats\n"
            "  · xGI/90, saves/90, xGC/90, pts per £m\n\n"
            "🗓 Fixture difficulty\n"
            "  · FDR next GW + avg FDR over 3 and 5 GWs\n\n"
            "↔️ Home/away split\n"
            "  · Avg pts at home vs away\n\n"
            "⚡ Consistency  ·  🏟 Team strength  ·  🟡 Availability\n\n"
            "Model accuracy: CV MAE ≈ 0.97 pts (trained on 27,231 samples)"
        )
        await edit_or_reply(query, text, reply_markup=get_transfers_submenu_markup())
