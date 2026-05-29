from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.strategy.transfer_engine import TransferPlan

POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
POSITION_EMOJIS = {
    "GK": "🧤",
    "DEF": "🛡",
    "MID": "🎯",
    "FWD": "⚽",
}
POSITION_LABELS = {
    "GK": "GOALKEEPER",
    "DEF": "DEFENCE",
    "MID": "MIDFIELD",
    "FWD": "ATTACK",
}
STATUS_EMOJIS = {
    "a": "✅",
    "d": "⚠️",
    "i": "🤕",
    "s": "🚫",
    "u": "🚫",
}
SWING_EMOJIS = {
    "FIXTURE_EASES": "🟢",
    "FIXTURE_HARDENS": "🔴",
}


def _as_money(value: object) -> str:
    try:
        return f"£{float(value):.1f}m"
    except (TypeError, ValueError):
        return "£?"


def _availability_dot(player: dict) -> str:
    """Return 🟢 / 🟡 / 🔴 based on status and chance of playing."""
    status = str(player.get("status") or "a").lower()
    chance = player.get("chance_of_playing_next_round")

    if status in ("i", "s", "u"):
        return "🔴"
    if status == "d":
        if chance is not None:
            return "🟢" if int(chance) >= 75 else ("🟡" if int(chance) >= 25 else "🔴")
        return "🟡"
    # status == 'a'
    if chance is not None and int(chance) < 75:
        return "🟡"
    return "🟢"


def format_squad(players: list[dict]) -> str:
    """Clean pitch-layout used after team import (no availability data yet)."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in players:
        grouped[p.get("position", "UNK")].append(p)

    lines: list[str] = ["📋 YOUR SQUAD\n"]
    for pos in POSITION_ORDER:
        pos_players = grouped.get(pos, [])
        if not pos_players:
            continue
        emoji = POSITION_EMOJIS.get(pos, "•")
        label = POSITION_LABELS.get(pos, pos)
        lines.append(f"{emoji} {label}")
        for p in sorted(pos_players, key=lambda x: x.get("squad_position", 99)):
            markers = []
            if p.get("is_captain"):
                markers.append("(C)")
            if p.get("is_vice_captain"):
                markers.append("(VC)")
            suffix = " " + " ".join(markers) if markers else ""
            lines.append(f"  · {p.get('web_name', '?')} {_as_money(p.get('now_cost'))}{suffix}")
        lines.append("")
    return "\n".join(lines).strip()


def format_player_news(players: list[dict]) -> str:
    """
    Combined squad status + news view.
    Grouped by position with 🟢/🟡/🔴 dots, then a 'Concerns' section
    for anyone with news text or sub-75% availability.
    """
    if not players:
        return "No players found. Import your team first."

    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in players:
        grouped[p.get("position", "UNK")].append(p)

    lines: list[str] = ["⚽ SQUAD STATUS\n"]

    for pos in POSITION_ORDER:
        pos_players = grouped.get(pos, [])
        if not pos_players:
            continue
        emoji = POSITION_EMOJIS.get(pos, "•")
        label = POSITION_LABELS.get(pos, pos)
        lines.append(f"{emoji} {label}")
        for p in sorted(pos_players, key=lambda x: x.get("web_name", "")):
            dot = _availability_dot(p)
            mins = p.get("recent_minutes")
            mins_text = f"{int(mins)} min" if mins not in (None, "") else "—"
            cap = ""
            if p.get("is_captain"):
                cap = " (C)"
            elif p.get("is_vice_captain"):
                cap = " (VC)"
            lines.append(
                f"  {dot} {p.get('web_name', '?')}{cap} · {_as_money(p.get('now_cost'))} · {mins_text}"
            )
        lines.append("")

    # Concerns section — players with news or reduced availability
    concerns = [
        p for p in players
        if (p.get("news") or "").strip()
        or str(p.get("status") or "a").lower() != "a"
        or (p.get("chance_of_playing_next_round") is not None
            and int(p["chance_of_playing_next_round"]) < 100)
    ]

    if concerns:
        lines.append("📋 CONCERNS\n")
        for p in concerns:
            dot = _availability_dot(p)
            chance = p.get("chance_of_playing_next_round")
            chance_text = f"{int(chance)}% chance" if chance is not None else "chance unknown"
            news = (p.get("news") or "").strip() or "No details."
            lines.append(f"{dot} {p.get('web_name', '?')} — {chance_text}")
            lines.append(f"   {news}")
            lines.append("")
    else:
        lines.append("✅ All players available — no concerns.")

    return "\n".join(lines).strip()


def format_transfer_suggestions(plan: TransferPlan) -> str:
    lines = [f"🔄 TRANSFER SUGGESTIONS — GW{plan.current_gw}\n", plan.recommendation]
    if plan.suggestions:
        lines.append("")
    for i, s in enumerate(plan.suggestions, 1):
        hit_tag = " 🔴 -4 HIT" if s.hit_required else ""
        lines.extend([
            f"{i}.{hit_tag}",
            f"  OUT  {s.player_out['web_name']} {_as_money(s.player_out['cost'])} · xPts {s.player_out['xpts']:.1f}",
            f"  IN   {s.player_in['web_name']} {_as_money(s.player_in['cost'])} · xPts {s.player_in['xpts']:.1f}",
            f"  Gain +{s.expected_gain_1gw:.1f}pt (1GW)  +{s.expected_gain_3gw:.1f}pt (3GW)  {s.confidence.upper()} confidence",
            f"  {s.reasoning}",
            "",
        ])
    return "\n".join(lines).strip()


def format_fixture_swings(alerts: list[dict]) -> str:
    if not alerts:
        return "No major fixture swings detected right now."

    lines = ["📈 FIXTURE SWINGS\n"]
    for a in alerts:
        emoji = SWING_EMOJIS.get(a.get("alert_type"), "📅")
        delta = float(a.get("delta", 0))
        lines.append(f"{emoji} {a.get('team', '?')} — {a.get('message', '')}  (Δ {delta:+.2f})")
    return "\n".join(lines)


def format_easiest_fixtures(rows: list[dict]) -> str:
    if not rows:
        return "No fixture data available."

    lines = ["📊 TOP 5 EASIEST FIXTURE RUNS  (next 5 GWs)\n"]
    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, row in enumerate(rows):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        avg = row.get("avg_fdr", 0)
        team = row.get("team_name", "?")
        fixtures = row.get("fixtures", "")
        lines.append(f"{medal} {team}  avg FDR {avg:.2f}")
        lines.append(f"   {fixtures}")
        lines.append("")
    return "\n".join(lines).strip()


def format_popular_missing(players: list[dict]) -> str:
    if not players:
        return "No popular players outside your squad (or squad not imported)."

    lines = ["👑 POPULAR PLAYERS YOU DON'T OWN\n"]
    pos_order = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    players = sorted(
        players,
        key=lambda p: (pos_order.get(p.get("position", ""), 5), -float(p.get("selected_by_percent", 0))),
    )
    current_pos = None
    for p in players:
        pos = p.get("position", "?")
        if pos != current_pos:
            if current_pos is not None:
                lines.append("")
            lines.append(f"{POSITION_EMOJIS.get(pos, '•')} {POSITION_LABELS.get(pos, pos)}")
            current_pos = pos
        ownership = float(p.get("selected_by_percent", 0))
        xpts = float(p.get("xpts", 0))
        lines.append(
            f"  · {p.get('web_name', '?')} {_as_money(p.get('now_cost'))} "
            f"| {ownership:.1f}% owned | xPts {xpts:.1f}"
        )
    return "\n".join(lines)


def format_squad_for_news(players: list[dict]) -> str:
    if not players:
        return "No players available."
    return "\n".join(
        f"{i}. {p.get('web_name', '?')} ({p.get('position', '?')})"
        for i, p in enumerate(players, 1)
    )


def format_team_of_gw(result: dict | None, owned_ids: set[int] | None = None) -> str:
    if result is None:
        return (
            "⚠️ Not enough data to build a Team of the GW yet.\n\n"
            "Make sure the xPts model has been trained by importing your team or triggering a data refresh."
        )

    gw = result.get("gameweek", "?")
    formation = result.get("formation", "?")
    total = result.get("total_xpts", 0.0)

    lines = [
        f"🏆 TEAM OF GW{gw}",
        f"Formation: {formation} | Total xPts: {total:.1f}",
        "",
    ]

    totgw_ids: list[int] = []
    for pos in ("GK", "DEF", "MID", "FWD"):
        for p in result.get(pos, []):
            pid = p.get("player_id")
            if pid is not None:
                totgw_ids.append(int(pid))

    for pos in ("GK", "DEF", "MID", "FWD"):
        players = result.get(pos, [])
        if not players:
            continue
        label = POSITION_LABELS.get(pos, pos)
        emoji = POSITION_EMOJIS.get(pos, "•")
        lines.append(f"{emoji} {label}")
        for p in players:
            name = p.get("web_name", "?")
            team = p.get("team_short") or p.get("team_name", "?")
            xpts = float(p.get("xpts", 0))
            cost = _as_money(p.get("now_cost"))
            dot = _availability_dot({"status": p.get("status", "a"), "chance_of_playing_next_round": p.get("chance_of_playing_next_round")})
            pid = int(p.get("player_id", 0))
            in_team = owned_ids is not None and pid in owned_ids
            tag = " ✅" if in_team else ""
            lines.append(f"  {dot} {name} ({team}) {cost} — xPts: {xpts:.1f}{tag}")
        lines.append("")

    if owned_ids is not None:
        matches = sum(1 for pid in totgw_ids if pid in owned_ids)
        total_players = len(totgw_ids)
        pct = int(matches / total_players * 100) if total_players else 0
        bar = "█" * matches + "░" * (total_players - matches)
        lines.append(f"📊 YOUR SCORE: {matches}/{total_players} players  ({pct}%)")
        lines.append(f"   [{bar}]")
        lines.append("")

    lines.append("ℹ️ Based on xPts model + fixture difficulty for this GW.")
    lines.append("✅ = already in your squad")
    return "\n".join(lines).rstrip()
