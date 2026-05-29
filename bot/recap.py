"""
Gameweek recap: points scored, rank change, and over/underachievers.

Fetches manager data from the FPL API and compares actual GW points
against the xPts model predictions stored in the local DB.
"""
from __future__ import annotations

from loguru import logger

from src.api.fpl_client import FPLClient
from src.database.db import get_connection


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_xpts_for_gw(gw: int, player_ids: list[int]) -> dict[int, float]:
    if not player_ids:
        return {}
    con = get_connection(read_only=True)
    try:
        placeholders = ",".join("?" * len(player_ids))
        rows = con.execute(
            f"SELECT player_id, xpts FROM expected_points WHERE gameweek_id = ? AND player_id IN ({placeholders})",
            [gw, *player_ids],
        ).fetchall()
        return {int(r[0]): float(r[1]) for r in rows}
    finally:
        con.close()


def _load_player_names(player_ids: list[int]) -> dict[int, str]:
    if not player_ids:
        return {}
    con = get_connection(read_only=True)
    try:
        placeholders = ",".join("?" * len(player_ids))
        rows = con.execute(
            f"SELECT id, web_name FROM players WHERE id IN ({placeholders})",
            player_ids,
        ).fetchall()
        return {int(r[0]): str(r[1]) for r in rows}
    finally:
        con.close()


def get_last_finished_gw() -> int | None:
    con = get_connection(read_only=True)
    try:
        row = con.execute("SELECT MAX(id) FROM gameweeks WHERE is_finished = true").fetchone()
        return int(row[0]) if row and row[0] is not None else None
    finally:
        con.close()


# ── FPL API fetch ─────────────────────────────────────────────────────────────

async def fetch_gw_recap(fpl_id: int, gw: int) -> dict | None:
    """
    Fetches GW recap data for a manager from the FPL API.
    Returns None if data is unavailable.
    """
    try:
        async with FPLClient() as client:
            history_data, picks_data, live_data = await _fetch_all(client, fpl_id, gw)
    except Exception:
        logger.exception("Failed to fetch GW recap from FPL API for manager {}", fpl_id)
        return None

    # ── Manager GW history ────────────────────────────────────────────────
    gw_by_event = {h["event"]: h for h in history_data.get("current", [])}
    this_gw = gw_by_event.get(gw, {})
    prev_gw = gw_by_event.get(gw - 1, {})

    if not this_gw:
        logger.warning("No GW{} history found for manager {}", gw, fpl_id)
        return None

    gw_points: int = this_gw.get("points", 0)
    transfers_cost: int = this_gw.get("event_transfers_cost", 0)
    bench_pts: int = this_gw.get("points_on_bench", 0)
    overall_rank: int = this_gw.get("overall_rank", 0)
    prev_rank: int = prev_gw.get("overall_rank", 0)
    rank_change: int = prev_rank - overall_rank  # positive = rank improved

    # ── Picks ─────────────────────────────────────────────────────────────
    picks = picks_data.get("picks", [])
    # Only starting XI (positions 1–11)
    starters = {p["element"]: p for p in picks if p.get("position", 99) <= 11}
    all_picks = {p["element"]: p for p in picks}

    # ── Live points ───────────────────────────────────────────────────────
    live_pts: dict[int, int] = {
        e["id"]: e["stats"]["total_points"]
        for e in live_data.get("elements", [])
    }

    # ── xPts comparison ───────────────────────────────────────────────────
    all_player_ids = list(all_picks.keys())
    xpts_map = _load_xpts_for_gw(gw, all_player_ids)
    names_map = _load_player_names(all_player_ids)

    players: list[dict] = []
    for pid, pick in starters.items():
        multiplier = pick.get("multiplier", 1)
        raw_pts = live_pts.get(pid, 0)
        actual = raw_pts * multiplier
        xpts = xpts_map.get(pid, 0.0)
        delta = round(actual - xpts, 1)
        players.append({
            "player_id": pid,
            "name": names_map.get(pid, f"#{pid}"),
            "actual_pts": actual,
            "raw_pts": raw_pts,
            "xpts": xpts,
            "delta": delta,
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "multiplier": multiplier,
        })

    players.sort(key=lambda x: x["delta"], reverse=True)
    has_xpts = any(p["xpts"] > 0 for p in players)

    return {
        "gw": gw,
        "gw_points": gw_points,
        "transfers_cost": transfers_cost,
        "bench_pts": bench_pts,
        "overall_rank": overall_rank,
        "rank_change": rank_change,
        "has_prev_rank": prev_rank > 0,
        "players": players,
        "overachievers": [p for p in players if p["delta"] >= 3] if has_xpts else [],
        "underachievers": [p for p in players if p["delta"] <= -3] if has_xpts else [],
        "has_xpts": has_xpts,
    }


async def _fetch_all(client: FPLClient, fpl_id: int, gw: int):
    import asyncio
    return await asyncio.gather(
        client.get_entry_history(fpl_id),
        client.get_entry_picks(fpl_id, gw),
        client.get_gameweek_live(gw),
    )


# ── Formatter ─────────────────────────────────────────────────────────────────

def format_gw_recap(data: dict) -> str:
    gw = data["gw"]
    pts = data["gw_points"]
    cost = data["transfers_cost"]
    bench = data["bench_pts"]
    rank = data["overall_rank"]
    change = data["rank_change"]
    net_pts = pts - cost

    lines = [f"📊 GW{gw} RECAP", ""]

    # Points
    lines.append(f"⚽ Points this GW:  {pts}")
    if cost:
        lines.append(f"   (incl. -{cost} transfer hit → net {net_pts})")
    lines.append(f"🪑 Points on bench: {bench}")
    lines.append("")

    # Rank
    lines.append(f"🏆 Overall rank: {rank:,}")
    if data["has_prev_rank"]:
        if change > 0:
            lines.append(f"📈 Rank change:  +{change:,} ▲")
        elif change < 0:
            lines.append(f"📉 Rank change:  {change:,} ▼")
        else:
            lines.append("➡️ Rank change:  no change")
    lines.append("")

    # Over/underachievers
    if data["has_xpts"]:
        over = data["overachievers"]
        under = data["underachievers"]

        if over:
            lines.append("⭐ OVERACHIEVERS")
            for p in over[:4]:
                cap = " (C)" if p["is_captain"] else (" (VC)" if p["is_vice_captain"] else "")
                lines.append(
                    f"  🟢 {p['name']}{cap} — {p['actual_pts']} pts "
                    f"(xPts {p['xpts']:.1f}, +{p['delta']:.1f})"
                )
            lines.append("")

        if under:
            lines.append("😬 UNDERACHIEVERS")
            for p in under[:4]:
                cap = " (C)" if p["is_captain"] else (" (VC)" if p["is_vice_captain"] else "")
                lines.append(
                    f"  🔴 {p['name']}{cap} — {p['actual_pts']} pts "
                    f"(xPts {p['xpts']:.1f}, {p['delta']:.1f})"
                )
            lines.append("")
    else:
        lines.append("ℹ️ xPts comparison not available for this GW.")
        lines.append("")

    # All starters
    lines.append("📋 STARTING XI")
    for p in sorted(data["players"], key=lambda x: -x["actual_pts"]):
        cap = " 🅒" if p["is_captain"] else (" 🅥" if p["is_vice_captain"] else "")
        lines.append(f"  {p['name']}{cap}  {p['actual_pts']} pts")

    return "\n".join(lines)
