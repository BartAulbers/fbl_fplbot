"""
FPL Squad Importer — fetches a manager's current squad directly
from the FPL API using their Manager ID.

Resolves player IDs, looks up purchase prices from transfer history,
and writes everything to the my_squad table.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import httpx
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.database.db import get_connection

BASE = settings.fpl_base_url
TIMEOUT = 20


@dataclass
class ImportResult:
    success: bool
    imported: int               # players successfully added
    skipped: int                # players not found in local DB
    message: str
    players: list[dict]         # list of imported player dicts


def import_squad_from_fpl(manager_id: int, gameweek: Optional[int] = None, user_id: int = 0) -> ImportResult:
    """
    Synchronous import (safe to call from Streamlit).

    1. Fetches /entry/{manager_id}/event/{gw}/picks/
    2. Fetches /entry/{manager_id}/transfers/ for purchase prices
    3. Upserts all 15 players into my_squad table

    Returns ImportResult with details.
    """
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            # ── Resolve current gameweek if not provided ──────────────────
            if gameweek is None:
                gameweek = _get_current_gw_from_db()

            logger.info("Importing squad for manager {} GW{}", manager_id, gameweek)

            # ── Fetch picks ───────────────────────────────────────────────
            try:
                picks_resp = client.get(f"{BASE}/entry/{manager_id}/event/{gameweek}/picks/")
                picks_resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    return ImportResult(
                        False, 0, 0,
                        f"No picks found for GW{gameweek}. Either the manager ID is wrong, "
                        f"or (more likely if this is early in the season) the GW{gameweek} deadline "
                        f"hasn't passed yet so FPL hasn't published squads for it.",
                        [],
                    )
                raise
            picks_data = picks_resp.json()

            picks = picks_data.get("picks", [])
            if not picks:
                return ImportResult(False, 0, 0, "No picks found for this GW.", [])

            entry_history = picks_data.get("entry_history", {})
            bank = entry_history.get("bank", 0) / 10.0  # £m in bank

            # ── Fetch transfer history for purchase prices ────────────────
            transfers_resp = client.get(f"{BASE}/entry/{manager_id}/transfers/")
            transfers_resp.raise_for_status()
            transfers = transfers_resp.json()

            # Build purchase price lookup: element_id → purchase_price
            # Most recent 'in' transfer for each player = purchase price
            purchase_prices: dict[int, float] = {}
            for t in sorted(transfers, key=lambda x: x.get("entry_transfer_deadline", ""), reverse=False):
                elem_in = t.get("element_in")
                cost_in = t.get("element_in_cost")
                if elem_in and cost_in:
                    purchase_prices[elem_in] = cost_in / 10.0

            # ── Match FPL element IDs to our DB ───────────────────────────
            element_ids = [p["element"] for p in picks]
            con = get_connection(read_only=True)
            placeholders = ",".join("?" * len(element_ids))
            players_df = con.execute(
                f"SELECT id, web_name, position, now_cost, team_id FROM players WHERE id IN ({placeholders})",
                element_ids,
            ).df()
            con.close()

            player_map = {row["id"]: row for _, row in players_df.iterrows()}

            # ── Write to my_squad ─────────────────────────────────────────
            imported_players = []
            skipped = 0

            con = get_connection()
            con.execute("DELETE FROM my_squad WHERE user_id = ?", [int(user_id)])  # clear only this user's squad

            for pick in picks:
                eid = pick["element"]
                player = player_map.get(eid)

                if player is None:
                    logger.warning("Player ID {} not found in local DB — run pipeline first", eid)
                    skipped += 1
                    continue

                # Purchase price: from transfer history, else current price
                pp = purchase_prices.get(eid, player["now_cost"])

                con.execute(
                    """
                    INSERT OR REPLACE INTO my_squad
                        (user_id, player_id, purchase_price, is_captain, is_vice_captain, added_gameweek, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        int(user_id),
                        int(eid),
                        float(pp),
                        bool(pick.get("is_captain", False)),
                        bool(pick.get("is_vice_captain", False)),
                        int(gameweek),
                        f"Imported GW{gameweek}",
                    ],
                )

                imported_players.append({
                    "player_id": int(eid),
                    "web_name": player["web_name"],
                    "position": player["position"],
                    "now_cost": float(player["now_cost"]),
                    "purchase_price": float(pp),
                    "is_captain": bool(pick.get("is_captain", False)),
                    "is_vice_captain": bool(pick.get("is_vice_captain", False)),
                    "squad_position": int(pick.get("position", 0)),
                })

            con.close()

            msg = (
                f"Imported {len(imported_players)} players for GW{gameweek}. "
                f"Bank: £{bank:.1f}m."
            )
            if skipped:
                msg += f" {skipped} player(s) not found locally — re-run the pipeline."

            logger.success(msg)
            return ImportResult(
                success=True,
                imported=len(imported_players),
                skipped=skipped,
                message=msg,
                players=imported_players,
            )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return ImportResult(False, 0, 0,
                f"Manager ID {manager_id} not found. Check your FPL Team ID.", [])
        return ImportResult(False, 0, 0, f"FPL API error: {e.response.status_code}", [])
    except httpx.RequestError as e:
        return ImportResult(False, 0, 0, f"Network error: {e}", [])
    except Exception as e:
        logger.exception("Squad import failed")
        return ImportResult(False, 0, 0, f"Import failed: {e}", [])


def _get_current_gw_from_db() -> int:
    try:
        con = get_connection(read_only=True)
        row = con.execute("SELECT id FROM gameweeks WHERE is_current = true LIMIT 1").fetchone()
        con.close()
        if row:
            return int(row[0])
        con2 = get_connection(read_only=True)
        row2 = con2.execute("SELECT MAX(id) FROM gameweeks WHERE is_finished = true").fetchone()
        con2.close()
        return int(row2[0] or 1)
    except Exception:
        return 1


def get_manager_info(manager_id: int) -> Optional[dict]:
    """Fetch basic manager info to validate the ID before importing."""
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            r = client.get(f"{BASE}/entry/{manager_id}/")
            r.raise_for_status()
            data = r.json()
            return {
                "name": f"{data.get('player_first_name', '')} {data.get('player_last_name', '')}".strip(),
                "team_name": data.get("name", "Unknown"),
                "overall_rank": data.get("summary_overall_rank"),
                "overall_points": data.get("summary_overall_points"),
                "last_gw_points": data.get("summary_event_points"),
            }
    except Exception:
        return None
