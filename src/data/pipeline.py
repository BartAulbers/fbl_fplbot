"""
Data ingestion pipeline: FPL API → DuckDB.
Handles bootstrap data, fixtures, and per-player history.
"""
import asyncio
import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# allow running as a script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.fpl_client import FPLClient
from src.database.db import get_connection, init_db
from config.settings import settings

# ── Position mapping ───────────────────────────────────────────────────────────
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ── Teams ──────────────────────────────────────────────────────────────────────

def _upsert_teams(con, teams: list[dict]) -> None:
    rows = []
    for t in teams:
        overall_home = t.get("strength_overall_home") or 0
        overall_away = t.get("strength_overall_away") or 0
        attack_home = t.get("strength_attack_home") or 0
        attack_away = t.get("strength_attack_away") or 0
        defence_home = t.get("strength_defence_home") or 0
        defence_away = t.get("strength_defence_away") or 0
        strength = t.get("strength")

        # FPL doesn't publish granular attack/defence strength splits until a
        # few gameweeks into the season (they're 0 preseason / at GW1). Fall
        # back to strength_overall_home/away — available from day one — so
        # the model always has a meaningful team-strength signal instead of
        # a flat 0 for every team.
        attack_home = attack_home or overall_home
        attack_away = attack_away or overall_away
        defence_home = defence_home or overall_home
        defence_away = defence_away or overall_away
        if not strength and (overall_home or overall_away):
            strength = round((overall_home + overall_away) / 2)

        rows.append({
            "id": t["id"],
            "name": t["name"],
            "short_name": t["short_name"],
            "strength": strength,
            "strength_attack_home": attack_home,
            "strength_attack_away": attack_away,
            "strength_defence_home": defence_home,
            "strength_defence_away": defence_away,
            "updated_at": pd.Timestamp.now(),
        })
    df = pd.DataFrame(rows)
    con.execute("DELETE FROM teams")
    con.execute("INSERT INTO teams SELECT * FROM df")
    logger.info("Upserted {} teams", len(df))


# ── Players ────────────────────────────────────────────────────────────────────

def _upsert_players(con, elements: list[dict]) -> None:
    rows = []
    for e in elements:
        rows.append({
            "id": e["id"],
            "first_name": e["first_name"],
            "second_name": e["second_name"],
            "web_name": e["web_name"],
            "team_id": e["team"],
            "position": POSITION_MAP.get(e["element_type"], "UNK"),
            "now_cost": e["now_cost"] / 10.0,
            "status": e["status"],
            "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
            "chance_of_playing_this_round": e.get("chance_of_playing_this_round"),
            "total_points": e["total_points"],
            "form": float(e["form"] or 0),
            "points_per_game": float(e["points_per_game"] or 0),
            "selected_by_percent": float(e["selected_by_percent"] or 0),
            "minutes": e["minutes"],
            "goals_scored": e["goals_scored"],
            "assists": e["assists"],
            "clean_sheets": e["clean_sheets"],
            "goals_conceded": e["goals_conceded"],
            "yellow_cards": e["yellow_cards"],
            "red_cards": e["red_cards"],
            "saves": e["saves"],
            "bonus": e["bonus"],
            "bps": e["bps"],
            "influence": float(e["influence"] or 0),
            "creativity": float(e["creativity"] or 0),
            "threat": float(e["threat"] or 0),
            "ict_index": float(e["ict_index"] or 0),
            "expected_goals": float(e.get("expected_goals") or 0),
            "expected_assists": float(e.get("expected_assists") or 0),
            "expected_goal_involvements": float(e.get("expected_goal_involvements") or 0),
            "expected_goals_conceded": float(e.get("expected_goals_conceded") or 0),
            "defensive_contribution": float(e.get("defensive_contribution") or 0),
            "transfers_in_event": e.get("transfers_in_event", 0),
            "transfers_out_event": e.get("transfers_out_event", 0),
            "value_form": float(e.get("value_form") or 0),
            "value_season": float(e.get("value_season") or 0),
            "news": e.get("news", ""),
            "news_added": e.get("news_added", ""),
            "updated_at": pd.Timestamp.now(),
        })
    df = pd.DataFrame(rows)
    # Clear dependent tables first to satisfy FK constraints before re-inserting players
    con.execute("DELETE FROM player_metrics")
    con.execute("DELETE FROM players")
    con.execute("""
        INSERT INTO players (
            id, first_name, second_name, web_name, team_id, position, now_cost, status,
            chance_of_playing_next_round, chance_of_playing_this_round, total_points, form,
            points_per_game, selected_by_percent, minutes, goals_scored, assists, clean_sheets,
            goals_conceded, yellow_cards, red_cards, saves, bonus, bps,
            influence, creativity, threat, ict_index,
            expected_goals, expected_assists, expected_goal_involvements, expected_goals_conceded,
            defensive_contribution,
            transfers_in_event, transfers_out_event, value_form, value_season,
            news, news_added, updated_at
        )
        SELECT
            id, first_name, second_name, web_name, team_id, position, now_cost, status,
            chance_of_playing_next_round, chance_of_playing_this_round, total_points, form,
            points_per_game, selected_by_percent, minutes, goals_scored, assists, clean_sheets,
            goals_conceded, yellow_cards, red_cards, saves, bonus, bps,
            influence, creativity, threat, ict_index,
            expected_goals, expected_assists, expected_goal_involvements, expected_goals_conceded,
            defensive_contribution,
            transfers_in_event, transfers_out_event, value_form, value_season,
            news, news_added, updated_at
        FROM df
    """)
    logger.info("Upserted {} players", len(df))


# ── Gameweeks ──────────────────────────────────────────────────────────────────

def _upsert_gameweeks(con, events: list[dict]) -> None:
    rows = []
    for e in events:
        rows.append({
            "id": e["id"],
            "name": e["name"],
            "deadline_time": e["deadline_time"],
            "average_entry_score": e.get("average_entry_score"),
            "highest_score": e.get("highest_score"),
            "is_finished": e["finished"],
            "is_current": e["is_current"],
            "is_next": e["is_next"],
            "updated_at": pd.Timestamp.now(),
        })
    df = pd.DataFrame(rows)
    con.execute("DELETE FROM gameweeks")
    con.execute("INSERT INTO gameweeks SELECT * FROM df")
    logger.info("Upserted {} gameweeks", len(df))


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _upsert_fixtures(con, fixtures: list[dict]) -> None:
    rows = []
    for f in fixtures:
        rows.append({
            "id": f["id"],
            "gameweek_id": f.get("event"),
            "team_h": f["team_h"],
            "team_a": f["team_a"],
            "team_h_score": f.get("team_h_score"),
            "team_a_score": f.get("team_a_score"),
            "team_h_difficulty": f.get("team_h_difficulty"),
            "team_a_difficulty": f.get("team_a_difficulty"),
            "kickoff_time": f.get("kickoff_time"),
            "finished": f.get("finished", False),
            "updated_at": pd.Timestamp.now(),
        })
    df = pd.DataFrame(rows)
    con.execute("DELETE FROM fixtures")
    con.execute("INSERT INTO fixtures SELECT * FROM df")
    logger.info("Upserted {} fixtures", len(df))


# ── Player GW history ──────────────────────────────────────────────────────────

def _upsert_player_history(con, player_id: int, history: list[dict]) -> None:
    if not history:
        return
    rows = []
    for h in history:
        rows.append({
            "player_id": player_id,
            "gameweek_id": h["round"],
            "total_points": h["total_points"],
            "minutes": h["minutes"],
            "goals_scored": h["goals_scored"],
            "assists": h["assists"],
            "clean_sheets": h["clean_sheets"],
            "goals_conceded": h["goals_conceded"],
            "own_goals": h["own_goals"],
            "penalties_saved": h["penalties_saved"],
            "penalties_missed": h["penalties_missed"],
            "yellow_cards": h["yellow_cards"],
            "red_cards": h["red_cards"],
            "saves": h["saves"],
            "bonus": h["bonus"],
            "bps": h["bps"],
            "influence": float(h.get("influence") or 0),
            "creativity": float(h.get("creativity") or 0),
            "threat": float(h.get("threat") or 0),
            "ict_index": float(h.get("ict_index") or 0),
            "expected_goals": float(h.get("expected_goals") or 0),
            "expected_assists": float(h.get("expected_assists") or 0),
            "expected_goal_involvements": float(h.get("expected_goal_involvements") or 0),
            "expected_goals_conceded": float(h.get("expected_goals_conceded") or 0),
            "defensive_contribution": float(h.get("defensive_contribution") or 0),
            "value": h["value"] / 10.0,
            "selected": float(h.get("selected") or 0),
            "was_home": h["was_home"],
            "round": h["round"],
        })
    df = pd.DataFrame(rows)

    # Aggregate double-gameweek rows: sum counting stats, average prices/flags
    sum_cols = [
        "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
        "yellow_cards", "red_cards", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded",
        "defensive_contribution",
    ]
    mean_cols = ["value", "selected"]

    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "mean" for c in mean_cols})
    agg["was_home"] = "last"  # keep last fixture's home flag
    agg["round"] = "first"

    df = df.groupby(["player_id", "gameweek_id"], as_index=False).agg(agg)
    history_columns = [
        "player_id", "gameweek_id", "total_points", "minutes", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
        "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence", "creativity",
        "threat", "ict_index", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded", "value", "selected",
        "was_home", "round", "defensive_contribution",
    ]
    df = df[history_columns]

    con.execute(f"DELETE FROM player_gw_history WHERE player_id = {player_id}")
    con.execute("INSERT INTO player_gw_history SELECT * FROM df")


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_full_pipeline(include_player_history: bool = True) -> None:
    """Fetch everything from the FPL API and store in DuckDB."""
    init_db()

    async with FPLClient() as client:
        logger.info("Fetching bootstrap data...")
        bootstrap = await client.get_bootstrap()

        logger.info("Fetching fixtures...")
        fixtures = await client.get_fixtures()

        con = get_connection()
        _upsert_teams(con, bootstrap["teams"])
        _upsert_gameweeks(con, bootstrap["events"])
        _upsert_players(con, bootstrap["elements"])
        _upsert_fixtures(con, fixtures)

        if include_player_history:
            player_ids = [p["id"] for p in bootstrap["elements"]]
            logger.info("Fetching history for {} players...", len(player_ids))
            histories = await client.get_all_player_summaries(player_ids)

            inserted = 0
            for pid, summary in histories.items():
                history = summary.get("history", [])
                _upsert_player_history(con, pid, history)
                inserted += len(history)

            logger.info("Inserted {} player-gameweek rows", inserted)

        con.close()

    logger.success("Pipeline complete.")


if __name__ == "__main__":
    asyncio.run(run_full_pipeline())
