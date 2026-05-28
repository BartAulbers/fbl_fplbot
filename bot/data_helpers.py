from __future__ import annotations

import pandas as pd
from loguru import logger

from src.analytics.analytics import compute_player_metrics, fixture_swing_alerts
from src.data.pipeline import run_full_pipeline
from src.database.db import get_connection
from src.models.expected_points import predict_multi_gw


def get_current_gw() -> int:
    con = get_connection(read_only=True)
    try:
        current = con.execute(
            "SELECT id FROM gameweeks WHERE is_current = true AND is_finished = false LIMIT 1"
        ).fetchone()
        if current:
            return int(current[0])

        next_up = con.execute(
            "SELECT MIN(id) FROM gameweeks WHERE is_finished = false"
        ).fetchone()
        if next_up and next_up[0] is not None:
            return int(next_up[0])

        latest = con.execute("SELECT MAX(id) FROM gameweeks").fetchone()
        return int(latest[0] or 1)
    finally:
        con.close()


def database_has_player_data() -> bool:
    con = get_connection(read_only=True)
    try:
        row = con.execute("SELECT COUNT(*) FROM players").fetchone()
        return bool(row and row[0])
    finally:
        con.close()


def is_season_over() -> bool:
    """Returns True if all gameweeks are finished and no upcoming fixtures exist."""
    con = get_connection(read_only=True)
    try:
        unfinished_gw = con.execute(
            "SELECT COUNT(*) FROM gameweeks WHERE is_finished = false"
        ).fetchone()[0]
        unfinished_fix = con.execute(
            "SELECT COUNT(*) FROM fixtures WHERE finished = false"
        ).fetchone()[0]
        return unfinished_gw == 0 and unfinished_fix == 0
    finally:
        con.close()


def squad_exists(user_id: int = 0) -> bool:
    con = get_connection(read_only=True)
    try:
        row = con.execute("SELECT COUNT(*) FROM my_squad WHERE user_id = ?", [int(user_id)]).fetchone()
        return bool(row and row[0])
    finally:
        con.close()


def load_players() -> pd.DataFrame:
    con = get_connection(read_only=True)
    try:
        return con.execute("SELECT * FROM players").df()
    finally:
        con.close()


def load_teams() -> pd.DataFrame:
    con = get_connection(read_only=True)
    try:
        return con.execute("SELECT * FROM teams").df()
    finally:
        con.close()


def load_fixtures() -> pd.DataFrame:
    con = get_connection(read_only=True)
    try:
        return con.execute("SELECT * FROM fixtures").df()
    finally:
        con.close()


def load_history() -> pd.DataFrame:
    con = get_connection(read_only=True)
    try:
        return con.execute("SELECT * FROM player_gw_history").df()
    finally:
        con.close()


async def refresh_fpl_data(include_predictions: bool = True) -> None:
    await run_full_pipeline(include_player_history=True)

    players = load_players()
    teams = load_teams()
    fixtures = load_fixtures()
    history = load_history()
    if players.empty or teams.empty or fixtures.empty or history.empty:
        logger.warning("Skipping analytics refresh because one or more tables are empty.")
        return

    current_gw = get_current_gw()
    metrics = compute_player_metrics(players, history, fixtures, current_gw, teams)
    metrics["updated_at"] = pd.Timestamp.now()

    con = get_connection()
    try:
        con.execute("DELETE FROM player_metrics")
        con.execute("""
            INSERT INTO player_metrics (
                player_id, pts_per_90, pts_per_million, form_score,
                fixture_score_3gw, fixture_score_5gw, consistency, home_away_delta,
                ownership_inefficiency, bonus_rate, xgi_per_90, rotation_risk, updated_at
            )
            SELECT
                player_id, pts_per_90, pts_per_million, form_score,
                fixture_score_3gw, fixture_score_5gw, consistency, home_away_delta,
                ownership_inefficiency, bonus_rate, xgi_per_90, rotation_risk, updated_at
            FROM metrics
        """)
    finally:
        con.close()

    if not include_predictions:
        from bot import cache
        cache.write("fpl_data_refreshed", {"gw": current_gw})
        return

    try:
        xpts = predict_multi_gw(history, fixtures, players, teams, current_gw)
        xpts["gameweek_id"] = current_gw
        xpts["model_version"] = "v1"
        xpts["confidence"] = 0.7
        xpts["created_at"] = pd.Timestamp.now()

        con = get_connection()
        try:
            con.execute("DELETE FROM expected_points WHERE gameweek_id = ?", [current_gw])
            con.execute("""
                INSERT INTO expected_points (
                    player_id, gameweek_id, xpts, xpts_3gw, xpts_5gw,
                    model_version, confidence, created_at
                )
                SELECT
                    player_id, gameweek_id, xpts, xpts_3gw, xpts_5gw,
                    model_version, confidence, created_at
                FROM xpts
            """)
        finally:
            con.close()
    except FileNotFoundError:
        logger.warning("Expected points model not trained yet. Transfer suggestions will fall back to zero xPts.")
    except Exception:
        logger.exception("Failed to refresh expected points predictions")

    from bot import cache
    cache.write("fpl_data_refreshed", {"gw": current_gw})
    logger.info("FPL data refresh complete — cache timestamp written")


async def nightly_refresh() -> None:
    """
    Full refresh + pre-compute fixture caches.
    Called by the scheduler at 19:00 daily.
    Falls back gracefully on any error so the scheduler keeps running.
    """
    from bot import cache
    from src.analytics.analytics import analyse_fixture_runs, fixture_swing_alerts

    logger.info("Nightly refresh starting...")
    try:
        await refresh_fpl_data(include_predictions=True)
    except Exception:
        logger.exception("Nightly refresh: pipeline/predictions failed")
        return

    try:
        teams = load_teams()
        fixtures = load_fixtures()
        current_gw = get_current_gw()

        if not teams.empty and not fixtures.empty:
            swings = fixture_swing_alerts(teams, fixtures, current_gw)
            cache.write("fixture_swings", swings)

            easiest = analyse_fixture_runs(teams, fixtures, current_gw, n_gws=5)
            cache.write("easiest_fixtures", easiest.head(5).to_dict(orient="records"))

            logger.info("Nightly refresh: fixture caches written (GW{})", current_gw)
    except Exception:
        logger.exception("Nightly refresh: fixture cache pre-computation failed")

    try:
        totgw = load_team_of_gw()
        if totgw:
            cache.write("team_of_gw", totgw)
            logger.info("Nightly refresh: team of GW cache written")
    except Exception:
        logger.exception("Nightly refresh: team of GW cache pre-computation failed")

    logger.info("Nightly refresh complete")


def load_squad_with_enrichment(user_id: int = 0) -> pd.DataFrame | None:
    current_gw = get_current_gw()
    con = get_connection(read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                ms.player_id,
                p.web_name,
                p.position,
                p.now_cost,
                ms.purchase_price,
                ms.is_captain,
                ms.is_vice_captain,
                ms.added_gameweek,
                p.selected_by_percent,
                p.team_id,
                p.status,
                p.chance_of_playing_next_round,
                p.news,
                COALESCE(ep.xpts, 0) AS xpts,
                COALESCE(ep.xpts_3gw, 0) AS xpts_3gw,
                COALESCE(ep.xpts_5gw, 0) AS xpts_5gw,
                COALESCE(pm.consistency, 0.5) AS consistency,
                COALESCE(pm.fixture_score_3gw, 3.0) AS fdr_avg_3gw
            FROM my_squad ms
            JOIN players p ON ms.player_id = p.id
            LEFT JOIN expected_points ep
                ON ms.player_id = ep.player_id AND ep.gameweek_id = ?
            LEFT JOIN player_metrics pm
                ON ms.player_id = pm.player_id
            WHERE ms.user_id = ?
            ORDER BY CASE p.position WHEN 'GK' THEN 1 WHEN 'DEF' THEN 2 WHEN 'MID' THEN 3 WHEN 'FWD' THEN 4 ELSE 5 END,
                     p.web_name
            """,
            [current_gw, int(user_id)],
        ).df()
        return df if not df.empty else None
    finally:
        con.close()


def load_all_players_with_enrichment() -> pd.DataFrame:
    current_gw = get_current_gw()
    con = get_connection(read_only=True)
    try:
        return con.execute(
            """
            SELECT
                p.id AS player_id,
                p.web_name,
                p.position,
                p.now_cost,
                p.now_cost AS purchase_price,
                COALESCE(ep.xpts, 0) AS xpts,
                COALESCE(ep.xpts_3gw, 0) AS xpts_3gw,
                COALESCE(ep.xpts_5gw, 0) AS xpts_5gw,
                COALESCE(pm.consistency, 0.5) AS consistency,
                1 AS added_gameweek,
                p.selected_by_percent,
                COALESCE(pm.fixture_score_3gw, 3.0) AS fdr_avg_3gw,
                p.team_id
            FROM players p
            LEFT JOIN expected_points ep
                ON p.id = ep.player_id AND ep.gameweek_id = ?
            LEFT JOIN player_metrics pm
                ON p.id = pm.player_id
            """,
            [current_gw],
        ).df()
    finally:
        con.close()


def _recent_minutes_bounds() -> tuple[int, int]:
    current_gw = get_current_gw()
    return max(current_gw - 3, 1), max(current_gw - 1, 0)


def load_news_for_my_squad(user_id: int = 0) -> list[dict]:
    lower, upper = _recent_minutes_bounds()
    con = get_connection(read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                p.id AS player_id,
                p.web_name,
                p.position,
                p.status,
                p.now_cost,
                p.chance_of_playing_next_round,
                p.news,
                ms.is_captain,
                ms.is_vice_captain,
                COALESCE(r.recent_minutes, 0) AS recent_minutes
            FROM my_squad ms
            JOIN players p ON ms.player_id = p.id
            LEFT JOIN (
                SELECT player_id, SUM(minutes) AS recent_minutes
                FROM player_gw_history
                WHERE gameweek_id BETWEEN ? AND ?
                GROUP BY player_id
            ) r ON p.id = r.player_id
            WHERE ms.user_id = ?
            ORDER BY CASE p.position WHEN 'GK' THEN 1 WHEN 'DEF' THEN 2 WHEN 'MID' THEN 3 WHEN 'FWD' THEN 4 ELSE 5 END,
                     p.web_name
            """,
            [lower, upper, int(user_id)],
        ).df()
        return df.to_dict(orient="records")
    finally:
        con.close()


def search_players_by_name(name: str, limit: int = 5) -> list[dict]:
    lower, upper = _recent_minutes_bounds()
    search = name.strip().lower()
    if not search:
        return []

    con = get_connection(read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                p.id AS player_id,
                p.web_name,
                p.position,
                p.status,
                p.chance_of_playing_next_round,
                p.news,
                p.selected_by_percent,
                COALESCE(r.recent_minutes, 0) AS recent_minutes
            FROM players p
            LEFT JOIN (
                SELECT player_id, SUM(minutes) AS recent_minutes
                FROM player_gw_history
                WHERE gameweek_id BETWEEN ? AND ?
                GROUP BY player_id
            ) r ON p.id = r.player_id
            WHERE lower(p.web_name) LIKE ?
            ORDER BY
                CASE
                    WHEN lower(p.web_name) = ? THEN 0
                    WHEN lower(p.web_name) LIKE ? THEN 1
                    ELSE 2
                END,
                p.selected_by_percent DESC,
                p.web_name ASC
            LIMIT ?
            """,
            [lower, upper, f"%{search}%", search, f"{search}%", int(limit)],
        ).df()
        return df.to_dict(orient="records")
    finally:
        con.close()


def load_player_news(player_id: int) -> list[dict]:
    lower, upper = _recent_minutes_bounds()
    con = get_connection(read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                p.id AS player_id,
                p.web_name,
                p.position,
                p.status,
                p.chance_of_playing_next_round,
                p.news,
                COALESCE(r.recent_minutes, 0) AS recent_minutes
            FROM players p
            LEFT JOIN (
                SELECT player_id, SUM(minutes) AS recent_minutes
                FROM player_gw_history
                WHERE gameweek_id BETWEEN ? AND ?
                GROUP BY player_id
            ) r ON p.id = r.player_id
            WHERE p.id = ?
            """,
            [lower, upper, int(player_id)],
        ).df()
        return df.to_dict(orient="records")
    finally:
        con.close()


def load_fixture_swings() -> list[dict]:
    from bot import cache
    from src.analytics.analytics import fixture_swing_alerts
    cached = cache.read("fixture_swings")
    if cached is not None:
        logger.debug("load_fixture_swings: using cache")
        return cached
    teams = load_teams()
    fixtures = load_fixtures()
    if teams.empty or fixtures.empty:
        return []
    return fixture_swing_alerts(teams, fixtures, get_current_gw())


def load_easiest_fixtures(top_n: int = 5) -> list[dict]:
    from bot import cache
    from src.analytics.analytics import analyse_fixture_runs
    cached = cache.read("easiest_fixtures")
    if cached is not None:
        logger.debug("load_easiest_fixtures: using cache")
        return cached[:top_n]
    teams = load_teams()
    fixtures = load_fixtures()
    if teams.empty or fixtures.empty:
        return []
    df = analyse_fixture_runs(teams, fixtures, get_current_gw(), n_gws=5)
    return df.head(top_n).to_dict(orient="records")


def load_popular_missing_players(user_id: int = 0, min_ownership: float = 15.0, limit: int = 15) -> list[dict]:
    current_gw = get_current_gw()
    con = get_connection(read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                p.id AS player_id,
                p.web_name,
                p.position,
                p.now_cost,
                p.selected_by_percent,
                COALESCE(ep.xpts, 0) AS xpts,
                COALESCE(ep.xpts_3gw, 0) AS xpts_3gw
            FROM players p
            LEFT JOIN expected_points ep
                ON p.id = ep.player_id AND ep.gameweek_id = ?
            WHERE p.selected_by_percent >= ?
              AND p.id NOT IN (SELECT player_id FROM my_squad WHERE user_id = ?)
            ORDER BY p.selected_by_percent DESC
            LIMIT ?
            """,
            [current_gw, min_ownership, int(user_id), limit],
        ).df()
        return df.to_dict(orient="records")
    finally:
        con.close()


def load_team_of_gw() -> dict | None:
    """
    Selects the best 11 players for the next GW using xPts + FDR.
    Checks cache first; falls back to live computation.
    """
    from bot import cache
    cached = cache.read("team_of_gw")
    if cached is not None:
        logger.debug("load_team_of_gw: using cache")
        return cached
    return _compute_team_of_gw()


def _compute_team_of_gw() -> dict | None:
    current_gw = get_current_gw()
    con = get_connection(read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                p.id AS player_id,
                p.web_name,
                p.position,
                p.team_id,
                t.name AS team_name,
                t.short_name AS team_short,
                p.now_cost,
                p.status,
                p.chance_of_playing_next_round,
                COALESCE(ep.xpts, 0)     AS xpts,
                COALESCE(ep.xpts_3gw, 0) AS xpts_3gw
            FROM players p
            JOIN teams t ON p.team_id = t.id
            LEFT JOIN expected_points ep
                ON p.id = ep.player_id AND ep.gameweek_id = ?
            WHERE p.status IN ('a', 'd')
              AND (p.chance_of_playing_next_round IS NULL
                   OR p.chance_of_playing_next_round >= 50)
            ORDER BY xpts DESC
            """,
            [current_gw],
        ).df()
    finally:
        con.close()

    if df.empty:
        return None

    by_pos = {pos: grp.reset_index(drop=True) for pos, grp in df.groupby("position")}
    for pos in ("GK", "DEF", "MID", "FWD"):
        if pos not in by_pos:
            by_pos[pos] = pd.DataFrame(columns=df.columns)

    # All valid FPL formations: (def, mid, fwd)
    formations = [
        (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
        (4, 5, 1), (5, 3, 2), (5, 4, 1), (5, 2, 3),
    ]

    best_total = -1.0
    best_result = None

    for n_def, n_mid, n_fwd in formations:
        gk_pool  = by_pos["GK"].head(1)
        def_pool = by_pos["DEF"].head(n_def)
        mid_pool = by_pos["MID"].head(n_mid)
        fwd_pool = by_pos["FWD"].head(n_fwd)

        if len(gk_pool) < 1 or len(def_pool) < n_def or len(mid_pool) < n_mid or len(fwd_pool) < n_fwd:
            continue

        total = (
            gk_pool["xpts"].sum()
            + def_pool["xpts"].sum()
            + mid_pool["xpts"].sum()
            + fwd_pool["xpts"].sum()
        )
        if total > best_total:
            best_total = float(total)
            best_result = {
                "formation": f"{n_def}-{n_mid}-{n_fwd}",
                "gameweek": current_gw,
                "total_xpts": round(best_total, 1),
                "GK":  gk_pool.to_dict(orient="records"),
                "DEF": def_pool.to_dict(orient="records"),
                "MID": mid_pool.to_dict(orient="records"),
                "FWD": fwd_pool.to_dict(orient="records"),
            }

    return best_result


def get_my_squad_player_ids(user_id: int) -> set[int]:
    """Returns the set of player IDs in the user's saved squad."""
    con = get_connection(read_only=True)
    try:
        rows = con.execute(
            "SELECT player_id FROM my_squad WHERE user_id = ?", [int(user_id)]
        ).fetchall()
        return {int(r[0]) for r in rows}
    finally:
        con.close()
