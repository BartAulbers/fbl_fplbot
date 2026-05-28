"""
Shared dependencies injected into all routers:
- Database connection
- Current gameweek
- Pre-loaded DataFrames (cached per request cycle)
"""
import sys
from pathlib import Path
from functools import lru_cache
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.db import get_connection


def load_players() -> pd.DataFrame:
    con = get_connection(read_only=True)
    df = con.execute("SELECT * FROM players").df()
    con.close()
    return df


def load_teams() -> pd.DataFrame:
    con = get_connection(read_only=True)
    df = con.execute("SELECT * FROM teams").df()
    con.close()
    return df


def load_fixtures() -> pd.DataFrame:
    con = get_connection(read_only=True)
    df = con.execute("SELECT * FROM fixtures").df()
    con.close()
    return df


def load_history() -> pd.DataFrame:
    con = get_connection(read_only=True)
    df = con.execute("SELECT * FROM player_gw_history").df()
    con.close()
    return df


def load_my_squad() -> pd.DataFrame:
    con = get_connection(read_only=True)
    df = con.execute("""
        SELECT ms.*, p.web_name, p.position, p.now_cost, p.team_id,
               p.selected_by_percent
        FROM my_squad ms
        JOIN players p ON ms.player_id = p.id
    """).df()
    con.close()
    return df


def get_current_gw() -> int:
    con = get_connection(read_only=True)
    # Prefer: current GW that's not yet finished (= next round to play)
    result = con.execute(
        "SELECT id FROM gameweeks WHERE is_current = true AND is_finished = false LIMIT 1"
    ).fetchone()
    if result:
        con.close()
        return int(result[0])
    # Fallback: first unfinished GW
    result2 = con.execute(
        "SELECT MIN(id) FROM gameweeks WHERE is_finished = false"
    ).fetchone()
    if result2 and result2[0]:
        con.close()
        return int(result2[0])
    # Final fallback: highest finished GW + 1
    result3 = con.execute(
        "SELECT MAX(id) FROM gameweeks WHERE is_finished = true"
    ).fetchone()
    con.close()
    return int(result3[0] or 1) + 1


def load_xpts(current_gw: int) -> pd.DataFrame:
    con = get_connection(read_only=True)
    df = con.execute(
        "SELECT * FROM expected_points WHERE gameweek_id = ?", [current_gw]
    ).df()
    con.close()
    return df


def load_metrics() -> pd.DataFrame:
    con = get_connection(read_only=True)
    df = con.execute("SELECT * FROM player_metrics").df()
    con.close()
    return df
