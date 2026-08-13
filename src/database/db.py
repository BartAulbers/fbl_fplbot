"""
DuckDB database layer — schema creation, connection management, and helpers.
DuckDB is a fast in-process OLAP database (no server needed).
"""
import duckdb
import pandas as pd
from pathlib import Path
from loguru import logger

from config.settings import settings


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(settings.db_path, read_only=read_only)


def init_db() -> None:
    """Create all tables if they don't exist and apply lightweight migrations."""
    con = get_connection()
    con.execute(SCHEMA_SQL)
    for migration in MIGRATION_SQL:
        con.execute(migration)
    _add_column_if_missing(con, "players", "defensive_contribution", "FLOAT DEFAULT 0")
    _add_column_if_missing(con, "player_gw_history", "defensive_contribution", "FLOAT DEFAULT 0")
    _migrate_my_squad_user_id(con)
    con.close()
    logger.info("Database initialised at {}", settings.db_path)


def _migrate_my_squad_user_id(con: duckdb.DuckDBPyConnection) -> None:
    """Add user_id to my_squad if missing (one-time migration)."""
    cols = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'my_squad'"
        ).fetchall()
    }
    if "user_id" in cols:
        return
    logger.info("Migrating my_squad to add user_id column (multi-user support)")
    con.execute("""
        CREATE TABLE my_squad_new (
            user_id         BIGINT  NOT NULL DEFAULT 0,
            player_id       INTEGER NOT NULL,
            purchase_price  FLOAT,
            is_captain      BOOLEAN DEFAULT FALSE,
            is_vice_captain BOOLEAN DEFAULT FALSE,
            added_gameweek  INTEGER,
            notes           VARCHAR,
            PRIMARY KEY (user_id, player_id)
        )
    """)
    con.execute("""
        INSERT INTO my_squad_new
        SELECT 0, player_id, purchase_price, is_captain, is_vice_captain, added_gameweek, notes
        FROM my_squad
    """)
    con.execute("DROP TABLE my_squad")
    con.execute("ALTER TABLE my_squad_new RENAME TO my_squad")
    logger.info("my_squad migration complete")


def _add_column_if_missing(con, table: str, column: str, definition: str) -> None:
    columns = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table],
        ).fetchall()
    }
    if column not in columns:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- ── Teams ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS teams (
    id              INTEGER PRIMARY KEY,
    name            VARCHAR NOT NULL,
    short_name      VARCHAR,
    strength        INTEGER,         -- FPL overall strength 1-5
    strength_attack_home    INTEGER,
    strength_attack_away    INTEGER,
    strength_defence_home   INTEGER,
    strength_defence_away   INTEGER,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Players (master) ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS players (
    id              INTEGER PRIMARY KEY,
    first_name      VARCHAR,
    second_name     VARCHAR,
    web_name        VARCHAR,
    team_id         INTEGER,
    position        VARCHAR,          -- GK DEF MID FWD
    now_cost        FLOAT,            -- £m
    status          VARCHAR,          -- a=available, d=doubtful, i=injured, s=suspended, u=unavailable
    chance_of_playing_next_round INTEGER,
    chance_of_playing_this_round INTEGER,
    total_points    INTEGER,
    form            FLOAT,
    points_per_game FLOAT,
    selected_by_percent FLOAT,        -- ownership %
    minutes         INTEGER,
    goals_scored    INTEGER,
    assists         INTEGER,
    clean_sheets    INTEGER,
    goals_conceded  INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    saves           INTEGER,
    bonus           INTEGER,
    bps             INTEGER,
    influence       FLOAT,
    creativity      FLOAT,
    threat          FLOAT,
    ict_index       FLOAT,
    expected_goals  FLOAT,
    expected_assists FLOAT,
    expected_goal_involvements FLOAT,
    expected_goals_conceded FLOAT,
    defensive_contribution FLOAT DEFAULT 0,
    transfers_in_event  INTEGER,
    transfers_out_event INTEGER,
    value_form      FLOAT,
    value_season    FLOAT,
    news            VARCHAR,
    news_added      VARCHAR,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Gameweeks ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gameweeks (
    id              INTEGER PRIMARY KEY,
    name            VARCHAR,
    deadline_time   TIMESTAMP,
    average_entry_score INTEGER,
    highest_score   INTEGER,
    is_finished     BOOLEAN,
    is_current      BOOLEAN,
    is_next         BOOLEAN,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Fixtures ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fixtures (
    id              INTEGER PRIMARY KEY,
    gameweek_id     INTEGER,
    team_h          INTEGER,
    team_a          INTEGER,
    team_h_score    INTEGER,
    team_a_score    INTEGER,
    team_h_difficulty INTEGER,   -- FDR 1-5
    team_a_difficulty INTEGER,
    kickoff_time    TIMESTAMP,
    finished        BOOLEAN,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Player gameweek history ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_gw_history (
    player_id       INTEGER,
    gameweek_id     INTEGER,
    total_points    INTEGER,
    minutes         INTEGER,
    goals_scored    INTEGER,
    assists         INTEGER,
    clean_sheets    INTEGER,
    goals_conceded  INTEGER,
    own_goals       INTEGER,
    penalties_saved INTEGER,
    penalties_missed INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    saves           INTEGER,
    bonus           INTEGER,
    bps             INTEGER,
    influence       FLOAT,
    creativity      FLOAT,
    threat          FLOAT,
    ict_index       FLOAT,
    expected_goals  FLOAT,
    expected_assists FLOAT,
    expected_goal_involvements FLOAT,
    expected_goals_conceded FLOAT,
    value           FLOAT,           -- price during that GW
    selected        FLOAT,           -- ownership %
    was_home        BOOLEAN,
    round           INTEGER,
    defensive_contribution FLOAT DEFAULT 0,
    PRIMARY KEY (player_id, gameweek_id)
);

-- ── Expected points predictions ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS expected_points (
    player_id       INTEGER,
    gameweek_id     INTEGER,
    xpts            FLOAT,           -- predicted points for 1 GW
    xpts_3gw        FLOAT,           -- sum of next 3 GW predictions
    xpts_5gw        FLOAT,           -- sum of next 5 GW predictions
    model_version   VARCHAR,
    confidence      FLOAT,           -- 0-1
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, gameweek_id)
);

-- ── My squad ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS my_squad (
    player_id       INTEGER PRIMARY KEY,
    purchase_price  FLOAT,
    is_captain      BOOLEAN DEFAULT FALSE,
    is_vice_captain BOOLEAN DEFAULT FALSE,
    added_gameweek  INTEGER,
    notes           VARCHAR
);

-- ── Transfer log ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transfer_log (
    id              INTEGER PRIMARY KEY,
    gameweek_id     INTEGER,
    player_out_id   INTEGER,
    player_in_id    INTEGER,
    hit_taken       BOOLEAN,
    expected_gain   FLOAT,
    actual_gain     FLOAT,           -- filled after the GW
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── Custom metrics cache ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_metrics (
    player_id       INTEGER PRIMARY KEY,
    pts_per_90          FLOAT,
    pts_per_million     FLOAT,
    form_score          FLOAT,       -- weighted form last N GWs
    fixture_score_3gw   FLOAT,       -- avg difficulty next 3 GWs (lower = easier)
    fixture_score_5gw   FLOAT,
    consistency         FLOAT,       -- 1 - coefficient of variation
    home_away_delta     FLOAT,       -- pts home minus pts away
    ownership_inefficiency FLOAT,    -- high pts + low ownership
    bonus_rate          FLOAT,       -- bonus pts per 90
    xgi_per_90          FLOAT,       -- xG + xA per 90
    rotation_risk       FLOAT,       -- 0-1 (1 = high risk of being benched)
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telegram_users (
    user_id     BIGINT PRIMARY KEY,
    chat_id     BIGINT,
    fpl_manager_id INTEGER,
    deadline_reminder BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

MIGRATION_SQL = [
    "ALTER TABLE players ADD COLUMN IF NOT EXISTS news VARCHAR",
    "ALTER TABLE players ADD COLUMN IF NOT EXISTS news_added VARCHAR",
]
