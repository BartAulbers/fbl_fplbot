"""
Central configuration. Override via .env file or environment variables.
"""
from pydantic_settings import BaseSettings
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────────
    db_path: str = str(ROOT_DIR / "data" / "fpl.duckdb")
    raw_data_dir: str = str(ROOT_DIR / "data" / "raw")
    kaggle_data_dir: str = str(ROOT_DIR / "data" / "kaggle")
    model_dir: str = str(ROOT_DIR / "data" / "models")

    # ── FPL API ────────────────────────────────────────────────────────────
    fpl_base_url: str = "https://fantasy.premierleague.com/api"
    fpl_request_timeout: int = 30

    # ── Squad constraints ──────────────────────────────────────────────────
    squad_budget: float = 100.0          # £m
    squad_gk: int = 2
    squad_def: int = 5
    squad_mid: int = 5
    squad_fwd: int = 3
    max_players_per_team: int = 3
    starting_xi_size: int = 11

    # ── Model parameters ───────────────────────────────────────────────────
    form_window: int = 5                 # last N gameweeks for form
    fixture_lookahead: int = 5           # GWs ahead for fixture scoring
    min_minutes_threshold: int = 45      # filter near-guaranteed starters

    # ── Strategy thresholds ────────────────────────────────────────────────
    hit_threshold_pts: float = 5.0       # min expected gain to justify -4 hit
    differential_ownership_cap: float = 10.0   # % below which = differential
    churn_prevention_min_gw_hold: int = 5      # min GWs to hold before transfer

    # ── Risk appetite 0.0 (safe) → 1.0 (aggressive differentials) ─────────
    risk_appetite: float = 0.5

    # ── API server ─────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # ── Telegram bot ────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
