"""
Automated scheduler — runs the FPL data pipeline and metric refresh
on a schedule. No Docker needed; uses APScheduler in-process.

Usage: python scripts/scheduler.py

Schedule:
- Full pipeline: every day at 08:00 and 20:00
- Metrics refresh: after each pipeline run
- Model predictions: daily at 01:00 Europe/London
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from src.data.pipeline import run_full_pipeline
from src.database.db import get_connection
from app.dependencies import (
    load_players, load_teams, load_fixtures, load_history, get_current_gw
)
from src.analytics.analytics import compute_player_metrics
import pandas as pd


def run_pipeline_job():
    logger.info("Scheduler: Running pipeline...")
    asyncio.run(run_full_pipeline(include_player_history=True))
    _refresh_metrics()


def _refresh_metrics():
    logger.info("Scheduler: Refreshing metrics...")
    try:
        players = load_players()
        history = load_history()
        fixtures = load_fixtures()
        current_gw = get_current_gw()
        metrics = compute_player_metrics(players, history, fixtures, current_gw)
        metrics["updated_at"] = pd.Timestamp.now()
        con = get_connection()
        con.execute("DELETE FROM player_metrics")
        con.execute("INSERT INTO player_metrics SELECT * FROM metrics")
        con.close()
        logger.success("Metrics refreshed for {} players", len(metrics))
    except Exception as e:
        logger.error("Metrics refresh failed: {}", e)


def run_predictions_job():
    logger.info("Scheduler: Running xPts predictions...")
    try:
        from src.models.expected_points import predict_multi_gw, train
        players = load_players()
        history = load_history()
        fixtures = load_fixtures()
        teams = load_teams()
        current_gw = get_current_gw()

        # Retrain weekly
        try:
            train(history, fixtures, players, teams)
        except Exception as e:
            logger.warning("Model training skipped: {}", e)

        xpts = predict_multi_gw(history, fixtures, players, teams, current_gw)
        xpts["model_version"] = "v1"
        xpts["confidence"] = 0.7
        xpts["created_at"] = pd.Timestamp.now()

        # Map columns for DB insert
        xpts = xpts.rename(columns={
            "xpts": "xpts",
            "xpts_3gw": "xpts_3gw",
            "xpts_5gw": "xpts_5gw",
        })
        xpts["gameweek_id"] = current_gw

        con = get_connection()
        con.execute(f"DELETE FROM expected_points WHERE gameweek_id = {current_gw}")
        con.execute("INSERT INTO expected_points SELECT * FROM xpts")
        con.close()
        logger.success("Predictions stored for GW{}", current_gw)

    except Exception as e:
        logger.error("Prediction job failed: {}", e)


def load_teams():
    con = get_connection(read_only=True)
    df = con.execute("SELECT * FROM teams").df()
    con.close()
    return df


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Europe/London")

    # Pipeline runs twice daily
    scheduler.add_job(run_pipeline_job, CronTrigger(hour="8,20", minute=0), id="pipeline")

    # Predictions daily at 01:00
    scheduler.add_job(run_predictions_job, CronTrigger(hour=1, minute=0), id="predictions")

    logger.info("Scheduler started. Jobs: pipeline (8:00, 20:00), predictions (01:00 Europe/London)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
