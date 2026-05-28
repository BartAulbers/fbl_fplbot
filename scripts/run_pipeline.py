from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.data.pipeline import run_full_pipeline
from src.database.db import init_db

if __name__ == "__main__":
    logger.info("Initialising database...")
    init_db()
    logger.info("Running FPL data pipeline...")
    asyncio.run(run_full_pipeline(include_player_history=True))
    logger.success("Done! You can now start the Telegram bot with: python -m bot.main")
