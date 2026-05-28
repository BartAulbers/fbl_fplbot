# FBL Telegram Bot

A Fantasy Premier League analytics bot that pulls live FPL data into DuckDB and delivers squad imports, transfer suggestions, player news, fixture swings, and deadline reminders through Telegram.

## Project Overview

- `src/` contains the FPL API client, data pipeline, analytics, xPts model, and transfer engine.
- `bot/` contains the Telegram bot, handlers, formatting, user state, and deadline scheduler.
- `data/` stores the DuckDB database, trained models, and bot state.
- `app/` and `dashboard/` are deprecated legacy frontend code and can be removed once no longer needed.

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and set at least:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ALLOWED_USER_IDS=
```

Leave `TELEGRAM_ALLOWED_USER_IDS` empty to allow all users, or set a comma-separated allowlist.

### 3. Pull latest FPL data

```bash
python scripts/run_pipeline.py
```

This initializes DuckDB and fetches the latest bootstrap, fixtures, and player history.

### 4. Start the Telegram bot

```bash
python -m bot.main
```

The bot runs in polling mode, which is ideal for home servers and Synology NAS setups behind double NAT.

## Bot Features

Use `/start` to open the main menu.

- Import Team - save your FPL manager ID and import your squad.
- Transfer Suggestions - refresh data and get ranked transfer advice.
- Player News - check availability updates for your squad or search any player.
- Fixture Swings - identify teams whose fixtures are improving or worsening.
- Deadline Reminder - enable 1-hour pre-deadline notifications.

## Docker / Synology NAS Setup

Polling mode means you do not need webhooks, public HTTPS, reverse proxies, or port forwarding.

### 1. Build and run with Docker Compose

```bash
docker compose up -d --build
```

### 2. Example compose setup

`docker-compose.yml` is included and mounts `./data` into the container so DuckDB, models, and user state persist across restarts.

### 3. Synology notes for double-NAT environments

- Use polling mode only.
- Store your bot token in `.env` on the NAS.
- Mount the `data/` directory to persistent storage.
- Set restart policy to `unless-stopped` so the bot resumes after NAS reboots.
- No inbound ports are required for Telegram polling.

## Common Commands

```bash
python scripts/run_pipeline.py
python -m bot.main
docker compose up -d --build
docker compose logs -f fbl-bot
```

## Notes

- Transfer suggestions and fixture swings refresh FPL data before responding.
- Player news is pulled from the `players.news` field stored in DuckDB.
- Deadline reminder preferences are stored in `data/user_state.json`.
