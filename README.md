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

Polling mode means you do **not** need webhooks, public HTTPS, reverse proxies, or port forwarding. Works perfectly behind double NAT.

### 1. SSH into your NAS and create the data folders

The bind-mount paths must exist **before** you start the container, otherwise Docker will throw a "does not exist" error.

```bash
mkdir -p /volume1/docker/fbl-bot/data/models
mkdir -p /volume1/docker/fbl-bot/data/raw
mkdir -p /volume1/docker/fbl-bot/data/processed
mkdir -p /volume1/docker/fbl-bot/data/kaggle
mkdir -p /volume1/docker/fbl-bot/data/cache
mkdir -p /volume1/docker/fbl-bot/logs
```

### 2. Copy the project to your NAS

```bash
git clone https://github.com/BartAulbers/fbl_fplbot.git /volume1/docker/fbl-bot
cd /volume1/docker/fbl-bot
```

### 3. Create your `.env` file

```bash
cp .env.example .env
vi .env   # or nano .env
```

Set at minimum:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ALLOWED_USER_IDS=   # leave empty to allow all, or add your user ID
```

### 4. Update `docker-compose.yml` with absolute NAS paths

Edit the volumes section to use absolute paths:

```yaml
volumes:
  - /volume1/docker/fbl-bot/data:/app/data
  - /volume1/docker/fbl-bot/logs:/app/logs
```

### 5. Build and start

```bash
docker compose up -d --build
```

### 6. Check logs

```bash
docker compose logs -f fbl-bot
# Or tail the activity log directly:
tail -f /volume1/docker/fbl-bot/logs/activity.log
```

### Synology notes

- Polling mode only — no inbound ports, no port forwarding needed.
- Set restart policy to `unless-stopped` (already in compose) so the bot survives NAS reboots.
- DuckDB, trained models, user state, and log files all live in the mounted `data/` and `logs/` folders and persist across container rebuilds.
- The 7 AM data refresh and 9 AM GW recap run automatically via the built-in scheduler.

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
