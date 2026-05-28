"""
FPL official API client — async, with retry logic and full endpoint coverage.
Reference: https://github.com/jeppe-smith/fpl-api
"""
import asyncio
from typing import Any, Optional
import httpx
from loguru import logger

from config.settings import settings

BASE = settings.fpl_base_url
TIMEOUT = settings.fpl_request_timeout


class FPLClient:
    """Async HTTP client for the FPL API. Use as a context manager."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=BASE,
            timeout=TIMEOUT,
            headers={"User-Agent": "FBL-Analytics/1.0"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_):
        await self._client.aclose()

    async def _get(self, path: str, retries: int = 3) -> Any:
        for attempt in range(retries):
            try:
                r = await self._client.get(path)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                logger.warning("HTTP {} on {} (attempt {})", e.response.status_code, path, attempt + 1)
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
            except httpx.RequestError as e:
                logger.warning("Request error on {}: {} (attempt {})", path, e, attempt + 1)
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    # ── Core endpoints ────────────────────────────────────────────────────

    async def get_bootstrap(self) -> dict:
        """Main bootstrap — players, teams, gameweeks, element types."""
        return await self._get("/bootstrap-static/")

    async def get_fixtures(self, gameweek: Optional[int] = None) -> list:
        """All fixtures, or filtered by GW."""
        path = "/fixtures/"
        if gameweek:
            path += f"?event={gameweek}"
        return await self._get(path)

    async def get_player_summary(self, player_id: int) -> dict:
        """Per-player detail: history, fixtures, history_past."""
        return await self._get(f"/element-summary/{player_id}/")

    async def get_gameweek_live(self, gameweek: int) -> dict:
        """Live points data for a gameweek."""
        return await self._get(f"/event/{gameweek}/live/")

    async def get_entry(self, team_id: int) -> dict:
        """A manager's team info."""
        return await self._get(f"/entry/{team_id}/")

    async def get_entry_picks(self, team_id: int, gameweek: int) -> dict:
        """A manager's picks for a specific GW."""
        return await self._get(f"/entry/{team_id}/event/{gameweek}/picks/")

    async def get_entry_transfers(self, team_id: int) -> list:
        """Transfer history for a manager."""
        return await self._get(f"/entry/{team_id}/transfers/")

    async def get_dream_team(self, gameweek: int) -> dict:
        return await self._get(f"/dream-team/{gameweek}/")

    # ── Batch helpers ─────────────────────────────────────────────────────

    async def get_all_player_summaries(self, player_ids: list[int]) -> dict[int, dict]:
        """Fetch summaries for many players concurrently (rate-limited)."""
        sem = asyncio.Semaphore(10)

        async def fetch_one(pid: int) -> tuple[int, dict]:
            async with sem:
                data = await self.get_player_summary(pid)
                return pid, data

        tasks = [fetch_one(pid) for pid in player_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = {}
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Failed player fetch: {}", r)
            else:
                pid, data = r
                out[pid] = data
        return out
