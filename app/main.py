"""
FastAPI main application.
All analytics modules are wired together here.
No Docker needed — run with: uvicorn app.main:app --reload
"""
import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.database.db import init_db, get_connection
from src.data.pipeline import run_full_pipeline
from src.optimization.squad_optimizer import optimize_squad, format_squad, SquadPlayer
from src.strategy.transfer_engine import recommend_transfers, TransferPlan
from src.analytics.analytics import (
    compute_player_metrics,
    analyse_fixture_runs,
    fixture_swing_alerts,
    find_differentials,
    pick_captain,
    correlation_analysis,
)
from src.models.expected_points import (
    predict_multi_gw,
    feature_importance,
    explain_player,
)
from app.dependencies import (
    load_players, load_teams, load_fixtures, load_history,
    load_my_squad, get_current_gw, load_xpts, load_metrics,
)
from app.schemas.schemas import (
    OptimizeRequest, SquadResultOut, SquadPlayerOut,
    TransferRequest, TransferPlanOut, TransferSuggestionOut, TransferPlayerDict,
    FixtureRunRow, DifferentialRow, CaptainRow, PipelineStatus,
)

# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("FBL Analytics API started. DB at {}", settings.db_path)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="FBL Analytics — FPL Decision Engine",
    description="Data-driven Fantasy Premier League analytics. No more impulse transfers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════
# DATA PIPELINE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/pipeline/refresh", response_model=PipelineStatus, tags=["Pipeline"])
async def refresh_data(background_tasks: BackgroundTasks, history: bool = True):
    """Trigger a full FPL API data refresh in the background."""
    background_tasks.add_task(run_full_pipeline, include_player_history=history)
    return PipelineStatus(
        status="started",
        message="Data refresh running in background. Check /pipeline/status.",
        current_gw=None,
    )


@app.get("/pipeline/status", response_model=PipelineStatus, tags=["Pipeline"])
async def pipeline_status():
    """Check database status and current gameweek."""
    try:
        gw = get_current_gw()
        con = get_connection(read_only=True)
        n_players = con.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        n_history = con.execute("SELECT COUNT(*) FROM player_gw_history").fetchone()[0]
        con.close()
        return PipelineStatus(
            status="ok",
            message=f"DB healthy — {n_players} players, {n_history} GW records",
            current_gw=gw,
        )
    except Exception as e:
        return PipelineStatus(status="error", message=str(e), current_gw=None)


# ═══════════════════════════════════════════════════════════════════════
# SQUAD OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

@app.post("/squad/optimize", tags=["Squad"])
async def optimize(req: OptimizeRequest):
    """
    Solve the FPL squad selection ILP.
    Returns optimal 15-player squad + starting XI + captain.
    """
    try:
        players = load_players()
        current_gw = get_current_gw()
        xpts = load_xpts(current_gw)
        metrics = load_metrics()

        # Merge xpts and metrics into players
        df = players.rename(columns={"id": "player_id"})
        if not xpts.empty:
            df = df.merge(xpts[["player_id", "xpts", "xpts_3gw", "xpts_5gw"]], on="player_id", how="left")
        else:
            # Fallback: use form as proxy
            df["xpts"] = df["form"] * 1.5
            df["xpts_3gw"] = df["xpts"] * 3
            df["xpts_5gw"] = df["xpts"] * 5

        if not metrics.empty:
            df = df.merge(metrics[["player_id", "consistency"]], on="player_id", how="left")
        else:
            df["consistency"] = 0.5

        df = df.rename(columns={"selected_by_percent": "ownership"})
        df = df.fillna(0)

        result = optimize_squad(
            players_df=df,
            budget=req.budget,
            risk_appetite=req.risk_appetite,
            horizon=req.horizon,
            locked_player_ids=req.locked_player_ids,
            excluded_player_ids=req.excluded_player_ids,
        )

        squad_out = [
            SquadPlayerOut(
                player_id=p.player_id,
                web_name=p.web_name,
                position=p.position,
                cost=p.cost,
                xpts=p.xpts,
                xpts_3gw=p.xpts_3gw,
                ownership=p.ownership,
                consistency=p.consistency,
                is_starting=p.is_starting,
                is_captain=p.is_captain,
                is_vice=p.is_vice,
                bench_order=p.bench_order,
            )
            for p in result.squad
        ]

        return SquadResultOut(
            squad=squad_out,
            total_cost=result.total_cost,
            projected_pts_gw=result.projected_pts_gw,
            projected_pts_3gw=result.projected_pts_3gw,
            budget_remaining=result.budget_remaining,
            solver_status=result.solver_status,
        )
    except Exception as e:
        logger.exception("Squad optimization failed")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# TRANSFERS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/transfers/recommend", tags=["Transfers"])
async def transfer_recommendations(req: TransferRequest):
    """
    Generate ranked transfer recommendations for your current squad.
    Includes -4 hit analysis and churn prevention.
    """
    try:
        my_squad = load_my_squad()
        if my_squad.empty:
            raise HTTPException(400, "No squad found. Add players to my_squad first.")

        all_players = load_players()
        current_gw = get_current_gw()
        xpts = load_xpts(current_gw)
        metrics = load_metrics()

        # Build enriched player frame
        df = all_players.rename(columns={"id": "player_id"})
        if not xpts.empty:
            df = df.merge(xpts[["player_id", "xpts", "xpts_3gw"]], on="player_id", how="left")
        else:
            df["xpts"] = df["form"] * 1.5
            df["xpts_3gw"] = df["xpts"] * 3
        if not metrics.empty:
            df = df.merge(
                metrics[["player_id", "consistency", "fixture_score_3gw", "rotation_risk"]],
                on="player_id", how="left",
            )
            df = df.rename(columns={"fixture_score_3gw": "fdr_avg_3gw"})
        # Ensure columns always exist with sensible defaults
        if "fdr_avg_3gw" not in df.columns:
            df["fdr_avg_3gw"] = 3.0
        if "consistency" not in df.columns:
            df["consistency"] = 0.5
        if "rotation_risk" not in df.columns:
            df["rotation_risk"] = 0.3
        df = df.fillna({"fdr_avg_3gw": 3.0, "consistency": 0.5, "rotation_risk": 0.3}).fillna(0)

        # Enrich my_squad too
        my_squad_enr = my_squad.merge(
            df[["player_id", "xpts", "xpts_3gw", "fdr_avg_3gw", "consistency"]],
            on="player_id", how="left",
        ).fillna({"xpts": 0, "xpts_3gw": 0, "fdr_avg_3gw": 3.0, "consistency": 0.5})
        if "added_gameweek" not in my_squad_enr.columns:
            my_squad_enr["added_gameweek"] = 1
        my_squad_enr["added_gameweek"] = my_squad_enr["added_gameweek"].fillna(1).astype(int)

        plan = recommend_transfers(
            my_squad_df=my_squad_enr,
            all_players_df=df,
            free_transfers=req.free_transfers,
            current_gw=current_gw,
            risk_appetite=req.risk_appetite,
            max_suggestions=req.max_suggestions,
        )

        return TransferPlanOut(
            suggestions=[
                TransferSuggestionOut(
                    player_out=TransferPlayerDict(**s.player_out),
                    player_in=TransferPlayerDict(**s.player_in),
                    expected_gain_1gw=s.expected_gain_1gw,
                    expected_gain_3gw=s.expected_gain_3gw,
                    hit_required=s.hit_required,
                    net_gain=s.net_gain,
                    reasoning=s.reasoning,
                    confidence=s.confidence,
                )
                for s in plan.suggestions
            ],
            free_transfers_available=plan.free_transfers_available,
            current_gw=plan.current_gw,
            recommendation=plan.recommendation,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transfer recommendation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# EXPECTED POINTS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/model/train", tags=["Model"])
async def train_model(background_tasks: BackgroundTasks):
    """Trigger model training in the background."""
    from src.models.expected_points import train as train_xpts

    async def _train():
        players = load_players()
        history = load_history()
        fixtures = load_fixtures()
        teams = load_teams()
        metrics = train_xpts(history, fixtures, players, teams)
        logger.info("Training complete: {}", metrics)

    background_tasks.add_task(_train)
    return {"status": "training_started", "message": "Model training running in background."}


@app.get("/model/predict", tags=["Model"])
async def predict_xpts(gws: int = 5):
    """
    Run multi-GW xPts predictions and store to DB.
    Returns top 50 players by xpts for next GW.
    """
    try:
        players = load_players()
        history = load_history()
        fixtures = load_fixtures()
        teams = load_teams()
        current_gw = get_current_gw()

        xpts_df = predict_multi_gw(history, fixtures, players, teams, current_gw, n_gws=gws)

        # Persist to DB
        players_slim = players[["id", "web_name", "position", "now_cost",
                                  "selected_by_percent"]].rename(columns={"id": "player_id"})
        result = xpts_df.merge(players_slim, on="player_id", how="left")
        result = result.sort_values("xpts", ascending=False).head(50)

        return result.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(400, "Model not trained yet. POST /model/train first.")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, str(e))


@app.get("/model/explain/{player_id}", tags=["Model"])
async def explain(player_id: int):
    """SHAP explanation for a single player's expected points."""
    try:
        players = load_players()
        history = load_history()
        fixtures = load_fixtures()
        teams = load_teams()
        current_gw = get_current_gw()
        return explain_player(history, fixtures, players, teams, player_id, current_gw)
    except FileNotFoundError:
        raise HTTPException(400, "Model not trained yet.")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/model/feature_importance", tags=["Model"])
async def feat_importance(top_n: int = 20):
    """Return top N feature importances from the trained model."""
    try:
        return feature_importance(top_n).to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(400, "Model not trained yet.")


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/fixtures/runs", tags=["Fixtures"])
async def fixture_runs(n_gws: int = 5):
    """Fixture difficulty runs for all teams over next N GWs."""
    teams = load_teams()
    fixtures = load_fixtures()
    current_gw = get_current_gw()
    result = analyse_fixture_runs(teams, fixtures, current_gw, n_gws)
    return result.to_dict(orient="records")


@app.get("/fixtures/swings", tags=["Fixtures"])
async def fixture_swings():
    """Detect fixture swing alerts (easy→hard or hard→easy)."""
    teams = load_teams()
    fixtures = load_fixtures()
    current_gw = get_current_gw()
    return fixture_swing_alerts(teams, fixtures, current_gw)


# ═══════════════════════════════════════════════════════════════════════
# DIFFERENTIALS & CAPTAINCY
# ═══════════════════════════════════════════════════════════════════════

@app.get("/differentials", tags=["Analytics"])
async def differentials(ownership_cap: float = 10.0, top_n: int = 5):
    """Find low-owned, high-xpts differential picks per position."""
    players = load_players()
    current_gw = get_current_gw()
    xpts = load_xpts(current_gw)
    metrics = load_metrics()

    if xpts.empty or metrics.empty:
        raise HTTPException(400, "Run /model/predict and /analytics/metrics first.")

    result = find_differentials(players, metrics, xpts, ownership_cap, top_n)
    return result.to_dict(orient="records")


@app.get("/captain/recommend", tags=["Analytics"])
async def captain_recommend(risk_appetite: float = 0.5):
    """Rank captain candidates from your current squad."""
    my_squad = load_my_squad()
    if my_squad.empty:
        raise HTTPException(400, "No squad found.")
    current_gw = get_current_gw()
    xpts = load_xpts(current_gw)
    metrics = load_metrics()
    result = pick_captain(my_squad, xpts, metrics, risk_appetite)
    return result.to_dict(orient="records")


# ═══════════════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/analytics/metrics/refresh", tags=["Analytics"])
async def refresh_metrics():
    """Recompute and store all custom player metrics."""
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
    return {"status": "ok", "rows_written": len(metrics)}


@app.get("/analytics/correlations", tags=["Analytics"])
async def correlations():
    """Pearson correlation of all features with actual points (for validation)."""
    history = load_history()
    return correlation_analysis(history).to_dict(orient="records")


@app.get("/players/search", tags=["Players"])
async def search_players(
    name: Optional[str] = None,
    position: Optional[str] = None,
    team_id: Optional[int] = None,
    min_xpts: Optional[float] = None,
    max_cost: Optional[float] = None,
):
    """Search players with optional filters."""
    players = load_players()
    df = players.rename(columns={"id": "player_id"})
    current_gw = get_current_gw()
    xpts = load_xpts(current_gw)
    if not xpts.empty:
        df = df.merge(xpts[["player_id", "xpts", "xpts_3gw"]], on="player_id", how="left")
    else:
        df["xpts"] = df["form"] * 1.5
        df["xpts_3gw"] = df["xpts"] * 3

    if name:
        df = df[df["web_name"].str.contains(name, case=False, na=False)]
    if position:
        df = df[df["position"] == position.upper()]
    if team_id:
        df = df[df["team_id"] == team_id]
    if min_xpts:
        df = df[df["xpts"] >= min_xpts]
    if max_cost:
        df = df[df["now_cost"] <= max_cost]

    return df.sort_values("xpts", ascending=False).head(50).to_dict(orient="records")


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=settings.api_reload)
