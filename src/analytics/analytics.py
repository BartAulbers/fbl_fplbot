"""
Analytics: custom metrics, fixture analysis, differentials, captaincy.
All computations run on DataFrames (no DB coupling) for testability.
"""
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


# ═══════════════════════════════════════════════════════════════════════
# 1. CUSTOM METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_player_metrics(
    players: pd.DataFrame,
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    current_gw: int,
    teams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the player_metrics table with all custom KPIs.
    Returns one row per player with computed metrics.
    """
    df = players[["id", "position", "now_cost", "selected_by_percent",
                   "minutes", "total_points", "team_id"]].copy()
    df = df.rename(columns={"id": "player_id"})

    gw_played = df["minutes"].clip(lower=1) / 90.0

    # Points per 90 minutes
    df["pts_per_90"] = df["total_points"] / gw_played

    # Points per £million
    df["pts_per_million"] = df["total_points"] / (df["now_cost"] + 0.1)

    # ── Form score (last N GWs, exponentially weighted) ──────────────────
    recent = history[history["gameweek_id"] >= current_gw - settings.form_window].copy()
    form = (
        recent.groupby("player_id")["total_points"]
        .apply(lambda x: np.average(x, weights=np.exp(np.arange(len(x)))))
        .reset_index()
        .rename(columns={"total_points": "form_score"})
    )
    df = df.merge(form, on="player_id", how="left")

    # ── Consistency (1 - CV) ─────────────────────────────────────────────
    def cv_scalar(series):
        if len(series) < 3:
            return 0.5
        c = series.std() / (series.mean() + 1e-9)
        return float(max(0, 1 - c))

    cons = (
        history.groupby("player_id")["total_points"]
        .apply(cv_scalar)
        .reset_index()
        .rename(columns={"total_points": "consistency"})
    )
    df = df.merge(cons, on="player_id", how="left")

    # ── Home/away delta ───────────────────────────────────────────────────
    home_pts = history[history["was_home"]].groupby("player_id")["total_points"].mean()
    away_pts = history[~history["was_home"]].groupby("player_id")["total_points"].mean()
    ha = (home_pts - away_pts).reset_index().rename(columns={"total_points": "home_away_delta"})
    df = df.merge(ha, on="player_id", how="left")

    # ── Fixture score (next 3 and 5 GWs) ─────────────────────────────────
    dynamic_fdr = compute_dynamic_fdr(teams, fixtures, current_gw) if teams is not None else {}
    if dynamic_fdr:
        logger.debug("Using dynamic FDR (form-based) for player metrics")
    fix_scores = _compute_fixture_scores(df["player_id"], players, fixtures, current_gw, dynamic_fdr)
    df = df.merge(fix_scores, on="player_id", how="left")

    # ── Ownership inefficiency: high pts, low ownership ───────────────────
    # z-score both, ownership inefficiency = pts_z - ownership_z
    pts_z = (df["total_points"] - df["total_points"].mean()) / (df["total_points"].std() + 1e-9)
    own_z = (df["selected_by_percent"] - df["selected_by_percent"].mean()) / (df["selected_by_percent"].std() + 1e-9)
    df["ownership_inefficiency"] = pts_z - own_z

    # ── Bonus rate per 90 ────────────────────────────────────────────────
    bonus_total = history.groupby("player_id")["bonus"].sum().reset_index()
    df = df.merge(bonus_total.rename(columns={"bonus": "_bonus_total"}), on="player_id", how="left")
    df["bonus_rate"] = df["_bonus_total"].fillna(0) / gw_played

    # ── xGI per 90 ───────────────────────────────────────────────────────
    xgi = players[["id", "expected_goal_involvements"]].rename(columns={"id": "player_id"})
    df = df.merge(xgi, on="player_id", how="left")
    df["xgi_per_90"] = df["expected_goal_involvements"].fillna(0) / gw_played

    # ── Rotation risk (heuristic: low avg minutes = rotation risk) ───────
    avg_mins = history.groupby("player_id")["minutes"].mean().reset_index().rename(
        columns={"minutes": "avg_minutes"}
    )
    df = df.merge(avg_mins, on="player_id", how="left")
    df["rotation_risk"] = 1 - (df["avg_minutes"].fillna(0) / 90).clip(0, 1)

    # Clean up temp columns
    df = df.drop(columns=["_bonus_total", "expected_goal_involvements", "avg_minutes",
                           "total_points", "minutes"], errors="ignore")

    df[df.select_dtypes(include=[np.number]).columns] = (
        df.select_dtypes(include=[np.number]).fillna(0)
    )
    return df


def _compute_fixture_scores(
    player_ids: pd.Series,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    current_gw: int,
    dynamic_fdr: dict | None = None,
) -> pd.DataFrame:
    """Average FDR for next 3 and 5 GWs per player. Uses dynamic FDR when available."""
    rows = []
    team_map = players.set_index("id")["team_id"].to_dict()
    dyn = dynamic_fdr or {}

    for pid in player_ids:
        team_id = team_map.get(pid)
        if team_id is None:
            rows.append({"player_id": pid, "fixture_score_3gw": 3.0, "fixture_score_5gw": 3.0})
            continue

        team_fix = fixtures[
            ((fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id)) &
            (fixtures["gameweek_id"] >= current_gw)
        ].sort_values("gameweek_id")

        fdrs = [_get_fdr(f, team_id, dyn) for _, f in team_fix.iterrows()]

        rows.append({
            "player_id": pid,
            "fixture_score_3gw": float(np.mean(fdrs[:3])) if fdrs else 3.0,
            "fixture_score_5gw": float(np.mean(fdrs[:5])) if fdrs else 3.0,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 2. DYNAMIC FDR ENGINE
# ═══════════════════════════════════════════════════════════════════════

def compute_dynamic_fdr(
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    current_gw: int,
    min_gws: int = 5,
    n_recent: int = 5,
) -> dict[int, dict[str, float]]:
    """
    Compute form-based FDR from recent results (goals scored/conceded, home/away).

    Returns {team_id: {"home": fdr, "away": fdr}} where:
      - "home" = how hard it is to face this team when THEY are at home
      - "away" = how hard it is to face this team when THEY are away

    Falls back to {} (caller uses official FDR) if fewer than min_gws played.

    Score = goals_scored_rate + 1/(goals_conceded_rate + 0.5)
    High score = dangerous opponent = high FDR.
    Normalised to 1–5 per the standard FPL scale.
    """
    if current_gw <= min_gws:
        return {}

    recent_start = max(1, current_gw - n_recent)
    done = fixtures[
        (fixtures["gameweek_id"] >= recent_start)
        & (fixtures["gameweek_id"] < current_gw)
        & (fixtures["finished"] == True)
    ].copy()

    if len(done) < 10:
        return {}

    league_h_scored = done["team_h_score"].mean() or 1.5
    league_a_scored = done["team_a_score"].mean() or 1.0

    raw: dict[int, dict[str, float]] = {}
    for _, team in teams.iterrows():
        tid = int(team["id"])
        home_games = done[done["team_h"] == tid]
        away_games = done[done["team_a"] == tid]

        h_scored   = home_games["team_h_score"].mean() if len(home_games) >= 2 else league_h_scored
        h_conceded = home_games["team_a_score"].mean() if len(home_games) >= 2 else league_a_scored
        a_scored   = away_games["team_a_score"].mean() if len(away_games) >= 2 else league_a_scored
        a_conceded = away_games["team_h_score"].mean() if len(away_games) >= 2 else league_h_scored

        # Higher = more dangerous opponent
        raw[tid] = {
            "home": h_scored + 1.0 / (h_conceded + 0.5),
            "away": a_scored + 1.0 / (a_conceded + 0.5),
        }

    def _norm(val: float, vals: list[float]) -> float:
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return 3.0
        return round(1.0 + (val - mn) / (mx - mn) * 4.0, 2)

    home_vals = [v["home"] for v in raw.values()]
    away_vals = [v["away"] for v in raw.values()]

    return {
        tid: {
            "home": _norm(s["home"], home_vals),
            "away": _norm(s["away"], away_vals),
        }
        for tid, s in raw.items()
    }


def _get_fdr(
    fixture: pd.Series,
    team_id: int,
    dynamic_fdr: dict[int, dict[str, float]],
) -> float:
    """
    Return the FDR for team_id in this fixture.
    If dynamic_fdr is available, use opponent's home/away strength.
    Falls back to official FDR stored in the fixture row.
    """
    if dynamic_fdr:
        if fixture["team_h"] == team_id:
            # team_id is home → opponent plays away
            opp = int(fixture["team_a"])
            return dynamic_fdr.get(opp, {}).get("away", 3.0)
        else:
            # team_id is away → opponent plays home
            opp = int(fixture["team_h"])
            return dynamic_fdr.get(opp, {}).get("home", 3.0)
    # Official FDR fallback
    if fixture["team_h"] == team_id:
        return float(fixture.get("team_h_difficulty") or 3)
    return float(fixture.get("team_a_difficulty") or 3)




def analyse_fixture_runs(
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    current_gw: int,
    n_gws: int = 5,
) -> pd.DataFrame:
    """
    Score fixture difficulty for all teams over the next N gameweeks.
    Lower score = easier run. Uses dynamic FDR when ≥5 GWs have been played.
    Returns DataFrame sorted best→worst fixture run.
    """
    dynamic_fdr = compute_dynamic_fdr(teams, fixtures, current_gw)
    if dynamic_fdr:
        logger.debug("analyse_fixture_runs: using dynamic FDR")

    rows = []
    upcoming = fixtures[
        (fixtures["gameweek_id"] >= current_gw) &
        (fixtures["gameweek_id"] < current_gw + n_gws)
    ]

    for _, team in teams.iterrows():
        tid = team["id"]
        team_fix = upcoming[
            (upcoming["team_h"] == tid) | (upcoming["team_a"] == tid)
        ].sort_values("gameweek_id")

        fdrs, opponents, home_flags = [], [], []
        for _, f in team_fix.iterrows():
            fdrs.append(_get_fdr(f, tid, dynamic_fdr))
            if f["team_h"] == tid:
                opp = teams[teams["id"] == f["team_a"]]["short_name"].values
                opponents.append(opp[0] if len(opp) else "?")
                home_flags.append("H")
            else:
                opp = teams[teams["id"] == f["team_h"]]["short_name"].values
                opponents.append(opp[0] if len(opp) else "?")
                home_flags.append("A")

        fixture_str = " ".join(
            f"{'(H)' if h == 'H' else '(A)'}{o}[{d:.2f}]"
            for o, d, h in zip(opponents, fdrs, home_flags)
        )

        rows.append({
            "team_id": tid,
            "team_name": team["short_name"],
            "avg_fdr": float(np.mean(fdrs)) if fdrs else 3.0,
            "min_fdr": float(min(fdrs)) if fdrs else 3.0,
            "max_fdr": float(max(fdrs)) if fdrs else 3.0,
            "n_fixtures": len(fdrs),
            "fixtures": fixture_str,
            "has_blank": int(len(fdrs) < n_gws),
            "has_double": int(len(fdrs) > n_gws),
            "fdr_source": "dynamic" if dynamic_fdr else "official",
        })

    return pd.DataFrame(rows).sort_values("avg_fdr")


def fixture_swing_alerts(
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    current_gw: int,
) -> list[dict]:
    """
    Detect teams whose fixtures get significantly easier/harder
    in the next 3 GWs vs the 3 after that. Uses dynamic FDR when available.
    """
    dynamic_fdr = compute_dynamic_fdr(teams, fixtures, current_gw)

    alerts = []
    for _, team in teams.iterrows():
        tid = team["id"]

        def avg_fdr_range(start: int, end: int) -> float:
            f = fixtures[
                ((fixtures["team_h"] == tid) | (fixtures["team_a"] == tid)) &
                (fixtures["gameweek_id"].between(start, end))
            ]
            fdrs = [_get_fdr(fix, tid, dynamic_fdr) for _, fix in f.iterrows()]
            return float(np.mean(fdrs)) if fdrs else 3.0

        near = avg_fdr_range(current_gw, current_gw + 2)
        far  = avg_fdr_range(current_gw + 3, current_gw + 5)
        delta = far - near

        if delta > 1.0:
            alerts.append({
                "team": team["short_name"],
                "alert_type": "FIXTURE_EASES",
                "message": f"Fixtures get easier from GW{current_gw + 3} (FDR {near:.2f}→{far:.2f})",
                "delta": round(delta, 2),
            })
        elif delta < -1.0:
            alerts.append({
                "team": team["short_name"],
                "alert_type": "FIXTURE_HARDENS",
                "message": f"Fixtures get harder from GW{current_gw + 3} (FDR {near:.2f}→{far:.2f})",
                "delta": round(delta, 2),
            })

    return sorted(alerts, key=lambda a: abs(a["delta"]), reverse=True)


# ═══════════════════════════════════════════════════════════════════════
# 3. DIFFERENTIAL FINDER
# ═══════════════════════════════════════════════════════════════════════

def find_differentials(
    players: pd.DataFrame,
    metrics: pd.DataFrame,
    xpts: pd.DataFrame,
    ownership_cap: float = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Find low-owned, high-expected-points differentials per position.

    Returns top_n differentials sorted by expected_pts_per_ownership_pct.
    """
    ownership_cap = ownership_cap or settings.differential_ownership_cap

    df = players[["id", "web_name", "position", "now_cost", "selected_by_percent", "team_id"]].copy()
    df = df.rename(columns={"id": "player_id"})
    df = df.merge(xpts[["player_id", "xpts", "xpts_3gw"]], on="player_id", how="left")
    df = df.merge(
        metrics[["player_id", "fixture_score_3gw", "consistency", "rotation_risk"]],
        on="player_id", how="left",
    )
    df = df.fillna(0)

    diffs = df[df["selected_by_percent"] < ownership_cap].copy()
    diffs = diffs[diffs["xpts"] > 0]

    # Differential score: high xpts, low ownership, good fixtures
    diffs["diff_score"] = (
        diffs["xpts_3gw"]
        * (1 - diffs["selected_by_percent"] / 100.0)
        * (1 - diffs["rotation_risk"])
        * (diffs["consistency"] + 0.5)
    )

    return (
        diffs.sort_values("diff_score", ascending=False)
        .groupby("position")
        .head(top_n)
        .sort_values(["position", "diff_score"], ascending=[True, False])
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. CAPTAINCY OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

def pick_captain(
    my_squad: pd.DataFrame,
    xpts: pd.DataFrame,
    metrics: pd.DataFrame,
    risk_appetite: float = 0.5,
) -> pd.DataFrame:
    """
    Rank captain candidates in my squad.

    Scoring:
    - Safe: highest xpts * consistency
    - Differential: xpts * (1 - ownership) for template-beating

    Returns squad sorted by captain_score desc.
    """
    df = my_squad.merge(xpts[["player_id", "xpts", "xpts_3gw"]], on="player_id", how="left")
    df = df.merge(
        metrics[["player_id", "consistency", "fixture_score_3gw", "rotation_risk"]],
        on="player_id", how="left",
    )
    df = df.fillna(0)

    safe_score = df["xpts"] * df["consistency"].clip(0.1, 1)
    diff_score = df["xpts"] * (1 - df["selected_by_percent"].fillna(50) / 100.0)

    df["captain_score"] = safe_score * (1 - risk_appetite) + diff_score * risk_appetite
    df["vc_score"] = df["captain_score"] * 0.9  # VC is a hedge

    return df.sort_values("captain_score", ascending=False)[
        ["player_id", "web_name", "position", "xpts", "consistency",
         "fixture_score_3gw", "selected_by_percent", "captain_score"]
    ]


# ═══════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE / CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def correlation_analysis(
    history: pd.DataFrame,
    numeric_features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute Pearson correlation of all numeric features with total_points.
    Useful for exploratory analysis and metric validation.
    """
    if numeric_features is None:
        numeric_features = [
            "minutes", "goals_scored", "assists", "clean_sheets",
            "bonus", "bps", "influence", "creativity", "threat",
            "ict_index", "expected_goals", "expected_assists",
            "expected_goal_involvements", "expected_goals_conceded",
            "saves", "value",
        ]

    available = [c for c in numeric_features if c in history.columns]
    corr = (
        history[available + ["total_points"]]
        .corr()["total_points"]
        .drop("total_points")
        .abs()
        .sort_values(ascending=False)
        .reset_index()
    )
    corr.columns = ["feature", "correlation_with_points"]
    return corr
