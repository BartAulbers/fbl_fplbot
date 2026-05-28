"""
Feature engineering for the expected points model.
Builds a rich feature matrix from raw player/fixture data.
"""
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings


def build_feature_matrix(
    player_history: pd.DataFrame,
    fixtures: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    target_gw: int,
) -> pd.DataFrame:
    """
    Builds feature matrix for all players for `target_gw`.

    Parameters
    ----------
    player_history : rows from player_gw_history
    fixtures       : rows from fixtures
    players        : rows from players (current season stats)
    teams          : rows from teams
    target_gw      : gameweek to predict

    Returns
    -------
    DataFrame with one row per player, feature columns + 'player_id'
    """
    form_w = settings.form_window
    history = player_history.copy()
    history = history.sort_values(["player_id", "gameweek_id"])

    # ── 1. Form features (last N GWs weighted) ────────────────────────────
    weights = np.array([0.5 ** i for i in range(form_w)])[::-1]
    weights /= weights.sum()

    def weighted_form(grp: pd.DataFrame) -> pd.Series:
        recent = grp.tail(form_w)
        n = len(recent)
        w = weights[-n:] / weights[-n:].sum()
        return pd.Series({
            # ── Attacking form ────────────────────────────────────────────
            "form_pts":     np.dot(recent["total_points"].values, w),
            "form_minutes": np.dot(recent["minutes"].values, w),
            "form_goals":   np.dot(recent["goals_scored"].values, w),
            "form_assists": np.dot(recent["assists"].values, w),
            "form_cs":      np.dot(recent["clean_sheets"].values, w),
            "form_bonus":   np.dot(recent["bonus"].values, w),
            "form_xgi":     np.dot(
                (recent["expected_goals"] + recent["expected_assists"]).values, w
            ),
            "form_bps":     np.dot(recent["bps"].values, w),
            # ── Defensive form ────────────────────────────────────────────
            "form_saves":              np.dot(recent["saves"].values, w),
            "form_gc":                 np.dot(recent["goals_conceded"].values, w),
            "form_xgc":                np.dot(recent["expected_goals_conceded"].values, w),
            "form_clean_sheet_rate":   recent["clean_sheets"].mean(),
            # ── Deduction risk (applies to ALL positions incl FWDs) ───────
            "form_yellow_cards":       np.dot(recent["yellow_cards"].values, w),
            "form_red_cards":          np.dot(recent["red_cards"].values, w),
            "form_own_goals":          np.dot(recent["own_goals"].values, w),
            "form_penalties_missed":   np.dot(recent["penalties_missed"].values, w),
            "form_penalties_saved":    np.dot(recent["penalties_saved"].values, w),
            # ── Combined deduction score (weighted pts cost) ──────────────
            "form_deduction_risk":     np.dot(
                (recent["yellow_cards"] * 1
                 + recent["red_cards"] * 3
                 + recent["own_goals"] * 2
                 + recent["penalties_missed"] * 2).values, w
            ),
        })

    form_df = (
        history[history["gameweek_id"] < target_gw]
        .groupby("player_id")
        .apply(weighted_form, include_groups=False)
        .reset_index()
    )

    # ── 2. Consistency (1 - CV of points) ─────────────────────────────────
    def consistency(grp: pd.DataFrame) -> pd.Series:
        pts = grp["total_points"].values
        if len(pts) < 3:
            return pd.Series({"consistency": 0.5, "pts_variance": 0.0})
        cv = pts.std() / (pts.mean() + 1e-9)
        return pd.Series({"consistency": max(0, 1 - cv), "pts_variance": float(pts.var())})

    cons_df = (
        history[history["gameweek_id"] < target_gw]
        .groupby("player_id")
        .apply(consistency, include_groups=False)
        .reset_index()
    )

    # ── 3. Home vs away split ─────────────────────────────────────────────
    def home_away(grp: pd.DataFrame) -> pd.Series:
        home = grp[grp["was_home"] == True]["total_points"].mean()
        away = grp[grp["was_home"] == False]["total_points"].mean()
        return pd.Series({
            "avg_pts_home": home if not np.isnan(home) else 0,
            "avg_pts_away": away if not np.isnan(away) else 0,
            "home_away_delta": (home or 0) - (away or 0),
        })

    ha_df = (
        history[history["gameweek_id"] < target_gw]
        .groupby("player_id")
        .apply(home_away, include_groups=False)
        .reset_index()
    )

    # ── 4. Upcoming fixture difficulty ────────────────────────────────────
    from src.analytics.analytics import compute_dynamic_fdr, _get_fdr
    dynamic_fdr = compute_dynamic_fdr(teams, fixtures, target_gw)

    upcoming = fixtures[
        (fixtures["gameweek_id"].between(target_gw, target_gw + settings.fixture_lookahead - 1))
        & (~fixtures["finished"])
    ].copy()

    # Merge team strengths
    team_str = teams[["id", "strength_attack_home", "strength_attack_away",
                       "strength_defence_home", "strength_defence_away"]].copy()

    def get_fixture_score(player_team_id: int, is_home: bool) -> dict:
        team_fixtures = upcoming[
            (upcoming["team_h"] == player_team_id) | (upcoming["team_a"] == player_team_id)
        ].head(settings.fixture_lookahead)

        if len(team_fixtures) == 0:
            return {"fdr_next1": 3, "fdr_avg_3gw": 3, "fdr_avg_5gw": 3, "has_fixture_next": 0}

        difficulties = [_get_fdr(f, player_team_id, dynamic_fdr) for _, f in team_fixtures.iterrows()]

        d = difficulties
        return {
            "fdr_next1": d[0] if d else 3,
            "fdr_avg_3gw": np.mean(d[:3]),
            "fdr_avg_5gw": np.mean(d[:5]) if len(d) >= 5 else np.mean(d),
            "has_fixture_next": 1,
        }

    # Build per-player fixture features
    fix_rows = []
    for _, player in players.iterrows():
        fd = get_fixture_score(player["team_id"], True)
        fd["player_id"] = player["id"]
        fix_rows.append(fd)
    fix_df = pd.DataFrame(fix_rows)

    # ── 5. Current season stats (base features) ───────────────────────────
    base = players[[
        "id", "position", "now_cost", "selected_by_percent",
        "total_points", "minutes", "goals_scored", "assists",
        "clean_sheets", "goals_conceded", "saves", "bonus", "bps",
        "influence", "creativity", "threat", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "value_season", "team_id",
        "chance_of_playing_next_round",
    ]].copy().rename(columns={"id": "player_id"})

    # Aggregate deduction columns from history (not in players table)
    hist_season = history[history["gameweek_id"] < target_gw]
    deductions = (
        hist_season.groupby("player_id")[
            ["yellow_cards", "red_cards", "own_goals",
             "penalties_missed", "penalties_saved"]
        ].sum().reset_index()
    )
    base = base.merge(deductions, on="player_id", how="left")
    for col in ["yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved"]:
        base[col] = base[col].fillna(0)

    # ── Availability risk (0 = definitely plays, 1 = unknown) ────────────
    base["availability"] = base["chance_of_playing_next_round"].fillna(75) / 100.0

    # Points per million (value metric)
    base["pts_per_million"] = base["total_points"] / (base["now_cost"] + 0.1)

    gw_played = base["minutes"].clip(lower=1) / 90

    # xGI per 90
    base["xgi_per_90"] = base["expected_goal_involvements"] / gw_played

    # ── FPL clean-sheet scoring rules encoded as features ─────────────────
    # GK=4, DEF=4, MID=1, FWD=0  — make the rule explicit so XGBoost
    # doesn't have to learn it from data alone
    cs_pts_map  = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
    gc_pts_map  = {"GK": -0.5, "DEF": -0.5, "MID": 0.0, "FWD": 0.0}  # per goal conceded
    base["cs_pts_multiplier"] = base["position"].map(cs_pts_map).fillna(0)
    base["gc_pts_multiplier"] = base["position"].map(gc_pts_map).fillna(0)

    # ── Defensive derived features ────────────────────────────────────────
    # Saves per 90 (GKs only earn points here)
    base["saves_per_90"] = base["saves"].fillna(0) / gw_played

    # xGC per 90 (lower = stronger defence = more clean sheet probability)
    base["xgc_per_90"] = base["expected_goals_conceded"].fillna(0) / gw_played

    # Season clean sheet rate (meaningful for GK/DEF/MID, zero-signal for FWD)
    base["cs_rate"] = base["clean_sheets"].fillna(0) / gw_played.clip(lower=1)

    # Goals conceded per 90 (penalty only for GK/DEF)
    base["gc_per_90"] = base["goals_conceded"].fillna(0) / gw_played

    # Encode position
    pos_dummies = pd.get_dummies(base["position"], prefix="pos")
    for col in ["pos_GK", "pos_DEF", "pos_MID", "pos_FWD"]:
        if col not in pos_dummies.columns:
            pos_dummies[col] = 0
    base = pd.concat([base, pos_dummies], axis=1)

    # Merge team strength — both home AND away defence
    base = base.merge(
        teams[[
            "id",
            "strength_attack_home", "strength_attack_away",
            "strength_defence_home", "strength_defence_away",
        ]].rename(columns={"id": "team_id"}),
        on="team_id",
        how="left",
    )

    # ── 6. Team defensive form (recent CS rate per player) ────────────────
    recent_hist = history[history["gameweek_id"] >= target_gw - settings.form_window].copy()
    team_cs = (
        recent_hist.groupby("player_id")["clean_sheets"]
        .mean()
        .reset_index()
        .rename(columns={"clean_sheets": "team_cs_rate_recent"})
    )
    base = base.merge(team_cs, on="player_id", how="left")

    # ── 7. Assemble final feature matrix ──────────────────────────────────
    feat = base.merge(form_df, on="player_id", how="left")
    feat = feat.merge(cons_df, on="player_id", how="left")
    feat = feat.merge(ha_df, on="player_id", how="left")
    feat = feat.merge(fix_df, on="player_id", how="left")

    # ── 8. Position-adjusted form signals (encode FPL rules explicitly) ───
    # Expected CS pts contribution from recent form
    feat["form_cs_pts"]  = feat["form_cs"] * feat["cs_pts_multiplier"]
    # Expected GC pts cost from recent form (negative for GK/DEF)
    feat["form_gc_pts"]  = feat["form_gc"] * feat["gc_pts_multiplier"]
    # Expected save pts contribution (1pt per 3 saves, GKs only)
    feat["form_save_pts"] = feat["form_saves"] / 3.0
    # Penalty saved pts (5pts, GKs only)
    feat["form_pen_save_pts"] = feat["form_penalties_saved"] * 5.0

    # Fill NAs for new players with no history
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    feat[numeric_cols] = feat[numeric_cols].fillna(0)

    logger.debug("Feature matrix shape: {}", feat.shape)
    return feat


# ── Feature columns used by the model ─────────────────────────────────────────

FEATURE_COLS = [
    # ── Attacking form (last N GWs, exponentially weighted) ──────────────
    "form_pts", "form_minutes", "form_goals", "form_assists",
    "form_bonus", "form_xgi", "form_bps",
    # ── Position-adjusted defensive form (FPL rules encoded) ─────────────
    # form_cs_pts  = form_cs * multiplier (GK/DEF=4, MID=1, FWD=0)
    # form_gc_pts  = form_gc * multiplier (GK/DEF=-0.5/goal, MID/FWD=0)
    # form_save_pts = saves / 3 (GKs only in practice)
    # form_pen_save_pts = pen saves * 5 (GKs only)
    "form_cs_pts", "form_gc_pts", "form_save_pts", "form_pen_save_pts",
    # Raw defensive signals (position dummies let model weight these correctly)
    "form_cs", "form_saves", "form_gc", "form_xgc", "form_clean_sheet_rate",
    # ── Deduction risk (all positions — FWDs press, get booked) ──────────
    "form_yellow_cards", "form_red_cards",
    "form_own_goals", "form_penalties_missed",
    "form_deduction_risk",
    # ── Consistency ───────────────────────────────────────────────────────
    "consistency", "pts_variance",
    # ── Home/away split ───────────────────────────────────────────────────
    "avg_pts_home", "avg_pts_away", "home_away_delta",
    # ── Fixture difficulty ────────────────────────────────────────────────
    "fdr_next1", "fdr_avg_3gw", "fdr_avg_5gw", "has_fixture_next",
    # ── Season-level stats ────────────────────────────────────────────────
    "now_cost", "selected_by_percent", "minutes",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "value_season",
    "clean_sheets", "goals_conceded", "saves",
    "yellow_cards", "red_cards", "own_goals", "penalties_missed", "penalties_saved",
    # ── Derived per-90 stats ──────────────────────────────────────────────
    "availability", "pts_per_million",
    "xgi_per_90", "saves_per_90", "xgc_per_90", "gc_per_90", "cs_rate",
    "team_cs_rate_recent",
    # ── FPL scoring rule multipliers (explicit position encoding) ─────────
    "cs_pts_multiplier", "gc_pts_multiplier",
    # ── Position dummies ──────────────────────────────────────────────────
    "pos_GK", "pos_DEF", "pos_MID", "pos_FWD",
    # ── Team strength (all 4 combinations) ───────────────────────────────
    "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
]
