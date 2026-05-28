"""
Season simulation: walk-forward backtesting across the full 2025-26 season.

For each gameweek 5–37:
  - Features are built using ONLY history prior to that GW (no look-ahead)
  - A fresh model is trained on GWs 1 to (gw-4) to mimic real decision-making
  - Predictions are compared to actual points scored

Outputs
-------
- Per-GW metrics table (MAE, RMSE, top-10 hit rate, correlation)
- Season-level summary
- Top/worst predicted players table
- Position breakdown
- Saved to data/simulation_results.csv  +  data/simulation_summary.json
"""
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import RobustScaler

from src.database.db import get_connection
from src.models.features import build_feature_matrix, FEATURE_COLS
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_TRAIN_GWS = 4   # need at least this many GWs before we can predict
RESULTS_CSV   = Path("data/simulation_results.csv")
SUMMARY_JSON  = Path("data/simulation_summary.json")

# ── Load data ──────────────────────────────────────────────────────────────────
print("📦 Loading data from DB...")
con = get_connection(read_only=True)
players  = con.execute("SELECT * FROM players").fetchdf()
history  = con.execute("SELECT * FROM player_gw_history").fetchdf()
fixtures = con.execute("SELECT * FROM fixtures").fetchdf()
teams    = con.execute("SELECT * FROM teams").fetchdf()
con.close()

all_gws = sorted(history["gameweek_id"].unique())
print(f"   {len(players)} players | {len(all_gws)} gameweeks | {len(history):,} GW records")


def train_model_on_gws(up_to_gw: int):
    """Train a fresh XGBoost on all GWs up to (and including) up_to_gw."""
    rows = []
    for gw in all_gws:
        if gw > up_to_gw:
            break
        if gw <= MIN_TRAIN_GWS:
            continue
        feat = build_feature_matrix(history, fixtures, players, teams, target_gw=gw)
        actual = (
            history[history["gameweek_id"] == gw][["player_id", "total_points"]]
            .rename(columns={"total_points": "gw_points"})
        )
        feat = feat.merge(actual, on="player_id", how="inner")
        rows.append(feat)

    if not rows:
        return None, None

    data = pd.concat(rows, ignore_index=True).dropna(subset=["gw_points"])
    X = data[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y = data["gw_points"].values.astype(np.float32)

    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_s, y, verbose=False)
    return model, scaler


# ── Walk-forward simulation ────────────────────────────────────────────────────
# We predict GW starting from MIN_TRAIN_GWS+1
predict_from = MIN_TRAIN_GWS + 1
predict_gws  = [gw for gw in all_gws if gw >= predict_from]

print(f"\n🔄 Running walk-forward simulation for GWs {predict_gws[0]}–{predict_gws[-1]}...")
print(f"   ({len(predict_gws)} gameweeks to simulate)\n")

gw_results = []   # per-GW metric rows
all_preds  = []   # every player prediction across all GWs

for idx, target_gw in enumerate(predict_gws):
    train_up_to = target_gw - 1

    # Train on everything before this GW
    model, scaler = train_model_on_gws(train_up_to)
    if model is None:
        continue

    # Build features for target GW (using only prior history — causal)
    feat = build_feature_matrix(history, fixtures, players, teams, target_gw=target_gw)

    # Actual points this GW
    actual = (
        history[history["gameweek_id"] == target_gw][["player_id", "total_points"]]
        .rename(columns={"total_points": "actual_pts"})
    )

    feat = feat.merge(actual, on="player_id", how="inner")
    if feat.empty:
        continue

    X = feat[FEATURE_COLS].fillna(0).values.astype(np.float32)
    X_s = scaler.transform(X)
    preds = np.clip(model.predict(X_s), 0, 25)

    feat["xpts"]      = preds
    feat["gameweek"]  = target_gw
    feat["error"]     = feat["xpts"] - feat["actual_pts"]
    feat["abs_error"] = feat["error"].abs()

    # ── GW-level metrics ──────────────────────────────────────────────────
    mae  = mean_absolute_error(feat["actual_pts"], feat["xpts"])
    rmse = root_mean_squared_error(feat["actual_pts"], feat["xpts"])
    corr = feat[["xpts", "actual_pts"]].corr().iloc[0, 1]

    # Top-10 hit rate: did top-10 predicted players appear in top-20 actual?
    top10_pred   = set(feat.nlargest(10, "xpts")["player_id"])
    top20_actual = set(feat.nlargest(20, "actual_pts")["player_id"])
    hit_rate     = len(top10_pred & top20_actual) / 10

    # Rank correlation
    feat["pred_rank"]   = feat["xpts"].rank(ascending=False)
    feat["actual_rank"] = feat["actual_pts"].rank(ascending=False)
    rank_corr = feat[["pred_rank", "actual_rank"]].corr().iloc[0, 1]

    gw_results.append({
        "gameweek":   target_gw,
        "mae":        round(mae, 3),
        "rmse":       round(rmse, 3),
        "correlation": round(corr, 3),
        "rank_corr":  round(rank_corr, 3),
        "top10_hit_rate": round(hit_rate, 2),
        "n_players":  len(feat),
    })

    # Keep individual predictions
    keep_cols = ["player_id", "gameweek", "xpts", "actual_pts",
                 "error", "abs_error", "position", "now_cost"]
    all_preds.append(feat[[c for c in keep_cols if c in feat.columns]])

    pct = (idx + 1) / len(predict_gws) * 100
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  GW{target_gw:>2} [{bar}] {pct:>5.1f}%  MAE={mae:.3f}  corr={corr:.3f}  top10={hit_rate:.0%}",
          end="\r", flush=True)

print("\n\n✅ Simulation complete!")

# ── Compile results ────────────────────────────────────────────────────────────
gw_df   = pd.DataFrame(gw_results)
pred_df = pd.concat(all_preds, ignore_index=True)

# Merge player names
name_map = players[["id", "web_name"]].rename(columns={"id": "player_id"})
pred_df  = pred_df.merge(name_map, on="player_id", how="left")

# ── Season summary ─────────────────────────────────────────────────────────────
summary = {
    "season": "2025-26",
    "gameweeks_simulated": len(gw_df),
    "total_predictions":   len(pred_df),
    "overall_mae":         round(float(pred_df["abs_error"].mean()), 3),
    "overall_rmse":        round(float(root_mean_squared_error(
                               pred_df["actual_pts"], pred_df["xpts"])), 3),
    "overall_correlation": round(float(pred_df[["xpts", "actual_pts"]].corr().iloc[0, 1]), 3),
    "avg_top10_hit_rate":  round(float(gw_df["top10_hit_rate"].mean()), 3),
    "best_gw":             int(gw_df.loc[gw_df["mae"].idxmin(), "gameweek"]),
    "worst_gw":            int(gw_df.loc[gw_df["mae"].idxmax(), "gameweek"]),
}

# ── Position breakdown ─────────────────────────────────────────────────────────
pos_summary = (
    pred_df.groupby("position").agg(
        mae=("abs_error", "mean"),
        rmse_approx=("error", lambda x: float(np.sqrt((x**2).mean()))),
        avg_actual=("actual_pts", "mean"),
        avg_xpts=("xpts", "mean"),
        n=("player_id", "count"),
    ).round(3)
)

# ── Top predicted players (by avg abs_error ASC = most accurate) ──────────────
top_accurate = (
    pred_df.groupby(["player_id", "web_name", "position"])
    .agg(mae=("abs_error", "mean"),
         avg_xpts=("xpts", "mean"),
         avg_actual=("actual_pts", "mean"),
         gws=("gameweek", "count"))
    .query("gws >= 10")  # played at least 10 GWs
    .sort_values("mae")
    .head(20)
    .round(3)
)

# ── Most over/under predicted ──────────────────────────────────────────────────
avg_error = (
    pred_df.groupby(["player_id", "web_name", "position"])
    .agg(avg_error=("error", "mean"),
         avg_xpts=("xpts", "mean"),
         avg_actual=("actual_pts", "mean"),
         gws=("gameweek", "count"))
    .query("gws >= 10")
    .round(3)
)
most_over  = avg_error.sort_values("avg_error", ascending=False).head(10)
most_under = avg_error.sort_values("avg_error").head(10)

# ── Save ───────────────────────────────────────────────────────────────────────
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
gw_df.to_csv(RESULTS_CSV, index=False)
pred_df.to_csv("data/simulation_predictions.csv", index=False)
with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

# ── Print report ───────────────────────────────────────────────────────────────
SEP = "─" * 60

print(f"\n{'═'*60}")
print("  📊  2025-26 SEASON SIMULATION REPORT")
print(f"{'═'*60}\n")

print("OVERALL METRICS")
print(SEP)
print(f"  Gameweeks simulated : {summary['gameweeks_simulated']}")
print(f"  Total predictions   : {summary['total_predictions']:,}")
print(f"  Overall MAE         : {summary['overall_mae']:.3f} pts")
print(f"  Overall RMSE        : {summary['overall_rmse']:.3f} pts")
print(f"  Pearson correlation : {summary['overall_correlation']:.3f}")
print(f"  Avg top-10 hit rate : {summary['avg_top10_hit_rate']:.1%}  (top-10 pred in top-20 actual)")
print(f"  Best GW (lowest MAE): GW{summary['best_gw']}  ({gw_df[gw_df['gameweek']==summary['best_gw']]['mae'].values[0]:.3f})")
print(f"  Worst GW            : GW{summary['worst_gw']}  ({gw_df[gw_df['gameweek']==summary['worst_gw']]['mae'].values[0]:.3f})")

print(f"\nPER-POSITION BREAKDOWN")
print(SEP)
print(pos_summary.to_string())

print(f"\nMOST ACCURATELY PREDICTED (min 10 GWs, by MAE)")
print(SEP)
print(top_accurate[["web_name","position","mae","avg_xpts","avg_actual","gws"]].to_string(index=False))

print(f"\nMOST OVER-PREDICTED (model too optimistic)")
print(SEP)
print(most_over[["web_name","position","avg_error","avg_xpts","avg_actual","gws"]].to_string(index=False))

print(f"\nMOST UNDER-PREDICTED (hidden gems)")
print(SEP)
print(most_under[["web_name","position","avg_error","avg_xpts","avg_actual","gws"]].to_string(index=False))

print(f"\nPER-GAMEWEEK METRICS")
print(SEP)
print(gw_df.to_string(index=False))

print(f"\n💾 Saved: {RESULTS_CSV}")
print(f"💾 Saved: data/simulation_predictions.csv")
print(f"💾 Saved: {SUMMARY_JSON}")
