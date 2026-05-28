"""Print the simulation report from saved CSV/JSON files."""
import sys, json, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

gw_df   = pd.read_csv("data/simulation_results.csv")
pred_df = pd.read_csv("data/simulation_predictions.csv")
with open("data/simulation_summary.json") as f:
    summary = json.load(f)

SEP = "-" * 64

print("=" * 64)
print("  2025-26 SEASON SIMULATION  (walk-forward, strictly no lookahead)")
print("=" * 64)

print()
print("OVERALL METRICS")
print(SEP)
print(f"  Gameweeks simulated  : {summary['gameweeks_simulated']}  (GW5 to GW37)")
print(f"  Total predictions    : {summary['total_predictions']:,}")
print(f"  Overall MAE          : {summary['overall_mae']:.3f} pts  (avg error per player per GW)")
print(f"  Overall RMSE         : {summary['overall_rmse']:.3f} pts")
print(f"  Pearson correlation  : {summary['overall_correlation']:.3f}  (xPts vs actual)")
print(f"  Avg top-10 hit rate  : {summary['avg_top10_hit_rate']:.1%}  (top-10 predicted in top-20 actual)")
best_mae  = gw_df[gw_df["gameweek"] == summary["best_gw"]]["mae"].values[0]
worst_mae = gw_df[gw_df["gameweek"] == summary["worst_gw"]]["mae"].values[0]
print(f"  Best GW (lowest MAE) : GW{summary['best_gw']}  (MAE={best_mae:.3f})")
print(f"  Worst GW             : GW{summary['worst_gw']}  (MAE={worst_mae:.3f})")

print()
print("PER-POSITION BREAKDOWN")
print(SEP)
pos = (
    pred_df.groupby("position").agg(
        mae=("abs_error", "mean"),
        rmse=("error", lambda x: float(np.sqrt((x**2).mean()))),
        avg_actual=("actual_pts", "mean"),
        avg_xpts=("xpts", "mean"),
        n=("player_id", "count"),
    ).round(3)
)
print(pos.to_string())

print()
print("MOST ACCURATELY PREDICTED  (min 10 GWs, sorted by MAE)")
print(SEP)
top_acc = (
    pred_df.groupby(["player_id", "web_name", "position"])
    .agg(mae=("abs_error", "mean"),
         avg_xpts=("xpts", "mean"),
         avg_actual=("actual_pts", "mean"),
         gws=("gameweek", "count"))
    .reset_index()
    .query("gws >= 10")
    .sort_values("mae")
    .head(15)
    .round(3)
)
print(top_acc[["web_name", "position", "mae", "avg_xpts", "avg_actual", "gws"]].to_string(index=False))

avg_err = (
    pred_df.groupby(["player_id", "web_name", "position"])
    .agg(avg_error=("error", "mean"),
         avg_xpts=("xpts", "mean"),
         avg_actual=("actual_pts", "mean"),
         gws=("gameweek", "count"))
    .reset_index()
    .query("gws >= 10")
    .round(3)
)

print()
print("MOST OVER-PREDICTED  (model predicted too high)")
print(SEP)
print(
    avg_err.sort_values("avg_error", ascending=False).head(10)[
        ["web_name", "position", "avg_error", "avg_xpts", "avg_actual", "gws"]
    ].to_string(index=False)
)

print()
print("MOST UNDER-PREDICTED  (hidden gems - model missed)")
print(SEP)
print(
    avg_err.sort_values("avg_error").head(10)[
        ["web_name", "position", "avg_error", "avg_xpts", "avg_actual", "gws"]
    ].to_string(index=False)
)

print()
print("SEASON TREND  (MAE & correlation by gameweek)")
print(SEP)
print(gw_df[["gameweek", "mae", "rmse", "correlation", "rank_corr", "top10_hit_rate"]].to_string(index=False))

print()
early  = gw_df[gw_df["gameweek"] <= 19]["mae"].mean()
late   = gw_df[gw_df["gameweek"] > 19]["mae"].mean()
print(f"  Avg MAE GW5-19  : {early:.3f}  (less data, harder)")
print(f"  Avg MAE GW20-37 : {late:.3f}  (more history, better predictions)")
print()
print(f"Saved CSVs: data/simulation_results.csv | data/simulation_predictions.csv")
