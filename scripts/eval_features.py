"""Quick holdout evaluation of the new feature set."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from src.models.features import FEATURE_COLS, build_feature_matrix
from src.models.expected_points import train, predict_multi_gw
from app.dependencies import load_players, load_history, load_fixtures, load_teams, get_current_gw
from src.database.db import get_connection

pl = load_players(); h = load_history(); fx = load_fixtures(); tm = load_teams()
gw = get_current_gw()
print(f"Features: {len(FEATURE_COLS)}")

# ── Holdout: last 5 GWs ───────────────────────────────────────────────────────
all_gws = sorted(h["gameweek_id"].unique())
train_gws, test_gws = all_gws[:-5], all_gws[-5:]

rows = []
for g in train_gws[4:]:
    feat = build_feature_matrix(h, fx, pl, tm, g)
    actual = (h[h["gameweek_id"] == g][["player_id", "total_points"]]
              .rename(columns={"total_points": "gw_points"}))
    rows.append(feat.merge(actual, on="player_id", how="inner"))

data = pd.concat(rows, ignore_index=True).dropna(subset=["gw_points"])
X = data[FEATURE_COLS].fillna(0).values.astype("float32")
y = data["gw_points"].values.astype("float32")
scaler = RobustScaler()
m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                 n_jobs=-1, verbosity=0)
m.fit(scaler.fit_transform(X), y)

test_rows = []
for g in test_gws:
    feat = build_feature_matrix(h, fx, pl, tm, g)
    actual = (h[h["gameweek_id"] == g][["player_id", "total_points"]]
              .rename(columns={"total_points": "gw_points"}))
    test_rows.append(feat.merge(actual, on="player_id", how="inner"))

test = pd.concat(test_rows, ignore_index=True).dropna(subset=["gw_points"])
Xt = scaler.transform(test[FEATURE_COLS].fillna(0).values.astype("float32"))
test["xpts"] = m.predict(Xt)
test["err"]  = (test["xpts"] - test["gw_points"]).abs()

print(f"Overall MAE: {test['err'].mean():.3f}")
print()
print("Per position:")
print(test.groupby("position").agg(
    mae=("err", "mean"), n=("player_id", "count")).round(3).to_string())

print()
print("Top 15 features by importance:")
fi = (pd.DataFrame({"feature": FEATURE_COLS, "importance": m.feature_importances_})
        .sort_values("importance", ascending=False))
print(fi.head(15).to_string(index=False))

# ── Retrain full model + store predictions ────────────────────────────────────
print()
print("Retraining full production model...")
metrics = train(h, fx, pl, tm)
print(f"CV MAE: {metrics['mae_cv']:.3f}  RMSE: {metrics['rmse_cv']:.3f}  features: {metrics['n_features']}")

xdf = predict_multi_gw(h, fx, pl, tm, gw, n_gws=5)
xdf["model_version"] = "v3_fpl_rules"
xdf["confidence"]    = 0.74
xdf["created_at"]    = pd.Timestamp.now()
xdf["gameweek_id"]   = gw
xdf = xdf[["player_id", "gameweek_id", "xpts", "xpts_3gw", "xpts_5gw",
            "model_version", "confidence", "created_at"]]
con = get_connection()
con.execute(f"DELETE FROM expected_points WHERE gameweek_id = {gw}")
con.execute("INSERT INTO expected_points SELECT * FROM xdf")
con.close()
print(f"Stored {len(xdf)} predictions for GW{gw}")
