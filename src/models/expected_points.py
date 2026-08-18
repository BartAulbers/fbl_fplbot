"""
Expected Points model using XGBoost.

Architecture:
- Train on historical GW data (features → actual points)
- Predict for next GW, next 3 GW, next 5 GW
- SHAP for explainability
- Persists model weights to disk
"""
import sys
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from src.models.features import FEATURE_COLS, build_feature_matrix

MODEL_PATH = Path(settings.model_dir) / "xpts_model.pkl"
SCALER_PATH = Path(settings.model_dir) / "xpts_scaler.pkl"


# ── Model definition ──────────────────────────────────────────────────────────

def _build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    player_history: pd.DataFrame,
    fixtures: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    param_overrides: dict | None = None,
) -> dict:
    """
    Train the expected points model using time-series cross-validation.
    Returns evaluation metrics.
    """
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)

    all_gws = sorted(player_history["gameweek_id"].unique())
    if len(all_gws) < 5:
        raise ValueError("Need at least 5 gameweeks of history to train.")

    rows = []
    for gw in all_gws:  # include GW1-3 (zero/sparse history) so the model
                        # learns to use fixture/team-strength signals for
                        # the real cold-start case: predicting a new season's
                        # GW1 with no prior in-season form data available.
        feat = build_feature_matrix(player_history, fixtures, players, teams, target_gw=gw)
        # Actual points for this GW — rename to avoid collision with season total
        actual = (
            player_history[player_history["gameweek_id"] == gw][["player_id", "total_points"]]
            .rename(columns={"total_points": "gw_points"})
        )
        feat = feat.merge(actual, on="player_id", how="inner")
        feat["target_gw"] = gw
        rows.append(feat)

    data = pd.concat(rows, ignore_index=True)
    data = data.dropna(subset=["gw_points"])

    X = data[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y = data["gw_points"].values.astype(np.float32)

    # Apply any hyperparameter overrides
    base_params = {}
    if param_overrides:
        base_params = {k: v for k, v in param_overrides.items()}

    # Time-series split (no look-ahead)
    tscv = TimeSeriesSplit(n_splits=5)
    gw_idx = data["target_gw"].values

    mae_scores, rmse_scores = [], []
    model = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        m = _build_model()
        if base_params:
            m.set_params(**base_params)
        m.fit(
            X_tr_s, y_tr,
            eval_set=[(X_val_s, y_val)],
            verbose=False,
        )

        preds = m.predict(X_val_s)
        mae_scores.append(mean_absolute_error(y_val, preds))
        rmse_scores.append(root_mean_squared_error(y_val, preds))
        model = (m, scaler)  # keep last fold model (or retrain on all)

    # Retrain on full data for production
    final_scaler = RobustScaler()
    X_full = final_scaler.fit_transform(X)
    final_model = _build_model()
    if base_params:
        final_model.set_params(**base_params)

    # Can't use early stopping without eval set on full data
    final_model.set_params(early_stopping_rounds=None, n_estimators=500)
    final_model.fit(X_full, y, verbose=False)

    # Persist
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(final_scaler, f)

    metrics = {
        "mae_cv": float(np.mean(mae_scores)),
        "cv_mae": float(np.mean(mae_scores)),   # alias used by dashboard
        "rmse_cv": float(np.mean(rmse_scores)),
        "n_samples": len(data),
        "n_features": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
    }
    logger.success("Model trained. CV MAE={:.3f}, RMSE={:.3f}", metrics["mae_cv"], metrics["rmse_cv"])
    return metrics


# ── Prediction ────────────────────────────────────────────────────────────────

def load_model() -> tuple[XGBRegressor, RobustScaler]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train() first.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_next_gw(
    player_history: pd.DataFrame,
    fixtures: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    target_gw: int,
) -> pd.DataFrame:
    """
    Predict expected points for all players for `target_gw`.
    Returns DataFrame with player_id, xpts, confidence columns.
    """
    model, scaler = load_model()

    feat = build_feature_matrix(player_history, fixtures, players, teams, target_gw)
    player_ids = feat["player_id"].values

    X = feat[FEATURE_COLS].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)

    # Confidence: inverse of local prediction variance across trees
    # Use XGBoost's predict_contributions for rough uncertainty
    preds_clipped = np.clip(preds, 0, 25)

    # ── Hard override for injured/suspended/unavailable players ───────────
    # The model can under-weight this signal since it's rare in training
    # data; force it here regardless of what the model learned.
    preds_clipped = preds_clipped * feat["status_playing_score"].values

    result = pd.DataFrame({
        "player_id": player_ids,
        "xpts": preds_clipped,
        "gameweek_id": target_gw,
    })

    logger.info("Predicted xPts for {} players in GW{}", len(result), target_gw)
    return result


def predict_multi_gw(
    player_history: pd.DataFrame,
    fixtures: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    current_gw: int,
    n_gws: int = 5,
) -> pd.DataFrame:
    """
    Predict expected points for next `n_gws` gameweeks and aggregate.
    Returns DataFrame with player_id, xpts_1gw, xpts_3gw, xpts_5gw.
    """
    model, scaler = load_model()
    all_preds = []

    for gw_offset in range(n_gws):
        gw = current_gw + gw_offset
        feat = build_feature_matrix(player_history, fixtures, players, teams, target_gw=gw)
        player_ids = feat["player_id"].values
        X = feat[FEATURE_COLS].fillna(0).values.astype(np.float32)
        X_scaled = scaler.transform(X)
        preds = np.clip(model.predict(X_scaled), 0, 25)
        preds = preds * feat["status_playing_score"].values
        all_preds.append(pd.DataFrame({"player_id": player_ids, f"xpts_gw{gw}": preds}))

    merged = all_preds[0]
    for df in all_preds[1:]:
        merged = merged.merge(df, on="player_id", how="outer")

    gw_cols = [c for c in merged.columns if c.startswith("xpts_gw")]
    merged["xpts"] = merged[gw_cols[0]] if gw_cols else 0
    merged["xpts_3gw"] = merged[[c for c in gw_cols[:3]]].sum(axis=1)
    merged["xpts_5gw"] = merged[[c for c in gw_cols[:5]]].sum(axis=1)

    return merged[["player_id", "xpts", "xpts_3gw", "xpts_5gw"]]


# ── Feature importance + SHAP ─────────────────────────────────────────────────

def feature_importance(top_n: int = 20, importance_type: str = "gain") -> pd.DataFrame:
    """Return top N feature importances from the trained model.

    importance_type: 'gain' (default), 'weight', or 'cover'
    """
    model, _ = load_model()
    booster = model.get_booster()
    scores = booster.get_score(importance_type=importance_type)
    # scores is a dict feature_name -> score; if model was trained without feature names,
    # keys are 'f0', 'f1', ... — map back to FEATURE_COLS
    if scores and list(scores.keys())[0].startswith("f"):
        named = {FEATURE_COLS[int(k[1:])]: v for k, v in scores.items() if int(k[1:]) < len(FEATURE_COLS)}
    else:
        named = scores
    df = pd.DataFrame([{"feature": k, "importance": v} for k, v in named.items()])
    return df.sort_values("importance", ascending=False).head(top_n)


def explain_player(
    player_history: pd.DataFrame,
    fixtures: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    player_id: int,
    target_gw: int,
) -> dict:
    """
    SHAP-based explanation for a single player's expected points.
    Returns feature contributions sorted by absolute impact.
    """
    if not SHAP_AVAILABLE:
        return {"error": "shap not installed. Run: pip install shap", "player_id": player_id}

    model, scaler = load_model()
    feat = build_feature_matrix(player_history, fixtures, players, teams, target_gw)
    row = feat[feat["player_id"] == player_id]
    if row.empty:
        return {"error": f"Player {player_id} not found"}

    X = row[FEATURE_COLS].fillna(0).values.astype(np.float32)
    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_scaled)

    contributions = sorted(
        [{"feature": f, "shap_value": float(v)} for f, v in zip(FEATURE_COLS, shap_vals[0])],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    return {
        "player_id": player_id,
        "xpts": float(np.clip(model.predict(X_scaled)[0], 0, 25)),
        "base_value": float(explainer.expected_value),
        "top_drivers": contributions[:10],
    }
