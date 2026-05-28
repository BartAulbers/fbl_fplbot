"""
FBL Analytics — Streamlit Dashboard
Run with: python -m streamlit run dashboard/app.py

A calm, data-driven FPL manager interface.
"""
import sys
import os
import time
import json
import threading
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.db import get_connection, init_db
from app.dependencies import (
    load_players, load_teams, load_fixtures, load_history,
    load_my_squad, get_current_gw, load_xpts, load_metrics,
)
from src.analytics.analytics import (
    analyse_fixture_runs, fixture_swing_alerts,
    find_differentials, pick_captain,
)
from src.analytics.pitch_view import draw_pitch, squad_list_to_pitch_df
from src.optimization.squad_optimizer import optimize_squad, format_squad

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FBL Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #00ff87; }
    .metric-card { background: #1e1e2e; border-radius: 10px; padding: 1rem; }
    .position-gk { color: #f59e0b; font-weight: bold; }
    .position-def { color: #10b981; font-weight: bold; }
    .position-mid { color: #6366f1; font-weight: bold; }
    .position-fwd { color: #ef4444; font-weight: bold; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Safe data loaders (graceful on empty DB) ──────────────────────────────────

@st.cache_data(ttl=300)
def safe_load_players():
    try:
        return load_players()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def safe_load_teams():
    try:
        return load_teams()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def safe_load_fixtures():
    try:
        return load_fixtures()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def safe_load_history():
    try:
        return load_history()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def safe_load_xpts(gw):
    try:
        return load_xpts(gw)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def safe_load_metrics():
    try:
        return load_metrics()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def safe_load_my_squad():
    try:
        return load_my_squad()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def safe_get_current_gw():
    try:
        return get_current_gw()
    except Exception:
        return 1


_METRICS_TABLE_COLS = [
    "player_id", "pts_per_90", "pts_per_million", "form_score",
    "fixture_score_3gw", "fixture_score_5gw", "consistency",
    "home_away_delta", "ownership_inefficiency", "bonus_rate",
    "xgi_per_90", "rotation_risk", "updated_at",
]

def _refresh_metrics_and_xpts(gw: int) -> None:
    """Generate + store player_metrics and expected_points for `gw`."""
    from src.analytics.analytics import compute_player_metrics
    from src.models.expected_points import predict_multi_gw

    _pl = load_players(); _h = load_history()
    _fx = load_fixtures(); _tm = load_teams()
    con = get_connection()

    # ── Metrics ──────────────────────────────────────────────────────────
    m = compute_player_metrics(_pl, _h, _fx, gw)
    m["updated_at"] = pd.Timestamp.now()
    m = m[[c for c in _METRICS_TABLE_COLS if c in m.columns]]
    con.execute("DELETE FROM player_metrics")
    con.execute("INSERT INTO player_metrics SELECT * FROM m")

    # ── xPts (only if model exists) ───────────────────────────────────────
    from pathlib import Path as _P
    from config.settings import settings as _cfg
    if (_P(_cfg.model_dir) / "xpts_model.pkl").exists():
        xdf = predict_multi_gw(_h, _fx, _pl, _tm, gw, n_gws=5)
        xdf["model_version"] = "v1"
        xdf["confidence"]    = 0.7
        xdf["created_at"]    = pd.Timestamp.now()
        xdf["gameweek_id"]   = gw
        xdf = xdf[["player_id", "gameweek_id", "xpts", "xpts_3gw", "xpts_5gw",
                    "model_version", "confidence", "created_at"]]
        con.execute(f"DELETE FROM expected_points WHERE gameweek_id = {gw}")
        con.execute("INSERT INTO expected_points SELECT * FROM xdf")

    con.close()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ FBL Analytics")
    st.markdown("*A calm, data-driven FPL manager*")
    st.divider()

    page = st.selectbox("Navigate", [
        "🏠 Dashboard",
        "🎯 Squad Optimizer",
        "🔄 Transfer Engine",
        "📊 Expected Points",
        "📅 Fixture Analysis",
        "🎲 Differentials",
        "👑 Captaincy",
        "🤖 Model Training",
        "🔬 Feature Importance",
        "⚙️ My Squad",
    ])

    st.divider()

    current_gw = safe_get_current_gw()
    st.metric("Current GW", current_gw)

    risk_appetite = st.slider(
        "Risk Appetite", 0.0, 1.0, 0.5, 0.05,
        help="0 = safe template, 1 = full differentials"
    )

    if st.button("🔄 Refresh Data", width='stretch'):
        st.cache_data.clear()
        st.rerun()


# ── Load data ─────────────────────────────────────────────────────────────────
players = safe_load_players()
teams = safe_load_teams()
fixtures = safe_load_fixtures()
history = safe_load_history()
xpts = safe_load_xpts(current_gw)
metrics = safe_load_metrics()
my_squad = safe_load_my_squad()

db_ok = not players.empty


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: DASHBOARD
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">FBL Analytics Dashboard</h1>', unsafe_allow_html=True)

    if not db_ok:
        st.warning("⚠️ No data yet. Run the pipeline first: `python scripts/run_pipeline.py`")
        st.code("python scripts/run_pipeline.py", language="bash")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Players", len(players))
    col2.metric("Teams", len(teams))
    col3.metric("GW Records", len(history))
    col4.metric("xPts Ready", "✅" if not xpts.empty else "❌ Train model")

    st.divider()

    if not xpts.empty:
        top_players = players.rename(columns={"id": "player_id"})
        top_players = top_players.merge(xpts[["player_id", "xpts", "xpts_3gw"]], on="player_id", how="left")
        top_players = top_players.dropna(subset=["xpts"]).sort_values("xpts", ascending=False)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("🔍 Top 10 by xPts (Next GW)")
            fig = px.bar(
                top_players.head(10),
                x="web_name", y="xpts",
                color="position",
                color_discrete_map={"GK": "#f59e0b", "DEF": "#10b981", "MID": "#6366f1", "FWD": "#ef4444"},
                title="Expected Points — Next GW",
            )
            fig.update_layout(xaxis_title="", yaxis_title="xPts")
            st.plotly_chart(fig, width='stretch')

        with col_b:
            st.subheader("💰 Value (xPts / £m)")
            top_value = top_players.copy()
            top_value["value"] = top_value["xpts"] / (top_value["now_cost"] + 0.1)
            # Top 5 per position so all positions are represented
            top_value = (
                top_value.sort_values("value", ascending=False)
                .groupby("position", group_keys=False)
                .head(5)
            )
            fig2 = px.scatter(
                top_value, x="now_cost", y="xpts",
                color="position", size="value", hover_name="web_name",
                color_discrete_map={"GK": "#f59e0b", "DEF": "#10b981", "MID": "#6366f1", "FWD": "#ef4444"},
                title="Price vs Expected Points",
            )
            fig2.update_layout(xaxis_title="Price (£m)", yaxis_title="xPts")
            st.plotly_chart(fig2, width='stretch')

    # Fixture swings
    if not fixtures.empty and not teams.empty:
        st.subheader("⚡ Fixture Swing Alerts")
        alerts = fixture_swing_alerts(teams, fixtures, current_gw)
        if alerts:
            for alert in alerts[:5]:
                icon = "✅" if alert["alert_type"] == "FIXTURE_EASES" else "⚠️"
                st.info(f"{icon} **{alert['team']}** — {alert['message']}")
        else:
            st.success("No major fixture swings detected this week.")


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: SQUAD OPTIMIZER
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

elif page == "🎯 Squad Optimizer":
    st.title("🎯 Squad Optimizer")
    st.markdown("*ILP-powered squad selection — find the mathematically optimal 15.*")

    if not db_ok:
        st.warning("No data loaded yet.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    budget = col1.number_input("Budget (£m)", 80.0, 115.0, 100.0, 0.5)
    horizon = col2.selectbox("Horizon", ["1gw", "3gw", "5gw"])

    locked_input = st.text_input("Lock players (comma-separated IDs)", "")
    excluded_input = st.text_input("Exclude players (comma-separated IDs)", "")

    def parse_ids(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]

    if st.button("🚀 Optimize Squad", type="primary", width='stretch'):
        with st.spinner("Solving ILP..."):
            try:
                df = players.rename(columns={"id": "player_id"})
                if not xpts.empty:
                    df = df.merge(xpts[["player_id", "xpts", "xpts_3gw", "xpts_5gw"]], on="player_id", how="left")
                else:
                    df["xpts"] = df["form"] * 1.5
                    df["xpts_3gw"] = df["xpts"] * 3
                    df["xpts_5gw"] = df["xpts"] * 5
                if not metrics.empty:
                    df = df.merge(metrics[["player_id", "consistency"]], on="player_id", how="left")
                else:
                    df["consistency"] = 0.5

                df = df.rename(columns={"selected_by_percent": "ownership"}).fillna(0)

                result = optimize_squad(
                    players_df=df,
                    budget=budget,
                    risk_appetite=risk_appetite,
                    horizon=horizon,
                    locked_player_ids=parse_ids(locked_input),
                    excluded_player_ids=parse_ids(excluded_input),
                )

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Cost", f"£{result.total_cost:.1f}m")
                c2.metric("Budget Left", f"£{result.budget_remaining:.1f}m")
                c3.metric("xPts (Next GW)", f"{result.projected_pts_gw:.1f}")
                c4.metric("xPts (3GW)", f"{result.projected_pts_3gw:.1f}")

                # ── Pitch view ────────────────────────────────────────────
                pitch_df = squad_list_to_pitch_df(result.squad)
                fig_pitch = draw_pitch(pitch_df, title="Optimal Squad", show_xpts=True)
                st.plotly_chart(fig_pitch, width='stretch')

                # Starting XI table + bench
                tab_xi, tab_bench = st.tabs(["Starting XI", "Bench"])
                with tab_xi:
                    starting = sorted(
                        [p for p in result.squad if p.is_starting],
                        key=lambda p: ["GK","DEF","MID","FWD"].index(p.position)
                    )
                    xi_data = []
                    for p in starting:
                        flag = "🚩" if p.is_captain else "©" if p.is_vice else ""
                        xi_data.append({
                            "Name": f"{p.web_name} {flag}",
                            "Pos": p.position,
                            "Cost": f"£{p.cost:.1f}m",
                            "xPts": f"{p.xpts:.1f}",
                            "xPts 3GW": f"{p.xpts_3gw:.1f}",
                            "Own%": f"{p.ownership:.1f}%",
                            "Consistency": f"{p.consistency:.2f}",
                        })
                    st.dataframe(pd.DataFrame(xi_data), width='stretch', hide_index=True)

                with tab_bench:
                    bench = sorted([p for p in result.squad if not p.is_starting], key=lambda p: p.bench_order or 9)
                    bench_data = [{"Name": p.web_name, "Pos": p.position, "Cost": f"£{p.cost:.1f}m", "xPts": f"{p.xpts:.1f}"} for p in bench]
                    st.dataframe(pd.DataFrame(bench_data), width='stretch', hide_index=True)

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                st.info("Tip: Make sure to run the Model Training page first.")


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: TRANSFER ENGINE
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

elif page == "🔄 Transfer Engine":
    st.title("🔄 Transfer Engine")
    st.markdown("*The calm voice that stops you making impulse decisions.*")

    if my_squad.empty:
        st.warning("No squad loaded. Add your team via the My Squad page.")
        st.stop()

    col1, col2 = st.columns(2)
    free_transfers = col1.number_input("Free Transfers", 0, 5, 1)
    max_suggestions = col2.number_input("Max Suggestions", 1, 10, 5)

    if st.button("🔄 Analyse Transfers", type="primary", width='stretch'):
        from src.strategy.transfer_engine import recommend_transfers

        with st.spinner("Analysing..."):
            try:
                df = players.rename(columns={"id": "player_id"})
                if not xpts.empty:
                    df = df.merge(xpts[["player_id", "xpts", "xpts_3gw"]], on="player_id", how="left")
                else:
                    df["xpts"] = df["form"] * 1.5
                    df["xpts_3gw"] = df["xpts"] * 3
                if not metrics.empty:
                    df = df.merge(metrics[["player_id", "consistency", "fixture_score_3gw", "rotation_risk"]], on="player_id", how="left")
                    df = df.rename(columns={"fixture_score_3gw": "fdr_avg_3gw"})
                # Always ensure these columns exist with sensible defaults
                if "fdr_avg_3gw" not in df.columns:
                    df["fdr_avg_3gw"] = 3.0
                if "consistency" not in df.columns:
                    df["consistency"] = 0.5
                if "rotation_risk" not in df.columns:
                    df["rotation_risk"] = 0.3
                df = df.fillna({"fdr_avg_3gw": 3.0, "consistency": 0.5, "rotation_risk": 0.3}).fillna(0)

                my_squad_enr = my_squad.merge(
                    df[["player_id", "xpts", "xpts_3gw", "fdr_avg_3gw", "consistency"]],
                    on="player_id", how="left"
                ).fillna({"xpts": 0, "xpts_3gw": 0, "fdr_avg_3gw": 3.0, "consistency": 0.5})
                # Ensure added_gameweek has a default so churn guard doesn't fail
                if "added_gameweek" not in my_squad_enr.columns:
                    my_squad_enr["added_gameweek"] = 1
                my_squad_enr["added_gameweek"] = my_squad_enr["added_gameweek"].fillna(1).astype(int)

                plan = recommend_transfers(
                    my_squad_df=my_squad_enr,
                    all_players_df=df,
                    free_transfers=int(free_transfers),
                    current_gw=current_gw,
                    risk_appetite=risk_appetite,
                    max_suggestions=int(max_suggestions),
                )

                # Main recommendation banner
                if "ROLL" in plan.recommendation:
                    st.success(plan.recommendation)
                elif "HIT" in plan.recommendation:
                    st.warning(plan.recommendation)
                else:
                    st.info(plan.recommendation)

                st.divider()

                if plan.suggestions:
                    st.subheader("Transfer Suggestions (ranked)")
                    for i, s in enumerate(plan.suggestions, 1):
                        with st.expander(
                            f"#{i} OUT: {s.player_out['web_name']} → IN: {s.player_in['web_name']} "
                            f"| Net +{s.net_gain:.1f}pts {'⚠️ HIT' if s.hit_required else '✅ Free'}"
                        ):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"**OUT**: {s.player_out['web_name']}")
                                st.markdown(f"- Position: {s.player_out['position']}")
                                st.markdown(f"- Cost: £{s.player_out['cost']:.1f}m")
                                st.markdown(f"- xPts: {s.player_out['xpts']:.1f}")
                                st.markdown(f"- xPts 3GW: {s.player_out['xpts_3gw']:.1f}")
                            with c2:
                                st.markdown(f"**IN**: {s.player_in['web_name']}")
                                st.markdown(f"- Position: {s.player_in['position']}")
                                st.markdown(f"- Cost: £{s.player_in['cost']:.1f}m")
                                st.markdown(f"- xPts: {s.player_in['xpts']:.1f}")
                                st.markdown(f"- xPts 3GW: {s.player_in['xpts_3gw']:.1f}")
                                st.markdown(f"- Ownership: {s.player_in['ownership']:.1f}%")
                            st.markdown(f"**Reasoning**: {s.reasoning}")
                            st.markdown(f"**Confidence**: {s.confidence.upper()}")
                else:
                    st.success("No transfers beat the bar. Consider rolling your transfer.")

            except Exception as e:
                st.error(f"Transfer analysis failed: {e}")


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: EXPECTED POINTS
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

elif page == "📊 Expected Points":
    st.title("📊 Expected Points Model")

    if not db_ok:
        st.warning("No data.")
        st.stop()

    pos_filter = st.multiselect("Position", ["GK", "DEF", "MID", "FWD"], default=["MID", "FWD"])
    max_cost = st.slider("Max Cost (£m)", 4.0, 15.0, 15.0, 0.5)

    from pathlib import Path as _P
    from config.settings import settings as _cfg
    _model_exists = (_P(_cfg.model_dir) / "xpts_model.pkl").exists()

    if xpts.empty:
        if not _model_exists:
            st.error("Model not trained yet. Go to **🤖 Model Training** and click *Start Training* first.")
        else:
            st.info(f"Model is trained ✅ — predictions not yet generated for GW{current_gw}.")
            if st.button("🔮 Generate Predictions Now", type="primary"):
                with st.spinner("Generating predictions…"):
                    try:
                        _h = load_history(); _fx = load_fixtures()
                        _pl = load_players(); _tm = load_teams()
                        from src.models.expected_points import predict_multi_gw as _pmg
                        _xdf = _pmg(_h, _fx, _pl, _tm, current_gw, n_gws=5)
                        _xdf["model_version"] = "v1"
                        _xdf["confidence"] = 0.7
                        _xdf["created_at"] = pd.Timestamp.now()
                        _xdf["gameweek_id"] = current_gw
                        _xdf = _xdf[["player_id", "gameweek_id", "xpts", "xpts_3gw", "xpts_5gw",
                                      "model_version", "confidence", "created_at"]]
                        _con = get_connection()
                        _con.execute(f"DELETE FROM expected_points WHERE gameweek_id = {current_gw}")
                        _con.execute("INSERT INTO expected_points SELECT * FROM _xdf")
                        _con.close()
                        st.success(f"✅ Predictions generated for {len(_xdf)} players in GW{current_gw}!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as _e:
                        st.error(f"Failed: {_e}")
        st.stop()
    else:
        df = players.rename(columns={"id": "player_id"})
        df = df.merge(xpts[["player_id", "xpts", "xpts_3gw", "xpts_5gw"]], on="player_id", how="left")
        df = df[df["position"].isin(pos_filter)]
        df = df[df["now_cost"] <= max_cost]
        df = df.dropna(subset=["xpts"]).sort_values("xpts", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("xPts Rankings")
            display = df[["web_name", "position", "now_cost", "xpts", "xpts_3gw", "xpts_5gw",
                           "selected_by_percent"]].head(30)
            display.columns = ["Name", "Pos", "Cost", "xPts 1GW", "xPts 3GW", "xPts 5GW", "Own%"]
            st.dataframe(display, width='stretch', hide_index=True)

        with col2:
            st.subheader("xPts vs Price")
            fig = px.scatter(
                df.head(50), x="now_cost", y="xpts",
                color="position", hover_name="web_name",
                size="xpts_3gw",
                color_discrete_map={"GK": "#f59e0b", "DEF": "#10b981", "MID": "#6366f1", "FWD": "#ef4444"},
            )
            fig.update_layout(xaxis_title="Price (£m)", yaxis_title="xPts")
            st.plotly_chart(fig, width='stretch')


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: FIXTURE ANALYSIS
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

elif page == "📅 Fixture Analysis":
    st.title("📅 Fixture Analysis Engine")

    if not db_ok or fixtures.empty:
        st.warning("No fixture data.")
        st.stop()

    n_gws = st.slider("Gameweeks to analyse", 3, 8, 5)

    fixture_runs = analyse_fixture_runs(teams, fixtures, current_gw, n_gws)

    st.subheader("Team Fixture Difficulty (lower = easier)")
    fig = px.bar(
        fixture_runs.sort_values("avg_fdr"),
        x="team_name", y="avg_fdr",
        color="avg_fdr",
        color_continuous_scale="RdYlGn_r",
        title=f"Average FDR — Next {n_gws} GWs",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Avg FDR")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Full Fixture Table")
    st.dataframe(fixture_runs, width='stretch', hide_index=True)

    st.subheader("⚡ Swing Alerts")
    alerts = fixture_swing_alerts(teams, fixtures, current_gw)
    for alert in alerts[:8]:
        icon = "✅" if alert["alert_type"] == "FIXTURE_EASES" else "⚠️"
        st.markdown(f"{icon} **{alert['team']}**: {alert['message']}")


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: DIFFERENTIALS
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

elif page == "🎲 Differentials":
    st.title("🎲 Differential Finder")
    st.markdown("*Low-owned, high-expected-points — template-busting picks.*")

    if xpts.empty or metrics.empty:
        missing = []
        if xpts.empty:   missing.append("xPts predictions")
        if metrics.empty: missing.append("player metrics")
        st.info(f"Missing: {', '.join(missing)}. Click below to generate them.")
        if st.button("⚡ Compute Missing Data", type="primary"):
            _refresh_metrics_and_xpts(current_gw)
            st.cache_data.clear()
            st.rerun()
        st.stop()

    ownership_cap = st.slider("Max Ownership %", 1.0, 25.0, 10.0, 0.5)
    top_n = st.slider("Top N per position", 3, 15, 5)

    diffs = find_differentials(players, metrics, xpts, ownership_cap, top_n)

    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_diffs = diffs[diffs["position"] == pos]
        if not pos_diffs.empty:
            st.subheader(f"{'🧤' if pos == 'GK' else '🛡️' if pos == 'DEF' else '🎯' if pos == 'MID' else '⚡'} {pos}")
            display = pos_diffs[["web_name", "now_cost", "xpts", "xpts_3gw",
                                  "selected_by_percent", "diff_score"]].copy()
            display.columns = ["Name", "Cost", "xPts", "xPts 3GW", "Own%", "Diff Score"]
            st.dataframe(display, width='stretch', hide_index=True)


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: CAPTAINCY
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

elif page == "👑 Captaincy":
    st.title("👑 Captaincy Optimizer")

    if my_squad.empty:
        st.warning("No squad loaded. Add players in the My Squad page.")
        st.stop()

    if xpts.empty or metrics.empty:
        missing = []
        if xpts.empty:    missing.append("xPts predictions")
        if metrics.empty: missing.append("player metrics")
        st.info(f"Missing: {', '.join(missing)}. Click below to generate them.")
        if st.button("⚡ Compute Missing Data", type="primary", key="cap_refresh"):
            _refresh_metrics_and_xpts(current_gw)
            st.cache_data.clear()
            st.rerun()
        st.stop()

    cap_df = pick_captain(my_squad, xpts, metrics, risk_appetite)

    st.subheader("Captain Rankings")
    st.markdown(f"**Recommended Captain**: 👑 {cap_df.iloc[0]['web_name']} ({cap_df.iloc[0]['xpts']:.1f} xPts)")
    if len(cap_df) > 1:
        st.markdown(f"**Vice Captain**: © {cap_df.iloc[1]['web_name']} ({cap_df.iloc[1]['xpts']:.1f} xPts)")

    display = cap_df.copy()
    display["captain_score"] = display["captain_score"].round(2)
    st.dataframe(display, width='stretch', hide_index=True)

    fig = px.bar(
        cap_df, x="web_name", y="captain_score",
        color="position",
        color_discrete_map={"GK": "#f59e0b", "DEF": "#10b981", "MID": "#6366f1", "FWD": "#ef4444"},
        title="Captain Score by Player",
    )
    st.plotly_chart(fig, width='stretch')


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: FEATURE IMPORTANCE
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    st.markdown("*Train the XGBoost expected points model on your local FPL data.*")

    if not db_ok or history.empty:
        st.warning("No history data. Run the data pipeline first.")
        st.stop()

    # ── Model status ──────────────────────────────────────────────────────
    from pathlib import Path as _Path
    from config.settings import settings as _cfg

    model_path = _Path(_cfg.model_dir) / "xpts_model.pkl"
    model_exists = model_path.exists()

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Model Trained", "✅ Yes" if model_exists else "❌ No")
    col_s2.metric("Training GW records", f"{len(history):,}")
    col_s3.metric("Players", f"{len(players):,}")

    if model_exists:
        import os, time
        mtime = os.path.getmtime(model_path)
        last_trained = pd.Timestamp(mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        st.info(f"📋 Model last trained: **{last_trained}**")

    st.divider()

    # ── Training controls ─────────────────────────────────────────────────
    st.subheader("Train / Retrain Model")

    col_t1, col_t2 = st.columns(2)
    run_predictions = col_t1.checkbox("Run predictions after training", value=True)
    refresh_metrics_flag = col_t2.checkbox("Refresh custom metrics after training", value=True)

    n_gws_pred = st.slider("Predict ahead (GWs)", 1, 5, 5,
                            help="How many gameweeks to generate predictions for")

    if st.button("🚀 Start Training", type="primary", width='stretch'):
        # ── Run in background thread so Streamlit doesn't timeout ────────
        if st.session_state.get("training_running", False):
            st.warning("Training already in progress. Please wait.")
        else:
            st.session_state["training_running"] = True
            st.session_state["training_log"] = []
            st.session_state["training_metrics"] = None

            log_placeholder = st.empty()
            progress_bar = st.progress(0, text="Initialising...")

            def _train_job():
                log = st.session_state["training_log"]
                try:
                    from src.models.expected_points import train as _train, predict_multi_gw
                    from src.analytics.analytics import compute_player_metrics

                    _players = load_players()
                    _history = load_history()
                    _fixtures = load_fixtures()
                    _teams = load_teams()

                    log.append("📦 Data loaded")
                    metrics = _train(_history, _fixtures, _players, _teams)
                    st.session_state["training_metrics"] = metrics
                    log.append(f"✅ Model trained — CV MAE: {metrics['mae_cv']:.3f}, RMSE: {metrics['rmse_cv']:.3f}")

                    if run_predictions:
                        log.append("🔮 Running predictions...")
                        gw = get_current_gw()
                        xpts_df = predict_multi_gw(_history, _fixtures, _players, _teams, gw, n_gws=n_gws_pred)
                        xpts_df["model_version"] = "v1"
                        xpts_df["confidence"] = 0.7
                        xpts_df["created_at"] = pd.Timestamp.now()
                        xpts_df["gameweek_id"] = gw
                        # Reorder to match table schema: player_id, gameweek_id, xpts, xpts_3gw, xpts_5gw, model_version, confidence, created_at
                        xpts_df = xpts_df[["player_id", "gameweek_id", "xpts", "xpts_3gw", "xpts_5gw", "model_version", "confidence", "created_at"]]
                        con = get_connection()
                        con.execute(f"DELETE FROM expected_points WHERE gameweek_id = {gw}")
                        con.execute("INSERT INTO expected_points SELECT * FROM xpts_df")
                        con.close()
                        log.append(f"✅ Predictions stored for GW{gw} (+{n_gws_pred} GWs)")

                    if refresh_metrics_flag:
                        log.append("📊 Refreshing custom metrics...")
                        gw = get_current_gw()
                        m = compute_player_metrics(_players, _history, _fixtures, gw)
                        m["updated_at"] = pd.Timestamp.now()
                        m = m[[c for c in _METRICS_TABLE_COLS if c in m.columns]]
                        con = get_connection()
                        con.execute("DELETE FROM player_metrics")
                        con.execute("INSERT INTO player_metrics SELECT * FROM m")
                        con.close()
                        log.append(f"✅ Metrics refreshed for {len(m)} players")

                    log.append("🎉 All done! Refresh the dashboard to see updated predictions.")
                except Exception as e:
                    log.append(f"❌ Error: {e}")
                finally:
                    st.session_state["training_running"] = False

            thread = threading.Thread(target=_train_job, daemon=True)
            thread.start()

            # Show live log while training runs
            step_labels = [
                "Loading data...",
                "Training XGBoost...",
                "Running predictions...",
                "Refreshing metrics...",
                "Done!",
            ]
            step = 0
            while thread.is_alive() or st.session_state.get("training_running", False):
                log_lines = st.session_state.get("training_log", [])
                log_placeholder.code("\n".join(log_lines) if log_lines else "Starting...", language="")
                pct = min(len(log_lines) / max(len(step_labels), 1), 0.95)
                label = step_labels[min(len(log_lines), len(step_labels) - 1)]
                progress_bar.progress(pct, text=label)
                time.sleep(0.5)
                if not thread.is_alive():
                    break

            thread.join(timeout=5)
            progress_bar.progress(1.0, text="Complete ✅")
            log_lines = st.session_state.get("training_log", [])
            log_placeholder.code("\n".join(log_lines), language="")
            st.cache_data.clear()

    # ── Show training metrics if available ────────────────────────────────
    if st.session_state.get("training_metrics"):
        m = st.session_state["training_metrics"]
        st.divider()
        st.subheader("📈 Training Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("CV MAE", f"{m['mae_cv']:.3f} pts", help="Mean Absolute Error (lower = better)")
        c2.metric("CV RMSE", f"{m['rmse_cv']:.3f} pts")
        c3.metric("Training Samples", f"{m['n_samples']:,}")
        st.caption(f"Model trained on {m['n_features']} features using time-series cross-validation.")

    # ── Feature importance (if model exists) ─────────────────────────────
    if model_exists:
        st.divider()
        st.subheader("🔬 Feature Importance")
        try:
            from src.models.expected_points import feature_importance
            fi = feature_importance(top_n=20)
            fig = px.bar(
                fi, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Viridis",
                title="Top 20 Feature Importances",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False, height=500)
            st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.warning(f"Could not load feature importance: {e}")


elif page == "🔬 Feature Importance":
    st.title("🔬 Algorithm Explorer")
    st.markdown("*Understand, inspect and improve the Expected Points model.*")

    MODEL_PATH = "data/models/xpts_model.pkl"
    model_exists = os.path.exists(MODEL_PATH)

    tab_how, tab_fi, tab_shap, tab_params, tab_quality = st.tabs([
        "📖 How It Works",
        "📊 Feature Importance",
        "🎯 SHAP Explainer",
        "⚙️ Model Parameters",
        "📈 Prediction Quality",
    ])

    # ── TAB 1: HOW IT WORKS ───────────────────────────────────────────────
    with tab_how:
        st.subheader("How the Expected Points model works")
        st.markdown("""
### The Pipeline

```
FPL API data  ──►  Feature Engineering  ──►  XGBoost  ──►  xPts per player
```

Every gameweek, the model runs this sequence:
1. **Load data** — player history (all past GWs), fixtures, team strengths
2. **Build features** — 68 numeric features per player (see groups below)
3. **Predict** — XGBoost outputs expected points for the next 1 / 3 / 5 GWs

---
### Feature Groups

The 68 features are split into 7 groups:
        """)

        feature_groups = {
            "⚡ Attacking Form (recent, exponentially weighted)": {
                "description": "Last N gameweeks weighted so recent games count more (½ power decay). Captures current attacking threat.",
                "features": ["form_pts", "form_minutes", "form_goals", "form_assists", "form_bonus", "form_xgi", "form_bps"],
            },
            "🛡️ Defensive Form (FPL-rule adjusted)": {
                "description": "Clean sheets, saves, goals conceded — multiplied by FPL scoring rules per position (GK/DEF CS = 4pts, MID = 1pt, FWD = 0pt). This encodes game knowledge directly into features.",
                "features": ["form_cs_pts", "form_gc_pts", "form_save_pts", "form_pen_save_pts", "form_cs", "form_saves", "form_gc", "form_xgc", "form_clean_sheet_rate"],
            },
            "🟨 Discipline / Deduction Risk": {
                "description": "Yellow cards, reds, own goals, missed penalties — all positions can be deducted points. form_deduction_risk is a composite weighted score.",
                "features": ["form_yellow_cards", "form_red_cards", "form_own_goals", "form_penalties_missed", "form_deduction_risk"],
            },
            "📐 Consistency & Home/Away Split": {
                "description": "Consistency = 1 - coefficient of variation (stable performers score higher). Home/away delta identifies players who boom at home but disappear away.",
                "features": ["consistency", "pts_variance", "avg_pts_home", "avg_pts_away", "home_away_delta"],
            },
            "📅 Fixture Difficulty (FDR)": {
                "description": "Official FPL Fixture Difficulty Rating for next 1/3/5 GWs. Lower = easier fixtures = more expected points.",
                "features": ["fdr_next1", "fdr_avg_3gw", "fdr_avg_5gw", "has_fixture_next"],
            },
            "💰 Season Stats & Value": {
                "description": "Cumulative season stats, ICT index (Influence, Creativity, Threat), per-90 rates, price, and ownership percentage.",
                "features": ["now_cost", "selected_by_percent", "minutes", "influence", "creativity", "threat", "ict_index",
                             "expected_goals", "expected_assists", "expected_goal_involvements", "pts_per_million",
                             "xgi_per_90", "saves_per_90", "xgc_per_90", "gc_per_90", "cs_rate", "team_cs_rate_recent"],
            },
            "🏟️ Team & Position Context": {
                "description": "Team attacking and defensive strength (all 4 home/away combos). Position dummies let the model weight features differently per position. cs_pts_multiplier and gc_pts_multiplier are explicit FPL rule encodings.",
                "features": ["strength_attack_home", "strength_attack_away", "strength_defence_home", "strength_defence_away",
                             "pos_GK", "pos_DEF", "pos_MID", "pos_FWD", "cs_pts_multiplier", "gc_pts_multiplier"],
            },
        }

        for group_name, info in feature_groups.items():
            with st.expander(group_name, expanded=False):
                st.markdown(f"**{info['description']}**")
                st.code(", ".join(info["features"]))

        st.divider()
        st.subheader("Why XGBoost?")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**XGBoost advantages for FPL:**
- Handles missing data natively (new players have no history)
- Non-linear interactions (e.g. cheap player + easy fixture = outsized value)
- Feature importance built-in — explainable output
- Fast to retrain each gameweek
- Robust to outliers (a single 20-pt haul doesn't destroy the model)
            """)
        with col2:
            st.markdown("""
**Current model config:**
- Estimators: 300 trees
- Max depth: 5
- Learning rate: 0.05
- Subsampling: 80% rows & 70% columns per tree
- Objective: `reg:squarederror` (minimise MSE)
- Training: walk-forward — trained on GWs 1→N, predicts GW N+1
            """)

        st.divider()
        st.subheader("💡 Ideas to improve the model")
        st.markdown("""
| Idea | Expected Impact | Effort |
|------|----------------|--------|
| Add player age as feature | Moderate — younger players have more variance | Low |
| Add injury history (days out last season) | Moderate — captures rotation risk | Medium |
| Opponent's recent form (not just FDR) | High — FDR is static, form is dynamic | Medium |
| Add weather data (wind/rain affects scoring) | Low | High |
| Use LSTM/sequence model instead of XGBoost | Potentially high — captures temporal patterns | High |
| Add transfer data (Δ ownership per week) | Moderate — wisdom of crowds signal | Low |
| Separate model per position | Moderate — GKs and FWDs have very different signals | Low |
| Add penalty taker / set-piece taker flag | High for specific players | Medium |
        """)

    # ── TAB 2: FEATURE IMPORTANCE ─────────────────────────────────────────
    with tab_fi:
        if not model_exists:
            st.warning("Train the model first on the 🤖 Model Training page.")
        else:
            from src.models.expected_points import feature_importance

            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            top_n = col_ctrl1.slider("Number of features to show", 10, 68, 25)
            importance_type = col_ctrl2.selectbox(
                "Importance type",
                ["weight", "gain", "cover"],
                index=1,
                help="gain = average improvement per split (best for understanding). weight = # of times feature is used. cover = average samples per split.",
            )

            fi = feature_importance(top_n=top_n, importance_type=importance_type)

            # Assign colour by group
            group_colors = {
                "form_pts|form_minutes|form_goals|form_assists|form_bonus|form_xgi|form_bps": "#ef4444",
                "form_cs|form_gc|form_saves|form_xgc|form_clean_sheet_rate|form_cs_pts|form_gc_pts|form_save_pts|form_pen_save_pts": "#10b981",
                "form_yellow|form_red|form_own|form_penalt|form_deduction": "#f59e0b",
                "consistency|pts_variance|avg_pts|home_away": "#8b5cf6",
                "fdr": "#06b6d4",
                "pos_|cs_pts_mult|gc_pts_mult|strength": "#64748b",
            }

            def get_color(feat):
                for pattern, color in group_colors.items():
                    if any(feat.startswith(p) or p in feat for p in pattern.split("|")):
                        return color
                return "#6366f1"

            fi["color"] = fi["feature"].apply(get_color)

            fig = px.bar(
                fi, x="importance", y="feature", orientation="h",
                color="color",
                color_discrete_map="identity",
                title=f"Top {top_n} Features by {importance_type.title()}",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=max(400, top_n * 22),
            )
            st.plotly_chart(fig, width='stretch')

            st.caption("""
**Colour key:** 🔴 Attacking form · 🟢 Defensive form · 🟡 Discipline · 🟣 Consistency/H-A · 🔵 Fixture · ⚫ Team/Position
            """)

            if not history.empty:
                st.divider()
                st.subheader("Feature Correlation with Actual Points")
                from src.analytics.analytics import correlation_analysis
                corr = correlation_analysis(history)
                fig2 = px.bar(
                    corr.head(20), x="correlation_with_points", y="feature",
                    orientation="h", title="Pearson Correlation with GW Points",
                    color="correlation_with_points",
                    color_continuous_scale="RdYlGn",
                )
                fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
                st.plotly_chart(fig2, width='stretch')

    # ── TAB 3: SHAP EXPLAINER ─────────────────────────────────────────────
    with tab_shap:
        st.subheader("Why is this player predicted to score X points?")
        st.markdown("SHAP (SHapley Additive exPlanations) shows the exact contribution of each feature to a single player's prediction.")

        if not model_exists:
            st.warning("Train the model first on the 🤖 Model Training page.")
        elif xpts.empty:
            st.warning("Generate predictions first (Expected Points page).")
        else:
            # Player selector
            player_list = (
                players.rename(columns={"id": "player_id"})
                .merge(xpts[["player_id", "xpts"]], on="player_id", how="inner")
                .sort_values("xpts", ascending=False)
            )
            player_options = player_list["web_name"].tolist()
            selected_name = st.selectbox("Select a player", player_options)

            if selected_name:
                sel = player_list[player_list["web_name"] == selected_name].iloc[0]
                player_id = int(sel["player_id"])
                pred_xpts = float(sel["xpts"])

                st.metric("Predicted xPts", f"{pred_xpts:.2f}")

                try:
                    with st.spinner("Computing SHAP values..."):
                        import pickle, shap as shap_lib
                        from src.models.features import FEATURE_COLS, build_feature_matrix
                        from app.dependencies import load_players, load_history, load_fixtures, load_teams

                        model = pickle.load(open(MODEL_PATH, "rb"))
                        pl = load_players()
                        hi = load_history()
                        fx = load_fixtures()
                        tm = load_teams()

                        feat = build_feature_matrix(hi, fx, pl, tm, current_gw)
                        player_row = feat[feat["player_id"] == player_id]

                        if player_row.empty:
                            st.warning("No feature data for this player.")
                        else:
                            X = player_row[FEATURE_COLS].fillna(0).values.astype("float32")
                            explainer = shap_lib.TreeExplainer(model)
                            shap_vals = explainer.shap_values(X)[0]

                            shap_df = pd.DataFrame({
                                "feature": FEATURE_COLS,
                                "shap_value": shap_vals,
                                "feature_value": X[0],
                            }).sort_values("shap_value", key=abs, ascending=False).head(20)

                            shap_df["direction"] = shap_df["shap_value"].apply(
                                lambda v: "Positive ↑" if v > 0 else "Negative ↓"
                            )
                            shap_df["abs_impact"] = shap_df["shap_value"].abs()

                            fig_shap = px.bar(
                                shap_df,
                                x="shap_value", y="feature", orientation="h",
                                color="direction",
                                color_discrete_map={"Positive ↑": "#10b981", "Negative ↓": "#ef4444"},
                                hover_data={"feature_value": ":.3f", "shap_value": ":.3f"},
                                title=f"SHAP values for {selected_name}  (base + contributions = {pred_xpts:.2f} xPts)",
                                height=550,
                            )
                            fig_shap.update_layout(
                                yaxis={"categoryorder": "total ascending"},
                                xaxis_title="SHAP value (impact on prediction)",
                                showlegend=True,
                            )
                            st.plotly_chart(fig_shap, width='stretch')

                            st.caption("""
**How to read this:** Each bar shows how much a feature pushed the prediction up (green) or down (red)
from the baseline. The sum of all SHAP values + baseline = final xPts prediction.
                            """)

                            with st.expander("Raw SHAP values table"):
                                st.dataframe(
                                    shap_df[["feature", "feature_value", "shap_value", "direction"]].round(4),
                                    width='stretch',
                                )
                except Exception as e:
                    st.error(f"SHAP computation failed: {e}")

    # ── TAB 4: MODEL PARAMETERS ───────────────────────────────────────────
    with tab_params:
        st.subheader("Current XGBoost Hyperparameters")

        if not model_exists:
            st.warning("No trained model found.")
        else:
            import pickle
            model = pickle.load(open(MODEL_PATH, "rb"))
            params = model.get_params()

            # Display key params in metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trees (n_estimators)", params.get("n_estimators", "?"))
            c2.metric("Max depth", params.get("max_depth", "?"))
            c3.metric("Learning rate", params.get("learning_rate", "?"))
            c4.metric("Features", len(FEATURE_COLS) if "FEATURE_COLS" in dir() else "68")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Row subsample", params.get("subsample", "?"))
            c6.metric("Col subsample", params.get("colsample_bytree", "?"))
            c7.metric("Min child weight", params.get("min_child_weight", "?"))
            c8.metric("Reg alpha (L1)", params.get("reg_alpha", "?"))

            st.divider()
            st.subheader("✏️ Tune & Retrain")
            st.markdown("Adjust hyperparameters and retrain the model. The current predictions will be replaced.")

            with st.form("retrain_form"):
                col_p1, col_p2, col_p3 = st.columns(3)
                new_n_est  = col_p1.number_input("n_estimators", 50, 1000, int(params.get("n_estimators", 300)), 50)
                new_depth  = col_p2.number_input("max_depth", 2, 12, int(params.get("max_depth", 5)))
                new_lr     = col_p3.number_input("learning_rate", 0.01, 0.5, float(params.get("learning_rate", 0.05)), 0.01, format="%.3f")

                col_p4, col_p5, col_p6 = st.columns(3)
                new_sub    = col_p4.slider("subsample", 0.4, 1.0, float(params.get("subsample", 0.8)), 0.05)
                new_col    = col_p5.slider("colsample_bytree", 0.4, 1.0, float(params.get("colsample_bytree", 0.7)), 0.05)
                new_mcw    = col_p6.number_input("min_child_weight", 1, 20, int(params.get("min_child_weight", 3)))

                submitted = st.form_submit_button("🔄 Retrain with new parameters", type="primary")

            if submitted:
                import threading
                from src.models.expected_points import train, predict_next_gw, predict_multi_gw
                from app.dependencies import load_players, load_history, load_fixtures, load_teams

                override = dict(
                    n_estimators=new_n_est, max_depth=new_depth, learning_rate=new_lr,
                    subsample=new_sub, colsample_bytree=new_col, min_child_weight=new_mcw,
                )

                retrain_log = st.empty()
                retrain_bar = st.progress(0)

                def _retrain():
                    try:
                        retrain_log.info("Loading data...")
                        retrain_bar.progress(15)
                        pl = load_players(); hi = load_history()
                        fx = load_fixtures(); tm = load_teams()
                        retrain_log.info("Training with custom params...")
                        retrain_bar.progress(40)
                        metrics = train(hi, fx, pl, tm, param_overrides=override)
                        retrain_bar.progress(75)
                        retrain_log.info("Generating predictions...")
                        preds = predict_next_gw(hi, fx, pl, tm, current_gw)
                        multi = predict_multi_gw(hi, fx, pl, tm, current_gw)
                        _refresh_metrics_and_xpts(current_gw)
                        retrain_bar.progress(100)
                        retrain_log.success(
                            f"✅ Done! CV MAE: {metrics['cv_mae']:.3f}  "
                            f"(n_estimators={new_n_est}, depth={new_depth}, lr={new_lr:.3f})"
                        )
                    except Exception as e:
                        retrain_log.error(f"Retrain failed: {e}")

                t = threading.Thread(target=_retrain, daemon=True)
                t.start()
                t.join(timeout=120)

            st.divider()
            st.subheader("📋 All Parameters")
            st.json({k: v for k, v in params.items() if v is not None})

    # ── TAB 5: PREDICTION QUALITY ─────────────────────────────────────────
    with tab_quality:
        st.subheader("How accurate are the predictions?")

        sim_path = "data/simulation_predictions.csv"
        sim_summary = "data/simulation_summary.json"

        if os.path.exists(sim_summary):
            with open(sim_summary) as f:
                summary = json.load(f)

            c1, c2, c3, c4 = st.columns(4)

            def _fmt(v, fmt=".3f"):
                return format(v, fmt) if isinstance(v, (int, float)) else str(v)

            c1.metric("Season MAE", _fmt(summary.get("overall_mae", "?")))
            c2.metric("Season RMSE", _fmt(summary.get("overall_rmse", "?")))
            c3.metric("Correlation", _fmt(summary.get("overall_correlation", summary.get("pearson_correlation", "?"))))
            c4.metric("GWs simulated", summary.get("gameweeks_simulated", summary.get("n_gameweeks", "?")))

            if os.path.exists(sim_path):
                sim_df = pd.read_csv(sim_path)

                col_pq1, col_pq2 = st.columns([1, 1])

                with col_pq1:
                    st.markdown("**Predicted vs Actual (scatter)**")
                    pos_filter = st.multiselect(
                        "Filter by position", ["GK", "DEF", "MID", "FWD"],
                        default=["GK", "DEF", "MID", "FWD"],
                        key="pq_pos_filter",
                    )
                    sim_filtered = sim_df[sim_df["position"].isin(pos_filter)] if "position" in sim_df.columns else sim_df
                    sample = sim_filtered.sample(min(2000, len(sim_filtered)), random_state=42)

                    fig_scatter = px.scatter(
                        sample, x="actual_pts", y="xpts",
                        color="position" if "position" in sample.columns else None,
                        color_discrete_map={"GK": "#f59e0b", "DEF": "#10b981", "MID": "#6366f1", "FWD": "#ef4444"},
                        opacity=0.4,
                        title="Predicted vs Actual Points",
                        labels={"actual_pts": "Actual GW Points", "xpts": "Predicted xPts"},
                    )
                    # Add perfect-prediction line
                    max_val = max(sample["actual_pts"].max(), sample["xpts"].max())
                    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                          line=dict(color="white", dash="dash", width=1))
                    fig_scatter.update_layout(height=450)
                    st.plotly_chart(fig_scatter, width='stretch')

                with col_pq2:
                    st.markdown("**Error distribution (residuals)**")
                    sim_filtered2 = sim_df[sim_df["position"].isin(pos_filter)] if "position" in sim_df.columns else sim_df
                    sim_filtered2 = sim_filtered2.copy()
                    # Use existing error column or compute it
                    if "error" not in sim_filtered2.columns:
                        sim_filtered2["error"] = sim_filtered2["xpts"] - sim_filtered2["actual_pts"]
                    fig_hist = px.histogram(
                        sim_filtered2, x="error", nbins=60,
                        color="position" if "position" in sim_filtered2.columns else None,
                        color_discrete_map={"GK": "#f59e0b", "DEF": "#10b981", "MID": "#6366f1", "FWD": "#ef4444"},
                        title="Prediction Error Distribution",
                        labels={"error": "Error (predicted - actual)"},
                        barmode="overlay", opacity=0.6,
                    )
                    fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
                    fig_hist.update_layout(height=450)
                    st.plotly_chart(fig_hist, width='stretch')

                # Per-GW MAE trend
                if os.path.exists("data/simulation_results.csv"):
                    gw_results = pd.read_csv("data/simulation_results.csv")
                    fig_trend = px.line(
                        gw_results, x="gameweek", y="mae",
                        title="Model MAE over the season (lower = better)",
                        markers=True,
                        labels={"mae": "MAE", "gameweek": "Gameweek"},
                    )
                    fig_trend.add_hline(y=gw_results["mae"].mean(), line_dash="dash",
                                        annotation_text=f"Avg {gw_results['mae'].mean():.3f}",
                                        line_color="rgba(255,255,255,0.5)")
                    st.plotly_chart(fig_trend, width='stretch')

        else:
            st.info("No simulation data found. Run the season simulation to see prediction quality:")
            st.code("python scripts/season_simulation.py", language="bash")


# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
# PAGE: MY SQUAD
# ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?

elif page == "⚙️ My Squad":
    st.title("⚙️ My Squad Management")
    from src.data.squad_importer import import_squad_from_fpl, get_manager_info

    # ── Tab layout ────────────────────────────────────────────────────────
    tab_import, tab_manual, tab_view = st.tabs([
        "🔗 Import from FPL", "➕ Add Manually", "📋 View / Remove"
    ])

    # ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
    # TAB 1: IMPORT FROM FPL
    # ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
    with tab_import:
        st.markdown("### Import your squad directly from the FPL website")
        st.markdown(
            "Enter your **FPL Manager ID** — find it in the URL when you visit your team page: "
            "`https://fantasy.premierleague.com/entry/**{your_id}**/event/1`"
        )

        col1, col2 = st.columns([2, 1])
        manager_id = col1.number_input(
            "FPL Manager ID", min_value=1, max_value=99_999_999,
            value=st.session_state.get("fpl_manager_id", 1),
            step=1,
        )
        st.session_state["fpl_manager_id"] = manager_id

        gw_override = col2.number_input(
            "Gameweek (0 = current)", min_value=0, max_value=38,
            value=0, step=1,
        )

        # Preview manager info
        if st.button("👁 Preview Manager", width='stretch'):
            with st.spinner("Looking up manager..."):
                info = get_manager_info(int(manager_id))
                if info:
                    st.session_state["manager_info"] = info
                else:
                    st.error("Manager not found. Double-check your FPL Manager ID.")
                    st.session_state.pop("manager_info", None)

        if "manager_info" in st.session_state:
            info = st.session_state["manager_info"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Manager", info["name"])
            c2.metric("Team Name", info["team_name"])
            c3.metric("Overall Rank", f"{info['overall_rank']:,}" if info.get("overall_rank") else "—")

            col_pts1, col_pts2, _ = st.columns(3)
            col_pts1.metric("Overall Pts", info.get("overall_points", "—"))
            col_pts2.metric("Last GW Pts", info.get("last_gw_points", "—"))

        st.divider()

        if st.button("⬇️ Import Squad Now", type="primary", width='stretch'):
            gw = int(gw_override) if gw_override > 0 else None
            with st.spinner(f"Importing squad for manager {manager_id}..."):
                result = import_squad_from_fpl(int(manager_id), gameweek=gw)

            if result.success:
                st.success(f"✅ {result.message}")

                # Show imported squad grouped by position
                pos_order = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
                imported_df = pd.DataFrame(result.players)
                imported_df["pos_order"] = imported_df["position"].map(pos_order)
                imported_df = imported_df.sort_values(["pos_order", "squad_position"])

                for pos, label in [("GK", "🧤 Goalkeepers"), ("DEF", "🛡️ Defenders"),
                                    ("MID", "🎯 Midfielders"), ("FWD", "⚡ Forwards")]:
                    pos_df = imported_df[imported_df["position"] == pos]
                    if not pos_df.empty:
                        st.markdown(f"**{label}**")
                        disp = pos_df[["web_name", "now_cost", "purchase_price",
                                        "is_captain", "is_vice_captain"]].copy()
                        disp.columns = ["Name", "Current Price", "Purchase Price", "Captain", "Vice"]
                        disp["Current Price"] = disp["Current Price"].apply(lambda x: f"£{x:.1f}m")
                        disp["Purchase Price"] = disp["Purchase Price"].apply(lambda x: f"£{x:.1f}m")
                        st.dataframe(disp, width='stretch', hide_index=True)

                if result.skipped:
                    st.warning(
                        f"⚠️ {result.skipped} player(s) weren't found in the local database. "
                        "Run the data pipeline to refresh player data."
                    )
                st.cache_data.clear()

            else:
                st.error(f"❌ {result.message}")
                if "not found" in result.message.lower():
                    st.info(
                        "**Where to find your Manager ID:**\n\n"
                        "1. Go to https://fantasy.premierleague.com/\n"
                        "2. Click 'My Team'\n"
                        "3. Look at the URL — it will be: `.../entry/{manager_id}/...`"
                    )

    # ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
    # TAB 2: ADD MANUALLY
    # ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
    with tab_manual:
        st.markdown("### Add players one by one")
        if not players.empty:
            col_pos, col_search = st.columns([1, 3])
            pos_filter_m = col_pos.selectbox("Position", ["All", "GK", "DEF", "MID", "FWD"], key="manual_pos")

            player_options = players[["id", "web_name", "position", "now_cost", "team_id"]].copy()
            if pos_filter_m != "All":
                player_options = player_options[player_options["position"] == pos_filter_m]

            player_options["display"] = player_options.apply(
                lambda r: f"{r['web_name']} ({r['position']}, £{r['now_cost']:.1f}m)", axis=1
            )
            selected = st.selectbox("Select Player", player_options["display"].tolist(), key="manual_player")
            purchase_price = st.number_input("Purchase Price (£m)", 3.5, 15.0, 7.0, 0.1, key="manual_pp")

            if st.button("Add Player", key="manual_add"):
                pid = int(player_options[player_options["display"] == selected]["id"].values[0])
                try:
                    con = get_connection()
                    con.execute(
                        "INSERT OR REPLACE INTO my_squad (player_id, purchase_price, added_gameweek) VALUES (?, ?, ?)",
                        [pid, purchase_price, current_gw]
                    )
                    con.close()
                    st.success(f"Added {selected}!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("No player data. Run the pipeline first.")

    # ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
    # TAB 3: VIEW / REMOVE
    # ───────────────────────────────────?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?�?
    with tab_view:
        if my_squad.empty:
            st.info("No squad loaded yet. Use the Import tab or add players manually.")
        else:
            st.markdown(f"**{len(my_squad)} players in squad**")

            # ── Pitch visualisation ───────────────────────────────────────
            # Build pitch-ready DataFrame: assume first 11 (by squad position) start
            pitch_ready = my_squad.copy()
            if not xpts.empty:
                pitch_ready = pitch_ready.merge(
                    xpts[["player_id", "xpts"]], on="player_id", how="left"
                )
            else:
                pitch_ready["xpts"] = None

            # Determine starters: GK + 4 DEF + (fill to 11)
            pitch_ready = pitch_ready.sort_values("player_id")
            pitch_ready["is_starting"] = False
            pitch_ready["is_captain"] = pitch_ready.get("is_captain", False)
            pitch_ready["is_vice_captain"] = pitch_ready.get("is_vice_captain", False)

            # Mark first 1 GK, up to 5 DEF, up to 5 MID, up to 3 FWD as potential starters
            # Use a simple heuristic: fill positions to reach 11
            count_starts = 0
            for pos, max_pos in [("GK", 1), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
                pos_idx = pitch_ready[pitch_ready["position"] == pos].index
                allowed = min(len(pos_idx), max_pos)
                if count_starts + allowed > 11:
                    allowed = 11 - count_starts
                pitch_ready.loc[pos_idx[:allowed], "is_starting"] = True
                count_starts += allowed
                if count_starts >= 11:
                    break

            fig_my_pitch = draw_pitch(
                pitch_ready.rename(columns={"selected_by_percent": "ownership"}),
                title="My Squad",
                show_xpts=not xpts.empty,
            )
            st.plotly_chart(fig_my_pitch, width='stretch')

            # Table below pitch
            pos_order = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
            display_df = my_squad.copy()
            display_df["pos_order"] = display_df["position"].map(pos_order).fillna(5)

            for pos, label in [("GK", "🧤 Goalkeepers"), ("DEF", "🛡️ Defenders"),
                                ("MID", "🎯 Midfielders"), ("FWD", "⚡ Forwards")]:
                pos_df = display_df[display_df["position"] == pos].sort_values("pos_order")
                if not pos_df.empty:
                    st.markdown(f"**{label}**")
                    show_cols = ["web_name", "now_cost", "purchase_price",
                                 "selected_by_percent", "added_gameweek"]
                    show_cols = [c for c in show_cols if c in pos_df.columns]
                    disp = pos_df[show_cols].copy()
                    disp.columns = [c.replace("_", " ").title() for c in show_cols]
                    st.dataframe(disp, width='stretch', hide_index=True)

            st.divider()
            st.markdown("**Remove a player**")
            remove_options = my_squad["web_name"].tolist()
            to_remove = st.selectbox("Select player to remove", remove_options, key="remove_player")
            col_r1, col_r2 = st.columns([1, 3])
            if col_r1.button("🗑️ Remove", type="secondary"):
                pid = int(my_squad[my_squad["web_name"] == to_remove]["player_id"].values[0])
                con = get_connection()
                con.execute("DELETE FROM my_squad WHERE player_id = ?", [pid])
                con.close()
                st.success(f"Removed {to_remove}")
                st.cache_data.clear()
                st.rerun()

            st.divider()
            if st.button("🗑️ Clear Entire Squad", type="secondary"):
                con = get_connection()
                con.execute("DELETE FROM my_squad")
                con.close()
                st.success("Squad cleared.")
                st.cache_data.clear()
                st.rerun()

