"""
Football pitch visualisation using Plotly.

Draws a realistic green pitch (portrait orientation, attack upward) and places
players according to their formation.  Works for both the optimised squad and
the current my_squad.
"""
import math
from typing import Optional
import pandas as pd
import plotly.graph_objects as go

# ── Pitch dimensions — portrait (W=width, H=length) ──────────────────────────
# Real pitch: 68m wide, 105m long. We keep these proportions.
W, H = 68, 105

POS_COLOR = {
    "GK":  "#f59e0b",   # amber
    "DEF": "#10b981",   # green
    "MID": "#6366f1",   # indigo
    "FWD": "#ef4444",   # red
}

# Y positions: GK near bottom, FWD near top
ROLE_Y = {"GK": 8, "DEF": 26, "MID": 55, "FWD": 80}


def _detect_formation(starters: pd.DataFrame) -> str:
    counts = starters[starters["position"] != "GK"]["position"].value_counts()
    d = counts.get("DEF", 0)
    m = counts.get("MID", 0)
    f = counts.get("FWD", 0)
    return f"{d}-{m}-{f}"


def _player_x_positions(n: int) -> list[float]:
    """Evenly space n players across pitch width with margins."""
    margin = 7
    if n == 1:
        return [W / 2]
    step = (W - 2 * margin) / (n - 1)
    return [margin + i * step for i in range(n)]


def _pitch_shapes() -> list[dict]:
    """Return Plotly shape dicts for a portrait football pitch."""
    shapes = []

    def rect(x0, y0, x1, y1, fill=None):
        d = dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                 line=dict(color="white", width=1.5))
        if fill:
            d["fillcolor"] = fill
        return d

    def circle(cx, cy, r):
        return dict(type="circle",
                    x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
                    line=dict(color="white", width=1.5))

    def spot(cx, cy):
        return dict(type="circle",
                    x0=cx - 0.6, y0=cy - 0.6, x1=cx + 0.6, y1=cy + 0.6,
                    line=dict(color="white", width=1),
                    fillcolor="white")

    # Outer pitch boundary
    shapes.append(rect(0, 0, W, H))

    # Halfway line (horizontal)
    shapes.append(dict(type="line", x0=0, y0=H/2, x1=W, y1=H/2,
                       line=dict(color="white", width=1.5)))

    # Centre circle & spot
    shapes.append(circle(W/2, H/2, 9.15))
    shapes.append(spot(W/2, H/2))

    # ── Bottom penalty area (our end) ─────────────────────────────────────
    pa_w, pa_h = 40.32, 16.5
    pa_x0 = (W - pa_w) / 2
    shapes.append(rect(pa_x0, 0, pa_x0 + pa_w, pa_h))

    # Bottom six-yard box
    sb_w, sb_h = 18.32, 5.5
    sb_x0 = (W - sb_w) / 2
    shapes.append(rect(sb_x0, 0, sb_x0 + sb_w, sb_h))

    # Bottom penalty spot & arc
    shapes.append(spot(W/2, 11))
    shapes.append(dict(type="circle",
                       x0=W/2 - 9.15, y0=11 - 9.15,
                       x1=W/2 + 9.15, y1=11 + 9.15,
                       line=dict(color="white", width=1.5)))

    # ── Top penalty area (attacking end) ──────────────────────────────────
    shapes.append(rect(pa_x0, H - pa_h, pa_x0 + pa_w, H))
    shapes.append(rect(sb_x0, H - sb_h, sb_x0 + sb_w, H))
    shapes.append(spot(W/2, H - 11))
    shapes.append(dict(type="circle",
                       x0=W/2 - 9.15, y0=H - 11 - 9.15,
                       x1=W/2 + 9.15, y1=H - 11 + 9.15,
                       line=dict(color="white", width=1.5)))

    # Corner arcs (radius 1)
    for cx, cy in [(0, 0), (W, 0), (0, H), (W, H)]:
        shapes.append(dict(type="circle",
                           x0=cx - 1, y0=cy - 1, x1=cx + 1, y1=cy + 1,
                           line=dict(color="white", width=1.5)))

    return shapes


def draw_pitch(
    squad_df: pd.DataFrame,
    title: str = "Squad",
    show_xpts: bool = True,
) -> go.Figure:
    """
    Draw a portrait football pitch and place players on it.

    squad_df must have: web_name, position, is_starting (bool), is_captain,
    now_cost (or cost), xpts (optional), selected_by_percent (optional).
    """
    if "cost" in squad_df.columns and "now_cost" not in squad_df.columns:
        squad_df = squad_df.rename(columns={"cost": "now_cost"})

    starters = squad_df[squad_df["is_starting"] == True].copy()
    bench    = squad_df[squad_df["is_starting"] == False].copy()

    formation = _detect_formation(starters)

    fig = go.Figure()

    # ── Grass stripes (horizontal bands) ─────────────────────────────────
    stripe_h = 8
    for i in range(int(H / stripe_h) + 1):
        y0 = i * stripe_h
        y1 = min(y0 + stripe_h / 2, H)
        fig.add_shape(type="rect", x0=0, y0=y0, x1=W, y1=y1,
                      fillcolor="rgba(0,0,0,0.07)", line=dict(width=0), layer="below")

    fig.update_layout(
        plot_bgcolor="#2d7a2d",
        paper_bgcolor="#1a1a2e",
        shapes=_pitch_shapes(),
        showlegend=False,
        title=dict(text=f"<b>{title}</b>  ({formation})",
                   font=dict(color="white", size=16)),
        margin=dict(l=10, r=10, t=50, b=80),
        xaxis=dict(range=[-3, W + 3], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-18, H + 3], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        height=750,
    )

    # ── Place starters ────────────────────────────────────────────────────
    for pos in ["GK", "DEF", "MID", "FWD"]:
        group = starters[starters["position"] == pos].reset_index(drop=True)
        if group.empty:
            continue
        xs = _player_x_positions(len(group))
        y = ROLE_Y[pos]
        for i, row in group.iterrows():
            _add_player_marker(fig, xs[i], y, row, show_xpts)

    # ── Bench (below pitch) ───────────────────────────────────────────────
    bench = bench.reset_index(drop=True)
    bench_xs = _player_x_positions(max(len(bench), 1))
    bench_y = -9
    for i, row in bench.iterrows():
        _add_player_marker(fig, bench_xs[i], bench_y, row,
                           show_xpts=False, alpha=0.55)

    fig.add_annotation(x=W/2, y=-15, text="BENCH", showarrow=False,
                       font=dict(color="rgba(255,255,255,0.5)", size=11))

    return fig


def _add_player_marker(
    fig: go.Figure,
    x: float,
    y: float,
    row: pd.Series,
    show_xpts: bool = True,
    alpha: float = 1.0,
) -> None:
    pos       = row.get("position", "MID")
    color     = POS_COLOR.get(pos, "#888")
    name      = str(row.get("web_name", "?"))
    is_cap    = bool(row.get("is_captain", False))
    is_vice   = bool(row.get("is_vice_captain", False))
    xpts      = row.get("xpts", None)
    cost      = row.get("now_cost", None)
    ownership = row.get("selected_by_percent", row.get("ownership", None))

    # Bubble
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode="markers",
        marker=dict(
            size=26,
            color=color,
            opacity=alpha,
            line=dict(color="white", width=2.5 if is_cap else 1.5),
        ),
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"Position: {pos}<br>"
            + (f"£{cost:.1f}m<br>" if cost else "")
            + (f"xPts: {xpts:.1f}<br>" if xpts is not None else "")
            + (f"Ownership: {ownership:.1f}%<br>" if ownership is not None else "")
            + "<extra></extra>"
        ),
        showlegend=False,
    ))

    # Captain / VC badge
    badge = " ©" if is_cap else (" v" if is_vice else "")

    # Name label below bubble
    short = name if len(name) <= 11 else name[:10] + "."
    fig.add_annotation(
        x=x, y=y - 4.2,
        text=f"{short}{badge}",
        showarrow=False,
        font=dict(color="white", size=9, family="Arial"),
        bgcolor="rgba(0,0,0,0.6)",
        borderpad=2,
        opacity=alpha,
    )

    # xPts badge above bubble
    if show_xpts and xpts is not None:
        fig.add_annotation(
            x=x, y=y + 4.0,
            text=f"{xpts:.1f}",
            showarrow=False,
            font=dict(color="white", size=8),
            bgcolor=color,
            borderpad=2,
            opacity=alpha,
        )


def squad_list_to_pitch_df(squad_players: list) -> pd.DataFrame:
    """Convert a list of SquadPlayer dataclasses to a pitch-ready DataFrame."""
    rows = []
    for p in squad_players:
        rows.append({
            "web_name":           p.web_name,
            "position":           p.position,
            "now_cost":           p.cost,
            "xpts":               p.xpts,
            "is_starting":        p.is_starting,
            "is_captain":         p.is_captain,
            "is_vice_captain":    p.is_vice,
            "selected_by_percent": p.ownership,
        })
    return pd.DataFrame(rows)

