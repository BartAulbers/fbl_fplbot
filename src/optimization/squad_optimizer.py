"""
Squad Optimizer using Integer Linear Programming (PuLP).

Selects optimal 15-player squad + starting XI + captain
subject to FPL rules and budget constraints.

Risk appetite parameter (0.0 = safe, 1.0 = differential):
- Adjusts weight between xpts (expected) vs consistency vs ownership
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import pulp
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


@dataclass
class SquadPlayer:
    player_id: int
    web_name: str
    team_id: int
    position: str
    cost: float
    xpts: float
    xpts_3gw: float
    ownership: float
    consistency: float
    is_starting: bool
    is_captain: bool
    is_vice: bool
    bench_order: Optional[int]  # None if starting


@dataclass
class SquadResult:
    squad: list[SquadPlayer]
    total_cost: float
    projected_pts_gw: float
    projected_pts_3gw: float
    budget_remaining: float
    solver_status: str


# ── Score calculation ─────────────────────────────────────────────────────────

def compute_selection_score(
    row: pd.Series,
    risk_appetite: float,
    horizon: str = "1gw",
) -> float:
    """
    Composite selection score blending:
      - Expected points (horizon-adjusted)
      - Consistency (safe picks)
      - Differential bonus (low-owned high scorers)

    risk_appetite: 0.0 = pure expected + consistency
                   1.0 = pure differential + upside
    """
    xpts = row["xpts"] if horizon == "1gw" else row.get("xpts_3gw", row["xpts"] * 3)

    # Base score: expected points
    score = xpts

    # Safe pick bonus: consistency reduces variance risk
    safe_bonus = row.get("consistency", 0.5) * xpts * 0.2
    score += safe_bonus * (1 - risk_appetite)

    # Differential bonus: low ownership + high xpts = value
    diff_bonus = (1 - row.get("ownership", 50) / 100.0) * xpts * 0.3
    score += diff_bonus * risk_appetite

    return float(score)


# ── ILP Optimizer ─────────────────────────────────────────────────────────────

def optimize_squad(
    players_df: pd.DataFrame,
    budget: float = 100.0,
    risk_appetite: float = 0.5,
    horizon: str = "1gw",
    locked_player_ids: Optional[list[int]] = None,
    excluded_player_ids: Optional[list[int]] = None,
) -> SquadResult:
    """
    Solve the FPL squad selection ILP.

    Parameters
    ----------
    players_df : must contain columns:
        player_id, web_name, team_id, position, now_cost,
        xpts, xpts_3gw, selected_by_percent, consistency
    budget : total budget in £m
    risk_appetite : 0.0–1.0
    horizon : '1gw' | '3gw' | '5gw'
    locked_player_ids : force these players into the squad
    excluded_player_ids : exclude these players

    Returns SquadResult
    """
    locked = set(locked_player_ids or [])
    excluded = set(excluded_player_ids or [])

    # Filter available players
    df = players_df[~players_df["player_id"].isin(excluded)].copy()
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    df = df.dropna(subset=["now_cost", "xpts"])
    df = df.reset_index(drop=True)
    n = len(df)

    # Scores
    df["score"] = df.apply(
        lambda r: compute_selection_score(r, risk_appetite, horizon), axis=1
    )

    # ── Decision variables ────────────────────────────────────────────────
    prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

    # x[i] = 1 if player i is in the squad (15)
    x = pulp.LpVariable.dicts("x", range(n), cat="Binary")
    # s[i] = 1 if player i is in the starting XI
    s = pulp.LpVariable.dicts("s", range(n), cat="Binary")
    # c[i] = 1 if player i is captain (doubles points)
    cap = pulp.LpVariable.dicts("cap", range(n), cat="Binary")
    # vc[i] = 1 if player i is vice-captain
    vc = pulp.LpVariable.dicts("vc", range(n), cat="Binary")

    # ── Objective: maximise total starting XI score + captain bonus ───────
    # Captain gets +0.5 * xpts effectively (doubles one player)
    prob += pulp.lpSum([
        df.loc[i, "score"] * s[i]
        + df.loc[i, "score"] * cap[i]   # captain gets their score doubled
        for i in range(n)
    ])

    # ── Squad size constraints ────────────────────────────────────────────
    prob += pulp.lpSum(x[i] for i in range(n)) == 15  # 15-player squad

    # Position constraints in squad
    for pos, req in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        mask = [i for i in range(n) if df.loc[i, "position"] == pos]
        prob += pulp.lpSum(x[i] for i in mask) == req

    # ── Budget constraint ─────────────────────────────────────────────────
    prob += pulp.lpSum(df.loc[i, "now_cost"] * x[i] for i in range(n)) <= budget

    # ── Max 3 per team ────────────────────────────────────────────────────
    for team_id in df["team_id"].unique():
        team_mask = [i for i in range(n) if df.loc[i, "team_id"] == team_id]
        if team_mask:
            prob += pulp.lpSum(x[i] for i in team_mask) <= settings.max_players_per_team

    # ── Starting XI constraints ───────────────────────────────────────────
    prob += pulp.lpSum(s[i] for i in range(n)) == 11

    # Can only start if in squad
    for i in range(n):
        prob += s[i] <= x[i]

    # Starting XI position rules (valid formations only)
    gk_start = [i for i in range(n) if df.loc[i, "position"] == "GK"]
    def_start = [i for i in range(n) if df.loc[i, "position"] == "DEF"]
    mid_start = [i for i in range(n) if df.loc[i, "position"] == "MID"]
    fwd_start = [i for i in range(n) if df.loc[i, "position"] == "FWD"]

    prob += pulp.lpSum(s[i] for i in gk_start) == 1         # exactly 1 GK starts
    prob += pulp.lpSum(s[i] for i in def_start) >= 3        # min 3 DEF
    prob += pulp.lpSum(s[i] for i in mid_start) >= 2        # min 2 MID
    prob += pulp.lpSum(s[i] for i in fwd_start) >= 1        # min 1 FWD

    # ── Captain constraints ───────────────────────────────────────────────
    prob += pulp.lpSum(cap[i] for i in range(n)) == 1       # exactly 1 captain
    prob += pulp.lpSum(vc[i] for i in range(n)) == 1        # exactly 1 vice

    for i in range(n):
        prob += cap[i] <= s[i]    # captain must start
        prob += vc[i] <= x[i]     # vice must be in squad
        prob += cap[i] + vc[i] <= 1  # can't be both

    # ── Lock constraints ──────────────────────────────────────────────────
    for pid in locked:
        lock_idx = df[df["player_id"] == pid].index.tolist()
        if lock_idx:
            prob += x[lock_idx[0]] == 1

    # ── Solve ──────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    logger.info("ILP solver status: {}", status)

    if prob.status != 1:  # not Optimal
        raise RuntimeError(f"Solver did not find optimal solution: {status}")

    # ── Extract solution ──────────────────────────────────────────────────
    squad_players = []
    for i in range(n):
        if pulp.value(x[i]) > 0.5:
            row = df.loc[i]
            is_starting = pulp.value(s[i]) > 0.5
            is_captain = pulp.value(cap[i]) > 0.5
            is_vice = pulp.value(vc[i]) > 0.5

            squad_players.append(SquadPlayer(
                player_id=int(row["player_id"]),
                web_name=str(row["web_name"]),
                team_id=int(row["team_id"]),
                position=str(row["position"]),
                cost=float(row["now_cost"]),
                xpts=float(row.get("xpts", 0)),
                xpts_3gw=float(row.get("xpts_3gw", 0)),
                ownership=float(row.get("selected_by_percent", 0)),
                consistency=float(row.get("consistency", 0.5)),
                is_starting=is_starting,
                is_captain=is_captain,
                is_vice=is_vice,
                bench_order=None if is_starting else _bench_order(
                    int(row["player_id"]), [
                        p for p in squad_players if not p.is_starting
                    ], row
                ),
            ))

    total_cost = sum(p.cost for p in squad_players)
    starting = [p for p in squad_players if p.is_starting]
    cap_player = next((p for p in squad_players if p.is_captain), None)

    projected_pts = sum(p.xpts for p in starting)
    if cap_player:
        projected_pts += cap_player.xpts  # captain double

    return SquadResult(
        squad=squad_players,
        total_cost=total_cost,
        projected_pts_gw=projected_pts,
        projected_pts_3gw=sum(p.xpts_3gw for p in starting),
        budget_remaining=budget - total_cost,
        solver_status=status,
    )


def _bench_order(player_id: int, bench_so_far: list, row: pd.Series) -> int:
    """Assign bench order: GK last, others by xpts desc."""
    if row["position"] == "GK":
        return 4  # bench GK always last
    return len(bench_so_far) + 1


# ── Pretty print helper ───────────────────────────────────────────────────────

def format_squad(result: SquadResult) -> str:
    lines = [
        f"\n{'='*60}",
        f"  OPTIMAL SQUAD  (£{result.total_cost:.1f}m / £{result.total_cost + result.budget_remaining:.1f}m)",
        f"  Projected GW pts: {result.projected_pts_gw:.1f}",
        f"  Projected 3GW pts: {result.projected_pts_3gw:.1f}",
        f"{'='*60}",
        "\n  STARTING XI",
        f"  {'Name':<20} {'Pos':<5} {'£':<6} {'xPts':>6} {'xPts3':>7} {'Own%':>6}",
        "  " + "-"*54,
    ]

    starting = sorted(
        [p for p in result.squad if p.is_starting],
        key=lambda p: ["GK", "DEF", "MID", "FWD"].index(p.position)
    )
    for p in starting:
        flag = " (C)" if p.is_captain else " (V)" if p.is_vice else ""
        lines.append(
            f"  {p.web_name + flag:<20} {p.position:<5} £{p.cost:<5.1f} {p.xpts:>6.1f} {p.xpts_3gw:>7.1f} {p.ownership:>5.1f}%"
        )

    lines += ["\n  BENCH", "  " + "-"*54]
    bench = sorted(
        [p for p in result.squad if not p.is_starting],
        key=lambda p: (p.bench_order or 99)
    )
    for p in bench:
        lines.append(
            f"  {p.web_name:<20} {p.position:<5} £{p.cost:<5.1f} {p.xpts:>6.1f}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
