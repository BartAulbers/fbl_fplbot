"""
Transfer Recommendation Engine

Philosophy:
  "A calm, data-driven manager that prevents bad impulsive decisions."

Rules:
1. Default: suggest FREE transfers only
2. -4 hit only if net expected gain > hit_threshold_pts (default 8pts)
3. Churn prevention: don't suggest selling a player held < min_gw_hold GWs
4. Consider next 3-5 GW fixture run, not just next week
5. Rank by expected points gain, penalised by uncertainty
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


@dataclass
class TransferSuggestion:
    player_out: dict          # {player_id, web_name, pos, cost, xpts, xpts_3gw, xpts_5gw}
    player_in: dict           # same
    expected_gain_1gw: float  # xpts_in - xpts_out for next GW
    expected_gain_3gw: float  # xpts_in_3gw - xpts_out_3gw
    expected_gain_5gw: float  # xpts_in_5gw - xpts_out_5gw
    hit_required: bool
    net_gain: float           # expected_gain - (4 if hit else 0)
    reasoning: str
    confidence: str           # 'high' | 'medium' | 'low'


@dataclass
class TransferPlan:
    suggestions: list[TransferSuggestion]
    free_transfers_available: int
    current_gw: int
    recommendation: str       # plain-language advice


# ── Core engine ───────────────────────────────────────────────────────────────

def recommend_transfers(
    my_squad_df: pd.DataFrame,
    all_players_df: pd.DataFrame,
    free_transfers: int = 1,
    current_gw: int = 1,
    risk_appetite: float = 0.5,
    max_suggestions: int = 5,
) -> TransferPlan:
    """
    Generate transfer recommendations.

    Parameters
    ----------
    my_squad_df : current squad with columns:
        player_id, web_name, position, now_cost, purchase_price,
        xpts, xpts_3gw, xpts_5gw, consistency, added_gameweek,
        selected_by_percent, fdr_avg_3gw, team_id
    all_players_df : all available players (same schema + ownership)
    free_transfers : number of free transfers this GW
    current_gw : current gameweek number
    risk_appetite : 0.0-1.0
    max_suggestions : top N suggestions to return
    """
    suggestions = []

    for _, out_player in my_squad_df.iterrows():
        # Churn prevention: don't sell if held too recently
        gws_held = current_gw - (out_player.get("added_gameweek") or 1)
        if gws_held < settings.churn_prevention_min_gw_hold:
            logger.debug("Churn guard: skipping {} (held {} GWs)", out_player["web_name"], gws_held)
            continue

        # Find replacements: same position, can afford, not in squad
        budget = out_player["now_cost"] + _available_funds(my_squad_df, settings.squad_budget)
        squad_ids = set(my_squad_df["player_id"].tolist())

        candidates = all_players_df[
            (all_players_df["position"] == out_player["position"])
            & (all_players_df["now_cost"] <= budget)
            & (~all_players_df["player_id"].isin(squad_ids))
            & (all_players_df["player_id"] != out_player["player_id"])
        ].copy()

        # Enforce max 3 per team rule
        team_counts = my_squad_df[
            my_squad_df["player_id"] != out_player["player_id"]
        ]["team_id"].value_counts()

        candidates = candidates[
            candidates["team_id"].map(
                lambda tid: team_counts.get(tid, 0) < settings.max_players_per_team
            )
        ]

        if candidates.empty:
            continue

        # Score candidates
        candidates["transfer_score"] = candidates.apply(
            lambda r: _score_candidate(r, risk_appetite), axis=1
        )
        candidates = candidates.sort_values("transfer_score", ascending=False)

        for _, in_player in candidates.head(3).iterrows():
            gain_1gw = in_player.get("xpts", 0) - out_player.get("xpts", 0)
            gain_3gw = in_player.get("xpts_3gw", 0) - out_player.get("xpts_3gw", 0)
            gain_5gw = in_player.get("xpts_5gw", 0) - out_player.get("xpts_5gw", 0)

            if gain_1gw <= 0.5 and gain_5gw <= 0.5:
                continue  # skip negligible gains across both planning horizons

            reasoning = _build_reasoning(out_player, in_player, gain_1gw, gain_3gw, gain_5gw, False)
            confidence = _assess_confidence(in_player, gain_1gw)

            suggestions.append(TransferSuggestion(
                player_out=_player_dict(out_player),
                player_in=_player_dict(in_player),
                expected_gain_1gw=round(gain_1gw, 2),
                expected_gain_3gw=round(gain_3gw, 2),
                expected_gain_5gw=round(gain_5gw, 2),
                hit_required=False,  # assigned below after sorting
                net_gain=round(gain_5gw, 2),
                reasoning=reasoning,
                confidence=confidence,
            ))

    # Sort best first; then assign hits to transfers beyond the free allowance
    suggestions.sort(key=lambda s: -s.net_gain)

    final: list[TransferSuggestion] = []
    for i, s in enumerate(suggestions):
        if i < free_transfers:
            # Free transfer — keep as-is
            final.append(s)
        else:
            # Would cost a -4 hit
            net_after_hit = s.expected_gain_5gw - 4
            if net_after_hit < settings.hit_threshold_pts:
                continue  # not worth the hit
            # Rebuild with hit flag
            hit_reasoning = _build_reasoning(
                my_squad_df[my_squad_df["player_id"] == s.player_out["player_id"]].iloc[0]
                if not my_squad_df[my_squad_df["player_id"] == s.player_out["player_id"]].empty
                else pd.Series(s.player_out),
                pd.Series(s.player_in),
                s.expected_gain_1gw,
                s.expected_gain_3gw,
                s.expected_gain_5gw,
                True,
            )
            final.append(TransferSuggestion(
                player_out=s.player_out,
                player_in=s.player_in,
                expected_gain_1gw=s.expected_gain_1gw,
                expected_gain_3gw=s.expected_gain_3gw,
                expected_gain_5gw=s.expected_gain_5gw,
                hit_required=True,
                net_gain=round(net_after_hit, 2),
                reasoning=hit_reasoning,
                confidence=s.confidence,
            ))
        if len(final) >= max_suggestions:
            break

    recommendation = _build_recommendation(final, free_transfers, current_gw)

    return TransferPlan(
        suggestions=final,
        free_transfers_available=free_transfers,
        current_gw=current_gw,
        recommendation=recommendation,
    )


def _available_funds(my_squad_df: pd.DataFrame, total_budget: float) -> float:
    """How much budget is unspent (ITB)."""
    squad_cost = my_squad_df["now_cost"].sum()
    return max(0, total_budget - squad_cost)


def _score_candidate(row: pd.Series, risk_appetite: float) -> float:
    """
    Scoring for transfer candidate:
    - Base: xpts_5gw (planning horizon)
    - Bonus: consistency, fixture, differential
    """
    xpts_5gw = row.get("xpts_5gw", 0)
    fdr = row.get("fdr_avg_3gw", 3)
    fixture_bonus = (5 - fdr) / 5.0 * xpts_5gw * 0.15

    consistency = row.get("consistency", 0.5)
    diff_bonus = (1 - row.get("selected_by_percent", 50) / 100.0) * xpts_5gw * 0.2

    safe_score = xpts_5gw + fixture_bonus + consistency * xpts_5gw * 0.1
    risky_score = xpts_5gw + fixture_bonus + diff_bonus

    return safe_score * (1 - risk_appetite) + risky_score * risk_appetite


def _player_dict(row: pd.Series) -> dict:
    return {
        "player_id": int(row["player_id"]),
        "web_name": str(row["web_name"]),
        "position": str(row["position"]),
        "cost": float(row["now_cost"]),
        "xpts": float(row.get("xpts", 0)),
        "xpts_3gw": float(row.get("xpts_3gw", 0)),
        "xpts_5gw": float(row.get("xpts_5gw", 0)),
        "ownership": float(row.get("selected_by_percent", 0)),
        "fdr_3gw": float(row.get("fdr_avg_3gw", 3)),
    }


def _build_reasoning(
    out: pd.Series,
    inp: pd.Series,
    g1: float,
    g3: float,
    g5: float,
    hit: bool,
) -> str:
    parts = []
    if g1 > 2:
        parts.append(f"Expected +{g1:.1f}pts next GW")
    if g3 > 5:
        parts.append(f"+{g3:.1f}pts over 3 GWs")
    if g5 > 8:
        parts.append(f"+{g5:.1f}pts over 5 GWs")
    fdr_out = out.get("fdr_avg_3gw", 3)
    fdr_in = inp.get("fdr_avg_3gw", 3)
    if fdr_in < fdr_out - 0.5:
        parts.append(f"Better fixtures ({fdr_in:.1f} vs {fdr_out:.1f} avg FDR)")
    if inp.get("consistency", 0) > out.get("consistency", 0) + 0.1:
        parts.append("More consistent returner")
    if inp.get("selected_by_percent", 50) < 10:
        parts.append(f"Differential ({inp['selected_by_percent']:.1f}% owned)")
    if hit:
        parts.append(f"Requires -4 hit (net gain: {g5 - 4:.1f}pts over 5 GWs)")
    return ". ".join(parts) if parts else "Marginal improvement"


def _assess_confidence(in_player: pd.Series, gain_1gw: float) -> str:
    if gain_1gw > 3 and in_player.get("consistency", 0) > 0.6:
        return "high"
    elif gain_1gw > 1.5:
        return "medium"
    return "low"


def _build_recommendation(
    suggestions: list[TransferSuggestion],
    free_transfers: int,
    current_gw: int,
) -> str:
    if not suggestions:
        return (
            f"🔒 ROLL TRANSFER — No compelling upgrades found this week. "
            f"Bank your free transfer for GW{current_gw + 1}."
        )

    free_moves = [s for s in suggestions if not s.hit_required]
    hit_moves = [s for s in suggestions if s.hit_required]

    if not free_moves:
        best = suggestions[0]
        if best.expected_gain_5gw >= settings.hit_threshold_pts + 4:
            return (
                f"⚠️ CONSIDER HIT — {best.player_in['web_name']} over {best.player_out['web_name']} "
                f"expected +{best.expected_gain_5gw:.1f}pts over 5GWs (net +{best.net_gain:.1f} after -4). "
                f"Only take if confident in fixtures."
            )
        return (
            f"🔒 ROLL TRANSFER — Hit not justified ({best.expected_gain_5gw:.1f}pts gain < "
            f"{settings.hit_threshold_pts + 4:.0f}pts threshold). Wait for a better opportunity."
        )

    best = free_moves[0]
    n_free = len(free_moves)

    if free_transfers > 1 and n_free >= free_transfers:
        # Enough good moves to use all free transfers
        names_in = ", ".join(s.player_in["web_name"] for s in free_moves[:free_transfers])
        total_gain = sum(s.expected_gain_5gw for s in free_moves[:free_transfers])
        return (
            f"✅ USE ALL {free_transfers} FTs — Bring in {names_in}. "
            f"Combined expected +{total_gain:.1f}pts over 5GWs."
            + (f" Also {len(hit_moves)} hit option(s) available." if hit_moves else "")
        )

    if free_transfers > 1 and n_free < free_transfers:
        # Fewer good moves than FTs available — suggest rolling the rest
        names_in = ", ".join(s.player_in["web_name"] for s in free_moves)
        return (
            f"✅ USE {n_free} of {free_transfers} FTs — {names_in}. "
            f"Roll the remaining {free_transfers - n_free} FT(s) to next week."
            + (f" Also {len(hit_moves)} hit option(s) available." if hit_moves else "")
        )

    # Single free transfer
    return (
        f"✅ FREE TRANSFER — {best.player_in['web_name']} in, {best.player_out['web_name']} out. "
        f"Expected +{best.expected_gain_1gw:.1f}pts GW, +{best.expected_gain_5gw:.1f}pts 5GW. "
        f"Confidence: {best.confidence.upper()}."
        + (f" Also {len(hit_moves)} hit option(s) available." if hit_moves else "")
    )
