"""
Quick demo: show how the system works end-to-end without a live API call.
Generates synthetic data → trains model → runs optimizer → transfer engine.

Usage: python scripts/demo.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


def make_synthetic_players(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic player set for testing."""
    rng = np.random.default_rng(seed)
    positions = ["GK"] * 20 + ["DEF"] * 60 + ["MID"] * 80 + ["FWD"] * 40
    teams = list(range(1, 21)) * (n // 20) + list(range(1, n % 20 + 1))

    price_ranges = {"GK": (4.0, 6.5), "DEF": (4.0, 7.0), "MID": (4.5, 13.0), "FWD": (5.0, 13.5)}

    rows = []
    for i in range(n):
        pos = positions[i % len(positions)]
        lo, hi = price_ranges[pos]
        cost = round(rng.uniform(lo, hi) * 2) / 2

        # Realistic points distribution
        mean_pts = {"GK": 4.5, "DEF": 4.2, "MID": 5.1, "FWD": 5.5}[pos]
        xpts = float(np.clip(rng.normal(mean_pts, 1.5), 0, 18))
        xpts_3gw = float(np.clip(rng.normal(xpts * 3, 3), 0, 50))
        consistency = float(np.clip(rng.beta(3, 2), 0.2, 0.95))

        rows.append({
            "player_id": i + 1,
            "web_name": f"Player_{i+1}",
            "position": pos,
            "team_id": teams[i % len(teams)],
            "now_cost": cost,
            "xpts": xpts,
            "xpts_3gw": xpts_3gw,
            "xpts_5gw": xpts_3gw * 5 / 3,
            "ownership": float(np.clip(rng.exponential(15), 0.1, 80)),
            "consistency": consistency,
            "fdr_avg_3gw": float(rng.uniform(1.5, 4.5)),
            "selected_by_percent": float(np.clip(rng.exponential(15), 0.1, 80)),
            "added_gameweek": 1,
        })

    return pd.DataFrame(rows)


def demo_squad_optimizer():
    from src.optimization.squad_optimizer import optimize_squad, format_squad

    console.rule("[bold green]SQUAD OPTIMIZER DEMO")
    players = make_synthetic_players()

    result = optimize_squad(
        players_df=players,
        budget=100.0,
        risk_appetite=0.4,
        horizon="3gw",
    )

    print(format_squad(result))

    console.print(f"\n[bold]Solver status:[/] {result.solver_status}")
    console.print(f"[bold]Total cost:[/] £{result.total_cost:.1f}m")
    console.print(f"[bold]xPts next GW:[/] {result.projected_pts_gw:.1f}")
    console.print(f"[bold]xPts 3GW:[/] {result.projected_pts_3gw:.1f}")


def demo_transfer_engine():
    from src.strategy.transfer_engine import recommend_transfers

    console.rule("[bold blue]TRANSFER ENGINE DEMO")

    all_players = make_synthetic_players()

    # Simulate current squad
    squad_ids = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 11, 31, 51, 71, 91]
    my_squad = all_players[all_players["player_id"].isin(squad_ids)].copy()
    my_squad["purchase_price"] = my_squad["now_cost"]
    my_squad["added_gameweek"] = 1

    plan = recommend_transfers(
        my_squad_df=my_squad,
        all_players_df=all_players,
        free_transfers=1,
        current_gw=10,
        risk_appetite=0.4,
    )

    console.print(f"\n[bold yellow]Recommendation:[/] {plan.recommendation}\n")

    if plan.suggestions:
        table = Table(title="Transfer Suggestions")
        table.add_column("Rank", style="dim")
        table.add_column("OUT")
        table.add_column("IN")
        table.add_column("+xPts 1GW", justify="right")
        table.add_column("+xPts 3GW", justify="right")
        table.add_column("Hit?")
        table.add_column("Net Gain", justify="right")
        table.add_column("Confidence")

        for i, s in enumerate(plan.suggestions, 1):
            table.add_row(
                str(i),
                s.player_out["web_name"],
                s.player_in["web_name"],
                f"{s.expected_gain_1gw:+.1f}",
                f"{s.expected_gain_3gw:+.1f}",
                "⚠️ YES" if s.hit_required else "✅ No",
                f"{s.net_gain:+.1f}",
                s.confidence.upper(),
            )
        console.print(table)

        console.print("\n[bold]Reasoning for top suggestion:[/]")
        console.print(f"  {plan.suggestions[0].reasoning}")


def demo_fixture_analysis():
    from src.analytics.analytics import analyse_fixture_runs

    console.rule("[bold cyan]FIXTURE ANALYSIS DEMO")

    # Synthetic teams and fixtures
    teams = pd.DataFrame([
        {"id": i, "name": f"Team_{i}", "short_name": f"T{i}",
         "strength": 3, "strength_attack_home": 3, "strength_attack_away": 3,
         "strength_defence_home": 3, "strength_defence_away": 3}
        for i in range(1, 21)
    ])

    rng = np.random.default_rng(99)
    fix_rows = []
    fid = 1
    for gw in range(10, 15):
        for match in range(10):
            h = match * 2 + 1
            a = match * 2 + 2
            fix_rows.append({
                "id": fid, "gameweek_id": gw,
                "team_h": h, "team_a": a,
                "team_h_score": None, "team_a_score": None,
                "team_h_difficulty": int(rng.integers(1, 6)),
                "team_a_difficulty": int(rng.integers(1, 6)),
                "kickoff_time": None, "finished": False,
            })
            fid += 1

    fixtures = pd.DataFrame(fix_rows)
    runs = analyse_fixture_runs(teams, fixtures, current_gw=10, n_gws=5)

    table = Table(title="Fixture Runs (GW10-14)")
    for col in ["team_name", "avg_fdr", "min_fdr", "max_fdr", "n_fixtures"]:
        table.add_column(col)
    for _, row in runs.head(10).iterrows():
        table.add_row(str(row["team_name"]), f"{row['avg_fdr']:.2f}", f"{row['min_fdr']:.1f}",
                      f"{row['max_fdr']:.1f}", str(row["n_fixtures"]))
    console.print(table)


if __name__ == "__main__":
    demo_squad_optimizer()
    demo_transfer_engine()
    demo_fixture_analysis()
