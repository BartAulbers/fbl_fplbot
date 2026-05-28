"""Quick health-check of all dashboard data."""
import sys, os, pickle
sys.path.insert(0, ".")
from src.database.db import get_connection
db = get_connection()

print("=== Predictions ===")
rows = db.execute(
    "SELECT model_version, COUNT(*) as n, ROUND(AVG(xpts),3) as avg_xpts "
    "FROM expected_points GROUP BY model_version"
).fetchall()
for r in rows:
    print(f"  {r}")

print("\n=== Player Metrics ===")
n = db.execute("SELECT COUNT(*) FROM player_metrics").fetchone()[0]
print(f"  {n} rows")

print("\n=== Top 5 xPts (Captaincy candidates) ===")
rows = db.execute(
    "SELECT p.web_name, ROUND(ep.xpts,2), ROUND(CAST(p.selected_by_percent AS FLOAT),1) "
    "FROM expected_points ep JOIN players p ON p.id=ep.player_id "
    "ORDER BY ep.xpts DESC NULLS LAST LIMIT 5"
).fetchall()
for r in rows:
    print(f"  {r}")

print("\n=== Top 5 Differentials (low owned, high xPts) ===")
rows = db.execute(
    "SELECT p.web_name, ROUND(CAST(p.selected_by_percent AS FLOAT),1), ROUND(ep.xpts,2) "
    "FROM expected_points ep JOIN players p ON p.id=ep.player_id "
    "WHERE CAST(p.selected_by_percent AS FLOAT) < 10 "
    "ORDER BY ep.xpts DESC NULLS LAST LIMIT 5"
).fetchall()
for r in rows:
    print(f"  {r}")

print("\n=== Current GW ===")
from app.dependencies import get_current_gw
try:
    gw = get_current_gw()
    print(f"  GW{gw}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== XGBoost Model ===")
model_path = "data/models/xpts_model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        m = pickle.load(f)
    print(f"  Type: {type(m).__name__}, features: {len(m.feature_names_in_)}")
    print(f"  File size: {os.path.getsize(model_path)//1024} KB")
else:
    print("  NO MODEL FILE")

print("\n=== Fixture data ===")
n_fx = db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
n_teams = db.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
print(f"  Fixtures: {n_fx}, Teams: {n_teams}")

print("\n=== Squad check (saved squads) ===")
try:
    n_sq = db.execute("SELECT COUNT(*) FROM my_squad").fetchone()[0]
    print(f"  my_squad rows: {n_sq}")
except Exception as e:
    print(f"  Error: {e}")

print("\nAll checks complete.")
