"""Full dashboard data test — all pages."""
import sys, os, pickle
sys.path.insert(0, ".")
from src.database.db import get_connection

db = get_connection()
errors = []

def check(name, fn):
    try:
        result = fn()
        print(f"  OK  {name}: {result}")
    except Exception as e:
        print(f"  ERR {name}: {e}")
        errors.append((name, str(e)))

print("=== 1. Data availability ===")
check("Players count", lambda: db.execute("SELECT COUNT(*) FROM players").fetchone()[0])
check("Teams count", lambda: db.execute("SELECT COUNT(*) FROM teams").fetchone()[0])
check("Fixtures count", lambda: db.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0])
check("History rows", lambda: db.execute("SELECT COUNT(*) FROM player_gw_history").fetchone()[0])
check("Gameweeks", lambda: db.execute("SELECT COUNT(*) FROM gameweeks").fetchone()[0])

print("\n=== 2. Models and predictions ===")
check("Model file exists", lambda: "YES" if os.path.exists("data/models/xpts_model.pkl") else "MISSING")
check("Model feature count", lambda: len(pickle.load(open("data/models/xpts_model.pkl","rb")).feature_names_in_))
check("Expected points rows", lambda: db.execute("SELECT COUNT(*) FROM expected_points").fetchone()[0])
check("Latest model version", lambda: db.execute("SELECT model_version FROM expected_points ORDER BY created_at DESC LIMIT 1").fetchone()[0])
check("Avg xPts", lambda: round(db.execute("SELECT AVG(xpts) FROM expected_points WHERE model_version='v3_fpl_rules'").fetchone()[0], 3))

print("\n=== 3. Analytics tables ===")
check("player_metrics rows", lambda: db.execute("SELECT COUNT(*) FROM player_metrics").fetchone()[0])
check("Metrics cols", lambda: [r[0] for r in db.execute("DESCRIBE player_metrics").fetchall()])

print("\n=== 4. Captaincy optimizer ===")
check("Top captain", lambda: db.execute(
    "SELECT p.web_name, ROUND(ep.xpts,2) FROM expected_points ep "
    "JOIN players p ON p.id=ep.player_id "
    "ORDER BY ep.xpts DESC LIMIT 1"
).fetchone())

print("\n=== 5. Differential finder ===")
check("Differentials count (<10%)", lambda: db.execute(
    "SELECT COUNT(*) FROM expected_points ep JOIN players p ON p.id=ep.player_id "
    "WHERE CAST(p.selected_by_percent AS FLOAT) < 10.0"
).fetchone()[0])

print("\n=== 6. My Squad ===")
check("my_squad rows", lambda: db.execute("SELECT COUNT(*) FROM my_squad").fetchone()[0])
check("my_squad cols", lambda: [r[0] for r in db.execute("DESCRIBE my_squad").fetchall()])

print("\n=== 7. Current GW ===")
from app.dependencies import get_current_gw
check("get_current_gw()", lambda: get_current_gw())

print("\n=== 8. Pitch view import ===")
check("pitch_view import", lambda: __import__("src.analytics.pitch_view", fromlist=["draw_pitch"]) and "OK")

print("\n=== 9. Optimizer import ===")
check("squad_optimizer import", lambda: __import__("src.optimization.squad_optimizer", fromlist=["optimize_squad"]) and "OK")

print("\n=== 10. Transfer engine import ===")
check("transfer_engine import", lambda: __import__("src.strategy.transfer_engine", fromlist=["suggest_transfers"]) and "OK")

print(f"\n{'='*40}")
if errors:
    print(f"FAILED: {len(errors)} errors")
    for name, err in errors:
        print(f"  - {name}: {err}")
else:
    print("ALL CHECKS PASSED")
