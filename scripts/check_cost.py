import sys; sys.path.insert(0, ".")
from src.database.db import get_connection
db = get_connection()
rows = db.execute("SELECT web_name, position, now_cost FROM players ORDER BY now_cost DESC LIMIT 8").fetchall()
for r in rows: print(r)
