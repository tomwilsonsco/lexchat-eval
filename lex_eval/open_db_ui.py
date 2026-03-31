import duckdb
import time
from pathlib import Path

db_path = Path(__file__).parent / "data/responses.db"
print(f"Opening {db_path}")

conn = duckdb.connect(str(db_path))

conn.execute("CALL start_ui_server();")

print("DuckDB UI running at http://localhost:4213")
print("Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    conn.close()
    print("Stopped.")
