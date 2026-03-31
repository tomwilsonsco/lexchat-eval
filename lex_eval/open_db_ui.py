#!/usr/bin/env python3
"""Start the DuckDB UI for responses.db and keep it running."""

import duckdb
import time
from pathlib import Path

db_path = Path(__file__).parent / "data/responses.db"
print(f"Opening {db_path}")

conn = duckdb.connect(str(db_path))

# Start the server without attempting to launch a desktop browser
conn.execute("CALL start_ui_server();")

print("DuckDB UI running at http://localhost:4213")
print("Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    conn.close()
    print("Stopped.")
