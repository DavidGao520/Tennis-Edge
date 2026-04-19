"""Import Phase 1 data (players, matches, rankings, glicko2_ratings)
from a portable SQLite file into the local tennis_edge.db WITHOUT
touching market_ticks.

Usage (on Mac mini):

    # tick-logger keeps running; this script is read/append only on
    # Phase 1 tables, does not touch market_ticks.
    python scripts/import_phase1_data.py ~/phase1_data.db

What it does:
  1. ATTACHes the incoming DB as 'src'
  2. For each Phase 1 table, DELETE-then-INSERT from src
     (idempotent: safe to re-run)
  3. Reports row counts before/after
  4. Never reads or writes market_ticks

Why DELETE-then-INSERT instead of DROP-then-CREATE: schema already
exists in the local DB (initialized on tick-logger startup); we just
want to refresh row contents. DROP would lose indexes.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


PHASE1_TABLES = ("players", "matches", "rankings", "glicko2_ratings")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: import_phase1_data.py <incoming.db>", file=sys.stderr)
        return 1

    src = Path(sys.argv[1]).expanduser().resolve()
    if not src.exists():
        print(f"source file not found: {src}", file=sys.stderr)
        return 1

    # Local DB in the current working directory (matches config default).
    local = Path("data/tennis_edge.db").resolve()
    if not local.exists():
        print(f"local DB not found at {local}", file=sys.stderr)
        print("run from the tennis-edge project root.", file=sys.stderr)
        return 1

    print(f"local DB: {local}")
    print(f"source  : {src}")

    conn = sqlite3.connect(local)
    conn.execute(f"ATTACH DATABASE '{src}' AS src")

    # Show current state.
    print("\nbefore:")
    for t in (*PHASE1_TABLES, "market_ticks"):
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
        except sqlite3.OperationalError:
            n = "N/A (table missing)"
        print(f"  {t:20s} {n}")

    print("\nimporting:")
    for t in PHASE1_TABLES:
        # Check the source has it.
        try:
            src_n = conn.execute(f"SELECT COUNT(*) FROM src.{t}").fetchone()[0]
        except sqlite3.OperationalError:
            print(f"  {t:20s} SKIP (not in source)")
            continue

        # Delete existing rows, then copy from src.
        conn.execute(f"DELETE FROM main.{t}")
        conn.execute(f"INSERT INTO main.{t} SELECT * FROM src.{t}")
        conn.commit()
        print(f"  {t:20s} imported {src_n:,} rows")

    print("\nafter:")
    for t in (*PHASE1_TABLES, "market_ticks"):
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM main.{t}").fetchone()[0]
            print(f"  {t:20s} {n:,}")
        except sqlite3.OperationalError as e:
            print(f"  {t:20s} N/A ({e})")

    conn.execute("DETACH DATABASE src")
    conn.close()
    print("\ndone. market_ticks was not touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
