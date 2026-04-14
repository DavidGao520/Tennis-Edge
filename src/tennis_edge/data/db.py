"""SQLite database connection manager."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import ALL_DDL


class Database:
    """Context-managed SQLite connection with WAL mode and helpers."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Database:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Use 'with Database(path) as db:'")
        return self._conn

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows: list) -> sqlite3.Cursor:
        return self.conn.executemany(sql, rows)

    def commit(self) -> None:
        self.conn.commit()

    def query_df(self, sql: str, params: tuple | dict = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def query_one(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        return self.execute(sql, params).fetchone()

    def query_all(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        return self.execute(sql, params).fetchall()

    def initialize(self) -> None:
        """Create all tables and indexes."""
        for ddl in ALL_DDL:
            self.execute(ddl)
        self.commit()

    def table_count(self, table: str) -> int:
        row = self.query_one(f"SELECT COUNT(*) as cnt FROM {table}")  # noqa: S608
        return row["cnt"] if row else 0
