"""Durable SQLite cache for SONIC point evaluations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS point_cache (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def put_cache(connection: sqlite3.Connection, cache_key: str, payload: dict[str, Any]) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO point_cache(cache_key, payload, created_at) VALUES (?, ?, ?)",
        (cache_key, json.dumps(payload, sort_keys=True), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()


def get_cache(connection: sqlite3.Connection, cache_key: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT payload FROM point_cache WHERE cache_key = ?", (cache_key,)).fetchone()
    return json.loads(row[0]) if row else None


def cache_stats(connection: sqlite3.Connection) -> dict[str, int]:
    count = int(connection.execute("SELECT COUNT(*) FROM point_cache").fetchone()[0])
    return {"entries": count}


def export_parquet(connection: sqlite3.Connection, path: Path) -> Path | None:
    """Export the cache as Parquet when the optional Arrow runtime is present."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError:
        return None
    rows = connection.execute("SELECT cache_key, payload, created_at FROM point_cache").fetchall()
    table = pa.table({"cache_key": [row[0] for row in rows], "payload": [row[1] for row in rows], "created_at": [row[2] for row in rows]})
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, path)
    return path


__all__ = ["cache_stats", "export_parquet", "get_cache", "open_cache", "put_cache"]
