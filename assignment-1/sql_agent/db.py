"""SQLite storage. Standard library only.

Two safety layers live here (the third is guard.py):
  * SELECTs open the file read-only (mode=ro), so a write cannot sneak in.
  * An sqlite3 authorizer allows only SELECT/INSERT on the `expenses` table.
"""

import sqlite3
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

CATEGORIES = ("food", "transport", "bills", "shopping", "health",
              "education", "entertainment", "transfer", "other")
MAX_ROWS = 100
QUERY_TIMEOUT_S = 5

_DDL = f"""
CREATE TABLE IF NOT EXISTS expenses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    spent_on   TEXT NOT NULL DEFAULT (date('now','localtime')),
    amount     REAL NOT NULL CHECK (amount > 0),
    category   TEXT NOT NULL DEFAULT 'other'
               CHECK (category IN ({", ".join(repr(c) for c in CATEGORIES)})),
    note       TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
)"""

# Shown to the model next to each column; this is what teaches it the formats.
_COLUMN_DOCS = {
    "id": "auto-generated, never set it",
    "spent_on": "date of the expense, YYYY-MM-DD text",
    "amount": "amount in BDT (taka), always positive",
    "category": "one of: " + ", ".join(CATEGORIES),
    "note": "short free text such as lunch or rickshaw",
    "created_at": "auto-generated, never set it",
}

# (days ago, amount, category, note)
_SEED = [
    (0, 180, "food", "lunch"), (0, 60, "transport", "rickshaw"),
    (1, 250, "food", "dinner with friends"), (2, 500, "transfer", "bkash to Rafi"),
    (3, 120, "food", "tea and snacks"), (4, 1200, "bills", "internet"),
    (6, 350, "shopping", "notebook and pens"), (8, 90, "transport", "bus"),
    (9, 300, "entertainment", "movie"), (12, 220, "food", "lunch"),
    (15, 800, "health", "pharmacy"), (20, 450, "education", "course material"),
]


class DatabaseError(Exception):
    pass


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    rowcount: int
    truncated: bool


def _authorizer(action, arg1, arg2, _db, _source):
    ok, deny = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_SELECT or action == sqlite3.SQLITE_RECURSIVE:
        return ok
    if action in (sqlite3.SQLITE_READ, sqlite3.SQLITE_INSERT):
        return ok if arg1 == "expenses" else deny
    if action == sqlite3.SQLITE_FUNCTION:
        return deny if (arg2 or "").lower() == "load_extension" else ok
    return deny  # UPDATE, DELETE, PRAGMA, ATTACH, DDL, transactions ...


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)

    # --- setup -----------------------------------------------------------

    def init(self, seed: bool = True) -> None:
        """Create the table; fill it with sample rows only if it is empty."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(_DDL)
            empty = conn.execute("SELECT COUNT(*) FROM expenses").fetchone()[0] == 0
            if seed and empty:
                today = date.today()
                conn.executemany(
                    "INSERT INTO expenses (spent_on, amount, category, note) VALUES (?,?,?,?)",
                    [((today - timedelta(days=d)).isoformat(), a, c, n) for d, a, c, n in _SEED],
                )
        conn.close()

    # --- what the model sees ---------------------------------------------

    def columns_signature(self) -> str:
        """Column names and types only. Used in the cache key, so adding rows never invalidates it."""
        with sqlite3.connect(self.path) as conn:
            info = conn.execute("PRAGMA table_info(expenses)").fetchall()
        conn.close()
        return ",".join(f"{r[1]}:{r[2]}" for r in info)

    def schema_text(self, sample_rows: int = 3) -> str:
        with sqlite3.connect(self.path) as conn:
            info = conn.execute("PRAGMA table_info(expenses)").fetchall()
            sample = conn.execute(
                "SELECT spent_on, amount, category, note FROM expenses ORDER BY id DESC LIMIT ?",
                (sample_rows,),
            ).fetchall()
        conn.close()
        lines = [f"  {r[1]} {r[2]}  -- {_COLUMN_DOCS.get(r[1], '')}" for r in info]
        text = "TABLE expenses (\n" + "\n".join(lines) + "\n)"
        if sample:
            text += "\n-- sample rows (spent_on, amount, category, note):\n"
            text += "\n".join(f"-- {list(row)}" for row in sample)
        return text

    # --- running SQL -----------------------------------------------------

    def run(self, sql: str, write: bool) -> QueryResult:
        uri = f"{self.path.resolve().as_uri()}?mode={'rw' if write else 'ro'}"
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, isolation_level=None)  # autocommit: one INSERT is atomic
            conn.set_authorizer(_authorizer)
            deadline = time.monotonic() + QUERY_TIMEOUT_S
            conn.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
            cur = conn.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            fetched = cur.fetchmany(MAX_ROWS + 1) if columns else []
            rowcount = cur.rowcount if write else len(fetched)
        except sqlite3.Error as exc:
            raise DatabaseError(str(exc)) from exc
        finally:
            if conn is not None:
                conn.close()
        return QueryResult(columns, [list(r) for r in fetched[:MAX_ROWS]], rowcount, len(fetched) > MAX_ROWS)
