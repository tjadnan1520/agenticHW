"""The two tools. They do not import or call each other; only the orchestrator links them."""

import time
from datetime import date

from . import guard
from .cache import SqlCache, make_key
from .db import Database, DatabaseError
from .llm import Ollama

SQL_PROMPT = """You write SQLite SQL for a personal expense table.
Reply with exactly ONE statement and nothing else: no markdown, no explanation.

Today is {today}.

Rules:
- Only SELECT or INSERT. Use only the table and columns in the schema.
- Dates are text 'YYYY-MM-DD'. Today is date('now','localtime'). Yesterday is date('now','localtime','-1 day').
- "last N days" means spent_on >= date('now','localtime','-N day'). "this month" means spent_on >= date('now','localtime','start of month').
- Date modifiers allowed: 'localtime', 'start of month', 'start of year', '-N day', '-N month'. Nothing else.
- To add several expenses, write ONE INSERT with several VALUES rows (spent_on, amount, category, note). Never set id or created_at.
- Amounts are plain numbers. category must be one of the values listed in the schema.
- If the request cannot be answered from this schema, reply exactly: SELECT 'UNSUPPORTED'

Schema:
{schema}

Examples:
Request: how much did I spend on food in the last 10 days
SQL: SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE category = 'food' AND spent_on >= date('now','localtime','-10 day')

Request: spending per category this month
SQL: SELECT category, SUM(amount) AS total FROM expenses WHERE spent_on >= date('now','localtime','start of month') GROUP BY category ORDER BY total DESC

Request: yesterday lunch 180, rickshaw 60
SQL: INSERT INTO expenses (spent_on, amount, category, note) VALUES (date('now','localtime','-1 day'), 180, 'food', 'lunch'), (date('now','localtime','-1 day'), 60, 'transport', 'rickshaw')
"""


class GenerateSql:
    name = "generate_sql"
    description = "Write one SQLite statement (SELECT or INSERT) for the user's request. Does not run it."

    def __init__(self, llm: Ollama, db: Database, cache: SqlCache):
        self._llm, self._db, self._cache = llm, db, cache

    def run(self, question: str, previous_sql: str | None = None,
            error: str | None = None, use_cache: bool = True) -> dict:
        key = make_key(question, date.today().isoformat(), self._llm.model, self._db.columns_signature())
        if use_cache:
            hit = self._cache.get(key)
            if hit:
                return {"sql": hit["sql"], "key": key, "cached": True, "saved_at": hit["created_at"],
                        "hits": hit.get("hits", 0) + 1, "ms": 0}

        system = SQL_PROMPT.format(today=date.today().isoformat(), schema=self._db.schema_text())
        user = f"Request: {question}"
        if error:
            user += f"\n\nYour previous SQL failed.\nSQL: {previous_sql}\nError: {error}\nWrite a corrected statement."
        started = time.perf_counter()
        raw = self._llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             temperature=0.1, num_predict=300)
        return {"sql": guard.extract_sql(raw), "key": key, "cached": False,
                "ms": round((time.perf_counter() - started) * 1000)}


class ExecuteSql:
    name = "execute_sql"
    description = "Run the most recently generated SQL on the database and return rows or an error."

    def __init__(self, db: Database):
        self._db = db

    def run(self, sql: str) -> dict:
        try:
            kind = guard.validate(sql)
        except guard.SqlRejected as exc:
            return {"ok": False, "error": f"rejected by guard: {exc}", "retryable": exc.retryable}
        started = time.perf_counter()
        try:
            result = self._db.run(sql, write=kind == "INSERT")
        except DatabaseError as exc:
            return {"ok": False, "error": str(exc), "retryable": True}
        return {"ok": True, "kind": kind, "columns": result.columns, "rows": result.rows,
                "rowcount": result.rowcount, "truncated": result.truncated,
                "ms": round((time.perf_counter() - started) * 1000)}
