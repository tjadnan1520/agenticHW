"""JSON cache of generated SQL (never of results, which change as rows are added).

Entry format, one per key:
    {"question": ..., "sql": ..., "model": ..., "created_at": ISO time,
     "last_used_at": ISO time, "hits": N}
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_key(question: str, day: str, model: str, columns: str) -> str:
    """Everything that decides the SQL: the question, today's date, the model, the table layout."""
    payload = json.dumps([" ".join(question.lower().split()), day, model, columns])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SqlCache:
    def __init__(self, path: Path, max_entries: int = 500, enabled: bool = True):
        self._path = Path(path)
        self._max = max_entries
        self._enabled = enabled
        self._entries: dict[str, dict] = self._read() if enabled else {}

    def get(self, key: str) -> dict | None:
        """Return the entry (with its previous last_used_at) and record this use."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        before = dict(entry)
        entry["hits"] = entry.get("hits", 0) + 1
        entry["last_used_at"] = _now()
        self._write()
        return before

    def put(self, key: str, question: str, sql: str, model: str) -> None:
        if not self._enabled:
            return
        now = _now()
        self._entries[key] = {"question": question, "sql": sql, "model": model,
                              "created_at": now, "last_used_at": now, "hits": 0}
        if len(self._entries) > self._max:
            oldest = sorted(self._entries, key=lambda k: self._entries[k]["last_used_at"])
            for stale in oldest[: len(self._entries) - self._max]:
                del self._entries[stale]
        self._write()

    def delete(self, key: str) -> None:
        if self._entries.pop(key, None) is not None:
            self._write()

    def _read(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}  # missing or corrupt file is just an empty cache

    def _write(self) -> None:
        if not self._enabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Temp file + rename, so a crash never leaves half a JSON file.
        with tempfile.NamedTemporaryFile("w", dir=self._path.parent, suffix=".tmp",
                                         delete=False, encoding="utf-8") as handle:
            json.dump(self._entries, handle, ensure_ascii=False, indent=2)
            name = handle.name
        os.replace(name, self._path)
