"""Long-term memory: durable facts that survive across sessions, backed by
SQLite so it's inspectable with any DB browser (a deliberate resume talking
point: "why SQLite and not a vector memory store?" -- because these are a
handful of short, structured facts, not a large unstructured corpus; a
full embedding-based memory would be over-engineering here).

Short-term memory is *not* implemented here -- it's just the LangGraph
run's own state (the message list + working scratchpad for the current
question), which disappears when the run ends. Long-term memory is the
explicit, curated subset of that state the agent decides is worth keeping.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from sentinelrag.config import settings
from sentinelrag.schema import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(repo, key)
);
"""


@dataclass
class MemoryFact:
    key: str
    value: str
    updated_at: str


class LongTermMemory:
    """One fact = one (repo, key) -> value row. Writing the same key again
    updates it in place, so memory doesn't grow unboundedly -- this is the
    guardrail the video calls out: "memory cannot be storing every sentence
    you've ever seen."
    """

    def __init__(self, repo: str, db_path: Path | None = None):
        self.repo = repo
        self.db_path = db_path or (settings.data_dir / "memory.sqlite3")
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def remember(self, key: str, value: str) -> None:
        now = now_iso()
        self._conn.execute(
            """
            INSERT INTO memory_facts (repo, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(repo, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (self.repo, key, value, now, now),
        )
        self._conn.commit()

    def recall(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM memory_facts WHERE repo = ? AND key = ?", (self.repo, key)
        ).fetchone()
        return row[0] if row else None

    def recall_all(self) -> list[MemoryFact]:
        rows = self._conn.execute(
            "SELECT key, value, updated_at FROM memory_facts WHERE repo = ? ORDER BY updated_at DESC",
            (self.repo,),
        ).fetchall()
        return [MemoryFact(*row) for row in rows]

    def forget(self, key: str) -> None:
        self._conn.execute("DELETE FROM memory_facts WHERE repo = ? AND key = ?", (self.repo, key))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
