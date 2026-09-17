"""Shared data types used across connectors, retrieval, agents and guardrails."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Document:
    """A single retrievable unit of evidence (one issue, PR, or commit)."""

    id: str  # stable id, e.g. "issue-482"
    source_type: str  # "issue" | "pull_request" | "commit"
    title: str
    body: str
    url: str
    updated_at: str  # ISO 8601
    labels: list[str] = field(default_factory=list)
    state: str | None = None  # "open" | "closed" | "merged" | None (commits)
    author: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Flatten into the text blob that gets indexed and shown to the LLM."""
        header = f"[{self.source_type.upper()} {self.id}] {self.title}"
        meta = f"(state={self.state}, labels={','.join(self.labels) or 'none'}, updated={self.updated_at})"
        return f"{header} {meta}\n{self.body}".strip()

    def citation(self) -> str:
        return f"{self.source_type}:{self.id}"


@dataclass
class Evidence:
    """A retrieved Document plus the retrieval score, used for citation tracing."""

    document: Document
    score: float


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
