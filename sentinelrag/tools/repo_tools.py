"""The tool surface exposed to the LLM. Kept as plain functions over a
`RepoContext` so they're trivially unit-testable without any LLM involved --
the agent's "reasoning" and the tools it calls are two separately-tested
layers.
"""
from __future__ import annotations

from dataclasses import dataclass

from sentinelrag.guardrails.injection import scan_evidence
from sentinelrag.guardrails.risky_actions import RiskyActionRegistry
from sentinelrag.memory.store import LongTermMemory
from sentinelrag.retrieval.index import BM25Index
from sentinelrag.schema import Document


@dataclass
class RepoContext:
    """Everything a tool call needs: the live document set, the retrieval
    index built on top of it, long-term memory, and the risky-action
    registry actions get proposed into (never executed directly).
    """

    documents: list[Document]
    index: BM25Index
    memory: LongTermMemory
    risky_actions: RiskyActionRegistry


def search_repo(ctx: RepoContext, query: str, source_type: str | None = None, top_k: int = 5) -> dict:
    hits = ctx.index.search(query, top_k=top_k * 3 if source_type else top_k)
    results = []
    injection_flags = []
    for chunk, score in hits:
        if source_type and chunk.source_type != source_type:
            continue
        verdict = scan_evidence(chunk.text)
        if verdict.is_suspicious:
            injection_flags.append(verdict.as_dict())
        results.append(
            {
                "citation": chunk.citation,
                "source_type": chunk.source_type,
                "text": chunk.text,
                "score": round(float(score), 3),
                "flagged_injection": verdict.is_suspicious,
            }
        )
        if len(results) >= top_k:
            break
    return {"results": results, "injection_flags": injection_flags}


def list_recent_commits(ctx: RepoContext, n: int = 5) -> dict:
    commits = [d for d in ctx.documents if d.source_type == "commit"][:n]
    return {"commits": [{"citation": d.citation(), "title": d.title, "updated_at": d.updated_at} for d in commits]}


def list_open_issues(ctx: RepoContext, label: str | None = None) -> dict:
    issues = [d for d in ctx.documents if d.source_type == "issue" and d.state == "open"]
    if label:
        issues = [d for d in issues if label in d.labels]
    return {
        "issues": [
            {"citation": d.citation(), "title": d.title, "labels": d.labels, "updated_at": d.updated_at}
            for d in issues
        ]
    }


def recall_memory(ctx: RepoContext, key: str) -> dict:
    value = ctx.memory.recall(key)
    return {"key": key, "value": value}


def remember_fact(ctx: RepoContext, key: str, value: str) -> dict:
    ctx.memory.remember(key, value)
    return {"stored": True, "key": key, "value": value}


def propose_risky_action(ctx: RepoContext, kind: str, target: str, reason: str) -> dict:
    action = ctx.risky_actions.propose(kind=kind, target=target, payload={}, reason=reason)
    return {
        "proposed": True,
        "action_id": action.id,
        "status": action.status.value,
        "note": "This action requires explicit human confirmation before it can execute.",
    }


TOOL_DISPATCH = {
    "search_repo": search_repo,
    "list_recent_commits": list_recent_commits,
    "list_open_issues": list_open_issues,
    "recall_memory": recall_memory,
    "remember_fact": remember_fact,
    "propose_risky_action": propose_risky_action,
}
