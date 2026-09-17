#!/usr/bin/env python3
"""Manual smoke test of the full graph using FakeLLMClient (no API key needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinelrag.agents.graph import run_question
from sentinelrag.agents.llm_client import FakeLLMClient, LLMResponse, TextBlock, ToolUseBlock
from sentinelrag.guardrails.risky_actions import RiskyActionRegistry
from sentinelrag.memory.store import LongTermMemory
from sentinelrag.retrieval.index import BM25Index
from sentinelrag.schema import Document
from sentinelrag.tools.repo_tools import RepoContext

docs = [
    Document(
        id="482",
        source_type="issue",
        title="Retriever returns stale results after refresh",
        body="After calling refresh(), search still returns cached docs. Looks like a caching bug in the BM25 index rebuild step.",
        url="https://github.com/acme/repo/issues/482",
        updated_at="2026-09-01T10:00:00Z",
        labels=["bug", "blocker"],
        state="open",
    ),
    Document(
        id="930",
        source_type="pull_request",
        title="Fix BM25 index rebuild on refresh",
        body="Rebuilds the BM25Okapi object whenever refresh() is called. Closes #482.",
        url="https://github.com/acme/repo/pulls/930",
        updated_at="2026-09-05T12:00:00Z",
        labels=[],
        state="open",
    ),
]

index = BM25Index.from_documents(docs)
memory = LongTermMemory(repo="acme/repo", db_path=Path("/tmp/smoke_memory.sqlite3"))
ctx = RepoContext(documents=docs, index=index, memory=memory, risky_actions=RiskyActionRegistry())

script = [
    LLMResponse(
        content=[ToolUseBlock(id="t1", name="search_repo", input={"query": "stale results caching bug"})],
        stop_reason="tool_use",
    ),
    LLMResponse(
        content=[
            TextBlock(
                text=(
                    "There is a known bug where the retriever returns stale results after refresh() "
                    "[issue:482]. It has already been fixed in an open pull request that rebuilds the "
                    "BM25 index on refresh [pull_request:930]."
                )
            )
        ],
        stop_reason="end_turn",
    ),
]
llm = FakeLLMClient(script=script)

result = run_question(llm, ctx, "Why does search return stale results, and has it been fixed?")
print("FINAL ANSWER:\n", result["final_answer"])
print("\nCITATION CHECK:", result["citation_check"])
print("PII REDACTIONS:", result["pii_redactions"])
print("ITERATIONS:", result["iterations"])
