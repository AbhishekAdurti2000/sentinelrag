#!/usr/bin/env python3
"""Interactive CLI: ask questions about a live GitHub repo's issues, PRs and
commits, backed by the full agentic RAG + guardrails + memory pipeline.

Usage:
    python -m sentinelrag.cli "What changed in this repo in the last week?"
    python -m sentinelrag.cli --repo langchain-ai/langgraph "Summarize open blockers"
    python -m sentinelrag.cli  # interactive loop
"""
from __future__ import annotations

import argparse
import sys

from sentinelrag.agents.graph import run_question
from sentinelrag.agents.llm_client import AnthropicClient
from sentinelrag.config import settings
from sentinelrag.connectors.github import GitHubConnector
from sentinelrag.guardrails.risky_actions import RiskyActionRegistry
from sentinelrag.memory.store import LongTermMemory
from sentinelrag.retrieval.index import BM25Index
from sentinelrag.tools.repo_tools import RepoContext
from sentinelrag.tracing.tracer import Tracer


def build_context(repo: str) -> RepoContext:
    connector = GitHubConnector(repo=repo)
    print(f"Fetching live data for {repo} ...", file=sys.stderr)
    docs = connector.load(max_age_seconds=300)
    print(f"Loaded {len(docs)} issues/PRs/commits.", file=sys.stderr)
    index = BM25Index.from_documents(docs, strategy="whole_document")
    memory = LongTermMemory(repo=repo)
    return RepoContext(documents=docs, index=index, memory=memory, risky_actions=RiskyActionRegistry())


def ask(ctx: RepoContext, question: str) -> None:
    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
        sys.exit(1)
    llm = AnthropicClient(api_key=settings.anthropic_api_key, model=settings.model)
    tracer = Tracer()
    result = run_question(llm, ctx, question, tracer)

    print("\n" + "=" * 70)
    print(result["final_answer"])
    print("=" * 70)
    print(f"citations: {result['citation_check']}")
    if result["injection_flags"]:
        print(f"⚠ injection flags raised during this run: {len(result['injection_flags'])}")
    pending = ctx.risky_actions.pending()
    if pending:
        print(f"⚠ {len(pending)} risky action(s) proposed and awaiting human confirmation:")
        for action in pending:
            print(f"   - [{action.id}] {action.kind} -> {action.target}: {action.reason}")
    print(f"trace saved to: {tracer.trace_dir / (tracer.run_id + '.json')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for an interactive loop.")
    parser.add_argument("--repo", default=settings.repo, help="owner/name of the GitHub repo to query.")
    args = parser.parse_args()

    ctx = build_context(args.repo)

    if args.question:
        ask(ctx, args.question)
        return

    print("Interactive mode. Ctrl-D to exit.")
    while True:
        try:
            question = input("\n> ")
        except EOFError:
            break
        if question.strip():
            ask(ctx, question)


if __name__ == "__main__":
    main()
