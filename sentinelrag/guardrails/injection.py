"""Prompt-injection detection.

Two attack surfaces, both covered:
  1. Direct injection -- the user's own query tries to override instructions.
  2. Indirect injection -- retrieved evidence (a GitHub issue/PR/commit body)
     contains attacker-planted instructions, since anyone can open an issue
     on a public repo. This is the more realistic and more interesting one
     for an agentic-RAG system, and the one most student projects miss
     entirely.

This is a deliberately deterministic, regex/heuristic classifier (not an
LLM call) so it: (a) has zero latency/cost, (b) is fully unit-testable and
CI-friendly without API keys, (c) can't itself be prompt-injected. The
README documents this as defense-in-depth layer 1; an optional LLM-based
classifier is noted as a natural layer 2 for phrasings this misses.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns are intentionally broad; false positives are cheap (we just flag
# for review), false negatives are the expensive failure mode for a security
# control, so we bias toward recall.
INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above|earlier) instructions",
    r"disregard (all|any|the)?\s*(previous|prior|above|earlier) (instructions|prompt)",
    r"you are now\b",
    r"new instructions?:",
    r"system prompt",
    r"act as (if you|a|an)\b",
    r"reveal (your|the) (system prompt|instructions|prompt)",
    r"do anything now",
    r"\bdan mode\b",
    r"override (your|all)? ?(rules|instructions|guardrails|safety)",
    r"pretend (you are|to be)\b",
    r"from now on,? you",
    r"do not (mention|tell|reveal) (this|that) to the user",
    r"\bexfiltrate\b",
    r"send (this|the) (data|information|contents?) to https?://",
    r"curl\s+https?://\S+\s*\|\s*(sh|bash)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class InjectionVerdict:
    is_suspicious: bool
    matched_patterns: list[str]
    source: str  # "user_query" | "retrieved_evidence"
    snippet: str = ""

    def as_dict(self) -> dict:
        return {
            "is_suspicious": self.is_suspicious,
            "matched_patterns": self.matched_patterns,
            "source": self.source,
            "snippet": self.snippet,
        }


def scan_text(text: str, source: str) -> InjectionVerdict:
    matched: list[str] = []
    snippet = ""
    for pattern, compiled in zip(INJECTION_PATTERNS, _COMPILED):
        m = compiled.search(text)
        if m:
            matched.append(pattern)
            if not snippet:
                start = max(m.start() - 30, 0)
                end = min(m.end() + 30, len(text))
                snippet = text[start:end]
    return InjectionVerdict(is_suspicious=bool(matched), matched_patterns=matched, source=source, snippet=snippet)


def scan_user_query(query: str) -> InjectionVerdict:
    return scan_text(query, source="user_query")


def scan_evidence(text: str) -> InjectionVerdict:
    return scan_text(text, source="retrieved_evidence")
