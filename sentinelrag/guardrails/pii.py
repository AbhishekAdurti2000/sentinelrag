"""Output PII / secret redaction.

Runs on the agent's final answer before it's returned to the user. GitHub
issue/commit text legitimately contains emails, tokens accidentally pasted
into bug reports, etc. -- if the agent quotes evidence verbatim, that content
should not leak into the answer unredacted.
"""
from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic_api_key": re.compile(r"(?i)(api[_-]?key|secret)[\"'\s:=]+[A-Za-z0-9_\-]{16,}"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Returns (redacted_text, counts_by_category)."""
    counts: dict[str, int] = {}
    out = text
    for label, pattern in PATTERNS.items():
        out, n = pattern.subn(f"[REDACTED_{label.upper()}]", out)
        if n:
            counts[label] = n
    return out, counts
