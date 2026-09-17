"""Citation verification: the model is instructed to cite evidence as
`[issue:482]` / `[pull_request:930]` / `[commit:a1b2c3d]`. This module checks
every citation the model actually produced against the evidence it was
actually given, so a hallucinated or fabricated citation is caught
mechanically rather than trusted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_RE = re.compile(r"\[(issue|pull_request|commit):([A-Za-z0-9]+)\]")


@dataclass
class CitationCheck:
    cited: list[str]
    valid: list[str]
    hallucinated: list[str]

    @property
    def all_valid(self) -> bool:
        return not self.hallucinated

    @property
    def accuracy(self) -> float:
        if not self.cited:
            return 1.0
        return len(self.valid) / len(self.cited)


def extract_citations(answer: str) -> list[str]:
    return [f"{kind}:{ident}" for kind, ident in CITATION_RE.findall(answer)]


def verify_citations(answer: str, allowed_citations: set[str]) -> CitationCheck:
    cited = extract_citations(answer)
    valid = [c for c in cited if c in allowed_citations]
    hallucinated = [c for c in cited if c not in allowed_citations]
    return CitationCheck(cited=cited, valid=valid, hallucinated=hallucinated)
