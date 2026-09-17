"""Chunking strategies for turning a Document's body into retrievable chunks.

Two strategies are implemented so the eval harness can compare them and
report a concrete retrieval-hit-rate delta in the README, exactly the kind
of "I measured this, not just built it" evidence recruiters look for.
"""
from __future__ import annotations

from dataclasses import dataclass

from sentinelrag.schema import Document


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source_type: str
    citation: str


def fixed_size_chunks(doc: Document, size: int = 400, overlap: int = 0) -> list[Chunk]:
    """Naive fixed-width character chunking. Fast, simple, ignores structure."""
    text = doc.to_text()
    chunks = []
    step = max(size - overlap, 1)
    idx = 0
    i = 0
    while i < len(text):
        piece = text[i : i + size]
        chunks.append(Chunk(doc.id, f"{doc.id}-{idx}", piece, doc.source_type, doc.citation()))
        idx += 1
        i += step
    return chunks or [Chunk(doc.id, f"{doc.id}-0", text, doc.source_type, doc.citation())]


def whole_document_chunks(doc: Document) -> list[Chunk]:
    """One chunk per document (issue/PR/commit). Preserves full context, which
    matters for short GitHub issues where splitting mid-thought hurts recall.
    """
    return [Chunk(doc.id, f"{doc.id}-0", doc.to_text(), doc.source_type, doc.citation())]


STRATEGIES = {
    "fixed_400": lambda doc: fixed_size_chunks(doc, size=400, overlap=0),
    "fixed_400_overlap_80": lambda doc: fixed_size_chunks(doc, size=400, overlap=80),
    "whole_document": whole_document_chunks,
}


def chunk_documents(docs: list[Document], strategy: str = "whole_document") -> list[Chunk]:
    fn = STRATEGIES[strategy]
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(fn(doc))
    return chunks
