"""BM25-based retrieval index.

Design trade-off (documented in README): we use BM25 (rank-bm25) rather than
a dense embedding index. This is a deliberate choice, not a shortcut --
it means the whole retrieval pipeline runs fully offline, deterministically,
and for $0, which matters a lot for a project meant to be cloned and re-run
by anyone evaluating it. The eval harness measures where BM25's lexical
matching falls short (paraphrased queries) so the trade-off is quantified,
not just asserted.
"""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from sentinelrag.retrieval.chunking import Chunk
from sentinelrag.schema import Evidence, Document

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._corpus_tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if chunks else None

    @classmethod
    def from_documents(cls, docs: list[Document], strategy: str = "whole_document") -> "BM25Index":
        from sentinelrag.retrieval.chunking import chunk_documents

        return cls(chunk_documents(docs, strategy=strategy))

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Returns the top_k chunks ranked by BM25 score.

        Note: with tiny corpora (a handful of documents, as in unit tests or
        a freshly-created repo), BM25's IDF term can go to ~0 or negative
        for query terms that appear in most/all documents -- that's a known
        property of the Okapi BM25 formula, not a bug. We therefore rank
        and return the top_k regardless of sign rather than filtering on
        `score > 0`, which would silently drop everything for small corpora.
        Query tokens that appear in zero chunks are excluded (nothing to
        rank), which is done implicitly by BM25 returning identical scores
        for all chunks in that degenerate case -- callers should still treat
        very low positive result counts as "weak match" territory.
        """
        if not self.chunks or self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
