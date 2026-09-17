from sentinelrag.retrieval.chunking import chunk_documents, fixed_size_chunks, whole_document_chunks
from sentinelrag.retrieval.index import BM25Index
from sentinelrag.schema import Document


def make_doc(id_="1", body="hello world " * 100) -> Document:
    return Document(
        id=id_,
        source_type="issue",
        title="Test issue",
        body=body,
        url="https://example.com",
        updated_at="2026-01-01T00:00:00Z",
    )


def test_whole_document_chunks_produces_one_chunk():
    doc = make_doc()
    chunks = whole_document_chunks(doc)
    assert len(chunks) == 1
    assert chunks[0].citation == "issue:1"


def test_fixed_size_chunks_splits_long_body():
    doc = make_doc(body="x" * 1000)
    chunks = fixed_size_chunks(doc, size=400, overlap=0)
    assert len(chunks) == 3  # 1000 chars / 400 -> ceil = 3
    assert all(c.citation == "issue:1" for c in chunks)


def test_fixed_size_chunks_overlap_increases_chunk_count():
    doc = make_doc(body="x" * 1000)
    no_overlap = fixed_size_chunks(doc, size=400, overlap=0)
    with_overlap = fixed_size_chunks(doc, size=400, overlap=80)
    assert len(with_overlap) >= len(no_overlap)


def test_bm25_index_finds_relevant_chunk():
    docs = [
        make_doc("1", "The retriever returns stale results after refresh due to a caching bug."),
        make_doc("2", "Unrelated: fixed a typo in the README installation instructions."),
    ]
    index = BM25Index.from_documents(docs, strategy="whole_document")
    top = index.search("stale caching bug", top_k=1)
    assert top[0][0].citation == "issue:1"


def test_bm25_index_empty_corpus_returns_empty():
    index = BM25Index([])
    assert index.search("anything") == []


def test_chunk_documents_dispatches_by_strategy_name():
    docs = [make_doc("1", "x" * 1000)]
    whole = chunk_documents(docs, strategy="whole_document")
    fixed = chunk_documents(docs, strategy="fixed_400")
    assert len(whole) == 1
    assert len(fixed) > 1
