#!/usr/bin/env python3
"""Deterministic evaluation suite -- no LLM calls, no API key, no network.

This is what CI runs on every push. It measures the parts of the system
that don't require a live model call: guardrail precision/recall against
labeled adversarial examples, and retrieval hit rate across chunking
strategies against a labeled QA set. The (optional, not-CI) LLM-in-the-loop
eval lives in `run_live_eval.py` and needs ANTHROPIC_API_KEY + network.

Usage:
    python evals/run_deterministic_evals.py [--fail-under 0.8]

Exits non-zero if any headline metric drops below --fail-under, so this can
gate a CI job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinelrag.guardrails.citations import verify_citations
from sentinelrag.guardrails.injection import scan_text
from sentinelrag.guardrails.pii import redact
from sentinelrag.retrieval.chunking import STRATEGIES
from sentinelrag.retrieval.index import BM25Index
from sentinelrag.schema import Document

DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------------
# 1. Prompt injection detection: precision / recall / F1
# ------------------------------------------------------------------------
def eval_injection() -> dict:
    rows = load_jsonl(DATASETS_DIR / "injection_eval.jsonl")
    tp = fp = tn = fn = 0
    for row in rows:
        verdict = scan_text(row["text"], source=row["source"])
        predicted = verdict.is_suspicious
        actual = row["is_injection"]
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ------------------------------------------------------------------------
# 2. PII redaction recall (did we catch every category present?)
# ------------------------------------------------------------------------
def eval_pii() -> dict:
    rows = load_jsonl(DATASETS_DIR / "pii_eval.jsonl")
    total_expected = 0
    total_caught = 0
    per_row = []
    for row in rows:
        _, counts = redact(row["text"])
        expected = set(row["expected_categories"])
        caught = set(counts.keys())
        total_expected += len(expected)
        total_caught += len(expected & caught)
        per_row.append({"text": row["text"][:60], "expected": sorted(expected), "caught": sorted(caught)})
    recall = total_caught / total_expected if total_expected else 1.0
    return {"n": len(rows), "category_recall": round(recall, 3), "details": per_row}


# ------------------------------------------------------------------------
# 3. Citation verification accuracy (do we correctly flag hallucinations?)
# ------------------------------------------------------------------------
def eval_citations() -> dict:
    rows = load_jsonl(DATASETS_DIR / "citation_eval.jsonl")
    correct = 0
    for row in rows:
        check = verify_citations(row["answer"], set(row["allowed_citations"]))
        if sorted(check.hallucinated) == sorted(row["expected_hallucinated"]):
            correct += 1
    return {"n": len(rows), "accuracy": round(correct / len(rows), 3) if rows else 1.0}


# ------------------------------------------------------------------------
# 4. Retrieval hit rate, compared across chunking strategies
# ------------------------------------------------------------------------
def eval_retrieval(top_k: int = 1) -> dict:
    """Retrieval hit rate @ top_k, compared across chunking strategies.

    top_k=1 is the headline number: it's the strictest test (does the single
    best-ranked chunk actually contain the answer?) and is where chunking
    strategy differences actually show up -- at top_k=3+ all three
    strategies saturate to ~1.0 on this dataset size and stop being
    distinguishing, which is itself a useful thing to have measured rather
    than assumed.
    """
    corpus_raw = json.loads((DATASETS_DIR / "retrieval_corpus.json").read_text())
    docs = [Document(**d) for d in corpus_raw]
    qa = load_jsonl(DATASETS_DIR / "retrieval_qa.jsonl")

    results = {}
    for strategy in STRATEGIES:
        index = BM25Index.from_documents(docs, strategy=strategy)
        hits = 0
        for item in qa:
            top = index.search(item["question"], top_k=top_k)
            retrieved_citations = {c.citation for c, _ in top}
            if retrieved_citations & set(item["expected_citations"]):
                hits += 1
        results[strategy] = {"hit_rate_at_k": round(hits / len(qa), 3), "top_k": top_k, "n_questions": len(qa)}
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-under", type=float, default=0.75)
    args = parser.parse_args()

    start = time.time()
    report = {
        "injection_detection": eval_injection(),
        "pii_redaction": eval_pii(),
        "citation_verification": eval_citations(),
        "retrieval_hit_rate_by_strategy_at_k1": eval_retrieval(top_k=1),
        "retrieval_hit_rate_by_strategy_at_k3": eval_retrieval(top_k=3),
        "elapsed_seconds": round(time.time() - start, 3),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "deterministic_results.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))

    best_retrieval = max(v["hit_rate_at_k"] for v in report["retrieval_hit_rate_by_strategy_at_k1"].values())
    headline = {
        "injection_f1": report["injection_detection"]["f1"],
        "pii_recall": report["pii_redaction"]["category_recall"],
        "citation_accuracy": report["citation_verification"]["accuracy"],
        "best_retrieval_hit_rate": best_retrieval,
    }
    print("\nHEADLINE METRICS:", json.dumps(headline, indent=2))

    failed = [k for k, v in headline.items() if v < args.fail_under]
    if failed:
        print(f"\nFAILED (below {args.fail_under}): {failed}", file=sys.stderr)
        return 1
    print(f"\nAll headline metrics >= {args.fail_under}. Eval suite PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
