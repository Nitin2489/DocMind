"""Retrieval evaluation on the sample documents.

For each question, a retrieval counts as a hit@k if one of the top-k chunks comes
from the expected document AND contains the expected answer text. Compares BM25
only, vector only and the hybrid (RRF) retriever.

Usage:  python eval/run_eval.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import config  # noqa: E402
from app.chunking import chunk_pages  # noqa: E402
from app.embeddings import HashingEmbedder  # noqa: E402
from app.ingest import extract_pages  # noqa: E402
from app.retrieval import HybridIndex  # noqa: E402
from app.store import Store  # noqa: E402


# Small chunks so each sample document splits into several chunks; with the
# production default (900 chars) every sample fits in one chunk and the test is trivial.
EVAL_CHUNK_SIZE, EVAL_CHUNK_OVERLAP = 300, 60


def main() -> None:
    embedder = HashingEmbedder(config.EMBED_DIM)
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(str(Path(tmp) / "eval.sqlite3"))
        for path in sorted((ROOT / "samples").iterdir()):
            chunks = chunk_pages(extract_pages(path.name, path.read_bytes()), EVAL_CHUNK_SIZE, EVAL_CHUNK_OVERLAP)
            store.add_document(path.name, chunks, embedder.embed([t for _, t in chunks]))
        index = HybridIndex(store.all_chunks(), embedder)
        store.close()
    cases = json.loads((ROOT / "eval" / "qa.json").read_text())

    print(f"{len(index.chunks)} chunks, {len(cases)} questions\n")
    print(f"{'method':<10} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7}   MRR")
    for method in ("bm25", "vector", "hybrid"):
        hits_at = {1: 0, 3: 0, 5: 0}
        rr = 0.0
        for case in cases:
            results = index.search(case["q"], k=5, method=method)
            rank = next(
                (
                    i
                    for i, h in enumerate(results, 1)
                    if h.chunk.doc_name == case["doc"] and case["must_contain"].lower() in h.chunk.text.lower()
                ),
                None,
            )
            if rank:
                rr += 1 / rank
                for k in hits_at:
                    hits_at[k] += rank <= k
        n = len(cases)
        print(f"{method:<10} {hits_at[1]/n:>7.0%} {hits_at[3]/n:>7.0%} {hits_at[5]/n:>7.0%}   {rr/n:.3f}")


if __name__ == "__main__":
    main()
