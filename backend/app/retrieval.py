"""Hybrid retrieval: BM25 keyword ranking + dense vector ranking, fused with RRF."""
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .embeddings import HashingEmbedder
from .store import ChunkRow
from .textutil import tokenize

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # standard reciprocal-rank-fusion constant


@dataclass
class Hit:
    chunk: ChunkRow
    score: float
    bm25_rank: int | None
    vector_rank: int | None


class HybridIndex:
    def __init__(self, chunks: list[ChunkRow], embedder: HashingEmbedder):
        self.chunks = chunks
        self.embedder = embedder
        self._tokens = [tokenize(c.text) for c in chunks]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = np.array([len(t) for t in self._tokens], dtype=np.float32)
        self._avgdl = float(self._len.mean()) if len(chunks) else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._matrix = (
            np.vstack([c.embedding for c in chunks]) if chunks else np.zeros((0, embedder.dim), np.float32)
        )

    def _bm25_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        for i, tf in enumerate(self._tf):
            norm = BM25_K1 * (1 - BM25_B + BM25_B * self._len[i] / (self._avgdl or 1.0))
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    scores[i] += self._idf[term] * f * (BM25_K1 + 1) / (f + norm)
        return scores

    @staticmethod
    def _ranks(scores: np.ndarray, allowed: np.ndarray) -> dict[int, int]:
        """Rank of every allowed chunk with a positive score (0 = best)."""
        order = [i for i in np.argsort(-scores) if allowed[i] and scores[i] > 0]
        return {int(i): r for r, i in enumerate(order)}

    def search(
        self, query: str, k: int = 5, doc_ids: set[int] | None = None, method: str = "hybrid"
    ) -> list[Hit]:
        """method: 'hybrid' (default), 'bm25' or 'vector' (the last two are for evaluation)."""
        if not self.chunks or not query.strip():
            return []
        allowed = np.array([doc_ids is None or c.doc_id in doc_ids for c in self.chunks])
        bm25_rank = self._ranks(self._bm25_scores(query), allowed) if method != "vector" else {}
        vec_rank = (
            self._ranks(self._matrix @ self.embedder.embed_one(query), allowed) if method != "bm25" else {}
        )
        fused: dict[int, float] = {}
        for ranks in (bm25_rank, vec_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(self.chunks[i], s, bm25_rank.get(i), vec_rank.get(i)) for i, s in top]
