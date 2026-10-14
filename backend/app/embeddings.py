"""Dependency-free dense embeddings using the hashing trick.

Each chunk becomes a fixed-size vector built from hashed unigrams and bigrams
(signed, sub-linear term frequency, L2-normalised), so cosine similarity works
without downloading a model. This captures phrase overlap that plain BM25 on
single words misses. The `Embedder` interface is small on purpose: swap in a
neural model (e.g. sentence-transformers) by implementing `embed`.
"""
import hashlib
import math
from collections import Counter

import numpy as np

from .textutil import tokenize


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed_one(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, count in Counter(features).items():
            idx, sign = self._bucket(feature)
            vec[idx] += sign * (1.0 + math.log(count))
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed_one(t) for t in texts])
