"""Lexical (BM25), dense (embeddings) and hybrid (reciprocal rank fusion) retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MODES = ("bm25", "dense", "hybrid")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None, embedder=None):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.search_text) or ["_"] for c in chunks])
        self.embeddings = embeddings
        self.embedder = embedder

    @property
    def has_dense(self) -> bool:
        return self.embeddings is not None and self.embedder is not None

    def bm25_rank(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def dense_rank(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([query], kind="query")[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def rank(self, query: str, k: int = 5, mode: str = "hybrid", pool: int = 20) -> list[tuple[int, float]]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "dense" and not self.has_dense:
            raise RuntimeError("Dense retrieval needs embeddings and an embedder")
        if mode == "bm25" or (mode == "hybrid" and not self.has_dense):
            return self.bm25_rank(query, k)
        if mode == "dense":
            return self.dense_rank(query, k)
        lexical = [i for i, _ in self.bm25_rank(query, pool)]
        semantic = [i for i, _ in self.dense_rank(query, pool)]
        return rrf([lexical, semantic])[:k]

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        return [
            Hit(self.chunks[i], score, rank)
            for rank, (i, score) in enumerate(self.rank(query, k, mode), start=1)
        ]
