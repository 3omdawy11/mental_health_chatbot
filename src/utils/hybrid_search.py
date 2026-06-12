"""
Hybrid retrieval: BM25 (keyword) + semantic (dense vector) search.

Combined score: alpha * bm25_norm + (1-alpha) * semantic_score
Default alpha=0.5 gives equal weight to both signals.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


_STOP_WORDS = frozenset({
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can","i","you",
    "he","she","it","we","they","this","that","these","those","my","your","our",
})

def _tokenise(text: str) -> list[str]:
    """Lowercase, remove punctuation, remove stop words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in _STOP_WORDS]


class HybridSearch:
    def __init__(
        self,
        chunks: list[dict],
        alpha: float = 0.5,
    ) -> None:
        if not chunks:
            raise ValueError("chunks must be a non-empty list")
        self.chunks   = chunks
        self.alpha    = alpha
        self._corpus_vecs: Optional[np.ndarray] = None  # (N, 384) once built

        # Build BM25 index immediately (cheap, CPU-only)
        tokenised  = [_tokenise(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info(f"HybridSearch: BM25 index built for {len(chunks)} chunks")


    def build_corpus_vectors(self, embedder) -> None:
        """
        Pre-compute and cache dense embeddings for all chunks.
        Call once before repeated semantic / hybrid searches.
        """
        logger.info(f"Building corpus embeddings for {len(self.chunks)} chunks …")
        texts = [c["text"] for c in self.chunks]
        self._corpus_vecs = embedder.embed_batch(texts, show_progress=True)
        logger.info("Corpus embeddings ready.")


    def bm25_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        tokens = _tokenise(query)
        if not tokens:
            return []
        raw = self._bm25.get_scores(tokens)
        top_idx = np.argsort(raw)[::-1][:k]
        max_score = raw.max() if raw.max() > 0 else 1.0
        return [(int(i), float(raw[i] / max_score)) for i in top_idx if raw[i] > 0]

    def semantic_search(
        self, query_vec: np.ndarray, k: int = 10
    ) -> list[tuple[int, float]]:
        if self._corpus_vecs is None:
            raise RuntimeError(
                "Call build_corpus_vectors(embedder) before semantic_search()."
            )
        scores  = self._corpus_vecs @ query_vec      # (N,)  — vecs are normalised
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def search(
        self,
        query: str,
        embedder,
        k: int = 5,
        alpha: Optional[float] = None,
        use_hyde: bool = False,
    ) -> list[dict]:
        """
        Hybrid BM25 + semantic search.

        List of dicts:
        [
            {
                "chunk":          {original chunk dict},
                "bm25_score":     float,   # 0-1 normalised
                "semantic_score": float,   # cosine similarity
                "combined_score": float,   # weighted average
                "rank":           int,     # 1-indexed
            },
            ...
        ]
        """
        _alpha = alpha if alpha is not None else self.alpha
        n      = len(self.chunks)
        pool   = max(k * 3, 20)    # retrieve larger pool, re-rank to k

        # BM25 scores (all chunks)
        tokens   = _tokenise(query)
        bm25_raw = self._bm25.get_scores(tokens) if tokens else np.zeros(n)
        bm25_max = bm25_raw.max() if bm25_raw.max() > 0 else 1.0
        bm25_norm = bm25_raw / bm25_max

        # Semantic scores
        if use_hyde:
            q_vec = embedder.embed_hyde(query)
        else:
            q_vec = embedder.embed_text(query)

        if self._corpus_vecs is None:
            self.build_corpus_vectors(embedder)

        sem_scores = self._corpus_vecs @ q_vec   # (N,)

        # Combined score
        combined = _alpha * bm25_norm + (1.0 - _alpha) * sem_scores

        # Top-k by combined score
        top_idx = np.argsort(combined)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_idx, 1):
            results.append({
                "chunk":          self.chunks[idx],
                "bm25_score":     round(float(bm25_norm[idx]),  4),
                "semantic_score": round(float(sem_scores[idx]), 4),
                "combined_score": round(float(combined[idx]),   4),
                "rank":           rank,
            })
        return results

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        vecs = "built" if self._corpus_vecs is not None else "not built"
        return (f"HybridSearch(chunks={len(self.chunks)}, "
                f"alpha={self.alpha}, corpus_vecs={vecs})")