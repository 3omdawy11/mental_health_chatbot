"""
src/utils/embedder.py
======================
Embedding pipeline using SentenceTransformer (all-MiniLM-L6-v2, 384-dim).
Includes HyDE (Hypothetical Document Embeddings) for improved retrieval.

Usage
-----
    from src.utils.embedder import Embedder
    emb = Embedder()
    vec = emb.embed_text("I feel anxious about work")         # (384,) ndarray
    vecs = emb.embed_batch(["text1", "text2"])                # (N, 384) ndarray
    hyde_vec = emb.embed_hyde("I feel anxious", groq_key=...) # HyDE embedding
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, 80 MB, fast CPU inference
EMBED_DIM  = 384

_HYDE_PROMPT = """You are a mental health counsellor writing a helpful response.
Write a concise 3-4 sentence response that directly addresses this concern.
Do NOT repeat the question. Write only the response, no preamble.

Concern: "{query}"
Response:"""


class Embedder:
    """
    Sentence embedding pipeline with optional HyDE.

    Parameters
    ----------
    model_name : SentenceTransformer model (default: all-MiniLM-L6-v2)
    device     : 'cpu', 'cuda', or None (auto-detect)
    groq_api_key : For HyDE generation. Falls back to GROQ_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ) -> None:
        self._model_name  = model_name
        self._device      = device
        self._groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._model       = None   # lazy load

    # ── Lazy model load ───────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None:
            return
        from pathlib import Path
        _ROOT = Path(__file__).resolve().parent.parent.parent
        local_path = _ROOT / "models" / "sentence_transformer_local"
        # Try local offline model first (no HuggingFace download needed)
        if local_path.exists():
            logger.info(f"Loading local BERT model from {local_path}")
            from transformers import BertModel, BertTokenizerFast
            import torch
            self._bert_tok = BertTokenizerFast(
                vocab_file=str(local_path / "vocab.txt"),
                do_lower_case=True
            )
            self._bert = BertModel.from_pretrained(str(local_path))
            self._bert.eval()
            self._use_local = True
            self._model = True   # sentinel — not None means loaded
        else:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer '{self._model_name}' …")
            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._use_local = False

    # ── Core embedding ────────────────────────────────────────────────────────

    def _local_encode(self, texts: list[str]) -> np.ndarray:
        """Mean-pool BERT hidden states → normalised (N, 384) embeddings."""
        import torch
        enc = self._bert_tok(
            texts, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        with torch.no_grad():
            out = self._bert(**enc)
        # Mean pool over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vecs = vecs.numpy().astype(np.float32)
        # L2 normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms > 0, norms, 1.0)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single string → (384,) float32 ndarray."""
        self._load()
        if not isinstance(text, str) or not text.strip():
            return np.zeros(EMBED_DIM, dtype=np.float32)
        if getattr(self, "_use_local", False):
            return self._local_encode([text])[0]
        vec = self._model.encode(text, normalize_embeddings=True,
                                  show_progress_bar=False)
        return vec.astype(np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of strings → (N, 384) float32 ndarray.
        Empty/invalid strings get zero vectors.
        """
        self._load()
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)

        valid_mask  = [isinstance(t, str) and bool(t.strip()) for t in texts]
        valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]

        if valid_texts:
            if getattr(self, "_use_local", False):
                # Process in sub-batches for the local model
                parts = []
                for i in range(0, len(valid_texts), batch_size):
                    parts.append(self._local_encode(valid_texts[i:i+batch_size]))
                vecs = np.concatenate(parts, axis=0).astype(np.float32)
            else:
                vecs = self._model.encode(
                    valid_texts, batch_size=batch_size,
                    normalize_embeddings=True, show_progress_bar=show_progress,
                ).astype(np.float32)
        else:
            vecs = np.empty((0, EMBED_DIM), dtype=np.float32)

        result  = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
        vi = 0
        for i, ok in enumerate(valid_mask):
            if ok:
                result[i] = vecs[vi]
                vi += 1
        return result

    # ── HyDE ─────────────────────────────────────────────────────────────────

    def _generate_hypothetical(self, query: str) -> Optional[str]:
        """
        Use Groq to generate a hypothetical counsellor response to the query.
        Returns None on failure.
        """
        if not self._groq_api_key:
            return None
        try:
            from groq import Groq
            client = Groq(api_key=self._groq_api_key, timeout=8.0)
            prompt = _HYDE_PROMPT.replace("{query}", query)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",   # fast model — just generating text
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            hyp = resp.choices[0].message.content or ""
            return hyp.strip() if hyp.strip() else None
        except Exception as exc:
            logger.warning(f"HyDE generation failed: {exc}")
            return None

    def embed_hyde(
        self,
        query: str,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        HyDE embedding: average of query embedding and hypothetical-doc embedding.

        Steps
        -----
        1. Embed the original query                  → q_vec
        2. Generate a hypothetical response via Groq → hyp_text
        3. Embed hyp_text                            → h_vec
        4. Return alpha*q_vec + (1-alpha)*h_vec      (both L2-normalised)

        Falls back to plain embed_text() if generation fails.

        Parameters
        ----------
        query : user query string
        alpha : weight on original query (0=all-hypothetical, 1=no-HyDE)
        """
        q_vec = self.embed_text(query)
        hyp   = self._generate_hypothetical(query)

        if hyp is None:
            logger.info("HyDE: no hypothetical generated — using plain embedding")
            return q_vec

        h_vec  = self.embed_text(hyp)
        combined = alpha * q_vec + (1.0 - alpha) * h_vec
        # Re-normalise so cosine similarity still works correctly
        norm = np.linalg.norm(combined)
        return (combined / norm).astype(np.float32) if norm > 0 else q_vec

    # ── Similarity helpers ────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors."""
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def top_k_similar(
        self,
        query_vec: np.ndarray,
        corpus_vecs: np.ndarray,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Return (index, score) pairs for the top-k most similar corpus vectors.
        corpus_vecs: (N, 384) array already L2-normalised.
        """
        scores = corpus_vecs @ query_vec          # dot product = cosine if normalised
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def __repr__(self) -> str:
        loaded = self._model is not None
        hyde   = "enabled" if self._groq_api_key else "disabled (no GROQ_API_KEY)"
        return f"Embedder(model='{self._model_name}', loaded={loaded}, HyDE={hyde})"