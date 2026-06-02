"""
src/utils/vector_db.py
=======================
Qdrant vector database manager for the mental health chatbot RAG pipeline.

Supports two modes:
  1. Cloud (QDRANT_URL + QDRANT_API_KEY in env) — production
  2. Local in-memory / on-disk            — development / testing

HNSW parameters used:
  m=16, ef_construct=200 → good recall (~98%) with fast search

Usage
-----
    from src.utils.vector_db import VectorDBManager
    db = VectorDBManager()                    # reads .env automatically
    db.create_collection()
    db.index_chunks(chunks, embeddings)
    results = db.search(query_vec, limit=5)
    results = db.search_by_text("I feel anxious", embedder)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import yaml
from dotenv import load_dotenv

if TYPE_CHECKING:
    from src.utils.embedder import Embedder

logger = logging.getLogger(__name__)

# Load .env on import (no-op if file absent)
load_dotenv()

_ROOT       = Path(__file__).resolve().parent.parent.parent
_CFG_PATH   = _ROOT / "configs" / "qdrant_config.yaml"


def _load_cfg() -> dict:
    with open(_CFG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────

class VectorDBManager:
    """
    Manages a Qdrant collection for the mental health knowledge base.

    Connection priority
    -------------------
    1. QDRANT_URL + QDRANT_API_KEY  → Qdrant Cloud
    2. QDRANT_URL only              → self-hosted Qdrant (no auth)
    3. Neither set                  → in-memory local instance (dev/test)

    Parameters
    ----------
    collection_name : override config value
    url             : override QDRANT_URL env var
    api_key         : override QDRANT_API_KEY env var
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        url:     Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._cfg        = _load_cfg()
        self._col_name   = collection_name or self._cfg["collection"]["name"]
        self._url        = url     or os.getenv(self._cfg["connection"]["url_env_var"],     "")
        self._api_key    = api_key or os.getenv(self._cfg["connection"]["api_key_env_var"], "")
        self._client     = None   # lazy init

    # ── Client (lazy, cached) ─────────────────────────────────────────────────

    def _get_client(self):
        if self._client is not None:
            return self._client

        from qdrant_client import QdrantClient

        if self._url:
            kwargs = dict(url=self._url, timeout=self._cfg["connection"]["timeout"])
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = QdrantClient(**kwargs)
            mode = f"cloud ({self._url[:40]}…)" if len(self._url) > 40 else f"cloud ({self._url})"
        else:
            # Local in-memory — perfect for development and testing
            local_path = str(_ROOT / self._cfg["connection"]["local_path"])
            Path(local_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=local_path)
            mode = f"local ({local_path})"

        logger.info(f"VectorDBManager: connected [{mode}]")
        return self._client

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the Qdrant collection with HNSW indexing.

        Parameters
        ----------
        recreate : if True, delete existing collection and recreate it.
                   Use carefully in production — deletes all indexed data.

        Returns True if created, False if already existed (and recreate=False).
        """
        from qdrant_client.models import (
            Distance, VectorParams, HnswConfigDiff,
            OptimizersConfigDiff,
        )

        client    = self._get_client()
        col_cfg   = self._cfg["collection"]
        hnsw_cfg  = self._cfg["hnsw"]

        dist_map  = {"Cosine": Distance.COSINE, "Euclid": Distance.EUCLID,
                     "Dot": Distance.DOT}
        distance  = dist_map.get(col_cfg["distance"], Distance.COSINE)

        # Check existence
        existing = [c.name for c in client.get_collections().collections]
        if self._col_name in existing:
            if not recreate:
                logger.info(f"Collection '{self._col_name}' already exists — skipping create.")
                return False
            client.delete_collection(self._col_name)
            logger.info(f"Deleted existing collection '{self._col_name}'.")

        client.create_collection(
            collection_name=self._col_name,
            vectors_config=VectorParams(
                size=col_cfg["vector_size"],
                distance=distance,
                hnsw_config=HnswConfigDiff(
                    m=hnsw_cfg["m"],
                    ef_construct=hnsw_cfg["ef_construct"],
                    full_scan_threshold=hnsw_cfg["full_scan_threshold"],
                ),
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0,   # index immediately (small collection)
            ),
        )
        logger.info(f"Created collection '{self._col_name}' "
                    f"[size={col_cfg['vector_size']}, dist={col_cfg['distance']}, "
                    f"m={hnsw_cfg['m']}, ef_construct={hnsw_cfg['ef_construct']}]")
        return True

    def collection_info(self) -> dict:
        """Return collection statistics."""
        client = self._get_client()
        try:
            info = client.get_collection(self._col_name)
            return {
                "name":    self._col_name,
                "count":   info.points_count,
                "status":  str(info.status),
                "vectors": str(info.config.params.vectors),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def delete_collection(self) -> None:
        self._get_client().delete_collection(self._col_name)
        logger.info(f"Deleted collection '{self._col_name}'.")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_chunks(
        self,
        chunks:     list[dict],
        embeddings: np.ndarray,
        show_progress: bool = True,
    ) -> dict:
        """
        Upload chunks + their embeddings to Qdrant.

        Parameters
        ----------
        chunks     : list of chunk dicts (must have 'id' and 'text' keys)
        embeddings : (N, 384) float32 ndarray — one row per chunk
        show_progress : print progress during upload

        Returns summary dict with total, batches, elapsed.
        """
        from qdrant_client.models import PointStruct

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match."
            )

        client     = self._get_client()
        batch_size = self._cfg["indexing"]["batch_size"]
        total      = len(chunks)
        uploaded   = 0
        t0         = time.time()

        for batch_start in range(0, total, batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            batch_vecs   = embeddings[batch_start : batch_start + batch_size]

            points = []
            for chunk, vec in zip(batch_chunks, batch_vecs):
                meta = chunk.get("metadata", {}) or {}

                payload = {
                    "text": chunk.get("text", ""),
                    "source": meta.get("source", chunk.get("source", "")),
                    "source_type": meta.get("source_type", chunk.get("source_type", "")),
                    "section": meta.get("section", chunk.get("section", "")),
                    "tokens": meta.get("tokens", chunk.get("tokens", 0)),
                    "chunk_id": chunk.get("chunk_id", chunk.get("id", "")),
                    "context_query": meta.get("context_query", ""),
                    "original_question": meta.get("original_question", ""),
                    "quality_rating": meta.get("quality_rating", None),
                }

                point_id = str(uuid.uuid4())

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=np.asarray(vec, dtype=np.float32).tolist(),
                        payload=payload,
                    )
                )
            client.upsert(
                collection_name=self._col_name,
                points=points,
                wait=self._cfg["indexing"]["wait"],
            )
            uploaded += len(batch_chunks)

            if show_progress:
                pct = uploaded / total * 100
                bar = "█" * int(pct / 5)
                print(f"\r  Indexing [{bar:<20}] {uploaded}/{total} ({pct:.0f}%)", end="", flush=True)

        elapsed = time.time() - t0
        if show_progress:
            print()   # newline after progress bar

        summary = {
            "total":    total,
            "uploaded": uploaded,
            "batches":  (total + batch_size - 1) // batch_size,
            "elapsed":  round(elapsed, 2),
            "rate":     round(total / max(elapsed, 0.001), 1),
        }
        logger.info(f"Indexed {uploaded}/{total} chunks in {elapsed:.1f}s "
                    f"({summary['rate']} chunks/s)")
        return summary

    def verify_count(self) -> int:
        """Return the number of indexed points in the collection."""
        info = self._get_client().get_collection(self._col_name)
        return info.points_count or 0

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        limit:           int   = None,
        score_threshold: float = None,
        filter_payload:  dict  = None,
    ) -> list[dict]:
        """
        Semantic search by pre-computed query vector.

        Parameters
        ----------
        query_vector    : (384,) float32 embedding of the query
        limit           : number of results (default from config)
        score_threshold : minimum cosine similarity (default from config)
        filter_payload  : optional Qdrant filter dict

        Returns
        -------
        List of result dicts:
        [{
            "score":   float,
            "text":    str,
            "source":  str,
            "section": str,
            "tokens":  int,
            "chunk_id": str,
            "source_type": str,
        }, ...]
        """
        from qdrant_client.models import SearchParams

        cfg_search = self._cfg["search"]
        _limit     = limit           if limit           is not None else cfg_search["default_limit"]
        _threshold = score_threshold if score_threshold is not None else cfg_search["score_threshold"]

        response = self._get_client().query_points(
            collection_name=self._col_name,
            query=query_vector.tolist(),
            limit=_limit,
            score_threshold=_threshold,
            with_payload=cfg_search["with_payload"],
            with_vectors=cfg_search["with_vectors"],
            search_params=SearchParams(hnsw_ef=self._cfg["search"]["ef"]),
            query_filter=filter_payload,
        )

        results = []
        for hit in response.points:
            payload = hit.payload or {}
            results.append({
                "score":       round(hit.score, 4),
                "text":        payload.get("text", ""),
                "source":      payload.get("source", ""),
                "source_type": payload.get("source_type", ""),
                "section":     payload.get("section", ""),
                "tokens":      payload.get("tokens", 0),
                "chunk_id":    payload.get("chunk_id", ""),
            })
        return results

    def search_by_text(
        self,
        text:            str,
        embedder:        "Embedder",
        limit:           int   = None,
        score_threshold: float = None,
        use_hyde:        bool  = False,
    ) -> list[dict]:
        """
        Convenience method: embed text then search.

        Parameters
        ----------
        text     : raw query string
        embedder : Embedder instance (Phase 5)
        use_hyde : use HyDE embedding (requires GROQ_API_KEY)
        """
        if use_hyde:
            vec = embedder.embed_hyde(text)
        else:
            vec = embedder.embed_text(text)
        return self.search(vec, limit=limit, score_threshold=score_threshold)

    # ── Utility ───────────────────────────────────────────────────────────────

    @property
    def collection_name(self) -> str:
        return self._col_name

    @property
    def is_cloud(self) -> bool:
        return bool(self._url)

    def __repr__(self) -> str:
        mode = f"cloud({self._url[:30]}…)" if self._url else "local"
        return f"VectorDBManager(collection='{self._col_name}', mode={mode})"