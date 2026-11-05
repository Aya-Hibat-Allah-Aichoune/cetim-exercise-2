"""Recuperación híbrida Porqué los papers están llenos de siglas, nombres de sensores y
cifras (NDVI, LoRaWAN, "0.92") que un embedding semántico difumina y un
índice léxico clava; y al revés, una pregunta en español sobre un texto en
inglés solo la resuelve el embedding multilingüe. RRF combina ambos sin
tener que calibrar escalas de puntuación.

>> El embedder de consulta se reconstruye a partir de meta.json, para que coincida
siempre con el que generó el índice (Gemini u Ollama). Si no está disponible al
arrancar (sin clave / sin Ollama), el recuperador funciona solo con BM25 (modo
degradado, se indica en /health).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import CANDIDATES_PER_RETRIEVER, INDEX_DIR, TOP_K
from .embeddings import Embedder, embedder_from_name

_TOKEN = re.compile(r"[a-záéíóúñü0-9]+(?:[-.][a-záéíóúñü0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class Hit:
    chunk_id: str
    doc: str
    title: str
    page: int
    text: str
    score: float          
    dense_rank: int | None
    bm25_rank: int | None


class HybridRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        chunks_path = index_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"No hay índice en {index_dir}. Ejecuta: python scripts/build_index.py"
            )
        self.chunks: list[dict] = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(index_dir / "embeddings.npy")
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        embedder = embedder_from_name(self.meta["embedding_model"], dim=self.meta.get("dim"))
        self.embedder: Embedder | None = embedder if embedder.available() else None
        self.embedding_model = self.meta["embedding_model"]

    @property
    def mode(self) -> str:
        return "hybrid" if self.embedder else "bm25-only"

    # ---- recuperadores individuales
    def dense_ranking(self, query: str, n: int) -> list[int]:
        if self.embedder is None:
            return []
        sims = self.embeddings @ self.embedder.embed_query(query)
        return [int(i) for i in np.argsort(-sims)[:n]]

    def bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]

    # ---- fusión
    def search(self, query: str, top_k: int = TOP_K, candidates: int = CANDIDATES_PER_RETRIEVER,
               rrf_k: int = 60) -> list[Hit]:
        dense = self.dense_ranking(query, candidates)
        sparse = self.bm25_ranking(query, candidates)
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        hits = []
        for idx, score in best:
            c = self.chunks[idx]
            hits.append(Hit(
                chunk_id=c["chunk_id"], doc=c["doc"], title=c["title"], page=c["page"], text=c["text"],
                score=round(score, 5),
                dense_rank=dense.index(idx) + 1 if idx in dense else None,
                bm25_rank=sparse.index(idx) + 1 if idx in sparse else None,
            ))
        return hits

    def documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for c in self.chunks:
            d = seen.setdefault(c["doc"], {"doc": c["doc"], "title": c["title"], "pages": 0, "chunks": 0})
            d["chunks"] += 1
            d["pages"] = max(d["pages"], c["page"])
        return list(seen.values())
