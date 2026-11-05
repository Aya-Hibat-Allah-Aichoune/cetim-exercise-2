"""Embeddings de pasajes y preguntas.

- GeminiEmbedder: API de Google (gemini-embedding-001) vía REST. Necesita GEMINI_API_KEY.
- OllamaEmbedder: modelo local (bge-m3) vía /api/embed. Sin claves.

Todos devuelven vectores con norma 1, así el producto escalar es la similitud coseno.
Cada embedder tiene un `name` ("gemini:gemini-embedding-001", "ollama:bge-m3") que se
guarda en el índice; al cargarlo se reconstruye el mismo embedder, de modo que las
preguntas siempre se codifican con el modelo con el que se indexaron los documentos.
"""
from __future__ import annotations

import time
from typing import Protocol

import httpx
import numpy as np

from .config import (
    EMBEDDING_BACKEND,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_DIM,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_URL,
)


def _normalize(vectors: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


class Embedder(Protocol):
    name: str

    def available(self) -> bool: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class GeminiEmbedder:
    """gemini-embedding-001 distingue pasajes (RETRIEVAL_DOCUMENT) de preguntas
    (RETRIEVAL_QUERY); usar el task_type adecuado mejora la recuperación."""

    def __init__(self, model: str = GEMINI_EMBEDDING_MODEL, dim: int = GEMINI_EMBEDDING_DIM,
                 api_key: str | None = GEMINI_API_KEY, url: str = GEMINI_URL):
        self.model = model
        self.dim = dim
        self.api_key = api_key
        self.url = url.rstrip("/")
        self.name = f"gemini:{model}"

    def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _retry_delay(r: httpx.Response, attempt: int) -> float:
        """Espera sugerida por la API (details[].retryDelay, p. ej. "17s") o backoff."""
        try:
            for d in r.json()["error"].get("details", []):
                if "retryDelay" in d:
                    return float(d["retryDelay"].rstrip("s")) + 1
        except Exception:
            pass
        return min(5 * 2 ** attempt, 60)

    def _post(self, payload: dict, retries: int = 6) -> dict:
        endpoint = f"{self.url}/models/{self.model}:batchEmbedContents"
        headers = {"x-goog-api-key": self.api_key or "", "Content-Type": "application/json"}
        for attempt in range(retries):
            r = httpx.post(endpoint, json=payload, headers=headers, timeout=120)
            if r.status_code in (429, 503) and attempt < retries - 1:
                # cuota por minuto del plan gratuito o servicio saturado: esperar y reintentar
                time.sleep(self._retry_delay(r, attempt))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Gemini: demasiados reintentos")

    def _embed(self, texts: list[str], task_type: str, batch_size: int = 25) -> np.ndarray:
        if not self.available():
            raise RuntimeError("Falta GEMINI_API_KEY (o GOOGLE_API_KEY) en el entorno")
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                        "outputDimensionality": self.dim,
                    }
                    for t in batch
                ]
            }
            data = self._post(payload)
            vectors.extend(e["values"] for e in data["embeddings"])
        return _normalize(vectors)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class OllamaEmbedder:
    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_EMBEDDING_MODEL):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.url}/api/tags", timeout=3)
            names = {m["name"].split(":")[0] for m in r.json().get("models", [])}
            return self.model.split(":")[0] in names
        except Exception:
            return False

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            r = httpx.post(f"{self.url}/api/embed", json={"model": self.model, "input": batch}, timeout=600)
            r.raise_for_status()
            vectors.extend(r.json()["embeddings"])
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def make_embedder(backend: str = EMBEDDING_BACKEND) -> Embedder:
    """Embedder para construir un índice nuevo, según RAG_EMBEDDING_BACKEND."""
    backend = backend.lower()
    if backend == "gemini":
        return GeminiEmbedder()
    if backend == "ollama":
        return OllamaEmbedder()
    if backend != "auto":
        raise ValueError(f"RAG_EMBEDDING_BACKEND desconocido: {backend}")
    return GeminiEmbedder() if GEMINI_API_KEY else OllamaEmbedder()


def embedder_from_name(name: str, dim: int | None = None) -> Embedder:
    """Reconstruye el embedder con el que se creó un índice a partir de meta.json.
    Nombres antiguos sin prefijo ("bge-m3") se interpretan como Ollama."""
    backend, _, model = name.partition(":")
    if not model:
        backend, model = "ollama", backend
    if backend == "gemini":
        return GeminiEmbedder(model=model, dim=dim or GEMINI_EMBEDDING_DIM)
    if backend == "ollama":
        return OllamaEmbedder(model=model)
    raise ValueError(f"Backend de embeddings desconocido en el índice: {name}")
