from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Carga KEY=VALUE de un .env (sin dependencias). No pisa variables ya definidas."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")

DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", ROOT / "docs"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "index"))

# --- Chunking -------------------------------------------------------------

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 200))

# --- Gemini (API de Google) ------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_URL = os.getenv("GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", 768))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# --- Embeddings -----------------------------------------------------------
# auto | gemini | ollama   (auto: gemini si hay clave, si no ollama)
EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING_BACKEND", "auto")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "bge-m3")

# --- Recuperación ---------------------------------------------------------
TOP_K = int(os.getenv("RAG_TOP_K", 5))
CANDIDATES_PER_RETRIEVER = int(os.getenv("RAG_CANDIDATES", 20))

# --- Generación -----------------------------------------------------------
# auto | gemini | anthropic | ollama | extractive
# (auto: gemini > anthropic > ollama > extractive, según claves/servicios disponibles)
LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "auto")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
MAX_ANSWER_TOKENS = int(os.getenv("RAG_MAX_ANSWER_TOKENS", 600))
