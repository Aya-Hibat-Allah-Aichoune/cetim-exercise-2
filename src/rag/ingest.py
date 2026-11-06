"""Ingesta: PDF -> páginas -> chunks -> embeddings -> índice en disco.

Decisiones:
- El texto se extrae por página con pypdf (Python puro, sin compilar nada)
  para poder citar documento y página exactos en cada fuente.
- Los chunks NO cruzan páginas. Se pierde alguna frase partida al final de
  página, a cambio de que cada cita apunte a una página real.
- Troceado por caracteres con corte preferente en fin de frase y solapamiento.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import logging

import numpy as np
from pypdf import PdfReader

# pypdf avisa por cada /MediaBox duplicado de los PDFs de MDPI; no afecta al texto
logging.getLogger("pypdf").setLevel(logging.ERROR)

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, INDEX_DIR
from .embeddings import Embedder, GeminiEmbedder, make_embedder


@dataclass
class Chunk:
    chunk_id: str
    doc: str        # nombre del fichero
    title: str      # título del paper (primeras líneas de la página 1)
    page: int       # 1-indexed
    text: str


# ---------------------------------------------------------------- extracción
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)   # "measure-\nment" -> "measurement"
    text = text.replace("\r", "")
    text = _MULTI_SPACE.sub(" ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # saltos de línea simples -> espacio
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


_AUTHOR_LINE = re.compile(r"(\*|@|\d{4}|;|,\s+[A-Z][\w.-]+\s+[A-Z])")
_SKIP_LINE = re.compile(r"^(sensors|article|review|citation|www\.|http|doi|received|academic editor|original research|published)", re.I)


def guess_title(first_page_text: str, metadata_title: str | None = None) -> str:
    """Título del paper: metadatos del PDF si son útiles; si no, las líneas de la
    primera página que van tras la etiqueta de tipo (Article...) y antes de los autores."""
    if metadata_title and len(metadata_title.strip()) >= 15 and not metadata_title.lower().startswith(("doi", "untitled")):
        return metadata_title.strip()
    lines = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]
    title_lines: list[str] = []
    for line in lines:
        if not title_lines:
            if len(line) >= 20 and not _SKIP_LINE.match(line) and not _AUTHOR_LINE.search(line):
                title_lines.append(line)
            continue
        if _AUTHOR_LINE.search(line) or len(line) < 10 or len(title_lines) >= 4:
            break
        title_lines.append(line)
    return " ".join(title_lines)


def extract_pages(pdf_path: Path) -> tuple[str, list[tuple[int, str]]]:
    reader = PdfReader(str(pdf_path))
    raw_pages = [page.extract_text() or "" for page in reader.pages]
    meta_title = reader.metadata.title if reader.metadata else None
    title = guess_title(raw_pages[0], meta_title) if raw_pages else ""
    pages = [(i + 1, clean_text(t)) for i, t in enumerate(raw_pages) if t.strip()]
    return title or pdf_path.stem, pages


# ------------------------------------------------------------------ chunking
_SENTENCE_END = re.compile(r"[.!?]\s")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Trocea en ventanas de ~size caracteres, cortando en fin de frase si lo hay
    en el último 30 % de la ventana, con solapamiento de `overlap` caracteres."""
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start + int(size * 0.7): end]
            cuts = [m.end() for m in _SENTENCE_END.finditer(window)]
            if cuts:
                end = start + int(size * 0.7) + cuts[-1]
        piece = text[start:end].strip()
        if len(piece) > 50:  # descartar restos insignificantes
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        title, pages = extract_pages(pdf)
        for page_no, text in pages:
            for j, piece in enumerate(split_text(text)):
                chunks.append(Chunk(f"{pdf.stem}:p{page_no}:c{j}", pdf.name, title, page_no, piece))
    return chunks


# ---------------------------------------------------------------- embeddings
def embed_passages(embedder: Embedder, texts: list[str]) -> np.ndarray:
    print(f"Calculando {len(texts)} embeddings con {embedder.name} ...")
    return embedder.embed(texts)


def build_index(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR,
                embedder: Embedder | None = None) -> dict:
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(docs_dir)
    embedder = embedder or make_embedder()
    if not embedder.available():
        if isinstance(embedder, GeminiEmbedder):
            raise RuntimeError("Falta GEMINI_API_KEY: expórtala o escríbela en un fichero .env")
        raise RuntimeError(
            f"Ollama no responde o no tiene el modelo {embedder.model}. "
            f"Ejecuta: ollama pull {embedder.model}  (o define GEMINI_API_KEY para usar Gemini)"
        )
    embeddings = embed_passages(embedder, [c.text for c in chunks])
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    meta = {
        "embedding_model": embedder.name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "n_docs": len({c.doc for c in chunks}),
        "n_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
