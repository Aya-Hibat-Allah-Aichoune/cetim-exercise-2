
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .rag.config import TOP_K
from .rag.pipeline import RagPipeline

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = RagPipeline()
    yield
    _state.clear()


app = FastAPI(
    title="RAG sobre documentación técnica (CETIM, prueba técnica ex 2 )",
    description="Pregunta en lenguaje natural sobre los papers indexados; la respuesta cita documento y página.",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, examples=["¿Qué índice de vegetación es más sensible al daño del saltahojas marrón en arroz?"])
    top_k: int = Field(TOP_K, ge=1, le=20, description="Número de fragmentos recuperados")


class SourceOut(BaseModel):
    n: int
    doc: str
    title: str
    page: int
    chunk_id: str
    score: float
    cited: bool
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    backend: str
    retrieval_ms: int
    generation_ms: int


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    p: RagPipeline = _state["pipeline"]
    return {
        "status": "ok",
        "generator": p.generator.name,
        "retrieval": p.retriever.mode,
        "embeddings": p.retriever.embedding_model,
        "n_chunks": len(p.retriever.chunks),
    }


@app.get("/documents")
def documents():
    return _state["pipeline"].retriever.documents()


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        return _state["pipeline"].query(req.question, top_k=req.top_k).to_dict()
    except Exception as exc:  
        raise HTTPException(status_code=502, detail=f"Error en el pipeline: {exc}") from exc
