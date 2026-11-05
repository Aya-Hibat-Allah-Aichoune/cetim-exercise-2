"""Pipeline completo: pregunta -> recuperación -> generación con citas -> fuentes."""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

from .config import TOP_K
from .generator import Generator, make_generator
from .retriever import Hit, HybridRetriever

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    n: int
    doc: str
    title: str
    page: int
    chunk_id: str
    score: float
    cited: bool
    excerpt: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source]
    backend: str
    retrieval_ms: int
    generation_ms: int

    def to_dict(self) -> dict:
        return asdict(self)


def format_context(hits: list[Hit]) -> list[str]:
    return [f"[{i}] ({h.doc}, página {h.page})\n{h.text}" for i, h in enumerate(hits, start=1)]


class RagPipeline:
    def __init__(self, retriever: HybridRetriever | None = None, generator: Generator | None = None):
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or make_generator()

    def query(self, question: str, top_k: int = TOP_K) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, top_k=top_k)
        t1 = time.perf_counter()
        answer_text = self.generator.generate(question, format_context(hits))
        t2 = time.perf_counter()

        cited = {int(n) for n in _CITATION.findall(answer_text)}
        sources = [
            Source(
                n=i, doc=h.doc, title=h.title, page=h.page, chunk_id=h.chunk_id, score=h.score,
                cited=i in cited, excerpt=h.text[:300] + ("..." if len(h.text) > 300 else ""),
            )
            for i, h in enumerate(hits, start=1)
        ]
        return Answer(
            question=question, answer=answer_text, sources=sources, backend=self.generator.name,
            retrieval_ms=int((t1 - t0) * 1000), generation_ms=int((t2 - t1) * 1000),
        )
