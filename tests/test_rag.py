import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.ingest import clean_text, split_text  # noqa: E402


def test_split_text_respects_size_and_overlap():
    text = ("Una frase de prueba con algo de contenido. " * 200).strip()
    chunks = split_text(text, size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[1][:40] in chunks[0]


def test_clean_text_joins_hyphenation_and_lines():
    assert clean_text("measure-\nment of soil\nmoisture") == "measurement of soil moisture"


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    monkeypatch_module.setenv("RAG_LLM_BACKEND", "extractive")
    if not (ROOT / "index" / "chunks.json").exists():
        pytest.skip("índice no construido; ejecuta scripts/build_index.py")
    from src import api
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_ui_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "/query" in r.text


def test_health_and_documents(client):
    assert client.get("/health").json()["status"] == "ok"
    docs = client.get("/documents").json()
    assert len(docs) == 5


def test_query_returns_sources_with_pages(client):
    r = client.post("/query", json={"question": "vegetation index water stress tomato", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["sources"]) == 3
    assert all(s["page"] >= 1 and s["doc"].endswith(".pdf") for s in body["sources"])
    assert body["answer"]


def test_query_validation(client):
    assert client.post("/query", json={"question": "ab"}).status_code == 422


# ---------------------------------------------------------------- Gemini (API simulada)
class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload, self.status_code = payload, status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_gemini_embedder_uses_task_types_and_normalizes(monkeypatch):
    from src.rag import embeddings

    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        n = len(json["requests"])
        return _FakeResponse({"embeddings": [{"values": [3.0, 4.0]} for _ in range(n)]})

    monkeypatch.setattr(embeddings.httpx, "post", fake_post)
    emb = embeddings.GeminiEmbedder(model="gemini-embedding-001", dim=2, api_key="test-key")
    vecs = emb.embed(["a", "b"])
    q = emb.embed_query("c")

    assert vecs.shape == (2, 2) and abs(float((vecs[0] ** 2).sum()) - 1.0) < 1e-6
    assert q.shape == (2,)
    assert calls[0][0].endswith("gemini-embedding-001:batchEmbedContents")
    assert calls[0][2]["x-goog-api-key"] == "test-key"
    assert calls[0][1]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert calls[1][1]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    assert calls[0][1]["requests"][0]["outputDimensionality"] == 2


def test_gemini_generator_builds_prompt_and_reads_text(monkeypatch):
    from src.rag import generator

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, body=json)
        return _FakeResponse({"candidates": [{"content": {"parts": [{"text": "Respuesta [1]."}]},
                                              "finishReason": "STOP"}]})

    monkeypatch.setattr(generator.httpx, "post", fake_post)
    gen = generator.GeminiGenerator(model="gemini-3.5-flash", api_key="test-key")
    out = gen.generate("¿Qué es X?", ["[1] (doc.pdf, página 2)\nX es Y."])

    assert out == "Respuesta [1]."
    assert captured["url"].endswith("gemini-3.5-flash:generateContent")
    assert "X es Y." in captured["body"]["contents"][0]["parts"][0]["text"]
    assert captured["body"]["systemInstruction"]["parts"][0]["text"] == generator.SYSTEM_PROMPT
    assert captured["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_gemini_generator_requires_key():
    from src.rag.generator import GeminiGenerator

    with pytest.raises(RuntimeError):
        GeminiGenerator(api_key=None)


def test_embedder_from_name_backwards_compatible():
    from src.rag.embeddings import GeminiEmbedder, OllamaEmbedder, embedder_from_name

    assert isinstance(embedder_from_name("bge-m3"), OllamaEmbedder)
    assert isinstance(embedder_from_name("ollama:bge-m3"), OllamaEmbedder)
    e = embedder_from_name("gemini:gemini-embedding-001", dim=768)
    assert isinstance(e, GeminiEmbedder) and e.dim == 768
