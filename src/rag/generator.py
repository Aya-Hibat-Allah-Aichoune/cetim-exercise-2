"""Generadores de respuesta. Todos reciben (pregunta, contexto numerado) y
devuelven texto con citas [n].

- GeminiGenerator:     API de Google (gemini-3.5-flash) vía REST. Necesita GEMINI_API_KEY.
- AnthropicGenerator:  API de Anthropic si hay ANTHROPIC_API_KEY.
- OllamaGenerator:     modelo local vía HTTP. Sin claves.
- ExtractiveGenerator: sin LLM. Devuelve los pasajes más relevantes tal cual.
  Garantiza que la API funciona aunque no haya ningún modelo disponible.

`make_generator()` elige según RAG_LLM_BACKEND (auto: gemini > anthropic > ollama > extractive).
"""
from __future__ import annotations

import os
import time
from typing import Protocol

import httpx

from .config import (
    ANTHROPIC_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_URL,
    LLM_BACKEND,
    MAX_ANSWER_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_URL,
)

SYSTEM_PROMPT = """Eres un asistente técnico que responde preguntas usando EXCLUSIVAMENTE los fragmentos de documentos proporcionados.

Reglas:
1. Responde en el mismo idioma en que está formulada la pregunta.
2. Cada afirmación debe ir seguida de la cita del fragmento que la sostiene, con el formato [n] (por ejemplo [2]). Puedes citar varios: [1][3].
3. Si los fragmentos no contienen la información necesaria, dilo claramente ("Los documentos no contienen información sobre...") y no inventes.
4. Sé conciso: entre dos y seis frases, sin introducciones ni despedidas."""


def build_user_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n".join(context_blocks)
    return (
        f"Fragmentos:\n\n{context}\n\nPregunta: {question}\n\n"
        "(Responde en el mismo idioma en que está escrita la pregunta.)"
    )


class Generator(Protocol):
    name: str

    def generate(self, question: str, context_blocks: list[str]) -> str: ...


class ExtractiveGenerator:
    name = "extractive"

    def generate(self, question: str, context_blocks: list[str]) -> str:
        top = context_blocks[:2]
        return (
            "Sin modelo generador disponible; se devuelven los fragmentos más relevantes:\n\n"
            + "\n\n".join(top)
        )


class GeminiGenerator:
    def __init__(self, model: str = GEMINI_MODEL, api_key: str | None = GEMINI_API_KEY, url: str = GEMINI_URL):
        if not api_key:
            raise RuntimeError("Falta GEMINI_API_KEY (o GOOGLE_API_KEY) en el entorno")
        self.model = model
        self.api_key = api_key
        self.url = url.rstrip("/")
        self.name = f"gemini:{model}"

    def _post(self, payload: dict, retries: int = 4) -> httpx.Response:
        for attempt in range(retries):
            r = httpx.post(
                f"{self.url}/models/{self.model}:generateContent",
                json=payload,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                timeout=120,
            )
            if r.status_code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)  # cuota del plan gratuito o modelo saturado
                continue
            return r
        return r

    def generate(self, question: str, context_blocks: list[str]) -> str:
        # Los modelos Gemini "piensan" por defecto; para una respuesta corta con citas no
        # hace falta y multiplica el tiempo (3-8 s frente a ~1 s). Si el modelo elegido
        # no admite desactivarlo (400), se reintenta sin esa opción.
        generation_config: dict = {
            "temperature": 0.1,
            "maxOutputTokens": max(MAX_ANSWER_TOKENS, 1024),
            "thinkingConfig": {"thinkingBudget": 0},
        }
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": build_user_prompt(question, context_blocks)}]}],
            "generationConfig": generation_config,
        }
        r = self._post(payload)
        if r.status_code == 400 and "thinking" in r.text.lower():
            generation_config.pop("thinkingConfig")
            r = self._post(payload)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "sin candidatos")
            return f"El modelo no ha devuelto respuesta ({reason})."
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            return f"El modelo no ha devuelto texto (finishReason={candidates[0].get('finishReason')})."
        return text


class OllamaGenerator:
    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"

    @staticmethod
    def available(url: str = OLLAMA_URL, model: str = OLLAMA_MODEL) -> bool:
        try:
            r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=3)
            names = [m["name"] for m in r.json().get("models", [])]
            return model in names or model.split(":")[0] in {n.split(":")[0] for n in names}
        except Exception:
            return False

    def generate(self, question: str, context_blocks: list[str]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, context_blocks)},
            ],
            "options": {"temperature": 0.1, "num_predict": MAX_ANSWER_TOKENS},
        }
        r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class AnthropicGenerator:
    def __init__(self, model: str = ANTHROPIC_MODEL):
        import anthropic

        self.client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY del entorno
        self.model = model
        self.name = f"anthropic:{model}"

    def generate(self, question: str, context_blocks: list[str]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max(MAX_ANSWER_TOKENS, 1024),
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(question, context_blocks)}],
        )
        if response.stop_reason == "refusal":
            return "El modelo ha rechazado responder a esta pregunta."
        return "".join(b.text for b in response.content if b.type == "text").strip()


def make_generator(backend: str = LLM_BACKEND) -> Generator:
    backend = backend.lower()
    if backend == "gemini":
        return GeminiGenerator()
    if backend == "anthropic":
        return AnthropicGenerator()
    if backend == "ollama":
        return OllamaGenerator()
    if backend == "extractive":
        return ExtractiveGenerator()
    if backend != "auto":
        raise ValueError(f"RAG_LLM_BACKEND desconocido: {backend}")
    if GEMINI_API_KEY:
        return GeminiGenerator()
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicGenerator()
    if OllamaGenerator.available():
        return OllamaGenerator()
    return ExtractiveGenerator()
