import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.retriever import HybridRetriever  # noqa: E402

K = 5


def evaluate(retriever: HybridRetriever, questions: list[dict]) -> dict:
    results = {"dense": [], "bm25": [], "hybrid": []}
    for q in questions:
        rankings = {
            "dense": retriever.dense_ranking(q["question"], K),
            "bm25": retriever.bm25_ranking(q["question"], K),
            "hybrid": [retriever.chunks.index(next(c for c in retriever.chunks if c["chunk_id"] == h.chunk_id))
                       for h in retriever.search(q["question"], top_k=K)],
        }
        for name, idxs in rankings.items():
            docs = [retriever.chunks[i]["doc"] for i in idxs]
            rank = docs.index(q["expected_doc"]) + 1 if q["expected_doc"] in docs else None
            results[name].append(rank)
    summary = {}
    for name, ranks in results.items():
        summary[name] = {
            f"hit@1": sum(r == 1 for r in ranks) / len(ranks),
            f"hit@{K}": sum(r is not None for r in ranks) / len(ranks),
            "mrr": sum(1 / r for r in ranks if r) / len(ranks),
        }
    return summary, results


if __name__ == "__main__":
    questions = json.loads((Path(__file__).resolve().parents[1] / "eval" / "questions.json").read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    if retriever.mode != "hybrid":
        sys.exit(f"Embedder {retriever.embedding_model} no disponible (clave de Gemini u Ollama): "
                 "no se puede evaluar el recuperador denso.")
    summary, per_question = evaluate(retriever, questions)
    print(f"{len(questions)} preguntas, k={K}, embeddings={retriever.embedding_model}\n")
    print(f"{'recuperador':12s} {'hit@1':>7s} {'hit@5':>7s} {'MRR':>7s}")
    for name, m in summary.items():
        print(f"{name:12s} {m['hit@1']:7.2f} {m['hit@5']:7.2f} {m['mrr']:7.2f}")
    print("\nposición del documento esperado por pregunta (None = no aparece):")
    for q, d, b, h in zip(questions, per_question["dense"], per_question["bm25"], per_question["hybrid"]):
        print(f"  dense={d} bm25={b} hybrid={h}  | {q['question'][:80]}")
    out = Path(__file__).resolve().parents[1] / "eval" / "results.json"
    out.write_text(json.dumps({"k": K, "embedding_model": retriever.embedding_model, "summary": summary, "per_question": per_question}, indent=2), encoding="utf-8")
    print(f"\nguardado en {out}")
