"""CLI for fictional contract search."""
import argparse
import json
from pathlib import Path
from .legal_rag import LegalHybridRAG


def main() -> int:
    parser = argparse.ArgumentParser(description="Search fictional contract clauses. This is not legal advice.")
    parser.add_argument("query", help="Question about the indexed fictional contracts")
    parser.add_argument("--data-dir", default="data/contracts")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-backend", choices=("tfidf", "sentence-transformers"), default="tfidf")
    parser.add_argument("--reranker", choices=("lexical", "cross-encoder"), default="lexical")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    paths = sorted(p for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in {".md", ".markdown", ".txt"})
    if not paths:
        parser.error(f"No supported contract documents found under {args.data_dir}")
    rag = LegalHybridRAG.from_paths(
        paths,
        vector_backend=args.vector_backend,
        reranker_backend=args.reranker,
        cross_encoder_model=args.cross_encoder_model,
    )
    answer = rag.answer(args.query, top_k=args.top_k)
    payload = {
        "query": answer.query,
        "answer": answer.answer,
        "citations": answer.citations,
        "abstained": answer.abstained,
        "vector_backend": rag.vector_backend,
        "reranker_backend": rag.reranker_backend,
        "retrieved": [
            {
                "source": item.clause.source,
                "clause_id": item.clause.clause_id,
                "title": item.clause.title,
                "bm25_score": round(item.bm25_score, 5),
                "vector_score": round(item.vector_score, 5),
                "rrf_score": round(item.rrf_score, 6),
                "rerank_score": round(item.rerank_score, 6),
            }
            for item in answer.retrieved
        ],
        "disclaimer": "Fictional-contract retrieval demonstration only; not legal advice.",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False) if args.json else answer.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
