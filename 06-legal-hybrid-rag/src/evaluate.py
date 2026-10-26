"""Evaluate hybrid retrieval on hand-authored queries for synthetic contracts."""
from pathlib import Path
from .legal_rag import LegalHybridRAG

CASES = [
    ("When are subscription invoices due for payment?", "fictional_saas_agreement.md", "4.1"),
    ("When may either party terminate for a material breach?", "fictional_saas_agreement.md", "7.1"),
    ("How quickly must a personal data breach be reported?", "fictional_data_processing_addendum.md", "3.1"),
    ("What security safeguards must the processor maintain?", "fictional_data_processing_addendum.md", "2.1"),
    ("What monthly availability target is in the SLA?", "fictional_service_level_agreement.md", "2.1"),
    ("How are service credits applied for availability shortfalls?", "fictional_service_level_agreement.md", "3.1"),
]


def evaluate(data_dir: str | Path = "data/contracts", top_k: int = 5) -> dict:
    paths = sorted(p for p in Path(data_dir).rglob("*") if p.suffix.lower() in {".md", ".markdown", ".txt"})
    rag = LegalHybridRAG.from_paths(paths)
    hits = 0
    reciprocal_ranks = []
    rows = []
    for query, expected_source, expected_clause in CASES:
        results = rag.retrieve(query, top_k=top_k)
        rank = next(
            (i + 1 for i, result in enumerate(results)
             if result.clause.source == expected_source and result.clause.clause_id == expected_clause),
            None,
        )
        hits += int(rank is not None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        rows.append({
            "query": query, "expected_source": expected_source, "expected_clause": expected_clause,
            "rank": rank,
            "top_result": (
                {"source": results[0].clause.source, "clause_id": results[0].clause.clause_id}
                if results else None
            ),
        })
    return {
        "cases": len(CASES),
        f"recall_at_{top_k}": round(hits / len(CASES), 4),
        f"mrr_at_{top_k}": round(sum(reciprocal_ranks) / len(CASES), 4),
        "k": top_k,
        "rows": rows,
        "note": "Small curated retrieval check over fictional contracts; not a measure of legal correctness or suitability for legal advice.",
    }


def main() -> int:
    print(__import__("json").dumps(evaluate(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
