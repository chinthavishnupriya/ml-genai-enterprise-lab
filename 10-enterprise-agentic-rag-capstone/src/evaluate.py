"""Small RAGAS-inspired retrieval/citation smoke evaluation without external dependencies."""
from __future__ import annotations
import json
from .agent import EnterpriseRAGAgent

CASES = [
    {
        "question": "What should operations do when a shipment is delayed?",
        "expected_citation": "delay_escalation.md#Delay escalation",
        "label": "policy_retrieval",
    },
    {
        "question": "Which documents are required for international customs clearance?",
        "expected_citation": "cross_border.md#Customs clearance",
        "label": "policy_retrieval",
    },
    {
        "question": "How should damaged goods be documented and reported?",
        "expected_citation": "damage_returns.md#Damaged goods",
        "label": "policy_retrieval",
    },
    {
        "question": "What precautions are required for temperature-controlled shipments?",
        "expected_citation": "temperature_control.md#Temperature control",
        "label": "policy_retrieval",
    },
    {
        "question": "What does delivered status mean in the tracking data?",
        "expected_citation": "shipment_status.md#Delivered",
        "label": "policy_retrieval",
    },
    {
        "question": "Track shipment SHP-1002 and tell me its current status.",
        "expected_citation": "shipment_events.jsonl#SHP-1002",
        "label": "shipment_lookup",
    },
    {
        "question": "Where is shipment SHP-1001 going?",
        "expected_citation": "shipment_events.jsonl#SHP-1001",
        "label": "shipment_lookup",
    },
    {
        "question": "What are the steps for a package waiting for pickup?",
        "expected_citation": "shipment_status.md#Awaiting pickup",
        "label": "policy_retrieval",
    },
    {
        "question": "How should cross-border delays be escalated?",
        "expected_citation": "cross_border.md#Escalation and holds",
        "label": "policy_retrieval",
    },
]


def evaluate() -> dict:
    agent = EnterpriseRAGAgent()
    results = []
    retrieved_relevant = 0
    total_retrieved = 0
    source_matches = 0
    cited_answers = 0
    for case in CASES:
        response = agent.answer(case["question"])
        relevant_in_retrieved = int(case["expected_citation"] in response.citations)
        source_matches += relevant_in_retrieved
        if response.citations:
            cited_answers += 1
        # The response citations are the passages actually used in the deterministic answer.
        total_retrieved += len(response.citations)
        retrieved_relevant += relevant_in_retrieved
        results.append({
            "question": case["question"],
            "label": case["label"],
            "expected_citation": case["expected_citation"],
            "matched": bool(relevant_in_retrieved),
            "abstained": response.abstained,
            "route": response.route,
            "citations": list(response.citations),
            "trace_steps": len(response.trace),
        })
    n = len(results)
    return {
        "cases": n,
        "expected_source_match_rate": round(source_matches / n, 4) if n else 0.0,
        "context_precision_at_answered_sources": round(retrieved_relevant / total_retrieved, 4) if total_retrieved else 0.0,
        "citation_coverage": round(cited_answers / n, 4) if n else 0.0,
        "results": results,
        "note": "RAGAS-inspired deterministic smoke metrics, not the RAGAS package. Labels are hand-authored; this small synthetic set measures routing/citation behavior, not semantic faithfulness or production answer quality.",
    }


def main() -> int:
    print(json.dumps(evaluate(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
