"""Deterministic agentic RAG orchestrator with safety gates and cited answers."""
from __future__ import annotations
from pathlib import Path
import re
from .guardrails import extract_citations, inspect_input, validate_cited_output
from .ingestion import DEFAULT_POLICY_DIR, DEFAULT_SHIPMENT_FILE, build_corpus
from .models import AgentResponse, SearchHit
from .retrieval import HybridRetriever
from .tools import SHIPMENT_ID_RE, ShipmentLookupTool, is_tracking_intent

MAX_CONTEXT_CHARS = 520


def _excerpt(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rsplit(" ", 1)[0] + "…"


class EnterpriseRAGAgent:
    """Local orchestrator that routes to a shipment tool or hybrid policy retriever.

    This offline capstone uses deterministic templates instead of an LLM planner so its
    tool permissions and behavior are reproducible. The same response contract can be
    extended with an optional language model behind the output-validation boundary.
    """
    def __init__(
        self,
        policy_dir: str | Path = DEFAULT_POLICY_DIR,
        shipment_file: str | Path = DEFAULT_SHIPMENT_FILE,
        *,
        top_k: int = 3,
        min_retrieval_score: float = 0.12,
    ) -> None:
        self.retriever = HybridRetriever(build_corpus(policy_dir))
        self.shipment_tool = ShipmentLookupTool(shipment_file)
        self.top_k = top_k
        self.min_retrieval_score = min_retrieval_score

    @staticmethod
    def _response(question, answer, route, citations=(), abstained=False, trace=(), safety=None):
        return AgentResponse(
            question=question,
            answer=answer,
            route=route,
            citations=tuple(citations),
            abstained=abstained,
            trace=tuple(trace),
            safety=safety or {"input_allowed": True, "redactions": 0, "output_check": "not_required"},
        )

    def answer(self, question: str) -> AgentResponse:
        checked = inspect_input(question)
        if not checked.allowed:
            return self._response(
                question if isinstance(question, str) else "",
                "I can only answer a logistics question using the approved local tools and knowledge base. This request was blocked by the input safety check.",
                "input_safety_block",
                abstained=True,
                trace=({"step": 1, "tool": "input_guard", "result": checked.blocked_reason},),
                safety={"input_allowed": False, "blocked_reason": checked.blocked_reason, "redactions": 0},
            )
        safe_question = checked.sanitized_text
        ids = SHIPMENT_ID_RE.findall(safe_question)
        trace = [{"step": 1, "tool": "input_guard", "result": "allowed", "redactions": checked.redactions}]
        safety = {"input_allowed": True, "redactions": checked.redactions, "output_check": "pending"}

        if ids and is_tracking_intent(safe_question):
            shipment_id = ids[0].upper()
            result = self.shipment_tool.lookup(shipment_id)
            trace.append({"step": 2, "tool": "shipment_lookup", "shipment_id": shipment_id, "success": result.success, "result": result.summary})
            if not result.success:
                return self._response(
                    safe_question,
                    f"I couldn't find {shipment_id} in the local synthetic shipment dataset. I can't infer its live status.",
                    "shipment_lookup",
                    abstained=True,
                    trace=trace,
                    safety={**safety, "output_check": "abstention_no_record"},
                )
            row = result.data["shipment"]
            answer = (
                f"**Shipment {row['shipment_id']} — {row['status']}**\n\n"
                f"- Origin: {row['origin']}\n"
                f"- Destination: {row['destination']}\n"
                f"- Estimated delivery (synthetic record): {row['estimated_delivery']}\n"
                f"- Last update: {row['last_update']}\n"
                f"- Update note: {row['update_note']}\n"
                f"- Carrier: {row['carrier']}\n\n"
                f"Source: [{result.source}]. This is a fictional local record, not live carrier tracking."
            )
            allowed = {result.source}
            ok, reason = validate_cited_output(answer, allowed)
            safety["output_check"] = reason
            if not ok:
                return self._response(
                    safe_question, "I couldn't produce a verifiable answer from the local record.",
                    "output_safety_fallback", abstained=True, trace=trace, safety=safety
                )
            return self._response(safe_question, answer, "shipment_lookup", (result.source,), False, trace, safety)

        hits = self.retriever.search(safe_question, top_k=self.top_k)
        hits = [hit for hit in hits if hit.score >= self.min_retrieval_score]
        trace.append({
            "step": 2, "tool": "hybrid_policy_retrieval",
            "query": safe_question, "top_k": self.top_k,
            "returned": len(hits),
            "sources": [hit.chunk.citation for hit in hits],
        })
        if not hits:
            answer = "I couldn't find enough relevant evidence in the local logistics knowledge base to answer this question. Please consult the current approved policy or an operations specialist."
            return self._response(
                safe_question, answer, "policy_retrieval", (), True, trace,
                {**safety, "output_check": "abstention_no_evidence"},
            )
        selected: list[SearchHit] = hits[:2]
        allowed = {hit.chunk.citation for hit in hits}
        sections = []
        for hit in selected:
            sections.append(f"- **{hit.chunk.section}**: {_excerpt(hit.chunk.text)} [{hit.chunk.citation}]")
        answer = (
            "I found the following relevant guidance in the local logistics knowledge base:\n\n"
            + "\n".join(sections)
            + "\n\nThis answer is based only on the synthetic local policy corpus. Confirm the current approved policy before acting."
        )
        ok, reason = validate_cited_output(answer, allowed)
        safety["output_check"] = reason
        if not ok:
            return self._response(
                safe_question, "I couldn't produce an answer with verifiable local citations.",
                "output_safety_fallback", (), True, trace, safety
            )
        citations = extract_citations(answer)
        return self._response(safe_question, answer, "hybrid_policy_retrieval", citations, False, trace, safety)
