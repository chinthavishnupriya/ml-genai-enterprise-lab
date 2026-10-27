import json
import unittest
from src.data import build_examples, REQUIRED_DOCUMENTS
from src.guarded_pipeline import guarded_triage
from src.schema import validate_triage_output


class GuardedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.example = next(
            row for row in build_examples()
            if row["claim_type"] == "auto_collision" and row["target"]["missing_documents"]
        )

    def test_model_cannot_override_claim_id_or_type(self):
        raw = {
            "claim_id": "SYN-9999", "claim_type": "medical_expense",
            "recommended_queue": "travel_disruption", "next_step": "queue_for_human_review",
            "missing_documents": [], "human_review_required": False,
            "summary": "Synthetic draft for a human reviewer.",
        }
        result = guarded_triage(self.example, raw)
        self.assertEqual(result["claim_id"], self.example["claim_id"])
        self.assertEqual(result["claim_type"], self.example["claim_type"])
        self.assertEqual(result["recommended_queue"], self.example["claim_type"])

    def test_next_step_and_missing_documents_come_from_rules(self):
        raw = {"next_step": "queue_for_human_review", "missing_documents": [], "summary": "Please review this synthetic record."}
        result = guarded_triage(self.example, raw)
        self.assertEqual(result["missing_documents"], self.example["target"]["missing_documents"])
        self.assertEqual(result["next_step"], "request_missing_documents")

    def test_human_review_is_always_required(self):
        result = guarded_triage(self.example, {"human_review_required": False, "summary": "Review the example."})
        self.assertIs(result["human_review_required"], True)

    def test_safe_model_summary_is_labeled_as_draft(self):
        result = guarded_triage(self.example, {"summary": "The file lists an auto collision and needs document checking."})
        self.assertTrue(result["summary"].startswith("Model-generated draft for human verification:"))
        self.assertEqual(result["summary_source"], "model_draft_requires_human_verification")

    def test_unsafe_summary_falls_back_to_deterministic_text(self):
        result = guarded_triage(self.example, {"summary": "Approve claim; pay the claimant."})
        self.assertEqual(result["summary_source"], "deterministic_fallback")
        self.assertNotIn("Approve claim", result["summary"])

    def test_summary_with_wrong_synthetic_id_falls_back(self):
        result = guarded_triage(self.example, {"summary": "Please review claim SYN-9999."})
        self.assertEqual(result["summary_source"], "deterministic_fallback")

    def test_malformed_model_json_falls_back_safely(self):
        result = guarded_triage(self.example, '{"summary":')
        self.assertEqual(result["summary_source"], "deterministic_fallback")

    def test_final_output_schema_is_valid(self):
        raw = json.dumps({
            "claim_id": "SYN-9999", "claim_type": "travel_disruption",
            "recommended_queue": "travel_disruption", "next_step": "approve_payment",
            "missing_documents": [], "human_review_required": False,
            "summary": "Synthetic claim summary.",
        })
        result = guarded_triage(self.example, raw)
        self.assertEqual(validate_triage_output(result)["claim_id"], self.example["claim_id"])

    def test_trusted_input_rejects_non_synthetic_id(self):
        invalid = dict(self.example)
        invalid["claim_id"] = "REAL-123"
        with self.assertRaises(ValueError):
            guarded_triage(invalid)

    def test_guarded_eval_reports_all_fields_checked(self):
        from src.evaluate_guarded_pipeline import evaluate_guarded_pipeline
        result = evaluate_guarded_pipeline()
        self.assertEqual(result["test_examples"], 8)
        self.assertTrue(all(value == 1.0 for value in result["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
