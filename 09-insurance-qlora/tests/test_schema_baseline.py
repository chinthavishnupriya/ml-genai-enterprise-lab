import unittest
from src.baseline import route_claim
from src.data import REQUIRED_DOCUMENTS, split_examples
from src.evaluate import evaluate_baseline
from src.schema import validate_triage_output


def valid_output():
    return {
        "claim_id": "SYN-9001",
        "claim_type": "auto_collision",
        "recommended_queue": "auto_collision",
        "next_step": "queue_for_human_review",
        "missing_documents": [],
        "human_review_required": True,
        "summary": "Synthetic claim routed to a qualified human reviewer.",
    }


class SchemaTests(unittest.TestCase):
    def test_valid_output_accepted(self):
        self.assertEqual(validate_triage_output(valid_output())["claim_id"], "SYN-9001")

    def test_missing_required_key_rejected(self):
        item = valid_output()
        del item["summary"]
        with self.assertRaises(ValueError):
            validate_triage_output(item)

    def test_non_synthetic_claim_id_rejected(self):
        item = valid_output()
        item["claim_id"] = "REAL-123"
        with self.assertRaises(ValueError):
            validate_triage_output(item)

    def test_unknown_claim_type_rejected(self):
        item = valid_output()
        item["claim_type"] = "fraud"
        with self.assertRaises(ValueError):
            validate_triage_output(item)

    def test_queue_must_match_claim_type(self):
        item = valid_output()
        item["recommended_queue"] = "property_damage"
        with self.assertRaises(ValueError):
            validate_triage_output(item)

    def test_human_review_is_mandatory(self):
        item = valid_output()
        item["human_review_required"] = False
        with self.assertRaises(ValueError):
            validate_triage_output(item)

    def test_unsupported_next_step_rejected(self):
        item = valid_output()
        item["next_step"] = "approve_payment"
        with self.assertRaises(ValueError):
            validate_triage_output(item)

    def test_automatic_claim_decision_language_rejected(self):
        item = valid_output()
        item["summary"] = "Approve claim and pay the claimant"
        with self.assertRaises(ValueError):
            validate_triage_output(item)


class BaselineTests(unittest.TestCase):
    def test_incomplete_documents_request_missing_items(self):
        required = REQUIRED_DOCUMENTS["auto_collision"]
        result = route_claim("auto_collision", "SYN-TEST1", required[:-1])
        self.assertEqual(result["next_step"], "request_missing_documents")
        self.assertEqual(result["missing_documents"], [required[-1]])
        self.assertTrue(result["human_review_required"])

    def test_complete_documents_route_to_human_review(self):
        required = REQUIRED_DOCUMENTS["property_damage"]
        result = route_claim("property_damage", "SYN-TEST2", required)
        self.assertEqual(result["next_step"], "queue_for_human_review")
        self.assertEqual(result["missing_documents"], [])
        self.assertTrue(result["human_review_required"])

    def test_unknown_route_rejected(self):
        with self.assertRaises(ValueError):
            route_claim("unknown", "SYN-TEST3", [])

    def test_baseline_evaluation_checks_heldout_scenario(self):
        result = evaluate_baseline()
        self.assertEqual(result["test_examples"], 8)
        self.assertEqual(result["route_accuracy"], 1.0)
        self.assertEqual(result["next_step_accuracy"], 1.0)
        self.assertEqual(result["missing_documents_accuracy"], 1.0)
        self.assertIn("not real claims accuracy", result["note"])


if __name__ == "__main__":
    unittest.main()
