import json
import unittest
from src.llm_provider import extract_result, OpenAICompatibleRouter, ProviderError
from src.schema import IntentCategory, Priority, SupportTicket

VALID_RESULT = {
    "ticket_id": "T-9",
    "category": "billing_payment",
    "priority": "medium",
    "confidence": 0.9,
    "escalation_required": False,
    "draft_response": "Thanks for contacting support. We will review the invoice and follow up with the available details.",
    "rationale": "Invoice keyword matched.",
}


class ProviderTests(unittest.TestCase):
    def test_parses_plain_json(self):
        result = extract_result(json.dumps(VALID_RESULT), "T-9")
        self.assertEqual(result.category, IntentCategory.BILLING_PAYMENT)

    def test_parses_markdown_fenced_json(self):
        fence = chr(96) * 3
        result = extract_result(fence + "json\n" + json.dumps(VALID_RESULT) + "\n" + fence, "T-9")
        self.assertEqual(result.ticket_id, "T-9")

    def test_rejects_invalid_schema(self):
        broken = dict(VALID_RESULT, priority="critical")
        with self.assertRaises(Exception):
            extract_result(json.dumps(broken), "T-9")

    def test_rejects_mismatched_ticket_id(self):
        with self.assertRaises(ValueError):
            extract_result(json.dumps(VALID_RESULT), "T-other")

    def test_retries_invalid_output_then_succeeds(self):
        outputs = iter(["not json", json.dumps({"choices": [{"message": {"content": json.dumps(VALID_RESULT)}}]})])
        def fake_transport(url, headers, body, timeout):
            return next(outputs)
        router = OpenAICompatibleRouter(api_key="test-key", max_attempts=2, backoff_seconds=0, transport=fake_transport, sleep=lambda _: None)
        ticket = SupportTicket(ticket_id="T-9", subject="Invoice", message="I need a copy of my invoice.")
        self.assertEqual(router.route(ticket).ticket_id, "T-9")

    def test_security_incident_cannot_be_downgraded_by_provider(self):
        bad_result = dict(
            VALID_RESULT,
            category="general_inquiry",
            priority="low",
            escalation_required=False,
            draft_response="Thanks for reaching out. We will review your general question.",
        )
        envelope = {"choices": [{"message": {"content": json.dumps(bad_result)}}]}
        ticket = SupportTicket(
            ticket_id="T-9",
            subject="My account was hacked",
            message="Someone accessed my account without authorization.",
        )
        router = OpenAICompatibleRouter(
            api_key="test-key",
            max_attempts=1,
            transport=lambda url, headers, body, timeout: json.dumps(envelope),
        )
        result = router.route(ticket)
        self.assertEqual(result.category, IntentCategory.SECURITY_PRIVACY)
        self.assertEqual(result.priority, Priority.URGENT)
        self.assertTrue(result.escalation_required)
        self.assertIn("security", result.draft_response.lower())

    def test_high_priority_rule_cannot_be_downgraded_by_provider(self):
        bad_result = dict(VALID_RESULT, priority="low", escalation_required=False)
        envelope = {"choices": [{"message": {"content": json.dumps(bad_result)}}]}
        ticket = SupportTicket(
            ticket_id="T-9",
            subject="Charged twice",
            message="My card was charged twice for the same order.",
        )
        router = OpenAICompatibleRouter(
            api_key="test-key",
            max_attempts=1,
            transport=lambda url, headers, body, timeout: json.dumps(envelope),
        )
        result = router.route(ticket)
        self.assertEqual(result.priority, Priority.HIGH)
        self.assertTrue(result.escalation_required)

    def test_missing_key_fails_clearly(self):
        router = OpenAICompatibleRouter(api_key="", max_attempts=1)
        ticket = SupportTicket(ticket_id="T-9", subject="Invoice", message="I need a copy of my invoice.")
        with self.assertRaises(ProviderError):
            router.route(ticket)


if __name__ == "__main__":
    unittest.main()
