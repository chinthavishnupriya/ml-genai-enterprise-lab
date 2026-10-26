import tempfile
import unittest
from pathlib import Path

from src.agent import FinancialAnalystAgent


class FinancialAnalystAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(__file__).resolve().parents[1]
        self.agent = FinancialAnalystAgent(database_path=Path(self.tmp.name) / "financials.sqlite", document_dir=base / "data" / "research")

    def tearDown(self):
        self.tmp.cleanup()

    def test_latest_quarter_summary_has_rows_and_sources(self):
        result = self.agent.answer("Summarize the latest financial results.")
        self.assertIn("2025-Q4", result.answer)
        self.assertEqual(len(result.citations), 3)
        self.assertEqual(result.trace[0].action, "read_only_sql")

    def test_company_margin_uses_calculator_tool(self):
        result = self.agent.answer("What was ACME Systems operating margin in Q4?")
        self.assertIn("15.00%", result.answer)
        self.assertEqual([e.action for e in result.trace], ["read_only_sql", "calculator"])
        self.assertIn("ACME Systems", result.citations[0])

    def test_revenue_growth_uses_sql_and_calculator(self):
        result = self.agent.answer("Compare Q3 to Q4 revenue growth across the companies.")
        self.assertIn("ACME Systems", result.answer)
        self.assertIn("Cobalt Manufacturing", result.answer)
        self.assertEqual(result.trace[0].action, "read_only_sql")
        self.assertGreaterEqual(sum(e.action == "calculator" for e in result.trace), 3)

    def test_risk_question_uses_document_search(self):
        result = self.agent.answer("What are the main liquidity and leverage risks?")
        self.assertIn("Liquidity and leverage", result.answer)
        self.assertEqual(result.trace[0].action, "document_search")
        self.assertTrue(any("risk_register.md" in c for c in result.citations))

    def test_investment_recommendation_abstains(self):
        result = self.agent.answer("Should I buy ACME Systems stock?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.trace, ())
        self.assertIn("cannot provide personalized investment recommendations", result.answer)

    def test_blank_question_rejected(self):
        with self.assertRaises(ValueError):
            self.agent.answer("  ")

    def test_period_filter_can_select_q2(self):
        result = self.agent.answer("Show financial metrics for Q2.")
        self.assertIn("2025-Q2", result.answer)
        self.assertNotIn("2025-Q4", result.answer)


if __name__ == "__main__":
    unittest.main()
