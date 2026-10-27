import unittest
from src.evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_evaluation_contains_nine_handwritten_cases(self):
        result = evaluate()
        self.assertEqual(result["cases"], 9)
        self.assertEqual(len(result["results"]), 9)

    def test_evaluation_metrics_are_bounded(self):
        result = evaluate()
        for metric in ("expected_source_match_rate", "context_precision_at_answered_sources", "citation_coverage"):
            self.assertGreaterEqual(result[metric], 0.0)
            self.assertLessEqual(result[metric], 1.0)


if __name__ == "__main__":
    unittest.main()
