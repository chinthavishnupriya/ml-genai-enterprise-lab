import unittest
import pandas as pd
from src.data import make_synthetic_transactions, validate_transactions, FEATURES, TARGET

class TransactionDataTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        a = make_synthetic_transactions(30, 200, 7)
        b = make_synthetic_transactions(30, 200, 7)
        pd.testing.assert_frame_equal(a, b)

    def test_generator_has_both_labels_and_required_columns(self):
        df = make_synthetic_transactions(60, 1000, 42)
        self.assertTrue(set(FEATURES + [TARGET]).issubset(df.columns))
        self.assertEqual(set(df[TARGET].unique()), {0, 1})
        self.assertLess(df[TARGET].mean(), 0.5)

    def test_validation_imputes_missing_values(self):
        df = make_synthetic_transactions(30, 200, 9)
        df.loc[0, "amount"] = float("nan")
        clean = validate_transactions(df)
        self.assertFalse(clean[FEATURES].isna().any().any())

    def test_missing_column_raises(self):
        df = make_synthetic_transactions(30, 200).drop(columns=["device_id"])
        with self.assertRaises(ValueError):
            validate_transactions(df)

if __name__ == "__main__":
    unittest.main()
