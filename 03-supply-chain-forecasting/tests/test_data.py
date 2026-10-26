import unittest
import pandas as pd
from src.data import make_synthetic_demand, validate_demand_data, split_by_date, FEATURES, TARGET

class DemandDataTests(unittest.TestCase):
    def test_generator_is_deterministic(self):
        a = make_synthetic_demand(8, 40, 11)
        b = make_synthetic_demand(8, 40, 11)
        pd.testing.assert_frame_equal(a, b)

    def test_required_features_and_nonnegative_target(self):
        df = make_synthetic_demand(8, 40, 11)
        self.assertTrue(set(FEATURES + [TARGET]).issubset(df.columns))
        self.assertTrue((df[TARGET] >= 0).all())
        self.assertFalse(df["lag_1_demand"].isna().any())

    def test_lag_feature_uses_previous_observation(self):
        df = make_synthetic_demand(8, 40, 12)
        row = df[df["sku_id"] == "SKU0001"].iloc[2]
        prior = df[(df["sku_id"] == "SKU0001") & (df["date"] < row["date"])].sort_values("date").iloc[-1]
        self.assertAlmostEqual(float(row["lag_1_demand"]), float(prior[TARGET]))

    def test_chronological_split_has_no_date_overlap(self):
        df = make_synthetic_demand(8, 40, 13)
        clean = validate_demand_data(df)
        train, validation, test = split_by_date(clean)
        self.assertLess(train["date"].max(), validation["date"].min())
        self.assertLess(validation["date"].max(), test["date"].min())

    def test_missing_required_column_raises(self):
        df = make_synthetic_demand(8, 40, 14).drop(columns=["sku_id"])
        with self.assertRaises(ValueError):
            validate_demand_data(df)

if __name__ == "__main__":
    unittest.main()
