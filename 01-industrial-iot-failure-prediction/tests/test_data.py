import unittest
import pandas as pd
from src.data import make_synthetic_telemetry, clean_telemetry, FEATURES

class DataTests(unittest.TestCase):
    def test_generator_is_deterministic(self):
        a = make_synthetic_telemetry(4, 30, 7)
        b = make_synthetic_telemetry(4, 30, 7)
        pd.testing.assert_frame_equal(a, b)

    def test_cleaning_preserves_missing_sensor_for_pipeline_imputation(self):
        df = make_synthetic_telemetry(4, 30, 7)
        df.loc[0, "vibration_hz"] = float("nan")
        cleaned = clean_telemetry(df)
        self.assertTrue(cleaned["vibration_hz"].isna().any())
        self.assertGreater(len(cleaned), 0)

    def test_missing_column_raises(self):
        df = make_synthetic_telemetry(4, 30, 7).drop(columns=["machine_id"])
        with self.assertRaises(ValueError):
            clean_telemetry(df)

if __name__ == "__main__":
    unittest.main()
