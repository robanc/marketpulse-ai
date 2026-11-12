import unittest
import pandas as pd

from src.pipeline.incremental_store import merge_incremental


class TestIncrementalMerge(unittest.TestCase):

    def setUp(self):
        self.existing = pd.DataFrame({
            "date": ["2026-11-10", "2026-11-11"],
            "ticker": ["AAPL", "AAPL"],
            "close": [270.0, 272.0],
        })

    def test_add_new_record(self):
        incoming = pd.DataFrame({
            "date": ["2026-11-12"],
            "ticker": ["AAPL"],
            "close": [275.0],
        })
        result = merge_incremental(
            self.existing, incoming, ["date", "ticker"]
        )
        self.assertEqual(len(result), 3)
        self.assertFalse(result.duplicated(["date", "ticker"]).any())

    def test_incoming_record_takes_precedence(self):
        incoming = pd.DataFrame({
            "date": ["2026-11-11"],
            "ticker": ["AAPL"],
            "close": [280.0],
        })
        result = merge_incremental(
            self.existing, incoming, ["date", "ticker"]
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[-1]["close"], 280.0)

    def test_schema_mismatch_rejected(self):
        incoming = pd.DataFrame({
            "date": ["2026-11-12"],
            "ticker": ["AAPL"],
            "price": [275.0],
        })
        with self.assertRaises(ValueError):
            merge_incremental(
                self.existing, incoming, ["date", "ticker"]
            )

    def test_null_primary_key_rejected(self):
        incoming = pd.DataFrame({
            "date": [None],
            "ticker": ["AAPL"],
            "close": [275.0],
        })
        with self.assertRaises(ValueError):
            merge_incremental(
                self.existing, incoming, ["date", "ticker"]
            )

    def test_duplicate_primary_keys_rejected(self):
        incoming = pd.DataFrame({
            "date": ["2026-11-12", "2026-11-12"],
            "ticker": ["AAPL", "AAPL"],
            "close": [275.0, 276.0],
        })
        with self.assertRaises(ValueError):
            merge_incremental(
                self.existing, incoming, ["date", "ticker"]
            )

    def test_existing_input_unchanged(self):
        original = self.existing.copy(deep=True)
        incoming = pd.DataFrame({
            "date": ["2026-11-12"],
            "ticker": ["AAPL"],
            "close": [275.0],
        })
        merge_incremental(
            self.existing, incoming, ["date", "ticker"]
        )
        pd.testing.assert_frame_equal(self.existing, original)


if __name__ == "__main__":
    unittest.main()
