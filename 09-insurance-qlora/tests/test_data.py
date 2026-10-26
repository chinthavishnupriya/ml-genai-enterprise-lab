import json
import tempfile
import unittest
from pathlib import Path

from src.data import build_examples, format_prompt, split_examples, write_splits
from src.schema import validate_triage_output


class SyntheticDataTests(unittest.TestCase):
    def test_builds_64_synthetic_examples(self):
        self.assertEqual(len(build_examples()), 64)

    def test_claim_ids_are_unique_and_synthetic(self):
        examples = build_examples()
        ids = [item["claim_id"] for item in examples]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(value.startswith("SYN-") for value in ids))

    def test_split_sizes_are_reproducible(self):
        splits = split_examples()
        self.assertEqual({key: len(value) for key, value in splits.items()}, {
            "train": 48, "validation": 8, "test": 8
        })
        self.assertEqual(splits, split_examples())

    def test_scenario_groups_do_not_cross_splits(self):
        splits = split_examples()
        groups = {name: {row["scenario_group"] for row in rows} for name, rows in splits.items()}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])

    def test_all_claim_types_appear_in_every_split(self):
        expected = {"auto_collision", "property_damage", "medical_expense", "travel_disruption"}
        for rows in split_examples().values():
            self.assertEqual({row["claim_type"] for row in rows}, expected)

    def test_prompt_marks_claim_as_synthetic_and_requires_human_review(self):
        prompt = format_prompt(build_examples()[0])
        self.assertIn("Synthetic claim record", prompt)
        self.assertIn("Always require human review", prompt)
        self.assertIn("never decide claim approval", prompt.lower())

    def test_generated_targets_follow_schema(self):
        for row in build_examples():
            validate_triage_output(row["target"])

    def test_write_splits_creates_valid_jsonl_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            counts = write_splits(tmp)
            self.assertEqual(counts, {"train": 48, "validation": 8, "test": 8})
            rows = [json.loads(line) for line in (Path(tmp) / "test.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            metadata = json.loads((Path(tmp) / "split_metadata.json").read_text())
            self.assertEqual(metadata["counts"]["train"], 48)


if __name__ == "__main__":
    unittest.main()
