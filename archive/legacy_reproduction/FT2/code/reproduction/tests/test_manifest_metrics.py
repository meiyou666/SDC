import unittest
from types import SimpleNamespace

from ft2_repro.manifest import (
    ModelShape,
    build_fault_manifest,
    manifest_sha256,
    validate_manifest,
)
from ft2_repro.metrics import (
    compare_token_ids,
    exact_mcnemar,
    paired_binary_counts,
    squad_reference_recall,
    squad_semantic_score,
    wilson_interval,
)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.shape = ModelShape.from_config(
            SimpleNamespace(
                hidden_size=8,
                num_attention_heads=4,
                num_key_value_heads=2,
                intermediate_size=20,
                num_hidden_layers=3,
            )
        )

    def test_projection_widths_include_grouped_query_attention(self):
        self.assertEqual(self.shape.projection_widths["q_proj"], 8)
        self.assertEqual(self.shape.projection_widths["k_proj"], 4)
        self.assertEqual(self.shape.projection_widths["v_proj"], 4)
        self.assertEqual(self.shape.projection_widths["up_proj"], 20)

    def test_manifest_is_deterministic_unique_and_valid(self):
        first = build_fault_manifest(
            [9605, 8683],
            [5, 1],
            trials_per_sample=3,
            shape=self.shape,
            seed=196,
        )
        second = build_fault_manifest(
            [9605, 8683],
            [5, 1],
            trials_per_sample=3,
            shape=self.shape,
            seed=196,
        )

        self.assertEqual(
            [entry.to_dict() for entry in first],
            [entry.to_dict() for entry in second],
        )
        self.assertEqual(manifest_sha256(first), manifest_sha256(second))
        self.assertEqual(len({entry.spec_id for entry in first}), 6)
        validate_manifest(
            first,
            sample_count=2,
            trials_per_sample=3,
            shape=self.shape,
            num_new_tokens=24,
        )

        for entry in first:
            fault = entry.fault
            width = self.shape.projection_widths[fault.projection]
            self.assertLess(fault.flat_index, width)


class MetricsTests(unittest.TestCase):
    def test_token_comparison_uses_ids(self):
        result = compare_token_ids([1, 2, 3], [1, 9, 3])
        self.assertFalse(result["equal"])
        self.assertEqual(result["first_different_token"], 1)
        self.assertEqual(result["different_token_count"], 1)

    def test_semantic_score_accepts_any_complete_alias(self):
        result = squad_semantic_score(
            "The answer is 5 people.",
            ["five individuals", "5 people"],
        )
        self.assertTrue(result["semantic_correct"])
        self.assertEqual(result["matched_reference_index"], 1)
        self.assertEqual(result["max_reference_recall"], 1.0)

    def test_semantic_score_rejects_partial_reference(self):
        result = squad_semantic_score(
            "Scottish rivers",
            ["gold panned from Scottish rivers"],
        )
        self.assertFalse(result["semantic_correct"])
        self.assertLess(result["max_reference_recall"], 1.0)

    def test_semantically_correct_divergence_is_masked(self):
        golden = "There are 5 people."
        faulted = "The number of people is 5."
        self.assertNotEqual(golden, faulted)
        self.assertTrue(
            squad_semantic_score(faulted, ["5 people"])[
                "semantic_correct"
            ]
        )

    def test_unanswerable_reference_requires_empty_prediction(self):
        self.assertEqual(squad_reference_recall("", ""), 1.0)
        self.assertEqual(squad_reference_recall("an answer", ""), 0.0)

    def test_wilson_reference_values(self):
        low, high = wilson_interval(0, 100)
        self.assertAlmostEqual(low, 0.0, places=6)
        self.assertAlmostEqual(high, 0.036993, places=6)

        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual(low, 0.403832, places=6)
        self.assertAlmostEqual(high, 0.596168, places=6)

    def test_exact_mcnemar_reference_values(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(5, 5), 1.0)
        self.assertEqual(exact_mcnemar(10, 0), 0.001953125)
        self.assertEqual(exact_mcnemar(9, 1), 0.021484375)
        self.assertEqual(exact_mcnemar(3, 7), exact_mcnemar(7, 3))

    def test_paired_binary_table(self):
        records = [
            {"spec_id": "a", "mode": "A", "failed": True},
            {"spec_id": "a", "mode": "B", "failed": False},
            {"spec_id": "b", "mode": "A", "failed": False},
            {"spec_id": "b", "mode": "B", "failed": True},
            {"spec_id": "c", "mode": "A", "failed": True},
            {"spec_id": "c", "mode": "B", "failed": True},
            {"spec_id": "d", "mode": "A", "failed": False},
            {"spec_id": "d", "mode": "B", "failed": False},
        ]
        counts = paired_binary_counts(records, "A", "B", "failed")
        self.assertEqual(
            counts,
            {"n00": 1, "n01": 1, "n10": 1, "n11": 1, "excluded": 0},
        )


if __name__ == "__main__":
    unittest.main()
