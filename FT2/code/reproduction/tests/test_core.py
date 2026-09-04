import unittest

import torch

from ft2_repro import (
    Bounds,
    FaultSpec,
    ProtectionMode,
    apply_protection,
    flip_fp16_bit,
)


class CoreTests(unittest.TestCase):
    def test_exact_bit_numbering(self):
        original = torch.tensor([1.0], dtype=torch.float16)
        original_bits = int(original.view(torch.int16).item()) & 0xFFFF
        self.assertEqual(original_bits, 0x3C00)

        for bit_index in (0, 14, 15):
            flipped = flip_fp16_bit(original, 0, bit_index)
            actual = int(flipped.view(torch.int16).item()) & 0xFFFF
            self.assertEqual(
                actual,
                original_bits ^ (1 << bit_index),
            )

        self.assertEqual(float(original.item()), 1.0)
        self.assertTrue(
            torch.isinf(flip_fp16_bit(original, 0, 14)).item()
        )
        self.assertEqual(
            float(flip_fp16_bit(original, 0, 15).item()),
            -1.0,
        )

    def test_noncontiguous_tensor_is_safe(self):
        original = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float16,
        ).t()
        self.assertFalse(original.is_contiguous())

        flipped = flip_fp16_bit(original, 0, 15)

        self.assertEqual(tuple(flipped.shape), tuple(original.shape))
        self.assertEqual(float(flipped[0, 0].item()), -1.0)
        self.assertEqual(float(original[0, 0].item()), 1.0)

    def test_repository_and_paper_semantics(self):
        values = torch.tensor(
            [
                float("-inf"),
                -3.0,
                -2.0,
                float("nan"),
                1.0,
                2.0,
                3.0,
                float("inf"),
            ],
            dtype=torch.float16,
        )
        bounds = Bounds(-2.0, 2.0)

        repository = apply_protection(
            values,
            ProtectionMode.REPOSITORY_ZERO,
            bounds,
        )
        paper = apply_protection(
            values,
            ProtectionMode.PAPER_CLAMP,
            bounds,
        )

        self.assertEqual(
            repository.tolist(),
            [0.0, 0.0, -2.0, 0.0, 1.0, 2.0, 0.0, 0.0],
        )
        self.assertEqual(
            paper.tolist(),
            [-2.0, -2.0, -2.0, 0.0, 1.0, 2.0, 2.0, 2.0],
        )

    def test_fault_hash_is_content_stable(self):
        first = FaultSpec(3, 7, 5, 2, "v_proj", 11, 14)
        same = FaultSpec(3, 7, 5, 2, "v_proj", 11, 14)
        changed = FaultSpec(3, 7, 5, 2, "v_proj", 11, 13)

        self.assertEqual(first.stable_hash, same.stable_hash)
        self.assertNotEqual(first.stable_hash, changed.stable_hash)
        self.assertEqual(len(first.stable_hash), 64)

    def test_step_zero_cannot_be_faulted(self):
        with self.assertRaises(ValueError):
            FaultSpec(0, 0, 0, 0, "v_proj", 0, 14)


if __name__ == "__main__":
    unittest.main()
