import math
import unittest

import torch
from torch import nn

from ft2_formal.adapters import ModelAdapter, ProjectionSite
from ft2_formal.engine import FT2HookEngine, OfflineBoundsProfiler
from ft2_formal.manifest import build_pair_manifest, manifest_sha256
from ft2_formal.schema import (
    Bounds,
    BoundsSource,
    Correction,
    FaultSpec,
    FaultType,
    ProtectionSpec,
    fp16_bits_at,
    flip_fp16_bits,
)
from ft2_formal.tasks import extract_last_number, score_output


def tensor_from_bits(bits):
    signed = bits if bits < 0x8000 else bits - 0x10000
    return torch.tensor([signed], dtype=torch.int16).view(torch.float16)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.noncritical = nn.Identity()
        self.critical = nn.Identity()

    def forward(self, value):
        return self.critical(self.noncritical(value))


def toy_adapter():
    model = ToyModel()
    sites = (
        ProjectionSite(
            key="layer.0.q_proj",
            layer_index=0,
            projection="q_proj",
            module_path="noncritical",
            module=model.noncritical,
            sampling_weight=1,
            critical=False,
            out_features=2,
        ),
        ProjectionSite(
            key="layer.0.v_proj",
            layer_index=0,
            projection="v_proj",
            module_path="critical",
            module=model.critical,
            sampling_weight=1,
            critical=True,
            out_features=2,
        ),
    )
    return model, ModelAdapter("toy", sites, 1, "toy")


def fault(
    *,
    step=0,
    site_key="layer.0.v_proj",
    projection="v_proj",
    seq=0,
    bits=(0,),
    fault_type=FaultType.FP16_1BIT,
    expected_sequence_length=1,
):
    return FaultSpec(
        model_key="toy",
        dataset_key="toy_data",
        dataset_index=0,
        sample_position=0,
        trial_index=0,
        fault_type=fault_type,
        target_step=step,
        layer_index=0,
        projection=projection,
        site_key=site_key,
        sequence_index=seq,
        feature_index=0,
        expected_sequence_length=expected_sequence_length,
        expected_out_features=2,
        bit_positions=bits,
        campaign_seed=196,
    )


class BitFaultTests(unittest.TestCase):
    def test_raw_bit_xor_across_special_patterns(self):
        patterns = (0x0000, 0x8000, 0x0001, 0x3C00, 0x7BFF, 0x7C00, 0xFC00, 0x7E01)
        for pattern in patterns:
            for bit in (0, 9, 10, 14, 15):
                with self.subTest(pattern=hex(pattern), bit=bit):
                    original = tensor_from_bits(pattern)
                    flipped = flip_fp16_bits(original, 0, (bit,))
                    self.assertEqual(fp16_bits_at(flipped, 0), pattern ^ (1 << bit))

    def test_two_bit_hamming_distance(self):
        original = tensor_from_bits(0x3C00)
        flipped = flip_fp16_bits(original, 0, (3, 15))
        changed = fp16_bits_at(original, 0) ^ fp16_bits_at(flipped, 0)
        self.assertEqual(changed.bit_count(), 2)

    def test_invalid_duplicate_and_exponent_bits(self):
        value = torch.ones(1, dtype=torch.float16)
        with self.assertRaises(ValueError):
            flip_fp16_bits(value, 0, (4, 4))
        with self.assertRaises(ValueError):
            fault(
                bits=(9,),
                fault_type=FaultType.FP16_EXPONENT_BIT,
            )


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.model, adapter = toy_adapter()
        self.engine = FT2HookEngine(adapter)
    def test_explicit_protected_subset(self):
        model, adapter = toy_adapter()
        default_engine = FT2HookEngine(adapter)
        self.assertEqual(
            default_engine._protected_keys,
            frozenset(("layer.0.v_proj",)),
        )
        with self.assertRaises(ValueError):
            FT2HookEngine(adapter, ("layer.0.unknown",))
        with self.assertRaises(ValueError):
            FT2HookEngine(
                adapter,
                ("layer.0.q_proj", "layer.0.q_proj"),
            )

        engine = FT2HookEngine(adapter, ("layer.0.q_proj",))
        engine.install()
        engine.start_inference(
            fault=None,
            protection=ProtectionSpec(
                BoundsSource.FIRST_TOKEN,
                Correction.PAPER_CLAMP,
                1.0,
            ),
        )
        engine.begin_step(0)
        model(torch.tensor([[[-1.0, 1.0]]], dtype=torch.float16))
        engine.begin_step(1)
        output = model(
            torch.tensor([[[-3.0, 3.0]]], dtype=torch.float16)
        )
        record = engine.finish_inference()
        engine.remove()
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[[-1.0, 1.0]]], dtype=torch.float16),
            )
        )
        self.assertEqual(
            set(record.online_bounds),
            {"layer.0.q_proj"},
        )
        self.assertEqual(record.correction_elements, 2)
    def test_empty_protected_set_is_noop(self):
        model, adapter = toy_adapter()
        engine = FT2HookEngine(adapter, ())
        engine.install()
        protection = ProtectionSpec(
            BoundsSource.FIRST_TOKEN,
            Correction.PAPER_CLAMP,
            1.0,
        )
        engine.start_inference(fault=None, protection=protection)
        engine.begin_step(0)
        model(torch.tensor([[[-1.0, 1.0]]], dtype=torch.float16))
        engine.begin_step(1)
        output = model(torch.tensor([[[-3.0, 3.0]]], dtype=torch.float16))
        record = engine.finish_inference()
        engine.remove()
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor(
                    [[[-3.0, 3.0]]],
                    dtype=torch.float16,
                ),
            )
        )
        self.assertEqual(record.online_bounds, {})
        self.assertEqual(record.correction_elements, 0)

    def test_unprotected_fault_is_still_injected(self):
        model, adapter = toy_adapter()
        spec = fault(
            step=1,
            site_key="layer.0.q_proj",
            projection="q_proj",
            bits=(0,),
            expected_sequence_length=1,
        )
        engine = FT2HookEngine(adapter)
        engine.install()
        protection = ProtectionSpec(
            BoundsSource.FIRST_TOKEN,
            Correction.PAPER_CLAMP,
            1.0,
        )
        engine.start_inference(fault=spec, protection=protection)
        engine.begin_step(0)
        model(torch.tensor([[[-1.0, 1.0]]], dtype=torch.float16))
        engine.begin_step(1)
        model(torch.tensor([[[-0.5, 0.5]]], dtype=torch.float16))
        record = engine.finish_inference()
        engine.remove()
        trace = record.injection_trace
        self.assertIsNotNone(trace)
        self.assertEqual(record.injection_count, 1)
        self.assertEqual(trace.site_key, "layer.0.q_proj")
        self.assertEqual(trace.hamming_distance, 1)
        self.assertEqual(trace.correction_action, "none")
        self.assertIsNone(trace.bounds)

    def test_truth_table(self):
        values = torch.tensor(
            [-2.0, -1.0, 0.5, 2.0, 3.0, float("nan"), float("inf"), -float("inf")],
            dtype=torch.float16,
        )
        bounds = Bounds(-1.0, 2.0, 1.0)
        paper, invalid, nan, low, high = self.engine._apply_correction(
            values, bounds, Correction.PAPER_CLAMP
        )
        expected_paper = torch.tensor(
            [-1.0, -1.0, 0.5, 2.0, 2.0, 0.0, 2.0, -1.0],
            dtype=torch.float16,
        )
        self.assertTrue(torch.equal(paper, expected_paper))
        repository, *_ = self.engine._apply_correction(
            values, bounds, Correction.REPOSITORY_ZERO
        )
        expected_repository = torch.tensor(
            [0.0, -1.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float16,
        )
        self.assertTrue(torch.equal(repository, expected_repository))
        self.assertEqual(int(invalid.sum()), 5)
        self.assertEqual(int(nan.sum()), 1)
        self.assertEqual(int(low.sum()), 2)
        self.assertEqual(int(high.sum()), 2)

    def test_step_zero_inject_nan_correct_then_calibrate(self):
        spec = fault(
            bits=(9, 14),
            fault_type=FaultType.FP16_2BIT,
            expected_sequence_length=3,
        )
        self.engine.install()
        self.engine.start_inference(
            fault=spec,
            protection=ProtectionSpec(
                BoundsSource.FIRST_TOKEN,
                Correction.PAPER_CLAMP,
                2.0,
            ),
        )
        self.engine.begin_step(0)
        value = torch.ones((1, 3, 2), dtype=torch.float16)
        output = self.model(value)
        record = self.engine.finish_inference()
        self.engine.remove()
        self.assertEqual(float(output[0, 0, 0]), 0.0)
        self.assertEqual(record.injection_count, 1)
        self.assertEqual(record.injection_trace.hamming_distance, 2)
        self.assertEqual(record.injection_trace.correction_action, "nan_to_zero")
        self.assertEqual(record.online_bounds["layer.0.v_proj"].raw_min, 0.0)
        self.assertEqual(record.online_bounds["layer.0.v_proj"].raw_max, 1.0)

    def test_first_token_bounds_reused_on_step_one(self):
        self.engine.install()
        self.engine.start_inference(
            fault=None,
            protection=ProtectionSpec(
                BoundsSource.FIRST_TOKEN,
                Correction.PAPER_CLAMP,
                2.0,
            ),
        )
        self.engine.begin_step(0)
        first = self.model(
            torch.tensor([[[-1.0, 1.0]]], dtype=torch.float16)
        )
        self.engine.begin_step(1)
        second = self.model(
            torch.tensor([[[-10.0, 10.0]]], dtype=torch.float16)
        )
        record = self.engine.finish_inference()
        self.engine.remove()
        self.assertTrue(torch.equal(first, torch.tensor([[[-1.0, 1.0]]], dtype=torch.float16)))
        self.assertTrue(torch.equal(second, torch.tensor([[[-2.0, 2.0]]], dtype=torch.float16)))
        self.assertEqual(record.correction_elements, 2)

    def test_offline_protects_step_zero(self):
        self.engine.install()
        self.engine.start_inference(
            fault=None,
            protection=ProtectionSpec(
                BoundsSource.OFFLINE,
                Correction.PAPER_CLAMP,
                1.0,
            ),
            offline_bounds={"layer.0.v_proj": Bounds(-1.0, 1.0, 2.0)},
        )
        self.engine.begin_step(0)
        output = self.model(
            torch.tensor([[[-2.0, 2.0]]], dtype=torch.float16)
        )
        record = self.engine.finish_inference()
        self.engine.remove()
        self.assertTrue(torch.equal(output, torch.tensor([[[-1.0, 1.0]]], dtype=torch.float16)))
        self.assertEqual(record.correction_elements, 2)

    def test_state_isolation_a_b_a(self):
        self.engine.install()

        def run(value):
            self.engine.start_inference(
                fault=None,
                protection=ProtectionSpec(
                    BoundsSource.FIRST_TOKEN,
                    Correction.PAPER_CLAMP,
                    2.0,
                ),
            )
            self.engine.begin_step(0)
            output = self.model(value)
            return output.clone(), self.engine.finish_inference()

        a1, record_a1 = run(torch.tensor([[[-1.0, 2.0]]], dtype=torch.float16))
        _, record_b = run(torch.tensor([[[-4.0, 3.0]]], dtype=torch.float16))
        a2, record_a2 = run(torch.tensor([[[-1.0, 2.0]]], dtype=torch.float16))
        self.engine.remove()
        self.assertTrue(torch.equal(a1, a2))
        self.assertEqual(record_a1.online_bounds, record_a2.online_bounds)
        self.assertNotEqual(record_a1.online_bounds, record_b.online_bounds)

    def test_unreached_fault_is_rejected(self):
        self.engine.install()
        self.engine.start_inference(
            fault=fault(step=1),
            protection=ProtectionSpec(BoundsSource.NONE, Correction.NONE),
        )
        self.engine.begin_step(0)
        self.model(torch.ones((1, 1, 2), dtype=torch.float16))
        with self.assertRaises(AssertionError):
            self.engine.finish_inference()
        self.engine.abort()
        self.engine.remove()


class ProfilerAndManifestTests(unittest.TestCase):
    def test_offline_global_extrema(self):
        model, adapter = toy_adapter()
        profiler = OfflineBoundsProfiler(adapter, scaling_factor=2.0)
        profiler.install()
        for value in (
            torch.tensor([[[-1.0, 3.0]]], dtype=torch.float16),
            torch.tensor([[[-4.0, 2.0]]], dtype=torch.float16),
        ):
            profiler.start_example()
            model(value)
            profiler.finish_example()
        result = profiler.finalize(expected_examples=2)
        profiler.remove()
        self.assertEqual(result["layer.0.v_proj"], Bounds(-4.0, 3.0, 2.0))

    def test_manifest_is_stable_and_has_all_fault_types(self):
        _, adapter = toy_adapter()
        first = build_pair_manifest(
            adapter=adapter,
            dataset_key="toy_data",
            dataset_indices=(3, 7),
            target_steps=(0, 5),
            trials_per_fault_type=2,
            generation_steps=10,
            campaign_seed=196,
            max_input_tokens=8,
        )
        second = build_pair_manifest(
            adapter=adapter,
            dataset_key="toy_data",
            dataset_indices=(3, 7),
            target_steps=(0, 5),
            trials_per_fault_type=2,
            generation_steps=10,
            campaign_seed=196,
            max_input_tokens=8,
        )
        self.assertEqual(first, second)
        self.assertEqual(manifest_sha256(first), manifest_sha256(second))
        self.assertEqual(len(first), 12)
        self.assertEqual({item.fault_type for item in first}, set(FaultType))
        for item in first:
            expected = 2 if item.fault_type is FaultType.FP16_2BIT else 1
            self.assertEqual(len(item.bit_positions), expected)


class TaskMetricTests(unittest.TestCase):
    def test_gsm_last_number(self):
        self.assertEqual(extract_last_number("work 12 then answer 1,234.5"), "2469/2")
        score = score_output("gsm8k", "reasoning... final 72", ("72",))
        self.assertTrue(score["task_correct"])

    def test_qa_reference_proxy(self):
        score = score_output(
            "squad_v2",
            "The answer is New York City.",
            ("New York City",),
        )
        self.assertTrue(score["task_correct"])


if __name__ == "__main__":
    unittest.main()
