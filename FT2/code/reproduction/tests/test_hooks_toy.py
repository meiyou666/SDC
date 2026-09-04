import unittest

import torch
from torch import nn

from ft2_repro import (
    CANDIDATE_PROJECTIONS,
    CRITICAL_PROJECTIONS,
    FaultSpec,
    ProtectionMode,
    RunState,
    install_qwen2_hooks,
)


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Identity()
        self.k_proj = nn.Identity()
        self.v_proj = nn.Identity()
        self.o_proj = nn.Identity()


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Identity()
        self.up_proj = nn.Identity()
        self.down_proj = nn.Identity()


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, value):
        self.self_attn.q_proj(value)
        self.self_attn.k_proj(value)
        target = self.self_attn.v_proj(value)
        self.self_attn.o_proj(value)
        self.mlp.gate_proj(value)
        self.mlp.up_proj(value)
        self.mlp.down_proj(value)
        return target


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer()])

    def forward(self, value):
        return self.layers[0](value)


class ToyQwen2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()

    def forward(self, value):
        return self.model(value)


class HookTests(unittest.TestCase):
    def setUp(self):
        self.fault = FaultSpec(
            sample_index=0,
            trial_index=0,
            token_step=1,
            layer_index=0,
            projection="v_proj",
            flat_index=0,
            bit_index=14,
        )

    def run_mode(self, mode):
        model = ToyQwen2().eval()
        state = RunState(mode=mode, faults=(self.fault,))
        value = torch.tensor([[[1.0]]], dtype=torch.float16)

        with install_qwen2_hooks(model, state):
            with torch.inference_mode():
                calibration_output = model(value)
                faulted_output = model(value)

            self.assertEqual(float(calibration_output.item()), 1.0)
            state.assert_complete()

        with torch.inference_mode():
            clean_after_removal = model(value)
        self.assertEqual(float(clean_after_removal.item()), 1.0)

        self.assertTrue(state.calibration_frozen)
        self.assertEqual(
            state.injection_counts[self.fault.stable_hash],
            1,
        )
        return faulted_output, state

    def test_required_three_mode_result(self):
        unprotected, state_a = self.run_mode(
            ProtectionMode.UNPROTECTED
        )
        repository, state_b = self.run_mode(
            ProtectionMode.REPOSITORY_ZERO
        )
        paper, state_c = self.run_mode(
            ProtectionMode.PAPER_CLAMP
        )

        self.assertTrue(torch.isposinf(unprotected).item())
        self.assertEqual(float(repository.item()), 0.0)
        self.assertEqual(float(paper.item()), 2.0)

        records = [
            state.injection_records[self.fault.stable_hash]
            for state in (state_a, state_b, state_c)
        ]
        for record in records:
            self.assertTrue(record.bit_flip_verified)
            self.assertEqual(record.before_bits, 0x3C00)
            self.assertEqual(record.after_bits, 0x7C00)
            self.assertEqual(record.output_shape, (1, 1, 1))
        self.assertEqual(records[0].protected_bits, 0x7C00)
        self.assertEqual(records[1].protected_bits, 0x0000)
        self.assertEqual(records[2].protected_bits, 0x4000)

        hashes = {
            state.faults[0].stable_hash
            for state in (state_a, state_b, state_c)
        }
        self.assertEqual(hashes, {self.fault.stable_hash})

        self.assertIsNot(
            state_a._injection_counts,
            state_b._injection_counts,
        )
        self.assertIsNot(state_a._bounds, state_b._bounds)

        telemetry = records[2].to_dict()
        self.assertEqual(telemetry["before_bits_hex"], "0x3c00")
        self.assertEqual(telemetry["after_class"], "positive_infinity")
        self.assertTrue(telemetry["protection_changed_value"])

    def test_paper_mode_corrects_first_token_nan_without_bounds(self):
        model = ToyQwen2().eval()
        state = RunState(mode=ProtectionMode.PAPER_CLAMP, faults=())
        value = torch.tensor([[[float("nan")]]], dtype=torch.float16)
        with install_qwen2_hooks(model, state):
            with torch.inference_mode():
                output = model(value)
            state.assert_complete()
        self.assertEqual(float(output.item()), 0.0)

    def test_unreached_fault_is_detected(self):
        future_fault = FaultSpec(
            sample_index=0,
            trial_index=1,
            token_step=2,
            layer_index=0,
            projection="v_proj",
            flat_index=0,
            bit_index=14,
        )
        model = ToyQwen2().eval()
        state = RunState(
            mode=ProtectionMode.UNPROTECTED,
            faults=(future_fault,),
        )
        value = torch.tensor([[[1.0]]], dtype=torch.float16)

        with install_qwen2_hooks(model, state):
            with torch.inference_mode():
                model(value)
                model(value)

        with self.assertRaisesRegex(
            AssertionError,
            "exactly once",
        ):
            state.assert_complete()

    def test_projection_sets_are_explicit(self):
        self.assertEqual(len(CANDIDATE_PROJECTIONS), 7)
        self.assertEqual(
            CRITICAL_PROJECTIONS,
            {"v_proj", "o_proj", "up_proj", "down_proj"},
        )


if __name__ == "__main__":
    unittest.main()
