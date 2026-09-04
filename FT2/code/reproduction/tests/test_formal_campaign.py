import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from ft2_formal.artifacts import (
    atomic_write_artifact,
    load_artifact,
    stable_run_id,
)
from ft2_formal.audit import _trace_invariants, _wilson
from ft2_formal.decoding import PreparedPrompt, fixed_greedy_generate
from ft2_formal.schema import FaultSpec, FaultType


class ArtifactTests(unittest.TestCase):
    def test_atomic_artifact_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            saved = atomic_write_artifact(path, {"value": 7})
            self.assertEqual(load_artifact(path), saved)
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["value"] = 8
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_artifact(path)

    def test_run_id_is_stable_and_mode_specific(self):
        arguments = {
            "campaign_fingerprint": "a" * 64,
            "spec_id": "b" * 64,
        }
        first = stable_run_id(**arguments, mode_id="no_protection")
        second = stable_run_id(**arguments, mode_id="no_protection")
        protected = stable_run_id(
            **arguments, mode_id="paper_clamp_first_token_bounds"
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, protected)

    def test_wilson_reference(self):
        low, high = _wilson(1, 4)
        self.assertAlmostEqual(low, 0.04558726080970055)
        self.assertAlmostEqual(high, 0.6993581574175981)


class TraceAuditTests(unittest.TestCase):
    def _spec(self):
        return FaultSpec(
            model_key="qwen2_math_7b",
            dataset_key="squad_v2",
            dataset_index=4,
            sample_position=0,
            trial_index=0,
            fault_type=FaultType.FP16_2BIT,
            target_step=0,
            layer_index=1,
            projection="q_proj",
            site_key="layer.1.q_proj",
            sequence_index=2,
            feature_index=3,
            expected_sequence_length=4,
            expected_out_features=8,
            bit_positions=(1, 12),
            campaign_seed=196,
        )

    def test_trace_recomputes_xor_and_hamming(self):
        spec = self._spec()
        before = 0x1234
        mask = (1 << 1) | (1 << 12)
        record = {
            "engine": {
                "injection_count": 1,
                "injection_trace": {
                    "before_bits_hex": f"0x{before:04x}",
                    "after_bits_hex": f"0x{before ^ mask:04x}",
                    "site_key": spec.site_key,
                    "observed_step": spec.target_step,
                    "flat_index": spec.flat_index,
                    "bit_positions": list(spec.bit_positions),
                },
            }
        }
        _trace_invariants(record, spec)
        record["engine"]["injection_trace"]["after_bits_hex"] = (
            f"0x{before ^ mask ^ 1:04x}"
        )
        with self.assertRaises(AssertionError):
            _trace_invariants(record, spec)


class FakeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, skip_special_tokens=True):
        return ",".join(str(value) for value in token_ids)


class FakeEngine:
    def __init__(self):
        self.steps = []

    def begin_step(self, step):
        self.steps.append(step)

    def finish_inference(self):
        result = SimpleNamespace(steps=tuple(self.steps))
        self.steps = []
        return result


class FakeDynamicCacheModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.forward_lengths = []
        self.cache_ids = []

    def _supports_default_dynamic_cache(self):
        return True

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        model_kwargs["cache_position"] = torch.arange(input_ids.shape[1])
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        *,
        past_key_values,
        attention_mask,
        cache_position,
        use_cache,
    ):
        self.assertIsDynamic(past_key_values)
        if past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            cache_position = cache_position[-1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "use_cache": use_cache,
        }

    @staticmethod
    def assertIsDynamic(value):
        if not isinstance(value, DynamicCache):
            raise AssertionError("Expected DynamicCache")

    def forward(
        self,
        input_ids,
        past_key_values,
        attention_mask,
        cache_position,
        use_cache,
        return_dict,
    ):
        previous = past_key_values.get_seq_length()
        self.forward_lengths.append(int(input_ids.shape[1]))
        self.cache_ids.append(id(past_key_values))
        key = torch.zeros((1, 1, input_ids.shape[1], 1))
        past_key_values.update(key, key.clone(), 0)
        logits = torch.full((1, input_ids.shape[1], 8), -10.0)
        logits[:, -1, 3 if previous == 0 else 4] = 10.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=past_key_values,
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        *,
        is_encoder_decoder,
        standardize_cache_format,
        num_new_tokens,
    ):
        model_kwargs["past_key_values"] = outputs.past_key_values
        mask = model_kwargs["attention_mask"]
        model_kwargs["attention_mask"] = torch.cat(
            (mask, mask.new_ones((1, 1))), dim=-1
        )
        model_kwargs["cache_position"] = (
            model_kwargs["cache_position"][-1:] + num_new_tokens
        )
        return model_kwargs


class DynamicCacheGenerationTests(unittest.TestCase):
    def test_dynamic_cache_is_initialized_and_isolated_per_inference(self):
        model = FakeDynamicCacheModel().eval()
        prepared = PreparedPrompt(
            prompt="x",
            prompt_sha256="0" * 64,
            input_ids=torch.tensor([[1, 2, 5, 6]]),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
            unpadded_token_count=4,
        )
        first = fixed_greedy_generate(
            model,
            FakeTokenizer(),
            prepared,
            FakeEngine(),
            num_new_tokens=2,
        )
        second = fixed_greedy_generate(
            model,
            FakeTokenizer(),
            prepared,
            FakeEngine(),
            num_new_tokens=2,
        )
        self.assertEqual(first.token_ids, (3, 4))
        self.assertEqual(second.token_ids, (3, 4))
        self.assertEqual(first.forward_input_lengths, (4, 1))
        self.assertEqual(second.forward_input_lengths, (4, 1))
        self.assertEqual(model.forward_lengths, [4, 1, 4, 1])
        self.assertEqual(model.cache_ids[0], model.cache_ids[1])
        self.assertEqual(model.cache_ids[2], model.cache_ids[3])
        self.assertNotEqual(model.cache_ids[0], model.cache_ids[2])


if __name__ == "__main__":
    unittest.main()
