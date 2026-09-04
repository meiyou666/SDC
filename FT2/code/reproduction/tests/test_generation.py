import unittest
from types import SimpleNamespace

import torch
from torch import nn

from ft2_repro.generation import (
    PreparedPrompt,
    generate_fixed,
)


class FakeTokenizer:
    eos_token_id = 2

    def decode(self, token_ids, skip_special_tokens=True):
        return ",".join(str(token) for token in token_ids)


class FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache,
        return_dict,
    ):
        self.calls.append(
            (int(input_ids.shape[1]), int(attention_mask.shape[1]))
        )
        step = len(self.calls) - 1
        chosen = (2, 3, 4, 1)[step]
        logits = torch.full(
            (1, input_ids.shape[1], 5),
            -100.0,
            dtype=torch.float32,
        )
        logits[:, -1, chosen] = 100.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=(step,),
        )


class GenerationTests(unittest.TestCase):
    def test_fixed_length_continues_after_eos_and_uses_cache_shape(self):
        model = FakeCausalLM().eval()
        prepared = PreparedPrompt(
            prompt="test",
            input_ids=torch.tensor([[0, 0, 5, 6, 7, 8]]),
            attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]),
            unpadded_token_count=4,
        )

        result = generate_fixed(
            model,
            FakeTokenizer(),
            prepared,
            num_new_tokens=4,
        )

        self.assertEqual(result.token_ids, (2, 3, 4, 1))
        self.assertEqual(result.first_eos_step, 0)
        self.assertEqual(result.forward_input_lengths, (6, 1, 1, 1))
        self.assertEqual(
            model.calls,
            [(6, 6), (1, 7), (1, 8), (1, 9)],
        )
        self.assertEqual(result.text, "2,3,4,1")


if __name__ == "__main__":
    unittest.main()
