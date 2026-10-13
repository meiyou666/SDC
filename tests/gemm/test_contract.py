import copy
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config, Qwen3ForCausalLM

from algorithms import Algorithm
from algorithms.classic.reference import checks, locate_correct, reference
from backends.reference import flip_raw
from gemm import ExecutionContext, Fault, gemm
from gemm.api import RecoveryError
from gemm.evaluate import bit_events, evaluate_case, rate
from gemm.linear import ProtectedLinear, install
from gemm.quantization import decode_reference, hadamard_pair, pack_reference, scale_offsets


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_dual_checks_and_exact_single_error():
    a = torch.arange(1., 25.).reshape(6, 4)
    b = torch.arange(1., 21.).reshape(4, 5)
    expected = reference(a, b)
    assert torch.equal(expected, checks(a @ b))
    for i in range(6):
        for j in range(5):
            bad = a @ b
            bad[i, j] += 256
            out, status, _, _ = locate_correct(bad, expected, Algorithm(atol=0, rtol=0))
            assert status.tolist() == [1, 1, i, j, 1, 0]
            assert torch.equal(out, a @ b)


def test_multiple_elements_are_not_single_error_repair():
    a, b = torch.eye(4), torch.ones(4, 4)
    bad = a @ b
    bad[0, 1] += 1
    bad[2, 3] += 1
    _, status, _, _ = locate_correct(bad, reference(a, b), Algorithm(atol=0, rtol=0))
    assert status.tolist() == [1, 0, -1, -1, 0, 1]


@pytest.mark.parametrize("width", [4, 8, 16, 32])
def test_every_bit_changes_only_its_target(width):
    original = torch.tensor([0xA5] * 16, dtype=torch.uint8)
    for bit in range(width):
        data = original.clone()
        metadata = flip_raw(data, 1, bit, width)
        assert int(metadata[0] ^ metadata[1]) == 1 << bit
        different = data ^ original
        assert sum(int(v).bit_count() for v in different) == 1
        flip_raw(data, 1, bit, width)
        assert torch.equal(data, original)
        flip_raw(data, 1, bit, width, False)
        assert torch.equal(data, original)


@pytest.mark.parametrize("group", [16, 32])
def test_scale_permutation(group):
    offsets = scale_offsets(256, 256, group)
    assert offsets.shape == (256, 256 // group)
    assert torch.equal(offsets.flatten().sort().values, torch.arange(offsets.numel()))
    assert offsets[0, 0] == 0 and offsets[1, 0] == 16 and offsets[32, 0] == 4


@pytest.mark.parametrize("profile", ["fp32", "bf16", "mxfp8", "nvfp4"])
def test_pack_decode_scale_and_padding(profile):
    x = torch.ones(17, 35) * 2
    p = pack_reference(x, profile)
    decoded = decode_reference(p)
    torch.testing.assert_close(decoded, x, rtol=.02, atol=0)
    assert p.padded_rows == 128 and p.padded_columns == 128
    zero = decode_reference(pack_reference(torch.zeros_like(x), profile))
    assert torch.isfinite(zero).all() and (zero == 0).all()


def test_nvfp4_weight_tile_and_stochastic_replay():
    torch.manual_seed(5)
    x = torch.randn(17, 23)
    p = pack_reference(x, "nvfp4", tile_scale=True, stochastic=True, seed=42)
    q = pack_reference(x, "nvfp4", tile_scale=True, stochastic=True, seed=42)
    changed = pack_reference(x, "nvfp4", tile_scale=True, stochastic=True, seed=43)
    assert torch.equal(p.codes, q.codes)
    assert not torch.equal(p.codes, changed.codes)
    sf = p.scales[scale_offsets(128, 128, 16)]
    assert torch.equal(sf[0], sf[15])


def test_hadamard_preserves_matmul_with_tail_and_does_not_touch_rng():
    torch.manual_seed(4)
    a, b = torch.randn(7, 19), torch.randn(19, 9)
    state = torch.get_rng_state()
    transformed_a, transformed_b = hadamard_pair(a, b, 3)
    torch.testing.assert_close(transformed_a @ transformed_b, a @ b, atol=2e-6, rtol=2e-5)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("profile", ["fp32", "bf16", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("site", ["a", "b", "accumulator"])
def test_injection_recovery_and_zero_mask(profile, site):
    a, b = torch.ones(4, 16), torch.ones(16, 5)
    options = dict(profile=profile, backend="reference", algorithm=Algorithm(atol=.01, rtol=.001))
    clean = gemm(a, b, **options)
    bit = {"fp32": 31, "bf16": 15, "mxfp8": 7, "nvfp4": 3}[profile] if site != "accumulator" else 31
    event = Fault(site, 1, 1, bit)
    control = gemm(a, b, fault=replace(event, enabled=False), **options)
    assert torch.equal(clean.output, control.output)
    bad = gemm(a, b, fault=event, **options)
    assert bad.detected
    assert bad.fault_bits[0] ^ bad.fault_bits[1] == 1 << bit
    recovered = gemm(a, b, fault=event, recovery="recompute", **options)
    assert recovered.recomputed and not recovered.unrecovered
    assert torch.equal(recovered.output, clean.output)
    if site == "accumulator":
        native = gemm(a, b, fault=event, recovery="native", **options)
        assert native.status[1] and not native.unrecovered
        torch.testing.assert_close(native.output, clean.output, atol=1e-5, rtol=1e-6)


def test_backend_and_shape_errors_are_explicit():
    a = torch.ones(2, 2)
    with pytest.raises(NotImplementedError, match="昇腾"):
        gemm(a, a, backend="ascend")
    with pytest.raises(RuntimeError, match="NVIDIA"):
        gemm(a, a, backend="nvidia")
    with pytest.raises(ValueError, match="bit"):
        gemm(a, a, backend="reference", fault=Fault("a", 0, 0, 32))
    with pytest.raises(ValueError, match="恢复"):
        gemm(a, a, backend="reference", algorithm="none", recovery="recompute")


def test_fault_value_decodes_scaled_operand():
    for profile, sign in (("fp32", 31), ("bf16", 15), ("mxfp8", 7), ("nvfp4", 3)):
        a = torch.full((4, 16), 2.)
        b = torch.full((16, 5), 3.)
        result = gemm(a, b, profile=profile, backend="reference", fault=Fault("a", 1, 2, sign))
        assert result.fault_values[0] > 0
        assert result.fault_values[0] == -result.fault_values[1]


def test_recompute_rejects_persistent_verification_failure():
    a = torch.full((2, 2), .1)
    with pytest.raises(RecoveryError, match="重算"):
        gemm(a, a, backend="reference", profile="fp32", algorithm=Algorithm(atol=0, rtol=0), recovery="recompute")


def test_nonfinite_fault_is_preserved_in_metrics():
    records = []
    summary = evaluate_case(torch.ones(1, 1), torch.ones(1, 1), profile="fp32", backend="reference",
                            algorithm="classic", events=[Fault("accumulator", 0, 0, 30)], writer=records.append)
    assert summary["nonfinite_events"] == 1
    assert summary["detection"]["value"] == 1
    assert summary["recomputation"]["value"] == 1
    assert records[0]["fault_values"] == [1., "inf"]


def test_quantization_roles_in_backward(monkeypatch):
    import backends.reference as backend

    calls = []
    original = backend.pack_reference

    def pack(x, profile, **kwargs):
        calls.append((kwargs["tile_scale"], kwargs["stochastic"]))
        return original(x, profile, **kwargs)

    monkeypatch.setattr(backend, "pack_reference", pack)
    for phase, expected in (("forward", [(False, False), (True, False)]),
                            ("dgrad", [(False, True), (True, False)]),
                            ("wgrad", [(False, True), (False, False)])):
        calls.clear()
        gemm(torch.ones(3, 16), torch.ones(16, 4), backend="reference", profile="nvfp4", phase=phase)
        assert calls == expected


def test_custom_detector_registration(monkeypatch):
    import algorithms
    from algorithms.classic.reference import verify
    from algorithms.template import PaperDetector

    class Detector(PaperDetector):
        def __init__(self, settings):
            super().__init__(settings)
            self.register_buffer("calls", torch.zeros((), dtype=torch.int64))

        def prepare(self, a, b, **kwargs):
            return reference(decode_reference(a), decode_reference(b).T)

        def verify(self, output, prepared, *, correct):
            self.calls += 1
            residual, threshold, bad = verify(output, prepared, self.settings)
            status = torch.tensor([int(bad.any()), 0, -1, -1, 0, int(bad.any())])
            return status, residual, threshold

    monkeypatch.setattr(algorithms, "_FACTORIES", {})
    algorithms.register("test_method", "reference", Detector)
    method = algorithms.Algorithm("test_method")
    instance = Detector(method)
    a = torch.ones(3, 4)
    b = torch.ones(4, 3)
    output = gemm(a, b, profile="fp32", algorithm=method, backend="reference", detector=instance,
                  fault=Fault("accumulator", 0, 0, 31), recovery="recompute")
    assert output.recomputed and instance.calls == 1
    torch.testing.assert_close(output.output, a @ b)


def test_call_denominators_and_seed_streams():
    a, b = torch.ones(3, 4), torch.ones(4, 3)
    records = []
    events = [Fault("accumulator", 0, 0, 31), Fault("accumulator", 0, 0, 0, False)]
    summary = evaluate_case(a, b, profile="fp32", algorithm="classic", backend="reference", events=events, writer=records.append)
    assert summary["fpr_call"]["denominator"] == 1
    assert summary["detection"]["denominator"] == 1
    assert summary["detection"]["value"] == 1
    assert summary["native_recovery"]["value"] == 1
    assert summary["misrepair"]["value"] == 0
    assert rate(0, 0)["value"] is None
    state = torch.get_rng_state()
    assert list(bit_events(3, 3, 4, "nvfp4")) == list(bit_events(3, 3, 4, "nvfp4"))
    assert torch.equal(state, torch.get_rng_state())
    assert ExecutionContext().quant_seed("forward") != ExecutionContext().quant_seed("wgrad")


@pytest.mark.parametrize("profile", ["fp32", "bf16", "mxfp8", "nvfp4"])
def test_autograd_all_phases_same_profile(profile, monkeypatch):
    import gemm.linear as linear_module

    called = []
    original_gemm = linear_module.gemm

    def tracked(a, b, **kwargs):
        called.append((kwargs["phase"], kwargs["profile"]))
        return original_gemm(a, b, **kwargs)

    monkeypatch.setattr(linear_module, "gemm", tracked)
    dense = nn.Linear(19, 17)
    protected = ProtectedLinear(dense, name="test", backend="reference", profile=profile)
    assert protected.weight is dense.weight and protected.bias is dense.bias
    x = torch.randn(2, 3, 19, requires_grad=True)
    protected(x).square().mean().backward()
    assert called == [("forward", profile), ("dgrad", profile), ("wgrad", profile)]
    assert torch.isfinite(x.grad).all() and torch.isfinite(dense.weight.grad).all()


def test_fp32_gradients_match_linear():
    torch.manual_seed(42)
    dense = nn.Linear(5, 7)
    protected = ProtectedLinear(copy.deepcopy(dense), name="test", backend="reference", profile="fp32")
    x = torch.randn(3, 5, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    dense(x).sum().backward()
    protected(y).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    torch.testing.assert_close(dense.weight.grad, protected.weight.grad)


def test_checkpoint_replay_nvfp4_and_state_restore():
    torch.manual_seed(8)
    first = ProtectedLinear(nn.Linear(17, 19), name="test", backend="reference", profile="nvfp4")
    second = copy.deepcopy(first)
    x = torch.randn(2, 17, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    rng = torch.get_rng_state()
    first(x).square().sum().backward()
    torch.set_rng_state(rng)
    checkpoint(second, y, use_reentrant=False).square().sum().backward()
    assert torch.equal(x.grad, y.grad)
    assert torch.equal(first.weight.grad, second.weight.grad)
    restored = copy.deepcopy(first)
    restored.load_state_dict(first.state_dict())
    rng = torch.get_rng_state()
    expected = first(x)
    torch.set_rng_state(rng)
    assert torch.equal(expected, restored(x))


def test_tiny_qwen_update_uses_protected_linears():
    torch.manual_seed(2)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                        head_dim=8, attention_dropout=0., pad_token_id=0))
    original = dict(model.named_parameters())
    install(model, backend="reference", profile="mxfp8")
    assert all(value is dict(model.named_parameters())[key] for key, value in original.items())
    assert sum(isinstance(module, ProtectedLinear) for module in model.modules()) == 3
    weight = model.model.layers[0].mlp.gate_proj.weight
    before = weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(1, 64, (2, 8))
    model(input_ids=tokens, labels=tokens).loss.backward()
    optimizer.step()
    assert not torch.equal(weight, before)
