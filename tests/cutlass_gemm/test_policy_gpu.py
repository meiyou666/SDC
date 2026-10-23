"""Compiled algorithm selection must preserve data and cover all four precisions."""
import pytest
import torch

from abft.operands import PreparedGemm
from cutlass_gemm import backend


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="Compiled algorithm policies require the configured SM120 GPU",
)
FORMATS = {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}


def _prepared(precision):
    backend.require_device(torch.device("cuda"))
    # Valid tail tiles, and an output near one for a deterministic exponent fault.
    a = torch.ones(136, 128, device="cuda")
    b = torch.full((128, 144), 1.0 / 128, device="cuda")
    return PreparedGemm.prepare(a, b, precision, FORMATS[precision], global_scale=1.0)


def _call(prepared, weights, algorithm, kind="reference", nonfinite=None, fault=(-1, -1, -1)):
    suffix = "reference_mm" if kind == "reference" else "mm"
    if prepared.precision in {"fp32", "bf16"}:
        op = getattr(torch.ops.sdc_cutlass, "abft_" + suffix)
        operands = (prepared.a, prepared.b)
    elif prepared.precision == "fp8":
        op = getattr(torch.ops.sdc_cutlass, "abft_fp8_" + suffix)
        operands = (prepared.a.view(torch.float8_e4m3fn), prepared.b.view(torch.float8_e4m3fn),
                    prepared.alpha)
    else:
        op = getattr(torch.ops.sdc_cutlass, "abft_nvfp4_" + suffix)
        operands = (prepared.a, prepared.b, prepared.sfa, prepared.sfb, prepared.k, prepared.alpha)
    return op(*operands, weights, *fault, algorithm, nonfinite)


@pytest.mark.parametrize("precision", FORMATS)
@pytest.mark.parametrize("kind", ["observe", "reference"])
def test_sum_policy_keeps_classic_first_projection_and_gemm_output(precision, kind):
    prepared = _prepared(precision)
    weights = torch.stack((torch.ones(prepared.m, device="cuda"),
                           torch.arange(1, prepared.m + 1, device="cuda", dtype=torch.float32)))
    classic = _call(prepared, weights, 0, kind)
    single = _call(prepared, weights[:1], 1, kind)
    torch.testing.assert_close(single[0], classic[0], rtol=0, atol=0)
    torch.testing.assert_close(single[0], prepared.raw_native(), rtol=0, atol=0)
    assert single[1].shape[0] == 1 and classic[1].shape[0] == 2
    torch.testing.assert_close(single[1], classic[1][:1], rtol=0, atol=0)
    assert single[2].item() == classic[2].item() == 0
    if kind == "reference":
        for index in (3, 4):
            torch.testing.assert_close(single[index], classic[index][:1], rtol=0, atol=0)


@pytest.mark.parametrize("precision", FORMATS)
def test_output_nonfinite_flag_is_reset_and_set_in_epilogue(precision):
    prepared = _prepared(precision)
    weights = torch.ones(1, prepared.m, device="cuda")
    flag = torch.full((1,), 7, dtype=torch.int32, device="cuda")
    clean = _call(prepared, weights, 1, nonfinite=flag)
    assert flag.item() == 0
    corrupted = _call(prepared, weights, 1, nonfinite=flag,
                      fault=(prepared.m - 1, prepared.n - 1, 30))
    assert corrupted[2].item() == 1
    assert flag.item() == 1
    assert not torch.isfinite(corrupted[0][-1, -1]).item()
    # The reference was generated from normal input loads before output injection.
    torch.testing.assert_close(corrupted[3], clean[3], rtol=0, atol=0)
    _call(prepared, weights, 1, nonfinite=flag)
    assert flag.item() == 0


@pytest.mark.parametrize("precision", FORMATS)
def test_nonfinite_flag_cannot_alias_readonly_operands(precision):
    prepared = _prepared(precision)
    weights = torch.ones(1, prepared.m, device="cuda")
    alias = prepared.a.view(torch.int32).reshape(-1)[:1]
    before = prepared.a.clone()
    with pytest.raises(RuntimeError):
        _call(prepared, weights, 1, nonfinite=alias)
    torch.testing.assert_close(prepared.a, before, rtol=0, atol=0)


@pytest.mark.parametrize("precision", FORMATS)
@pytest.mark.parametrize("algorithm", [-1, 2, 2**32])
def test_unknown_compiled_policy_is_rejected_without_integer_truncation(precision, algorithm):
    prepared = _prepared(precision)
    weights = torch.ones(2, prepared.m, device="cuda")
    with pytest.raises(RuntimeError, match="algorithm"):
        _call(prepared, weights, algorithm)


def test_augmented_sum_uses_one_checksum_without_changing_data_rows():
    prepared = _prepared("bf16")
    weights = torch.stack((torch.ones(prepared.m, device="cuda"),
                           torch.arange(1, prepared.m + 1, device="cuda"))).to(torch.bfloat16)
    op = torch.ops.sdc_cutlass.abft_augmented_compact
    classic = op(prepared.a, prepared.b, weights, -1, -1, -1, 0)
    single = op(prepared.a, prepared.b, weights[:1], -1, -1, -1, 1)
    assert single[0].shape == (136, 144)
    assert single[1].shape[1] == 1 and classic[1].shape[1] == 2
    torch.testing.assert_close(single[0], classic[0], rtol=0, atol=0)
    torch.testing.assert_close(single[1], classic[1][:, :1], rtol=0, atol=0)
    torch.testing.assert_close(single[2], classic[2], rtol=0, atol=0)
