"""Run all of this file on the target GPU with scripts/gemm-accept.sh."""

import pytest
import torch

from algorithms import Algorithm
from gemm import Fault, gemm
from gemm.benchmark import device_latency
from gemm.quantization import decode_reference, pack_reference


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0), reason="requires SM120 GPU")


@pytest.mark.parametrize("profile", ["fp32", "bf16", "mxfp8", "nvfp4"])
@pytest.mark.parametrize("shape", [(128, 128, 128), (17, 35, 19), (256, 128, 256)])
def test_native_output_and_pack_against_oracle(profile, shape):
    from backends.nvidia.runtime import require_device
    from gemm.types import PROFILES

    require_device(torch.device("cuda"))
    m, n, k = shape
    torch.manual_seed(42)
    a, b = torch.randn(m, k, device="cuda"), torch.randn(k, n, device="cuda")
    p = PROFILES[profile]
    # Compare the byte-level representation with an independent CPU oracle.
    native = torch.ops.sdc.pack(a, p.code, False, False, 42)
    oracle = pack_reference(a.cpu(), profile, seed=42)
    assert torch.equal(native[0].cpu(), oracle.codes)
    assert torch.equal(native[1].cpu(), oracle.scales)
    torch.testing.assert_close(native[2].cpu(), oracle.global_scale, atol=0, rtol=0)
    decoded = torch.ops.sdc.decode(*native, m, k, p.code)
    torch.testing.assert_close(decoded.cpu(), decode_reference(oracle), atol=0, rtol=0)
    pb = pack_reference(b.T.cpu(), profile, tile_scale=profile == "nvfp4", seed=0)
    expected = decode_reference(oracle).double() @ decode_reference(pb).T.double()
    actual = gemm(a, b, profile=profile, algorithm="none").output
    torch.testing.assert_close(actual.cpu().double(), expected, atol=.002, rtol=.003)


@pytest.mark.parametrize("profile", ["fp32", "bf16", "mxfp8", "nvfp4"])
def test_native_all_bits_and_device_graph(profile):
    a = torch.ones(16, 128, device="cuda")
    b = torch.ones(128, 32, device="cuda")
    options = dict(profile=profile, algorithm=Algorithm(atol=.01, rtol=.001))
    clean = gemm(a, b, **options)
    assert not clean.detected
    control = gemm(a, b, fault=Fault("accumulator", 1, 2, 31, False), **options)
    assert torch.equal(control.output, clean.output)
    for bit in range(32):
        out = gemm(a, b, fault=Fault("accumulator", 1, 2, bit), **options)
        assert int(out.fault_bits[0] ^ out.fault_bits[1]) == 1 << bit
        if bit == 31:
            assert out.detected and out.status[1]
    native = gemm(a, b, fault=Fault("accumulator", 1, 2, 31), recovery="native", **options)
    assert not native.unrecovered
    torch.testing.assert_close(native.output, clean.output)
    latency = device_latency(lambda: gemm(a, b, **options), warmup=2, repeats=3, replay_count=2)
    assert latency["median_ms"] > 0
