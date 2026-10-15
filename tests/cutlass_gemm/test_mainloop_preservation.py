"""Check copied consumer bodies against the installed pinned CUTLASS source."""

import os
from pathlib import Path
import re

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/cutlass_gemm"
CUTLASS = Path(os.environ.get("CUTLASS_PATH", "/opt/cutlass"))


def function_body(source, name):
    start = source.index(name + "(MainloopPipeline pipeline,")
    start = source.index("{", start)
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1:end - 1]


@pytest.mark.parametrize("local,official", [
    ("abft_fp8_mainloop.cuh", "sm120_mma_tma.hpp"),
    ("abft_nvfp4_mainloop.cuh", "sm120_blockscaled_mma_tma.hpp"),
])
def test_only_checksum_calls_are_added_to_official_mma(local, official):
    upstream = CUTLASS / "include/cutlass/gemm/collective" / official
    if not upstream.is_file():
        pytest.skip("Pinned CUTLASS source is installed in the NVIDIA image")
    original = function_body(upstream.read_text(), "mma")
    adapted = function_body((SOURCE / local).read_text(), "mma_with_reference")
    adapted, declarations = re.subn(
        r"typename Reference::Accumulators reference_registers;", "", adapted,
    )
    adapted, consumes = re.subn(r"Reference::consume\([^;]*\);", "", adapted)
    adapted, finishes = re.subn(r"Reference::finish\([^;]*\);", "", adapted)
    assert (declarations, consumes, finishes) == (1, 2, 1)
    assert re.sub(r"\s+", "", adapted) == re.sub(r"\s+", "", original)
