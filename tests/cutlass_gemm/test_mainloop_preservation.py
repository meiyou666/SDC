"""Check copied consumer bodies against the installed pinned CUTLASS source."""

import os
from pathlib import Path
import re

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/cutlass_gemm/csrc/abft"
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
    ("fp8/fp8_mainloop.cuh", "sm120_mma_tma.hpp"),
    ("nvfp4/nvfp4_mainloop.cuh", "sm120_blockscaled_mma_tma.hpp"),
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


def test_augmented_preserves_pinned_multistage_source(tmp_path):
    from cutlass_gemm.codegen import sm80

    upstream = CUTLASS / "include/cutlass/gemm/threadblock/mma_multistage.h"
    if not upstream.is_file():
        pytest.skip("Pinned CUTLASS source is installed in the NVIDIA environment")
    adapted = sm80.generate_augmented_header(CUTLASS, tmp_path).read_text()
    assert adapted.count(sm80.ALGORITHM_PARAMETER) == 1
    restored = adapted.replace(sm80.ALGORITHM_PARAMETER, "", 1).replace("AugmentedMmaMultistage", "MmaMultistage").replace(
        "  using Encoding = sdc_abft::AugmentedTileEncoding<Base, Algorithm>;\n"
        "  Encoding* encoding = nullptr;\n", "", 1,
    ).replace("\n      if (encoding) encoding->consume(smem_read_stage_idx_);", "", 1).replace(
        "      if (encoding) encoding->propagate(\n"
        "          warp_mma_k, pipe_state.warp_transformed_frag_B_[warp_mma_k % 2]);\n\n", "", 1,
    )
    assert restored == upstream.read_text()


@pytest.mark.parametrize("stem,name,stage", [
    ("mma_multistage", "MmaMultistage", "smem_read_stage_idx_"),
    ("mma_pipelined", "MmaPipelined", "smem_write_stage_idx ^ 1"),
])
def test_legacy_algorithm_adapter_preserves_pinned_source(tmp_path, stem, name, stage):
    from cutlass_gemm.codegen import sm80

    directory = CUTLASS / "include/cutlass/gemm/threadblock"
    if any(not (directory / (item + ".h")).is_file() for item in sm80.SOURCES):
        pytest.skip("Pinned CUTLASS source is installed in the NVIDIA environment")
    generated = sm80.generate_legacy_reference_headers(CUTLASS, tmp_path)
    adapted = next(path for path in generated if path.name == "abft_" + stem + ".h").read_text()
    assert adapted.count(sm80.ALGORITHM_PARAMETER) == 1
    restored = adapted.replace(sm80.ALGORITHM_PARAMETER, "", 1).replace("Abft" + name, name).replace(
        "  using Reference = sdc_abft::LegacyTileReference<Base, Algorithm>;\n"
        "  Reference* reference = nullptr;\n", "", 1,
    ).replace(f"\n      if (reference) reference->consume({stage});", "", 1)
    assert restored == (directory / (stem + ".h")).read_text()
