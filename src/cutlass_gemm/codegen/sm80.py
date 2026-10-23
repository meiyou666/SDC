"""Add reference hooks to pinned official legacy mainloops without changing scheduling."""

import hashlib
from pathlib import Path


SOURCES = {
    "mma_multistage": "3f89e410ca96b06e4a56ba19f90399e8fdd2744a28c45d458a60940708b4171b",
    "mma_pipelined": "fac90ae04135ccacc0ea48440d20c1059c555ad6355a8608df0e6a74a8a7b4e5",
}
ALGORITHM_PARAMETER = ",\n    class Algorithm = sdc_abft::policy::Classic"


def _add_algorithm_parameter(text):
    # Append after the official final Enable parameter, keeping every original
    # parameter/default in place. Both pinned legacy declarations use this site.
    anchor = "typename Enable = bool"
    if text.count(anchor) != 1:
        raise RuntimeError("Legacy algorithm template insertion site changed")
    return text.replace(anchor, anchor + ALGORITHM_PARAMETER, 1)


def generate_legacy_reference_headers(cutlass, destination):
    """Preserve official instructions/order; add one callback per real K tile."""
    targets = []
    for stem, digest in SOURCES.items():
        source = (Path(cutlass) / "include/cutlass/gemm/threadblock" / (stem + ".h")).read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise RuntimeError("Legacy mainloop differs from pinned CUTLASS: " + stem)
        text = _add_algorithm_parameter(source.decode("utf-8"))
        name = "MmaMultistage" if stem == "mma_multistage" else "MmaPipelined"
        text = text.replace(name, "Abft" + name)
        anchor = "  using Shape = Shape_;"
        if text.count(anchor) != 1:
            raise RuntimeError("Legacy reference type insertion site changed")
        text = text.replace(anchor, "  using Reference = sdc_abft::LegacyTileReference<Base, Algorithm>;\n"
                            "  Reference* reference = nullptr;\n" + anchor, 1)
        if stem == "mma_multistage":
            anchor = "    for (; gemm_k_iterations > (-Base::kStages + 1);) {"
            hook = "\n      if (reference) reference->consume(smem_read_stage_idx_);"
        else:
            anchor = "    for (; gemm_k_iterations > 0; --gemm_k_iterations) {"
            hook = "\n      if (reference) reference->consume(smem_write_stage_idx ^ 1);"
        if text.count(anchor) != 1:
            raise RuntimeError("Legacy reference stage insertion site changed")
        text = text.replace(anchor, anchor + hook, 1)
        target = Path(destination) / ("abft_" + stem + ".h")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
        targets.append(target)
    return targets


def generate_augmented_header(cutlass, destination):
    """Append checksum MMA fragments while preserving the original data tile.

    Encoding reads the existing shared stage. Compact propagation reads the
    already-transformed B registers; neither hook mutates official MMA operands.
    """
    source = (Path(cutlass) / "include/cutlass/gemm/threadblock/mma_multistage.h").read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCES["mma_multistage"]:
        raise RuntimeError("Augmented mainloop differs from pinned CUTLASS: mma_multistage")
    text = _add_algorithm_parameter(source.decode("utf-8")).replace("MmaMultistage", "AugmentedMmaMultistage")
    replacements = {
        "  using Shape = Shape_;": (
            "  using Encoding = sdc_abft::AugmentedTileEncoding<Base, Algorithm>;\n"
            "  Encoding* encoding = nullptr;\n"
            "  using Shape = Shape_;"
        ),
        "    for (; gemm_k_iterations > (-Base::kStages + 1);) {": (
            "    for (; gemm_k_iterations > (-Base::kStages + 1);) {\n"
            "      if (encoding) encoding->consume(smem_read_stage_idx_);"
        ),
        "      // Execute the current warp-tile of MMA operations": (
            "      if (encoding) encoding->propagate(\n"
            "          warp_mma_k, pipe_state.warp_transformed_frag_B_[warp_mma_k % 2]);\n\n"
            "      // Execute the current warp-tile of MMA operations"
        ),
    }
    for anchor, replacement in replacements.items():
        if text.count(anchor) != 1:
            raise RuntimeError("Augmented mainloop insertion site changed: " + anchor.strip())
        text = text.replace(anchor, replacement, 1)
    target = Path(destination) / "abft_augmented_mma_multistage.h"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target
