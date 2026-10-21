"""Legacy adapter preserves the pinned scheduling and rejects upstream drift."""

import hashlib

import pytest

from cutlass_gemm import legacy_adapter as adapter


def test_mainloops_only_gain_reference_pointer_and_stage_hook(tmp_path, monkeypatch):
    originals = {}
    hashes = {}
    for stem in adapter.SOURCES:
        name = "MmaMultistage" if stem == "mma_multistage" else "MmaPipelined"
        loop = ("    for (; gemm_k_iterations > (-Base::kStages + 1);) {"
                if stem == "mma_multistage" else
                "    for (; gemm_k_iterations > 0; --gemm_k_iterations) {")
        text = f"// untouched license\nclass {name} {{\n  using Shape = Shape_;\n{loop}\n// unchanged MMA\n}};\n"
        source = tmp_path / "include/cutlass/gemm/threadblock" / (stem + ".h")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(text)
        originals[stem] = text
        hashes[stem] = hashlib.sha256(text.encode()).hexdigest()
    monkeypatch.setattr(adapter, "SOURCES", hashes)
    targets = adapter.generate_legacy_reference_headers(tmp_path, tmp_path / "generated")
    for (stem, upstream), target in zip(originals.items(), targets, strict=True):
        name = "MmaMultistage" if stem == "mma_multistage" else "MmaPipelined"
        stage = "smem_read_stage_idx_" if stem == "mma_multistage" else "smem_write_stage_idx ^ 1"
        restored = target.read_text().replace("Abft" + name, name).replace(
            "  using Reference = sdc_abft::LegacyTileReference<Base>;\n"
            "  Reference* reference = nullptr;\n", "", 1
        ).replace(f"\n      if (reference) reference->consume({stage});", "", 1)
        assert restored == upstream
    mtimes = [target.stat().st_mtime_ns for target in targets]
    adapter.generate_legacy_reference_headers(tmp_path, tmp_path / "generated")
    assert [target.stat().st_mtime_ns for target in targets] == mtimes


def test_changed_legacy_source_is_rejected(tmp_path):
    source = tmp_path / "include/cutlass/gemm/threadblock/mma_multistage.h"
    source.parent.mkdir(parents=True)
    source.write_text("unrecognized source")
    with pytest.raises(RuntimeError, match="pinned CUTLASS"):
        adapter.generate_legacy_reference_headers(tmp_path, tmp_path / "generated")
    assert not (tmp_path / "generated").exists()


def test_augmented_append_keeps_original_data_mainloop(tmp_path, monkeypatch):
    text = ("class MmaMultistage {\n  using Shape = Shape_;\n"
            "    PipeState pipe_state;\n    first_warp_load();\n"
            "    for (; gemm_k_iterations > (-Base::kStages + 1);) {\n"
            "      original_data_mma();\n        advance_smem_read_stage();\n"
            "        next_warp_load();\n    }\n};\n")
    source = tmp_path / "include/cutlass/gemm/threadblock/mma_multistage.h"
    source.parent.mkdir(parents=True)
    source.write_text(text)
    monkeypatch.setitem(adapter.SOURCES, "mma_multistage", hashlib.sha256(text.encode()).hexdigest())
    target = adapter.generate_augmented_header(tmp_path, tmp_path / "generated")
    adapted = target.read_text()
    assert "consume(smem_read_stage_idx_);\n      original_data_mma();" in adapted
    restored = adapted.replace("AugmentedMmaMultistage", "MmaMultistage").replace(
        "  using Encoding = sdc_abft::AugmentedTileEncoding<Base>;\n"
        "  Encoding* encoding = nullptr;\n", "", 1,
    ).replace("\n      if (encoding) encoding->consume(smem_read_stage_idx_);", "", 1)
    assert restored == text
    mtime = target.stat().st_mtime_ns
    adapter.generate_augmented_header(tmp_path, target.parent)
    assert target.stat().st_mtime_ns == mtime


def test_augmented_rejects_unknown_upstream_before_writing(tmp_path):
    source = tmp_path / "include/cutlass/gemm/threadblock/mma_multistage.h"
    source.parent.mkdir(parents=True)
    source.write_text("unknown upstream")
    with pytest.raises(RuntimeError, match="pinned CUTLASS"):
        adapter.generate_augmented_header(tmp_path, tmp_path / "generated")
    assert not (tmp_path / "generated").exists()
