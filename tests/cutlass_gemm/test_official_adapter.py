"""The generated adapter must preserve every unrelated upstream byte."""

import hashlib

import pytest

from cutlass_gemm.codegen import sm120 as adapter


def test_adapter_only_changes_class_and_mma_call(tmp_path, monkeypatch):
    source = tmp_path / adapter.KERNEL
    source.parent.mkdir(parents=True)
    upstream = ("// license remains intact\nnamespace cutlass::gemm::kernel {\n"
                "template<class A> class GemmUniversal<A> {\n"
                "// TMA and scheduler remain intact\n" + adapter.ORIGINAL_CALL +
                "\n// epilogue remains intact\n};\n}\n")
    source.write_text(upstream)
    monkeypatch.setattr(adapter, "KERNEL_SHA256", hashlib.sha256(upstream.encode()).hexdigest())
    target = adapter.generate_reference_kernel(tmp_path, tmp_path / "generated")
    result = target.read_text()
    restored = result.replace("\n" + adapter.DECLARATION, "", 1).replace(
        "class AbftGemmUniversal<", "class GemmUniversal<", 1
    ).replace(adapter.REFERENCE_CALL, adapter.ORIGINAL_CALL, 1)
    assert restored == upstream
    assert source.read_text() == upstream
    mtime = target.stat().st_mtime_ns
    assert adapter.generate_reference_kernel(tmp_path, target.parent) == target
    assert target.stat().st_mtime_ns == mtime


def test_changed_official_source_is_rejected(tmp_path):
    source = tmp_path / adapter.KERNEL
    source.parent.mkdir(parents=True)
    source.write_text("modified upstream kernel")
    with pytest.raises(RuntimeError, match="pinned CUTLASS"):
        adapter.generate_reference_kernel(tmp_path, tmp_path / "generated")
    assert not (tmp_path / "generated").exists()
