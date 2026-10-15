"""Cached extensions and training identities must follow their own source inputs."""

from cutlass_gemm import backend


def test_headers_rebuild_extension_but_method_changes_only_training_identity(tmp_path, monkeypatch):
    source = tmp_path / "cutlass_gemm"
    methods = tmp_path / "abft"
    source.mkdir()
    methods.mkdir()
    (source / "kernel.cu").write_text("original kernel")
    header = source / "visitor.cuh"
    header.write_text("first visitor")
    method = methods / "methods.py"
    method.write_text("first method")
    monkeypatch.setattr(backend, "SOURCE", source)
    monkeypatch.setattr(backend, "settings", lambda: {"cuda_arch": "120a"})
    initial = backend.build_key(), backend._extension_key()

    header.write_text("changed visitor")
    with_header = backend.build_key(), backend._extension_key()
    assert with_header[0] != initial[0]
    assert with_header[1] != initial[1]

    method.write_text("changed method")
    with_method = backend.build_key(), backend._extension_key()
    assert with_method[0] != with_header[0]
    assert with_method[1] == with_header[1]

    generator = source / "official_adapter.py"
    generator.write_text("modified call-site adapter")
    assert backend._extension_key() != with_method[1]
