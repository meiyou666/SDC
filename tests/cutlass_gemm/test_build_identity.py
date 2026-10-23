"""Cached extensions and training identities follow nested source inputs."""

import pytest

from cutlass_gemm import backend


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    source = tmp_path / "src/cutlass_gemm"
    methods = tmp_path / "src/abft"
    (source / "csrc/gemm").mkdir(parents=True)
    (source / "codegen").mkdir()
    methods.mkdir()
    (source / "csrc/gemm/fp32_bf16.cu").write_text("original kernel")
    monkeypatch.setattr(backend, "SOURCE", source)
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    monkeypatch.setattr(backend, "settings", lambda: {"cuda_arch": "120a"})
    return source


def _keys():
    return backend.build_key(), backend._extension_key()


def test_nested_header_changes_rebuild_extension_and_training_identity(source_tree):
    header = source_tree / "csrc/abft/fp8/visitor.cuh"
    header.parent.mkdir(parents=True)
    header.write_text("first visitor")
    initial = _keys()

    header.write_text("changed visitor")
    changed = _keys()
    assert changed[0] != initial[0]
    assert changed[1] != initial[1]


def test_nested_codegen_changes_rebuild_extension_and_training_identity(source_tree):
    generator = source_tree / "codegen/helpers/template.py"
    generator.parent.mkdir()
    generator.write_text("first call-site generator")
    initial = _keys()

    generator.write_text("changed call-site generator")
    changed = _keys()
    assert changed[0] != initial[0]
    assert changed[1] != initial[1]


@pytest.mark.parametrize("relative", ["linear.py", "runtime/helpers.py", "../abft/methods.py"],
                         ids=["package-runtime", "nested-runtime", "abft-method"])
def test_runtime_python_changes_only_training_identity(source_tree, relative):
    runtime = source_tree / relative
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("first runtime implementation")
    initial = _keys()

    runtime.write_text("changed runtime implementation")
    changed = _keys()
    assert changed[0] != initial[0]
    assert changed[1] == initial[1]


def test_identical_basenames_and_contents_still_have_distinct_path_identity(source_tree):
    first = source_tree / "csrc/abft/fp8/reference.cuh"
    second = source_tree / "csrc/abft/nvfp4/reference.cuh"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_text("identical header contents")
    initial = _keys()

    # Basenames and contents remain identical. Only the relative path changes;
    # hashing path.name instead would incorrectly reuse the previous identity.
    moved = source_tree / "csrc/abft/common/reference.cuh"
    moved.parent.mkdir()
    first.rename(moved)
    relocated = _keys()
    assert relocated[0] != initial[0]
    assert relocated[1] != initial[1]

    # The other same-named header must remain independently tracked as well.
    second.write_text("changed NVFP4 reference")
    changed = _keys()
    assert changed[0] != relocated[0]
    assert changed[1] != relocated[1]
