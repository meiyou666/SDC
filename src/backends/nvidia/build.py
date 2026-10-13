"""Compile the shared CUDA extension inside the NVIDIA project container."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[3]
SOURCES = Path(__file__).parent


def build_key() -> tuple[str, dict]:
    manifest = json.loads((ROOT / "environment/gemm.json").read_text())
    contract = {**manifest, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "abi": torch._C._GLIBCXX_USE_CXX11_ABI}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode())
    for directory in (SOURCES, ROOT / "src/algorithms"):
        for file in sorted(directory.rglob("*")):
            if file.suffix in {".cpp", ".cu", ".cuh", ".h"}:
                digest.update(str(file.relative_to(ROOT)).encode())
                digest.update(file.read_bytes())
    return digest.hexdigest()[:16], contract


@lru_cache(maxsize=1)
def load_backend(verbose: bool = False):
    if not Path("/.dockerenv").exists() and not Path("/run/.containerenv").exists():
        raise RuntimeError("请在项目 NVIDIA Docker 容器中编译 GEMM。")
    if torch.version.cuda != "12.8":
        raise RuntimeError("GEMM 编译需要项目的 PyTorch cu128 环境。")
    cutlass = Path(os.environ.get("CUTLASS_PATH", "/opt/cutlass"))
    key, contract = build_key()
    if not (cutlass / "SDC_COMMIT").is_file() or (cutlass / "SDC_COMMIT").read_text().strip() != contract["cutlass_commit"]:
        raise RuntimeError("CUTLASS 版本未匹配；请重建 NVIDIA 镜像。")
    baked = Path("/opt/sdc/gemm.json")
    if not baked.is_file() or json.loads(baked.read_text()) != json.loads((ROOT / "environment/gemm.json").read_text()):
        raise RuntimeError("GEMM 编译配置已变化，请重建 NVIDIA 镜像。")
    os.environ.setdefault("MAX_JOBS", str(contract["max_build_jobs"]))
    files = [SOURCES / "bindings.cpp", SOURCES / "kernels.cu", SOURCES / "dense.cu",
             SOURCES / "mxfp8.cu", SOURCES / "nvfp4.cu"]
    module = load(
        name=f"sdc_gemm_{key}", sources=[str(p) for p in files],
        extra_include_paths=[str(cutlass / "include"), str(cutlass / "tools/util/include"), str(ROOT / "src")],
        extra_cflags=["-O2", f"-std={contract['cxx_standard']}"],
        extra_cuda_cflags=["-O2", f"-std={contract['cxx_standard']}", "--expt-relaxed-constexpr", "--expt-extended-lambda",
                          f"-gencode=arch=compute_{contract['cuda_arch']},code=sm_{contract['cuda_arch']}",
                          "--ftz=false", "-lineinfo"] + (["--use_fast_math"] if contract["fast_math"] else []),
        verbose=verbose,
    )
    return module


if __name__ == "__main__":
    load_backend(verbose=True)
    print(json.dumps({"extension": build_key()[0], **build_key()[1]}, indent=2))
