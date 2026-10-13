"""Build and call the unchanged CUTLASS device::Gemm implementation."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]


@lru_cache(maxsize=1)
def settings():
    return json.loads((ROOT / "environment/gemm.json").read_text())


def build_key():
    digest = hashlib.sha256(json.dumps(settings(), sort_keys=True).encode())
    digest.update(f"{torch.__version__}:{torch._C._GLIBCXX_USE_CXX11_ABI}".encode())
    for path in sorted(SOURCE.iterdir()):
        if path.suffix in {".cpp", ".cu", ".py"}:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def load_extension():
    if not Path("/.dockerenv").exists() and not Path("/run/.containerenv").exists():
        raise RuntimeError("请在项目 NVIDIA 容器中编译和运行。")
    if torch.version.cuda != "12.8":
        raise RuntimeError("请使用项目的 PyTorch cu128 环境。")
    config = settings()
    cutlass = Path(os.environ.get("CUTLASS_PATH", "/opt/cutlass"))
    commit_file = cutlass / "SDC_COMMIT"
    if not commit_file.is_file() or commit_file.read_text().strip() != config["cutlass_commit"]:
        raise RuntimeError("CUTLASS 版本不匹配，请重建 NVIDIA 镜像。")
    baked = Path("/opt/sdc/gemm.json")
    if not baked.is_file() or json.loads(baked.read_text()) != config:
        raise RuntimeError("编译配置已更新，请重建 NVIDIA 镜像。")
    os.environ.setdefault("MAX_JOBS", str(config["max_build_jobs"]))
    arch = config["cuda_arch"]
    return load(
        name=f"sdc_cutlass_{build_key()}",
        sources=[str(SOURCE / "bindings.cpp"), str(SOURCE / "gemm.cu")],
        extra_include_paths=[str(cutlass / "include")],
        extra_cflags=["-O2", f"-std={config['cxx_standard']}"],
        extra_cuda_cflags=["-O2", f"-std={config['cxx_standard']}",
                          f"-gencode=arch=compute_{arch},code=sm_{arch}",
                          "--expt-relaxed-constexpr", "--ftz=false"],
        verbose=True,
    )


@lru_cache(maxsize=None)
def require_device(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUTLASS 基线需要 NVIDIA CUDA 设备。")
    arch = int(settings()["cuda_arch"].rstrip("a"))
    if torch.cuda.get_device_capability(device) != (arch // 10, arch % 10):
        raise RuntimeError(f"当前构建目标为 SM{arch}，请在对应 GPU 上运行。")
    load_extension()


def mm(a, b):
    require_device(a.device)
    return torch.ops.sdc_cutlass.mm(a, b)


if __name__ == "__main__":
    load_extension()
    print(json.dumps({"build": build_key(), **settings()}, indent=2))
