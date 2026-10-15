"""Build thin bindings to the pinned official CUTLASS GEMM implementations."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import get_default_build_root, load


SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]


@lru_cache(maxsize=1)
def settings():
    return json.loads((ROOT / "environment/gemm.json").read_text())


def build_key():
    digest = hashlib.sha256(json.dumps(settings(), sort_keys=True).encode())
    digest.update(f"{torch.__version__}:{torch._C._GLIBCXX_USE_CXX11_ABI}".encode())
    for path in sorted(SOURCE.iterdir()):
        if path.suffix in {".cpp", ".cu", ".h", ".cuh", ".py"}:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    for path in sorted((SOURCE.parent / "abft").rglob("*.py")):
        digest.update(path.relative_to(SOURCE.parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _extension_key():
    digest = hashlib.sha256(json.dumps(settings(), sort_keys=True).encode())
    digest.update(f"{torch.__version__}:{torch._C._GLIBCXX_USE_CXX11_ABI}".encode())
    for path in sorted(SOURCE.iterdir()):
        if path.suffix in {".cpp", ".cu", ".h", ".cuh"} or path.name == "official_adapter.py":
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
    from .official_adapter import generate_reference_kernel

    generated = Path(os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root())) / "sdc_generated" / _extension_key()
    generate_reference_kernel(cutlass, generated)
    return load(
        name=f"sdc_cutlass_{_extension_key()}",
        sources=[str(SOURCE / name) for name in (
            "bindings.cpp", "gemm.cu", "fp8.cu", "nvfp4.cu",
            "abft_legacy.cu", "abft_fp8.cu", "abft_nvfp4.cu",
            "abft_fp8_reference.cu",
        )],
        extra_include_paths=[str(cutlass / "include"), str(cutlass / "tools/util/include"), str(generated)],
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


@lru_cache(maxsize=128)
def _scale_layout(m, n, k, device):
    """Use CUTLASS's own layout object; retain only the GPU index tables."""
    ia, ib, size_a, size_b = torch.ops.sdc_cutlass.nvfp4_scale_layout(m, n, k)
    return ia.to(device), ib.to(device), size_a, size_b


def _pack_scales(scales, indices, size):
    storage = torch.zeros(size, device=scales.device, dtype=torch.uint8)
    storage.scatter_(0, indices.flatten(), scales.view(torch.uint8).flatten())
    return storage


def gemm(a, b, precision, scaling, global_scale="amax"):
    """Compute A @ B with the official kernel selected by the experiment config."""
    expected = {"fp32": "none", "bf16": "none", "fp8": "tensorwise", "nvfp4": "block16"}
    if precision not in expected or scaling != expected[precision]:
        raise ValueError(f"GEMM 精度与缩放设置不匹配：{precision}/{scaling}。")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.device != b.device:
        raise ValueError("GEMM 输入必须是同设备上维度匹配的两个矩阵。")
    if precision in {"fp32", "bf16"}:
        dtype = torch.float32 if precision == "fp32" else torch.bfloat16
        return mm(a.to(dtype), b.to(dtype))

    require_device(a.device)
    from .quantization import quantize

    qa = quantize(a.contiguous(), precision, global_scale)
    qb = quantize(b.T.contiguous(), precision, global_scale)
    alpha = (qa.multiplier * qb.multiplier).reciprocal()
    if precision == "fp8":
        return torch.ops.sdc_cutlass.fp8_mm(
            qa.codes.view(torch.float8_e4m3fn), qb.codes.view(torch.float8_e4m3fn), alpha,
        )
    m, k = a.shape
    n = b.shape[1]
    ia, ib, size_a, size_b = _scale_layout(m, n, k, a.device)
    sa = _pack_scales(qa.scales, ia, size_a)
    sb = _pack_scales(qb.scales, ib, size_b)
    return torch.ops.sdc_cutlass.nvfp4_mm(qa.codes, qb.codes, sa, sb, k, alpha)


if __name__ == "__main__":
    load_extension()
    print(json.dumps({"build": build_key(), **settings()}, indent=2))
