"""Build an isolated, runtime-masked copy of the BF16 augmented GEMM.

This is a diagnostic kernel, never a training backend. N/K select the same
full-tile specialization as production; all masks for that shape use one CUDA
entry point with identical static resources. A common q sink keeps encoding
observable even when propagation is disabled; it also perturbs production timing.
"""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from cutlass_gemm import backend
from cutlass_gemm.legacy_adapter import generate_augmented_header


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/cutlass_gemm/abft_augmented.h"
SOURCE_SHA256 = "8e64ed1eb175e249a60b5edb0205c564d2d6de86b369a8c4352d824928f5a6b2"


def _replace_once(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise RuntimeError("Diagnostic insertion site changed: " + anchor[:100])
    return text.replace(anchor, replacement, 1)


def generate_headers(cutlass, destination):
    """Fail closed on production changes; never edit production or CUTLASS files."""
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Production augmented GEMM changed; review the diagnostic generator first")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    official = generate_augmented_header(cutlass, destination).read_text(encoding="utf-8")
    official = official.replace("sdc_abft::", "sdc_breakdown_detail::")
    official = official.replace("AugmentedMmaMultistage", "BreakdownMmaMultistage")
    text = source.decode("utf-8")
    text = _replace_once(text, "  int* tile_flags = nullptr;", """  int* tile_flags = nullptr;
  int diagnostic_mask = 15;
  float diagnostic_seed = 1.f;
  float* diagnostic_sink = nullptr;""")
    text = _replace_once(text, "  ChecksumFragment predicted;", """  ChecksumFragment predicted;
  float diagnostic_q_sink = 0.f;""")
    text = _replace_once(text, """    CUTLASS_PRAGMA_UNROLL
    for (int reduction = 0; reduction < 128; reduction += 16) {""", """    if (projection.diagnostic_mask & 1) {
    CUTLASS_PRAGMA_UNROLL
    for (int reduction = 0; reduction < 128; reduction += 16) {""")
    text = _replace_once(text, """    const int q_slot = reuse_pipeline_barriers ? k_tile % 2 : 0;""", """    } else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(q_fragment); ++i) {
        q_fragment(i) = projection.diagnostic_seed;
      }
    }
    const int q_slot = reuse_pipeline_barriers ? k_tile % 2 : 0;""")
    text = _replace_once(text, """        scratch.q[q_slot][p][k] = full_kn || k < valid_k
            ? convert(q_fragment(i)) : Element(0.f);""", """        const Element encoded = full_kn || k < valid_k
            ? convert(q_fragment(i)) : Element(0.f);
        scratch.q[q_slot][p][k] = encoded;
        // Every mode consumes all valid BF16 q elements, not just the last K tile.
        diagnostic_q_sink += float(encoded);""")
    text = _replace_once(text, """    // The two BF16 rows live beside A, not in its original 128 data rows.
    __syncthreads();""", """    // Only this extra publication barrier is masked. The host rejects qB
    // without it; all original pipeline and shared-union barriers stay intact.
    if (projection.diagnostic_mask & 8) __syncthreads();""")
    text = _replace_once(text, """    if (!reuse_pipeline_barriers) return;""", """    if (!reuse_pipeline_barriers || !(projection.diagnostic_mask & 2)) return;""")
    text = _replace_once(text, """  CUTLASS_DEVICE void finish() {
    ChecksumMma mma;""", """  CUTLASS_DEVICE void finish() {
    const int64_t cta = int64_t(tile_m) * (projection.padded_n / 128) + tile_n;
    projection.diagnostic_sink[cta * Threads + thread] = diagnostic_q_sink;
    ChecksumMma mma;""")
    # The native visitor has its own helper. Mask its actual projection work,
    # leaving data conversion, injection, and nonfinite checks in every mode.
    text = _replace_once(text, """      const float row_weights[2] = {
          float(params_.weights[row]),""", """      if (params_.diagnostic_mask & 4) {
      const float row_weights[2] = {
          float(params_.weights[row]),""")
    text = _replace_once(text, """            magnitude_[p][slot] += fabsf(term);
          }
        }
      }
    }
    OutputVector& output""", """            magnitude_[p][slot] += fabsf(term);
          }
        }
      }
      }
    }
    OutputVector& output""")
    text = _replace_once(text, """      for (int p = 0; p < 2; ++p) {
        shared_.sums[p][index] = sums_[p][c];
        shared_.magnitude[p][index] = magnitude_[p][c];
      }""", """      if (params_.diagnostic_mask & 4) for (int p = 0; p < 2; ++p) {
        shared_.sums[p][index] = sums_[p][c];
        shared_.magnitude[p][index] = magnitude_[p][c];
      }""")
    text = _replace_once(text, """          if (col < extent_.column()) {
            // Match the legacy thread-major/slot-major summation order exactly.""", """          if (col < extent_.column() && (params_.diagnostic_mask & 4)) {
            // Match the legacy thread-major/slot-major summation order exactly.""")
    text = text.replace("sdc_abft", "sdc_breakdown_detail")
    text = text.replace("AugmentedMmaMultistage", "BreakdownMmaMultistage")
    text = _replace_once(text, '#include "abft_augmented_mma_multistage.h"',
                         '#include "breakdown_mma_multistage.h"')
    for name, content in (("breakdown_mma_multistage.h", official), ("breakdown_augmented.h", text)):
        target = destination / name
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
    return destination


@lru_cache(maxsize=1)
def load_diagnostic():
    """Load only the isolated diagnostic translation unit in the project environment."""
    baked = backend.environment_settings_path()
    config = backend.settings()
    if torch.version.cuda != "12.8":
        raise RuntimeError("请使用项目的 PyTorch cu128 环境。")
    if not baked.is_file() or json.loads(baked.read_text()) != config:
        raise RuntimeError("编译配置已更新，请重新初始化平台环境或重建 NVIDIA 镜像。")
    cutlass = Path(os.environ.get("CUTLASS_PATH", "/opt/cutlass"))
    commit = cutlass / "SDC_COMMIT"
    if not commit.is_file() or commit.read_text().strip() != config["cutlass_commit"]:
        raise RuntimeError("CUTLASS 版本不匹配，请重新初始化平台环境或重建 NVIDIA 镜像。")
    wrapper = Path(__file__).with_name("gemm_breakdown_kernel.cu")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    digest.update(f"{torch.__version__}:{torch._C._GLIBCXX_USE_CXX11_ABI}".encode())
    for path in (Path(__file__), wrapper, SOURCE, ROOT / "src/cutlass_gemm/legacy_adapter.py"):
        digest.update(path.read_bytes())
    build = ROOT / ".sdc/diagnostics/gemm_breakdown" / digest.hexdigest()[:16]
    generated = generate_headers(cutlass, build / "generated")
    os.environ.setdefault("MAX_JOBS", str(config["max_build_jobs"]))
    arch = config["cuda_arch"]
    load(
        name="sdc_breakdown_" + build.name,
        sources=[str(wrapper)],
        build_directory=str(build),
        extra_include_paths=[str(cutlass / "include"), str(cutlass / "tools/util/include"), str(generated)],
        extra_cflags=["-O2", f"-std={config['cxx_standard']}"],
        extra_cuda_cflags=["-O2", f"-std={config['cxx_standard']}",
                          f"-gencode=arch=compute_{arch},code=sm_{arch}",
                          "--expt-relaxed-constexpr", "--ftz=false", "-lineinfo", "-Xptxas=-v",
                          f"-DSDC_BREAKDOWN_ARCH={int(arch.rstrip('a'))}"],
        is_python_module=False,
        verbose=False,
    )
    return torch.ops.sdc_breakdown
