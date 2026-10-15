"""Generate one checksum call-site extension of the pinned official kernel.

The installed CUTLASS tree and original GemmUniversal remain unchanged. The
generated class differs only in its name/declaration and the consumer MMA call.
"""

import hashlib
from pathlib import Path


KERNEL = "include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
KERNEL_SHA256 = "3fbacbdbc3b35dc6efae7604359474ac42f8cda71f2e55b1fdb2a3b40d2869c3"
DECLARATION = """template<class ProblemShape, class Mainloop, class Epilogue,
         class Scheduler = void, class Enable = void>
class AbftGemmUniversal;

"""
ORIGINAL_CALL = """          collective_mainloop.mma(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            accumulators,
            work_k_tile_count,
            mma_thread_idx,
            shared_storage.tensors.mainloop,
            params.mainloop
          );"""
REFERENCE_CALL = """          collective_mainloop.mma_with_reference(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            accumulators,
            work_k_tile_count,
            mma_thread_idx,
            shared_storage.tensors.mainloop,
            params.mainloop,
            sdc_abft::TileReferenceContext{
              params.mainloop.reference_partial,
              params.mainloop.reference_alpha,
              params.mainloop.reference_weights,
              params.mainloop.reference_magnitude,
              int(get<0>(problem_shape_MNKL)),
              int(get<1>(problem_shape_MNKL)),
              int(get<0>(blk_coord)), int(get<1>(blk_coord)),
              int((get<0>(problem_shape_MNKL) + size<0>(TileShape{}) - 1) / size<0>(TileShape{})),
              int(((get<1>(problem_shape_MNKL) + size<1>(TileShape{}) - 1) / size<1>(TileShape{})) * size<1>(TileShape{}))
            }
          );"""


def generate_reference_kernel(cutlass, destination):
    """Return a generated header; reject any change to the pinned upstream file."""
    source = (Path(cutlass) / KERNEL).read_bytes()
    if hashlib.sha256(source).hexdigest() != KERNEL_SHA256:
        raise RuntimeError("Official cooperative kernel differs from pinned CUTLASS source")
    text = source.decode("utf-8")
    namespace = "namespace cutlass::gemm::kernel {\n"
    replacements = (
        (namespace, namespace + "\n" + DECLARATION),
        ("class GemmUniversal<", "class AbftGemmUniversal<"),
        (ORIGINAL_CALL, REFERENCE_CALL),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError("Official kernel checksum extension requires exactly one matching site")
        text = text.replace(old, new, 1)
    target = Path(destination) / "abft_gemm_universal.hpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return target
