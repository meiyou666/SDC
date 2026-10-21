#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

namespace {
constexpr int kThreads = 256;
constexpr int kTileColumns = 128;
constexpr unsigned kDetected = 1u;
constexpr unsigned kNonfinite = 2u;
constexpr unsigned kHit = 4u;

using FinalizeOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor,
                                  at::Tensor, at::Tensor, at::Tensor>;

struct CombineFlags {
  __device__ unsigned operator()(unsigned left, unsigned right) const {
    return left | right;
  }
};

// One CTA consumes the small checksum arrays, never the GEMM output matrix.
// partials is [observed/predicted/magnitude, checksum, M tile, padded N].
// reference is [predicted/magnitude, checksum, N]. On a retry it is read-only
// evidence from the first execution; retry predictions are deliberately ignored.
__global__ void finalize_kernel(
    const float* partials, const int32_t* tile_flags, int64_t n,
    int64_t m_tiles, int64_t padded_n, int64_t n_tiles,
    float atol, float rtol, bool use_reference, bool diagnostics,
    float* reference, float* residual_out, float* threshold_out,
    bool* flags_out, int32_t* hit_out, int64_t* location_out) {
  using Reduction = cub::BlockReduce<unsigned, kThreads>;
  __shared__ typename Reduction::TempStorage reduction_storage;
  unsigned state = 0;
  const int64_t channel_stride = m_tiles * padded_n;
  const int64_t field_stride = 2 * channel_stride;

  for (int64_t index = threadIdx.x; index < m_tiles * n_tiles;
       index += kThreads) {
    const int32_t tile_state = tile_flags[index];
    if (tile_state & 1) state |= kHit;
    // The epilogue records BF16 output overflow as well as nonfinite data.
    // A finite pre-cast checksum cannot by itself detect that overflow.
    if (tile_state & 2) state |= kDetected | kNonfinite;
  }

  for (int64_t col = threadIdx.x; col < n; col += kThreads) {
#pragma unroll
    for (int channel = 0; channel < 2; ++channel) {
      float observed = 0.f;
      float predicted = 0.f;
      float magnitude = 0.f;
      const int64_t base = channel * channel_stride + col;
      for (int64_t tile = 0; tile < m_tiles; ++tile) {
        const int64_t offset = base + tile * padded_n;
        observed = __fadd_rn(observed, partials[offset]);
        if (!use_reference) {
          predicted = __fadd_rn(predicted, partials[field_stride + offset]);
          magnitude = __fadd_rn(magnitude, partials[2 * field_stride + offset]);
        }
      }
      const int64_t output_offset = channel * n + col;
      if (use_reference) {
        predicted = reference[output_offset];
        magnitude = reference[2 * n + output_offset];
      } else {
        reference[output_offset] = predicted;
        reference[2 * n + output_offset] = magnitude;
      }

      const float residual = __fsub_rn(observed, predicted);
      // Match the separate FP32 multiply and add used by the original tensor
      // expression, rather than silently replacing it with a fused operation.
      const float threshold = __fadd_rn(atol, __fmul_rn(rtol, magnitude));
      const bool nonfinite = !isfinite(observed) || !isfinite(predicted) ||
          !isfinite(magnitude) || !isfinite(residual) || !isfinite(threshold);
      if (nonfinite) state |= kDetected | kNonfinite;
      if (fabsf(residual) > threshold) state |= kDetected;
      if (diagnostics) {
        residual_out[output_offset] = residual;
        threshold_out[output_offset] = threshold;
      }
    }
  }

  const unsigned combined = Reduction(reduction_storage).Reduce(state, CombineFlags{});
  if (threadIdx.x == 0) {
    flags_out[0] = (combined & kDetected) != 0;
    flags_out[1] = (combined & kNonfinite) != 0;
    flags_out[2] = false;  // BF16 augmented encoding does not provide row localization.
    flags_out[3] = false;  // Recovery is controlled by the caller, not this check.
    flags_out[4] = false;
    hit_out[0] = (combined & kHit) != 0;
    location_out[0] = -1;
    location_out[1] = -1;
  }
}
}  // namespace

FinalizeOutput sdc_cutlass_abft_finalize(
    const at::Tensor& partials, const at::Tensor& tile_flags, int64_t n,
    double atol, double rtol, const std::optional<at::Tensor>& reference,
    bool diagnostics) {
  TORCH_CHECK(partials.is_cuda() && tile_flags.is_cuda() &&
                  partials.device() == tile_flags.device(),
              "ABFT finalize tensors must be on the same CUDA device");
  TORCH_CHECK(partials.layout() == at::kStrided && partials.is_contiguous() &&
                  partials.scalar_type() == at::kFloat && partials.dim() == 4 &&
                  partials.size(0) == 3 && partials.size(1) == 2 && partials.size(2) > 0,
              "ABFT finalize partials must be contiguous float32 [3,2,T,padded_N] with T>0");
  TORCH_CHECK(n > 0 && n <= std::numeric_limits<int>::max() - kTileColumns,
              "ABFT finalize N must be positive and fit padded int32 extents");
  const int64_t n_tiles = (n + kTileColumns - 1) / kTileColumns;
  const int64_t padded_n = n_tiles * kTileColumns;
  const int64_t m_tiles = partials.size(2);
  TORCH_CHECK(partials.size(3) == padded_n,
              "ABFT finalize padded_N must equal ceil(N/128)*128");
  TORCH_CHECK(tile_flags.layout() == at::kStrided && tile_flags.is_contiguous() &&
                  tile_flags.scalar_type() == at::kInt && tile_flags.dim() == 2 &&
                  tile_flags.size(0) == m_tiles && tile_flags.size(1) == n_tiles,
              "ABFT finalize tile_flags must be contiguous int32 [T,ceil(N/128)]");
  const double max_float = std::numeric_limits<float>::max();
  TORCH_CHECK(std::isfinite(atol) && std::isfinite(rtol) &&
                  atol >= 0. && rtol >= 0. && atol <= max_float && rtol <= max_float,
              "ABFT finalize tolerances must be finite nonnegative float32 values");
  if (reference.has_value()) {
    const auto& supplied = *reference;
    TORCH_CHECK(supplied.is_cuda() && supplied.device() == partials.device() &&
                    supplied.layout() == at::kStrided && supplied.is_contiguous() &&
                    supplied.scalar_type() == at::kFloat && supplied.dim() == 3 &&
                    supplied.size(0) == 2 && supplied.size(1) == 2 && supplied.size(2) == n,
                "ABFT finalize reference must be contiguous float32 [2,2,N] on the same CUDA device");
  }

  const c10::cuda::CUDAGuard guard(partials.device());
  auto output_reference = reference.has_value() ? *reference : at::empty({2, 2, n}, partials.options());
  auto residual = diagnostics ? at::empty({2, n}, partials.options()) : at::empty({0}, partials.options());
  auto threshold = diagnostics ? at::empty({2, n}, partials.options()) : at::empty({0}, partials.options());
  auto flags = at::empty({5}, partials.options().dtype(at::kBool));
  auto hit = at::empty({1}, partials.options().dtype(at::kInt));
  auto location = at::empty({2}, partials.options().dtype(at::kLong));
  const auto stream = at::cuda::getCurrentCUDAStream(partials.get_device());
  finalize_kernel<<<1, kThreads, 0, stream.stream()>>>(
      partials.data_ptr<float>(), tile_flags.data_ptr<int32_t>(), n, m_tiles, padded_n,
      n_tiles, static_cast<float>(atol), static_cast<float>(rtol), reference.has_value(), diagnostics,
      output_reference.data_ptr<float>(), residual.data_ptr<float>(), threshold.data_ptr<float>(),
      flags.data_ptr<bool>(), hit.data_ptr<int32_t>(), location.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_reference, residual, threshold, flags, hit, location};
}
