#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "abft/common/policy.cuh"

namespace {
constexpr int kThreads = 256;
constexpr int kParallelThreads = 128;
constexpr int64_t kParallelMinWork = 8192;
constexpr int kParallelMaxBlocks = 256;
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

__device__ void write_flags(unsigned combined, bool* flags_out,
                            int32_t* hit_out, int64_t* location_out) {
  flags_out[0] = (combined & kDetected) != 0;
  flags_out[1] = (combined & kNonfinite) != 0;
  flags_out[2] = false;  // These compiled detection policies do not localize errors.
  flags_out[3] = false;  // Recovery is controlled by the caller, not this check.
  flags_out[4] = false;
  hit_out[0] = (combined & kHit) != 0;
  location_out[0] = -1;
  location_out[1] = -1;
}

// Consume only the checksum arrays, never the GEMM output matrix. The small
// path uses one CTA; the parallel grid splits independent columns and channels.
// Each (channel, column) is still reduced in ascending M-tile order by one thread.
// Each input pointer describes [checksum, M tile, padded N]. Packed augmented
// input is adapted by pointer offsets without concatenating/copying tensors.
// reference is [predicted/magnitude, checksum, N]. On a retry it is read-only
// evidence from the first execution; retry predictions are deliberately ignored.
template <class Policy, bool Parallel, bool PackedTileFlags>
__global__ void finalize_kernel(
    const float* observed_partials, const float* predicted_partials,
    const float* magnitude_partials, const int32_t* tile_flags, int64_t flag_count,
    const int32_t* input_hit, const int32_t* input_nonfinite,
    int64_t n, int64_t m_tiles, int64_t padded_n,
    float atol, float rtol, bool use_reference, bool diagnostics,
    const float* thresholds, bool scalar_threshold,
    float* reference, float* residual_out, float* threshold_out,
    bool* flags_out, int32_t* hit_out, int64_t* location_out,
    int32_t* block_states) {
  constexpr int threads = Parallel ? kParallelThreads : kThreads;
  using Reduction = cub::BlockReduce<unsigned, threads>;
  __shared__ typename Reduction::TempStorage reduction_storage;
  unsigned state = 0;
  const int64_t block = Parallel ? int64_t(blockIdx.y) * gridDim.x + blockIdx.x : 0;
  const int64_t flag_thread = block * threads + threadIdx.x;
  const int64_t flag_stride = Parallel ? int64_t(gridDim.x) * gridDim.y * threads : threads;
  const int64_t column_thread = Parallel ? int64_t(blockIdx.x) * threads + threadIdx.x : threadIdx.x;
  const int64_t column_stride = Parallel ? int64_t(gridDim.x) * threads : threads;
  const int64_t channel_stride = m_tiles * padded_n;

  // Each tile flag belongs to exactly one thread in the entire grid.
  if constexpr (PackedTileFlags) {
    for (int64_t index = flag_thread; index < flag_count; index += flag_stride) {
      const int32_t tile_state = tile_flags[index];
      if (tile_state & 1) state |= kHit;
      // Includes output-cast overflow, even when pre-cast checksums are finite.
      if (tile_state & 2) state |= kDetected | kNonfinite;
    }
  } else if (block == 0 && threadIdx.x == 0) {
    // Runtime evidence affects report bits only, never the policy's arithmetic.
    if (*input_hit != 0) state |= kHit;
    if (*input_nonfinite != 0) state |= kDetected | kNonfinite;
  }

  for (int64_t col = column_thread; col < n; col += column_stride) {
#pragma unroll
    for (int entry = 0; entry < (Parallel ? 1 : Policy::kChecksums); ++entry) {
      const int channel = Parallel ? int(blockIdx.y) : entry;
      float observed = 0.f;
      float predicted = 0.f;
      float magnitude = 0.f;
      const int64_t base = channel * channel_stride + col;
      for (int64_t tile = 0; tile < m_tiles; ++tile) {
        const int64_t offset = base + tile * padded_n;
        observed = Policy::merge(observed, observed_partials[offset]);
        if (!use_reference) {
          predicted = Policy::merge(predicted, predicted_partials[offset]);
          magnitude = Policy::merge(magnitude, magnitude_partials[offset]);
        }
      }
      const int64_t output_offset = channel * n + col;
      if (use_reference) {
        predicted = reference[output_offset];
        magnitude = reference[Policy::kChecksums * n + output_offset];
      } else {
        reference[output_offset] = predicted;
        reference[Policy::kChecksums * n + output_offset] = magnitude;
      }

      const bool use_override = thresholds != nullptr;
      const float threshold_override = use_override ? thresholds[scalar_threshold ? 0 : output_offset] : 0.f;
      const auto decision = Policy::finalize(
          observed, predicted, magnitude, atol, rtol, threshold_override, use_override);
      if (decision.nonfinite) state |= kDetected | kNonfinite;
      if (decision.detected) state |= kDetected;
      if (diagnostics) {
        residual_out[output_offset] = decision.residual;
        threshold_out[output_offset] = decision.threshold;
      }
    }
  }

  const unsigned combined = Reduction(reduction_storage).Reduce(state, CombineFlags{});
  if (threadIdx.x == 0) {
    if constexpr (Parallel) {
      block_states[block] = static_cast<int32_t>(combined);
    } else {
      write_flags(combined, flags_out, hit_out, location_out);
    }
  }
}

// Only exact integer status bits cross CTA boundaries. This launch follows the
// column/channel grid on the same stream; no unordered floating-point atomics.
__global__ void finalize_flags_kernel(
    const int32_t* block_states, int blocks,
    bool* flags_out, int32_t* hit_out, int64_t* location_out) {
  using Reduction = cub::BlockReduce<unsigned, kThreads>;
  __shared__ typename Reduction::TempStorage reduction_storage;
  unsigned state = 0;
  for (int index = threadIdx.x; index < blocks; index += kThreads) {
    state |= static_cast<unsigned>(block_states[index]);
  }
  const unsigned combined = Reduction(reduction_storage).Reduce(state, CombineFlags{});
  if (threadIdx.x == 0) {
    write_flags(combined, flags_out, hit_out, location_out);
  }
}
void check_common(int64_t n, double atol, double rtol) {
  TORCH_CHECK(n > 0 && n <= std::numeric_limits<int>::max() - kTileColumns,
              "ABFT finalize N must be positive and fit padded int32 extents");
  const double max_float = std::numeric_limits<float>::max();
  TORCH_CHECK(std::isfinite(atol) && std::isfinite(rtol) &&
                  atol >= 0. && rtol >= 0. && atol <= max_float && rtol <= max_float,
              "ABFT finalize tolerances must be finite nonnegative float32 values");
}

void check_reference(const std::optional<at::Tensor>& reference,
                     const at::Tensor& prototype, int channels, int64_t n) {
  if (reference.has_value()) {
    const auto& supplied = *reference;
    TORCH_CHECK(supplied.is_cuda() && supplied.device() == prototype.device() &&
                    supplied.layout() == at::kStrided && supplied.is_contiguous() &&
                    supplied.scalar_type() == at::kFloat && supplied.dim() == 3 &&
                    supplied.size(0) == 2 && supplied.size(1) == channels && supplied.size(2) == n,
                "ABFT finalize reference must be contiguous float32 [2,channels,N] on the same CUDA device");
  }
}

void check_thresholds(const std::optional<at::Tensor>& thresholds,
                      const at::Tensor& prototype, int channels, int64_t n) {
  if (thresholds.has_value()) {
    const at::Tensor& supplied = *thresholds;
    TORCH_CHECK(supplied.is_cuda() && supplied.device() == prototype.device() &&
                    supplied.layout() == at::kStrided && supplied.is_contiguous() &&
                    supplied.scalar_type() == at::kFloat &&
                    ((supplied.dim() == 1 && supplied.size(0) == 1) ||
                     (supplied.dim() == 2 && supplied.size(0) == channels && supplied.size(1) == n)),
                "ABFT finalize thresholds must be contiguous CUDA float32 [1] or [channels,N] "
                "on the same CUDA device");
    // Value validity is checked by the device policy. No host read or extra
    // reduction kernel is needed for negative/nonfinite dynamic thresholds.
  }
}

template<class Policy, bool PackedTileFlags>
FinalizeOutput launch(
    const at::Tensor& prototype, const float* observed, const float* predicted,
    const float* magnitude, const int32_t* tile_flags, int64_t flag_count,
    const int32_t* input_hit, const int32_t* input_nonfinite,
    int64_t n, int64_t m_tiles, int64_t padded_n, double atol, double rtol,
    const std::optional<at::Tensor>& reference, bool diagnostics,
    const std::optional<at::Tensor>& thresholds) {
  constexpr int channels = Policy::kChecksums;
  static_assert(channels > 0 && channels <= kParallelMaxBlocks,
                "Finalizer policy must have a positive supported checksum count");
  const c10::cuda::CUDAGuard guard(prototype.device());
  at::Tensor output_reference = reference.has_value() ? *reference : at::empty({2, channels, n}, prototype.options());
  // Metadata-only overlap checking also covers a supplied read-only reference:
  // thresholds and first-call reference must remain independently frozen.
  if (thresholds.has_value()) at::assert_no_overlap(*thresholds, output_reference);
  const float* threshold_values = thresholds.has_value() ? thresholds->data_ptr<float>() : nullptr;
  const bool scalar_threshold = thresholds.has_value() && thresholds->numel() == 1;
  at::Tensor residual = diagnostics ? at::empty({channels, n}, prototype.options()) : at::empty({0}, prototype.options());
  at::Tensor threshold = diagnostics ? at::empty({channels, n}, prototype.options()) : at::empty({0}, prototype.options());
  at::Tensor flags = at::empty({5}, prototype.options().dtype(at::kBool));
  at::Tensor hit = at::empty({1}, prototype.options().dtype(at::kInt));
  at::Tensor location = at::empty({2}, prototype.options().dtype(at::kLong));
  const auto stream = at::cuda::getCurrentCUDAStream(prototype.get_device());
  if (n * m_tiles >= kParallelMinWork) {
    const int64_t column_blocks = (n + kParallelThreads - 1) / kParallelThreads;
    const int blocks_x = static_cast<int>(std::min<int64_t>(column_blocks, kParallelMaxBlocks / channels));
    const int blocks = channels * blocks_x;
    // Per-call scratch belongs to the current stream; no shared mutable cache.
    at::Tensor block_states = at::empty({blocks}, prototype.options().dtype(at::kInt));
    finalize_kernel<Policy, true, PackedTileFlags><<<dim3(blocks_x, channels), kParallelThreads, 0, stream.stream()>>>(
        observed, predicted, magnitude, tile_flags, flag_count, input_hit, input_nonfinite,
        n, m_tiles, padded_n, static_cast<float>(atol), static_cast<float>(rtol), reference.has_value(), diagnostics,
        threshold_values, scalar_threshold,
        output_reference.data_ptr<float>(), residual.data_ptr<float>(), threshold.data_ptr<float>(),
        flags.data_ptr<bool>(), hit.data_ptr<int32_t>(), location.data_ptr<int64_t>(),
        block_states.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finalize_flags_kernel<<<1, kThreads, 0, stream.stream()>>>(
        block_states.data_ptr<int32_t>(), blocks,
        flags.data_ptr<bool>(), hit.data_ptr<int32_t>(), location.data_ptr<int64_t>());
  } else {
    finalize_kernel<Policy, false, PackedTileFlags><<<1, kThreads, 0, stream.stream()>>>(
      observed, predicted, magnitude, tile_flags, flag_count, input_hit, input_nonfinite,
      n, m_tiles, padded_n, static_cast<float>(atol), static_cast<float>(rtol), reference.has_value(), diagnostics,
      threshold_values, scalar_threshold,
      output_reference.data_ptr<float>(), residual.data_ptr<float>(), threshold.data_ptr<float>(),
      flags.data_ptr<bool>(), hit.data_ptr<int32_t>(), location.data_ptr<int64_t>(), nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_reference, residual, threshold, flags, hit, location};
}

}  // namespace

FinalizeOutput sdc_cutlass_abft_finalize(
    const at::Tensor& partials, const at::Tensor& tile_flags, int64_t n,
    double atol, double rtol, const std::optional<at::Tensor>& reference,
    bool diagnostics, int64_t algorithm, const std::optional<at::Tensor>& thresholds) {
  return sdc_abft::policy::dispatch(algorithm, [&](auto tag) {
    using Policy = typename decltype(tag)::Type;
    constexpr int channels = Policy::kChecksums;
    check_common(n, atol, rtol);
    TORCH_CHECK(partials.is_cuda() && tile_flags.is_cuda() &&
                    partials.device() == tile_flags.device(),
                "ABFT finalize tensors must be on the same CUDA device");
    TORCH_CHECK(partials.layout() == at::kStrided && partials.is_contiguous() &&
                    partials.scalar_type() == at::kFloat && partials.dim() == 4 &&
                    partials.size(0) == 3 && partials.size(1) == channels && partials.size(2) > 0,
                "ABFT finalize partials must be contiguous float32 [3,channels,T,padded_N] with T>0");
    const int64_t n_tiles = (n + kTileColumns - 1) / kTileColumns;
    const int64_t padded_n = n_tiles * kTileColumns;
    const int64_t m_tiles = partials.size(2);
    TORCH_CHECK(partials.size(3) == padded_n,
                "ABFT finalize padded_N must equal ceil(N/128)*128");
    TORCH_CHECK(tile_flags.layout() == at::kStrided && tile_flags.is_contiguous() &&
                    tile_flags.scalar_type() == at::kInt && tile_flags.dim() == 2 &&
                    tile_flags.size(0) == m_tiles && tile_flags.size(1) == n_tiles,
                "ABFT finalize tile_flags must be contiguous int32 [T,ceil(N/128)]");
    check_reference(reference, partials, channels, n);
    check_thresholds(thresholds, partials, channels, n);
    const int64_t field_stride = channels * m_tiles * padded_n;
    const float* observed = partials.data_ptr<float>();
    return launch<Policy, true>(partials, observed, observed + field_stride, observed + 2 * field_stride,
        tile_flags.data_ptr<int32_t>(), m_tiles * n_tiles, nullptr, nullptr,
        n, m_tiles, padded_n, atol, rtol, reference, diagnostics, thresholds);
  });
}

FinalizeOutput sdc_cutlass_abft_finalize_reference(
    const at::Tensor& observed, const at::Tensor& predicted, const at::Tensor& magnitude,
    const at::Tensor& hit, const at::Tensor& nonfinite, int64_t n,
    double atol, double rtol, const std::optional<at::Tensor>& reference,
    bool diagnostics, int64_t algorithm, const std::optional<at::Tensor>& thresholds) {
  return sdc_abft::policy::dispatch(algorithm, [&](auto tag) {
    using Policy = typename decltype(tag)::Type;
    constexpr int channels = Policy::kChecksums;
    check_common(n, atol, rtol);
    TORCH_CHECK(observed.is_cuda(), "ABFT finalize observed must be a CUDA tensor");
    for (const auto* input : {&predicted, &magnitude, &hit, &nonfinite}) {
      TORCH_CHECK(input->is_cuda() && input->device() == observed.device(),
                  "ABFT finalize inputs must be on the same CUDA device");
    }
    TORCH_CHECK(observed.layout() == at::kStrided && observed.is_contiguous() &&
                    observed.scalar_type() == at::kFloat && observed.dim() == 3 &&
                    observed.size(0) == channels && observed.size(1) > 0,
                "ABFT finalize observed must be contiguous float32 [channels,T,padded_N] with T>0");
    for (const auto* input : {&predicted, &magnitude}) {
      TORCH_CHECK(input->layout() == at::kStrided && input->is_contiguous() &&
                      input->scalar_type() == at::kFloat && input->sizes() == observed.sizes(),
                  "ABFT finalize predicted/magnitude must match observed's contiguous float32 layout and shape");
    }
    const int64_t padded_n = observed.size(2);
    const int64_t padding64 = ((n + 63) / 64) * 64;
    const int64_t padding128 = ((n + 127) / 128) * 128;
    TORCH_CHECK(padded_n == padding64 || padded_n == padding128,
                "ABFT finalize padded_N must equal ceil(N/64)*64 or ceil(N/128)*128");
    for (const auto* flag : {&hit, &nonfinite}) {
      TORCH_CHECK(flag->layout() == at::kStrided && flag->is_contiguous() &&
                      flag->scalar_type() == at::kInt && flag->dim() == 1 && flag->size(0) == 1,
                  "ABFT finalize hit/nonfinite must be contiguous CUDA int32 [1] tensors");
    }
    check_reference(reference, observed, channels, n);
    check_thresholds(thresholds, observed, channels, n);
    return launch<Policy, false>(observed, observed.data_ptr<float>(), predicted.data_ptr<float>(),
        magnitude.data_ptr<float>(), nullptr, 0, hit.data_ptr<int32_t>(), nonfinite.data_ptr<int32_t>(),
        n, observed.size(1), padded_n, atol, rtol, reference, diagnostics, thresholds);
  });
}
