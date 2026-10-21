#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include <tuple>
#include <type_traits>
#include "cutlass/device_kernel.h"
#include "abft_augmented.h"

namespace {
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;
using Element = cutlass::bfloat16_t;
using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
using CompactResult = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
template<bool Compact>
using LaunchResult = std::conditional_t<Compact, CompactResult, Result>;

template<bool Compact, class LayoutA, class LayoutB, bool FullKN = false>
LaunchResult<Compact> launch(const at::Tensor& a, const at::Tensor& b,
                            const at::Tensor& weights, int row, int col, int bit) {
  // Same SM80 official BF16 main MMA, warp tile and three-stage schedule as
  // gemm.cu. The appended checksum rows use additional official MMA fragments;
  // the 128 original data rows retain all their input/output tile coordinates.
  using Gemm = cutlass::gemm::device::Gemm<Element, LayoutA, Element, LayoutB,
      Element, Row, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128,128,64>>;
  using Kernel = sdc_abft::AugmentedKernel<typename Gemm::GemmKernel, Compact, FullKN>;
  const int m = a.size(0), n = b.size(1), k = a.size(1);
  const int lda = a.stride(std::is_same_v<LayoutA, Row> ? 0 : 1);
  const int ldb = b.stride(std::is_same_v<LayoutB, Row> ? 0 : 1);
  const int m_tiles = (m + sdc_abft::kAugmentedDataRows - 1) / sdc_abft::kAugmentedDataRows;
  const int n_tiles = (n + 127) / 128;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(m_tiles <= properties->maxGridSize[0] && n_tiles <= properties->maxGridSize[1],
              "Augmented ABFT tile grid exceeds CUDA launch limits");
  auto d = at::empty({m, n}, a.options());
  at::Tensor observed, predicted, magnitude, hit, partials, tile_flags;
  if constexpr (Compact) {
    // Every CTA owns its complete tile slice, including padded N columns.
    // There are no atomics across CTAs and no pre-launch zeroing kernels.
    partials = at::empty({3, 2, m_tiles, n_tiles * 128}, a.options().dtype(at::kFloat));
    observed = partials.select(0, 0);
    predicted = partials.select(0, 1);
    magnitude = partials.select(0, 2);
    tile_flags = at::empty({m_tiles, n_tiles}, a.options().dtype(at::kInt));
  } else {
    observed = at::zeros({2, m_tiles, n_tiles * 128}, a.options().dtype(at::kFloat));
    predicted = at::zeros_like(observed);
    magnitude = at::zeros_like(observed);
    hit = at::zeros({1}, a.options().dtype(at::kInt));
  }
  const auto* ptr_a = reinterpret_cast<const Element*>(a.const_data_ptr());
  const auto* ptr_b = reinterpret_cast<const Element*>(b.const_data_ptr());
  auto* ptr_d = reinterpret_cast<Element*>(d.mutable_data_ptr());
  typename Gemm::Arguments original{
      {m, n, k}, {ptr_a, lda}, {ptr_b, ldb}, {ptr_d, n}, {ptr_d, n}, {1.f, 0.f}};
  const auto status = Gemm::can_implement(original);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS augmented ABFT can_implement: ",
              cutlassGetStatusString(status));
  typename Kernel::Params params{
      typename Kernel::Mma::IteratorA::Params(LayoutA(lda)),
      typename Kernel::Mma::IteratorB::Params(LayoutB(ldb)),
      typename Kernel::Iterator::Params(Row(n)), ptr_a, ptr_b, ptr_d, {m, n, k},
      {reinterpret_cast<const Element*>(weights.const_data_ptr()),
       observed.data_ptr<float>(), predicted.data_ptr<float>(), magnitude.data_ptr<float>(),
       Compact ? nullptr : hit.data_ptr<int>(), m_tiles, n_tiles * 128, m, row, col, bit,
       Compact ? tile_flags.data_ptr<int>() : nullptr}};
  const size_t shared_bytes = sizeof(typename Kernel::SharedStorage);
  TORCH_CHECK(shared_bytes <= properties->sharedMemPerBlockOptin,
              "Augmented ABFT kernel shared memory exceeds this GPU's limit");
  if (shared_bytes >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
  }
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  cutlass::Kernel<Kernel><<<dim3(m_tiles, n_tiles), Kernel::kThreads,
                            shared_bytes, stream.stream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if constexpr (Compact) return {d, partials, tile_flags};
  else return {d, observed, hit, predicted, magnitude};
}

template<bool Compact, class LayoutA, class LayoutB>
LaunchResult<Compact> launch_selected(const at::Tensor& a, const at::Tensor& b,
                                    const at::Tensor& weights, int row, int col, int bit) {
  if constexpr (Compact) {
    // Only remove K/N masks for complete native tiles. M tails continue to
    // use the same masks, checksum weights, and output footprint as before.
    if (b.size(1) % 128 == 0 && a.size(1) % 64 == 0) {
      return launch<Compact, LayoutA, LayoutB, true>(a, b, weights, row, col, bit);
    }
  }
  return launch<Compact, LayoutA, LayoutB, false>(a, b, weights, row, col, bit);
}

template<bool Compact>
LaunchResult<Compact> dispatch(const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& weights, int64_t row, int64_t col, int64_t bit) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && weights.is_cuda() &&
                  a.device() == b.device() && a.device() == weights.device(),
              "Augmented ABFT tensors must be on the same CUDA device");
  TORCH_CHECK(a.layout() == at::kStrided && b.layout() == at::kStrided &&
                  a.dim() == 2 && b.dim() == 2,
              "Augmented ABFT operands must be strided matrices");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
              "Augmented ABFT currently requires BF16 operands");
  TORCH_CHECK(a.size(1) == b.size(0), "Augmented ABFT inner dimensions must match");
  for (const auto* tensor : {&a, &b}) {
    for (int dim = 0; dim < 2; ++dim) {
      TORCH_CHECK(tensor->size(dim) > 0 && tensor->size(dim) <= std::numeric_limits<int>::max() - 128,
                  "Augmented ABFT dimensions must be positive and fit padded int32 extents");
      TORCH_CHECK(tensor->stride(dim) > 0 && tensor->stride(dim) <= std::numeric_limits<int>::max(),
                  "Augmented ABFT strides must be positive and fit int32");
    }
  }
  const bool row_a = a.stride(1) == 1 && a.stride(0) == a.size(1);
  TORCH_CHECK(row_a || (a.stride(0) == 1 && a.stride(1) == a.size(0)),
              "Augmented ABFT A requires packed row-major or column-major storage");
  const bool row_b = b.stride(1) == 1 && b.stride(0) == b.size(1);
  TORCH_CHECK(row_b || (b.stride(0) == 1 && b.stride(1) == b.size(0)),
              "Augmented ABFT B requires packed row-major or column-major storage");
  TORCH_CHECK(weights.layout() == at::kStrided && weights.scalar_type() == at::kBFloat16 &&
                  weights.dim() == 2 && weights.size(0) == 2 && weights.size(1) == a.size(0) &&
                  weights.is_contiguous(), "Augmented ABFT weights must be contiguous BF16 [2,M]");
  TORCH_CHECK((row == -1 && col == -1 && bit == -1) ||
                  (row >= 0 && row < a.size(0) && col >= 0 && col < b.size(1) && bit >= 0 && bit < 32),
              "Augmented ABFT fault must be all -1, or a valid row/column and FP32 bit 0..31");
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8,
              "Augmented ABFT GEMM requires SM80 or newer");
  if (row_a) {
    if (row_b) return launch_selected<Compact, Row, Row>(a, b, weights, row, col, bit);
    return launch_selected<Compact, Row, Col>(a, b, weights, row, col, bit);
  }
  if (row_b) return launch_selected<Compact, Col, Row>(a, b, weights, row, col, bit);
  return launch_selected<Compact, Col, Col>(a, b, weights, row, col, bit);
}
}  // namespace

Result sdc_cutlass_abft_augmented_mm(const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& weights, int64_t row, int64_t col, int64_t bit) {
  return dispatch<false>(a, b, weights, row, col, bit);
}

CompactResult sdc_cutlass_abft_augmented_compact(const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& weights, int64_t row, int64_t col, int64_t bit) {
  return dispatch<true>(a, b, weights, row, col, bit);
}
