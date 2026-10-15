#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include <tuple>
#include <type_traits>
#include "cutlass/device_kernel.h"
#include "abft_legacy_reference.h"

namespace {
using Row = cutlass::layout::RowMajor;
using Col = cutlass::layout::ColumnMajor;

bool is_row(const at::Tensor& tensor) {
  if (tensor.stride(1) == 1 && tensor.stride(0) == tensor.size(1)) return true;
  TORCH_CHECK(tensor.stride(0) == 1 && tensor.stride(1) == tensor.size(0),
              "ABFT operands require packed row-major or column-major storage");
  return false;
}

template <class Element, class OpClass, class LayoutA, class LayoutB>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> launch(
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& weights,
    int fault_row, int fault_col, int fault_bit) {
  // Identical base GEMM configuration; generated mainloop adds shared-stage reference hooks.
  using Default = cutlass::gemm::device::DefaultGemmConfiguration<
      OpClass, cutlass::arch::Sm80, Element, Element, Element, float>;
  using Tile = std::conditional_t<std::is_same_v<Element, cutlass::bfloat16_t>,
      cutlass::gemm::GemmShape<128, 128, 64>, typename Default::ThreadblockShape>;
  using Gemm = cutlass::gemm::device::Gemm<Element, LayoutA, Element, LayoutB,
      Element, Row, float, OpClass, cutlass::arch::Sm80, Tile>;
  using Kernel = sdc_abft::LegacyReferenceKernel<typename Gemm::GemmKernel>;
  static_assert(Tile::kM == 128 && Tile::kN == 128);
  const int m = a.size(0), n = b.size(1), k = a.size(1);
  const int lda = a.stride(std::is_same_v<LayoutA, Row> ? 0 : 1);
  const int ldb = b.stride(std::is_same_v<LayoutB, Row> ? 0 : 1);
  const int m_tiles = (m + 127) / 128, n_tiles = (n + 127) / 128;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(m_tiles <= properties->maxGridSize[0] && n_tiles <= properties->maxGridSize[1],
              "ABFT tile grid exceeds CUDA launch limits");
  auto d = at::empty({m, n}, a.options());
  auto partial = at::zeros({2, m_tiles, n_tiles * 128}, a.options().dtype(at::kFloat));
  auto predicted = at::empty_like(partial);
  auto magnitude = at::empty_like(partial);
  auto hit = at::zeros({1}, a.options().dtype(at::kInt));
  const auto* ptr_a = reinterpret_cast<const Element*>(a.const_data_ptr());
  const auto* ptr_b = reinterpret_cast<const Element*>(b.const_data_ptr());
  auto* ptr_d = reinterpret_cast<Element*>(d.mutable_data_ptr());
  typename Gemm::Arguments original{
      {m, n, k}, {ptr_a, lda}, {ptr_b, ldb}, {ptr_d, n}, {ptr_d, n}, {1.0f, 0.0f}};
  const auto status = Gemm::can_implement(original);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS ABFT can_implement: ",
              cutlassGetStatusString(status));
  typename Kernel::Params params{
      typename Kernel::Mma::IteratorA::Params(LayoutA(lda)),
      typename Kernel::Mma::IteratorB::Params(LayoutB(ldb)),
      typename Kernel::Iterator::Params(Row(n)), ptr_a, ptr_b, ptr_d, {m, n, k},
      {weights.const_data_ptr<float>(), partial.data_ptr<float>(), hit.data_ptr<int>(),
       m_tiles, n_tiles * 128, fault_row, fault_col, fault_bit},
      {predicted.data_ptr<float>(), magnitude.data_ptr<float>()}};
  const size_t shared_bytes = sizeof(typename Kernel::SharedStorage);
  TORCH_CHECK(shared_bytes <= size_t(properties->sharedMemPerBlockOptin),
              "ABFT reference shared storage exceeds device opt-in capacity");
  if (shared_bytes >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
  }
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  cutlass::Kernel<Kernel><<<dim3(m_tiles, n_tiles), Kernel::kThreads,
                            shared_bytes, stream.stream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d, partial, hit, predicted, magnitude};
}

template <class Element, class OpClass>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> dispatch(
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& weights,
    int row, int col, int bit) {
  if (is_row(a)) {
    if (is_row(b)) return launch<Element, OpClass, Row, Row>(a, b, weights, row, col, bit);
    return launch<Element, OpClass, Row, Col>(a, b, weights, row, col, bit);
  }
  if (is_row(b)) return launch<Element, OpClass, Col, Row>(a, b, weights, row, col, bit);
  return launch<Element, OpClass, Col, Col>(a, b, weights, row, col, bit);
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdc_cutlass_abft_reference_mm(
    const at::Tensor& a, const at::Tensor& b, const at::Tensor& weights,
    int64_t row, int64_t col, int64_t bit) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && weights.is_cuda() &&
                  a.device() == b.device() && a.device() == weights.device(),
              "ABFT tensors must be on the same CUDA device");
  TORCH_CHECK(a.layout() == at::kStrided && b.layout() == at::kStrided &&
                  a.dim() == 2 && b.dim() == 2, "ABFT operands must be strided matrices");
  TORCH_CHECK(a.scalar_type() == b.scalar_type() &&
                  (a.scalar_type() == at::kFloat || a.scalar_type() == at::kBFloat16),
              "ABFT legacy GEMM requires matching FP32 or BF16 operands");
  TORCH_CHECK(a.size(1) == b.size(0), "ABFT inner dimensions must match");
  for (const auto* tensor : {&a, &b}) {
    for (int dim = 0; dim < 2; ++dim) {
      TORCH_CHECK(tensor->size(dim) > 0 && tensor->size(dim) <= std::numeric_limits<int>::max() - 128,
                  "ABFT dimensions must be positive and fit padded int32 extents");
      TORCH_CHECK(tensor->stride(dim) > 0 && tensor->stride(dim) <= std::numeric_limits<int>::max(),
                  "ABFT strides must be positive and fit int32");
    }
  }
  TORCH_CHECK(weights.layout() == at::kStrided && weights.scalar_type() == at::kFloat &&
                  weights.dim() == 2 && weights.size(0) == 2 && weights.size(1) == a.size(0) &&
                  weights.is_contiguous(), "ABFT weights must be contiguous FP32 [2,M]");
  TORCH_CHECK((row == -1 && col == -1 && bit == -1) ||
                  (row >= 0 && row < a.size(0) && col >= 0 && col < b.size(1) && bit >= 0 && bit < 32),
              "ABFT fault must be all -1, or a valid global row/column and FP32 bit 0..31");
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8, "ABFT GEMM requires SM80 or newer");
  if (a.scalar_type() == at::kBFloat16) {
    return dispatch<cutlass::bfloat16_t, cutlass::arch::OpClassTensorOp>(a, b, weights, row, col, bit);
  }
  return dispatch<float, cutlass::arch::OpClassSimt>(a, b, weights, row, col, bit);
}
