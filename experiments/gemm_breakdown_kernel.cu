// Diagnostic-only runtime ablation of the pinned production BF16 kernel.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>
#include "cutlass/device_kernel.h"
#include "breakdown_augmented.h"

namespace {
using Row = cutlass::layout::RowMajor;
using Element = cutlass::bfloat16_t;
using Gemm = cutlass::gemm::device::Gemm<Element, Row, Element, Row,
    Element, Row, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,64>>;
template<bool FullKN>
using DiagnosticKernel =
    sdc_breakdown_detail::AugmentedKernel<typename Gemm::GemmKernel, true, FullKN>;
using Result = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

template<bool FullKN>
void check_device(const at::Tensor& a) {
  constexpr size_t kSharedBytes = sizeof(typename DiagnosticKernel<FullKN>::SharedStorage);
  TORCH_CHECK(a.is_cuda(), "GEMM breakdown requires a CUDA tensor");
  const auto* properties = at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(properties->major * 10 + properties->minor == SDC_BREAKDOWN_ARCH,
              "GEMM breakdown device does not match the configured CUDA architecture");
  TORCH_CHECK(kSharedBytes <= properties->sharedMemPerBlockOptin,
              "GEMM breakdown shared memory exceeds this GPU's limit");
}

template<bool FullKN>
void configure_kernel() {
  using Kernel = DiagnosticKernel<FullKN>;
  constexpr size_t kSharedBytes = sizeof(typename Kernel::SharedStorage);
  if (kSharedBytes >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes)));
  }
}

template<bool FullKN>
Result run_impl(const at::Tensor& a, const at::Tensor& b, const at::Tensor& weights,
                int64_t mask, double seed) {
  using Kernel = DiagnosticKernel<FullKN>;
  constexpr size_t kSharedBytes = sizeof(typename Kernel::SharedStorage);
  TORCH_CHECK(mask >= 0 && mask <= 15, "GEMM breakdown mask must be in [0,15]");
  TORCH_CHECK(!(mask & 2) || (mask & 8),
              "qB propagation requires the q publication barrier");
  TORCH_CHECK(std::isfinite(seed) && std::abs(seed) <= 0x1.fep127,
              "GEMM breakdown seed must fit finite BF16");
  const float encoded_seed = float(Element(float(seed)));
  TORCH_CHECK(std::isfinite(encoded_seed) && encoded_seed != 0.f,
              "GEMM breakdown seed must remain nonzero and finite in BF16");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && weights.is_cuda() &&
                  a.device() == b.device() && a.device() == weights.device(),
              "GEMM breakdown inputs must share one CUDA device");
  for (const auto* tensor : {&a, &b}) {
    TORCH_CHECK(tensor->layout() == at::kStrided && tensor->dim() == 2 &&
                    tensor->scalar_type() == at::kBFloat16 &&
                    tensor->stride(1) == 1 && tensor->stride(0) == tensor->size(1),
                "GEMM breakdown supports packed row-major BF16 matrices only");
    for (int dim = 0; dim < 2; ++dim) {
      TORCH_CHECK(tensor->size(dim) > 0 &&
                      tensor->size(dim) <= std::numeric_limits<int>::max() - 128,
                  "GEMM breakdown dimensions must be positive padded int32 extents");
    }
  }
  TORCH_CHECK(a.size(1) == b.size(0), "GEMM breakdown inner dimensions must match");
  TORCH_CHECK(weights.layout() == at::kStrided && weights.scalar_type() == at::kBFloat16 &&
                  weights.dim() == 2 && weights.size(0) == 2 &&
                  weights.size(1) == a.size(0) && weights.is_contiguous(),
              "GEMM breakdown weights must be contiguous BF16 [2,M]");
  const c10::cuda::CUDAGuard guard(a.device());
  check_device<FullKN>(a);
  const int m = a.size(0), n = b.size(1), k = a.size(1);
  const int m_tiles = (m + 127) / 128, n_tiles = (n + 127) / 128;
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(m_tiles <= properties->maxGridSize[0] && n_tiles <= properties->maxGridSize[1],
              "GEMM breakdown tile grid exceeds CUDA limits");
  auto d = at::empty({m, n}, a.options());
  auto partials = at::empty({3, 2, m_tiles, n_tiles * 128}, a.options().dtype(at::kFloat));
  auto flags = at::empty({m_tiles, n_tiles}, a.options().dtype(at::kInt));
  at::Tensor sink = at::empty({m_tiles, n_tiles, Kernel::kThreads}, a.options().dtype(at::kFloat));
  const auto* ptr_a = reinterpret_cast<const Element*>(a.const_data_ptr());
  const auto* ptr_b = reinterpret_cast<const Element*>(b.const_data_ptr());
  auto* ptr_d = reinterpret_cast<Element*>(d.mutable_data_ptr());
  typename Gemm::Arguments original{
      {m, n, k}, {ptr_a, k}, {ptr_b, n}, {ptr_d, n}, {ptr_d, n}, {1.f, 0.f}};
  const auto status = Gemm::can_implement(original);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "GEMM breakdown can_implement: ",
              cutlassGetStatusString(status));
  const int64_t field = int64_t(2) * m_tiles * n_tiles * 128;
  auto* evidence = partials.data_ptr<float>();
  typename Kernel::Params params{
      typename Kernel::Mma::IteratorA::Params(Row(k)),
      typename Kernel::Mma::IteratorB::Params(Row(n)),
      typename Kernel::Iterator::Params(Row(n)), ptr_a, ptr_b, ptr_d, {m, n, k},
      {reinterpret_cast<const Element*>(weights.const_data_ptr()),
       evidence, evidence + field, evidence + 2 * field, nullptr,
       m_tiles, n_tiles * 128, m, -1, -1, -1, flags.data_ptr<int>(),
       static_cast<int>(mask), static_cast<float>(seed), sink.data_ptr<float>()}};
  configure_kernel<FullKN>();
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  cutlass::Kernel<Kernel><<<dim3(m_tiles, n_tiles), Kernel::kThreads,
                            kSharedBytes, stream.stream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {d, partials, flags, sink};
}

Result run(const at::Tensor& a, const at::Tensor& b, const at::Tensor& weights,
           int64_t mask, double seed) {
  // Selection depends only on N/K, never on the runtime ablation mask. Keep
  // the same specialization as the production native dispatch for this shape.
  const bool full_kn = a.dim() == 2 && b.dim() == 2 &&
                       a.size(1) % 64 == 0 && b.size(1) % 128 == 0;
  if (full_kn) return run_impl<true>(a, b, weights, mask, seed);
  return run_impl<false>(a, b, weights, mask, seed);
}

template<bool FullKN>
std::vector<int64_t> resources_impl(const at::Tensor& a) {
  using Kernel = DiagnosticKernel<FullKN>;
  constexpr size_t kSharedBytes = sizeof(typename Kernel::SharedStorage);
  TORCH_CHECK(a.is_cuda(), "GEMM breakdown resources requires a CUDA tensor");
  const c10::cuda::CUDAGuard guard(a.device());
  check_device<FullKN>(a);
  configure_kernel<FullKN>();
  cudaFuncAttributes attributes{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, cutlass::Kernel<Kernel>));
  int active_blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, cutlass::Kernel<Kernel>, Kernel::kThreads, kSharedBytes));
  return {attributes.numRegs, static_cast<int64_t>(attributes.localSizeBytes),
          static_cast<int64_t>(kSharedBytes + attributes.sharedSizeBytes),
          attributes.maxThreadsPerBlock, active_blocks,
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount};
}

std::vector<int64_t> resources(const at::Tensor& a, int64_t n) {
  TORCH_CHECK(a.dim() == 2 && a.size(1) > 0 && n > 0 &&
                  n <= std::numeric_limits<int>::max() - 128,
              "GEMM breakdown resources requires a matrix and valid N");
  if (a.size(1) % 64 == 0 && n % 128 == 0) return resources_impl<true>(a);
  return resources_impl<false>(a);
}
}  // namespace

TORCH_LIBRARY(sdc_breakdown, library) {
  library.def("run(Tensor a, Tensor b, Tensor weights, int mask, float seed) -> (Tensor, Tensor, Tensor, Tensor)");
  library.def("resources(Tensor a, int n) -> int[]");
}

TORCH_LIBRARY_IMPL(sdc_breakdown, CUDA, library) {
  library.impl("run", &run);
  library.impl("resources", &resources);
}
