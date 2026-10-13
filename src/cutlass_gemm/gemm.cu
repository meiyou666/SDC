#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

namespace {

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

void check_status(cutlass::Status status, const char* operation) {
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS ", operation, " failed: ", cutlassGetStatusString(status));
}

bool row_major(const at::Tensor& tensor, const char* name) {
  if (tensor.stride(1) == 1 && tensor.stride(0) == tensor.size(1)) {
    return true;
  }
  if (tensor.stride(0) == 1 && tensor.stride(1) == tensor.size(0)) {
    return false;
  }
  TORCH_CHECK(false, name,
              " must be packed row-major or a simple transpose of a packed matrix; got strides ",
              tensor.strides());
}

template <typename Element, typename OperatorClass, typename LayoutA, typename LayoutB>
at::Tensor launch(const at::Tensor& a, const at::Tensor& b, bool a_row, bool b_row) {
  // Official CUTLASS v4.2.1 APIs:
  // include/cutlass/gemm/device/gemm.h and default_gemm_configuration.h.
  // Omitting all parameters after ArchTag deliberately preserves the official
  // DefaultGemmConfiguration tile, instruction, mainloop and epilogue defaults.
  using Gemm = cutlass::gemm::device::Gemm<
      Element, LayoutA, Element, LayoutB, Element, RowMajor,
      float, OperatorClass, cutlass::arch::Sm80>;

  const int m = static_cast<int>(a.size(0));
  const int n = static_cast<int>(b.size(1));
  const int k = static_cast<int>(a.size(1));
  const int lda = static_cast<int>(a.stride(a_row ? 0 : 1));
  const int ldb = static_cast<int>(b.stride(b_row ? 0 : 1));
  auto output = at::empty({m, n}, a.options());
  const auto* ptr_a = reinterpret_cast<const Element*>(a.const_data_ptr());
  const auto* ptr_b = reinterpret_cast<const Element*>(b.const_data_ptr());
  auto* ptr_output = reinterpret_cast<Element*>(output.mutable_data_ptr());

  // beta=0 makes the official epilogue independent of the C contents.
  typename Gemm::Arguments arguments{
      {m, n, k},
      {ptr_a, lda},
      {ptr_b, ldb},
      {ptr_output, n},
      {ptr_output, n},
      {1.0f, 0.0f}};
  check_status(Gemm::can_implement(arguments), "can_implement");
  auto workspace = at::empty(
      {static_cast<int64_t>(Gemm::get_workspace_size(arguments))},
      a.options().dtype(at::kByte));
  const auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  Gemm gemm;
  check_status(gemm.initialize(arguments, workspace.mutable_data_ptr(), stream.stream()),
               "initialize");
  check_status(gemm.run(stream.stream()), "run");
  return output;
}

template <typename Element, typename OperatorClass>
at::Tensor dispatch_layout(const at::Tensor& a, const at::Tensor& b,
                           bool a_row, bool b_row) {
  if (a_row && b_row) {
    return launch<Element, OperatorClass, RowMajor, RowMajor>(a, b, a_row, b_row);
  }
  if (a_row) {
    return launch<Element, OperatorClass, RowMajor, ColumnMajor>(a, b, a_row, b_row);
  }
  if (b_row) {
    return launch<Element, OperatorClass, ColumnMajor, RowMajor>(a, b, a_row, b_row);
  }
  return launch<Element, OperatorClass, ColumnMajor, ColumnMajor>(a, b, a_row, b_row);
}

}  // namespace

at::Tensor sdc_cutlass_mm(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(a.device() == b.device(), "A and B must be on the same CUDA device");
  TORCH_CHECK(a.layout() == at::kStrided && b.layout() == at::kStrided,
              "A and B must be strided tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "A and B must be two-dimensional");
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "A and B must have the same dtype");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kFloat,
              "Only BF16 and FP32 inputs are supported");
  TORCH_CHECK(a.size(1) == b.size(0), "A and B inner dimensions must match");
  for (const auto* tensor : {&a, &b}) {
    for (int dim = 0; dim < 2; ++dim) {
      TORCH_CHECK(tensor->size(dim) > 0 &&
                      tensor->size(dim) <= std::numeric_limits<int>::max(),
                  "Matrix dimensions must be positive and fit int32");
      TORCH_CHECK(tensor->stride(dim) > 0 &&
                      tensor->stride(dim) <= std::numeric_limits<int>::max(),
                  "Matrix strides must be positive and fit int32");
    }
  }
  const bool a_row = row_major(a, "A");
  const bool b_row = row_major(b, "B");
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 8,
              "This CUTLASS baseline requires an SM80 or newer GPU");
  if (a.scalar_type() == at::kBFloat16) {
    return dispatch_layout<cutlass::bfloat16_t, cutlass::arch::OpClassTensorOp>(
        a, b, a_row, b_row);
  }
  return dispatch_layout<float, cutlass::arch::OpClassSimt>(a, b, a_row, b_row);
}
