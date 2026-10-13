#include <torch/extension.h>
#include <torch/library.h>

at::Tensor sdc_cutlass_mm(const at::Tensor& a, const at::Tensor& b);

TORCH_LIBRARY(sdc_cutlass, library) {
  library.def("mm(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(sdc_cutlass, CUDA, library) {
  library.impl("mm", &sdc_cutlass_mm);
}

// Importing the extension registers torch.ops.sdc_cutlass.mm. Autograd is
// supplied by the Python training bridge, not by this raw GEMM operator.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
