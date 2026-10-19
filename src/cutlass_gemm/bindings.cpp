#include <torch/extension.h>
#include <torch/library.h>
#include <tuple>

at::Tensor sdc_cutlass_mm(const at::Tensor& a, const at::Tensor& b);
at::Tensor sdc_cutlass_fp8_mm(const at::Tensor&, const at::Tensor&, const at::Tensor&);
at::Tensor sdc_cutlass_nvfp4_mm(const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&);
std::tuple<at::Tensor, at::Tensor, int64_t, int64_t>
sdc_cutlass_nvfp4_scale_layout(int64_t, int64_t, int64_t);

TORCH_LIBRARY(sdc_cutlass, library) {
  library.def("mm(Tensor a, Tensor b) -> Tensor");
  library.def("fp8_mm(Tensor a, Tensor b_rows, Tensor alpha) -> Tensor");
  library.def("nvfp4_mm(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, int k, Tensor alpha) -> Tensor");
  // No tensor input: register the host-only layout helper as a catch-all.
  library.def("nvfp4_scale_layout(int m, int n, int k) -> (Tensor, Tensor, int, int)",
              &sdc_cutlass_nvfp4_scale_layout);
}

TORCH_LIBRARY_IMPL(sdc_cutlass, CUDA, library) {
  library.impl("mm", &sdc_cutlass_mm);
  library.impl("fp8_mm", &sdc_cutlass_fp8_mm);
  library.impl("nvfp4_mm", &sdc_cutlass_nvfp4_mm);
}

// Importing the extension registers torch.ops.sdc_cutlass operators. Autograd is
// supplied by the Python training bridge, not by this raw GEMM operator.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
