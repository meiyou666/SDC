#include <torch/extension.h>
#include <torch/library.h>
#include <tuple>
#include <optional>

using OptionalTensor = const std::optional<at::Tensor>&;
using AbftOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
using ReferenceOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
using FinalizeOutput = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

at::Tensor sdc_cutlass_mm(const at::Tensor& a, const at::Tensor& b);
at::Tensor sdc_cutlass_fp8_mm(const at::Tensor&, const at::Tensor&, const at::Tensor&);
at::Tensor sdc_cutlass_nvfp4_mm(const at::Tensor&, const at::Tensor&,
                              const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&);
std::tuple<at::Tensor, at::Tensor, int64_t, int64_t>
sdc_cutlass_nvfp4_scale_layout(int64_t, int64_t, int64_t);
ReferenceOutput sdc_cutlass_abft_augmented_mm(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t);
AbftOutput sdc_cutlass_abft_augmented_compact(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t);
FinalizeOutput sdc_cutlass_abft_finalize(
    const at::Tensor&, const at::Tensor&, int64_t, double, double, OptionalTensor, bool, int64_t, OptionalTensor);
FinalizeOutput sdc_cutlass_abft_finalize_reference(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t, double, double, OptionalTensor, bool, int64_t, OptionalTensor);
AbftOutput sdc_cutlass_abft_mm(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t,
    int64_t, OptionalTensor);
ReferenceOutput sdc_cutlass_abft_reference_mm(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t,
    int64_t, OptionalTensor);
AbftOutput sdc_cutlass_abft_fp8_mm(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t, int64_t, int64_t, int64_t, OptionalTensor);
ReferenceOutput sdc_cutlass_abft_fp8_reference_mm(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    int64_t, int64_t, int64_t, int64_t, OptionalTensor);
AbftOutput sdc_cutlass_abft_nvfp4_mm(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t,
    const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t, OptionalTensor);
ReferenceOutput sdc_cutlass_abft_nvfp4_reference_mm(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t,
    const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t, OptionalTensor);

TORCH_LIBRARY(sdc_cutlass, library) {
  library.def("mm(Tensor a, Tensor b) -> Tensor");
  library.def("fp8_mm(Tensor a, Tensor b_rows, Tensor alpha) -> Tensor");
  library.def("nvfp4_mm(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, int k, Tensor alpha) -> Tensor");
  library.def("abft_mm(Tensor a, Tensor b, Tensor weights, int row, int col, int bit, int algorithm=0, Tensor(a!)? nonfinite=None) -> (Tensor, Tensor, Tensor)");
  library.def("abft_reference_mm(Tensor a, Tensor b, Tensor weights, int row, int col, int bit, int algorithm=0, Tensor(a!)? nonfinite=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  library.def("abft_augmented_mm(Tensor a, Tensor b, Tensor weights, int row, int col, int bit, int algorithm=0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  library.def("abft_augmented_compact(Tensor a, Tensor b, Tensor weights, int row, int col, int bit, int algorithm=0) -> (Tensor, Tensor, Tensor)");
  library.def("abft_finalize(Tensor partials, Tensor tile_flags, int n, float atol, float rtol, Tensor(a)? reference, bool diagnostics, int algorithm=0, Tensor? thresholds=None) -> (Tensor(a), Tensor, Tensor, Tensor, Tensor, Tensor)");
  library.def("abft_finalize_reference(Tensor observed, Tensor predicted, Tensor magnitude, Tensor hit, Tensor nonfinite, int n, float atol, float rtol, Tensor(a)? reference, bool diagnostics, int algorithm=0, Tensor? thresholds=None) -> (Tensor(a), Tensor, Tensor, Tensor, Tensor, Tensor)");
  library.def("abft_fp8_mm(Tensor a, Tensor b_rows, Tensor alpha, Tensor weights, int row, int col, int bit, int algorithm=0, Tensor(a!)? nonfinite=None) -> (Tensor, Tensor, Tensor)");
  library.def("abft_fp8_reference_mm(Tensor a, Tensor b_rows, Tensor alpha, Tensor weights, int row, int col, int bit, int algorithm=0, Tensor(a!)? nonfinite=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  library.def("abft_nvfp4_mm(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, int k, Tensor alpha, Tensor weights, int row, int col, int bit, int algorithm=0, Tensor(a!)? nonfinite=None) -> (Tensor, Tensor, Tensor)");
  library.def("abft_nvfp4_reference_mm(Tensor a_packed, Tensor b_packed, Tensor sfa, Tensor sfb, int k, Tensor alpha, Tensor weights, int row, int col, int bit, int algorithm=0, Tensor(a!)? nonfinite=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  // No tensor input: register the host-only layout helper as a catch-all.
  library.def("nvfp4_scale_layout(int m, int n, int k) -> (Tensor, Tensor, int, int)",
              &sdc_cutlass_nvfp4_scale_layout);
}

TORCH_LIBRARY_IMPL(sdc_cutlass, CUDA, library) {
  library.impl("mm", &sdc_cutlass_mm);
  library.impl("fp8_mm", &sdc_cutlass_fp8_mm);
  library.impl("nvfp4_mm", &sdc_cutlass_nvfp4_mm);
  library.impl("abft_mm", &sdc_cutlass_abft_mm);
  library.impl("abft_reference_mm", &sdc_cutlass_abft_reference_mm);
  library.impl("abft_augmented_mm", &sdc_cutlass_abft_augmented_mm);
  library.impl("abft_augmented_compact", &sdc_cutlass_abft_augmented_compact);
  library.impl("abft_finalize", &sdc_cutlass_abft_finalize);
  library.impl("abft_finalize_reference", &sdc_cutlass_abft_finalize_reference);
  library.impl("abft_fp8_mm", &sdc_cutlass_abft_fp8_mm);
  library.impl("abft_fp8_reference_mm", &sdc_cutlass_abft_fp8_reference_mm);
  library.impl("abft_nvfp4_mm", &sdc_cutlass_abft_nvfp4_mm);
  library.impl("abft_nvfp4_reference_mm", &sdc_cutlass_abft_nvfp4_reference_mm);
}

// Autograd remains in the Python training bridge.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
