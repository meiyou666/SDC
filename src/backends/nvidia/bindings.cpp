#include <torch/extension.h>
#include "native.h"

TORCH_LIBRARY(sdc, m) {
  m.def("pack(Tensor x, int profile, bool tile_scale, bool stochastic, int seed) -> Tensor[]");
  m.def("decode(Tensor q, Tensor s, Tensor g, int rows, int cols, int profile) -> Tensor");
  m.def("reference(Tensor a, Tensor sa, Tensor ga, Tensor b, Tensor sb, Tensor gb, int m, int n, int k, int profile) -> Tensor");
  m.def("verify(Tensor(a!) c, Tensor expected, float atol, float rtol, bool correct) -> Tensor[]");
  m.def("flip(Tensor(a!) q, int index, int bit, int width, bool enabled) -> Tensor");
  m.def("mask(int m, int n, int row, int col, int bit, bool enabled, Tensor like) -> Tensor");
  m.def("dense(Tensor a, Tensor b, Tensor mask, Tensor(a!) bits, int m, int n, int k, int profile, bool instrumented) -> Tensor");
  m.def("mxfp8(Tensor a, Tensor sa, Tensor ga, Tensor b, Tensor sb, Tensor gb, Tensor mask, Tensor(a!) bits, int m, int n, int k, bool instrumented) -> Tensor");
  m.def("nvfp4(Tensor a, Tensor sa, Tensor ga, Tensor b, Tensor sb, Tensor gb, Tensor mask, Tensor(a!) bits, int m, int n, int k, bool instrumented) -> Tensor");
}
TORCH_LIBRARY_IMPL(sdc, CUDA, m) {
  m.impl("pack", pack_cuda); m.impl("decode", decode_cuda);
  m.impl("reference", reference_cuda); m.impl("verify", verify_cuda);
  m.impl("flip", flip_cuda); m.impl("mask", mask_cuda);
  m.impl("dense", dense_cuda); m.impl("mxfp8", mxfp8_cuda); m.impl("nvfp4", nvfp4_cuda);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
