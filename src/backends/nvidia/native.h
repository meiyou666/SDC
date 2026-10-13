#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <cmath>

std::vector<at::Tensor> pack_cuda(at::Tensor x, int64_t profile, bool tile_scale, bool stochastic, int64_t seed);
at::Tensor decode_cuda(at::Tensor q, at::Tensor s, at::Tensor g, int64_t rows, int64_t cols, int64_t profile);
at::Tensor reference_cuda(at::Tensor a, at::Tensor sa, at::Tensor ga, at::Tensor b, at::Tensor sb, at::Tensor gb,
                      int64_t m, int64_t n, int64_t k, int64_t profile);
std::vector<at::Tensor> verify_cuda(at::Tensor c, at::Tensor expected, double atol, double rtol, bool correct);
at::Tensor flip_cuda(at::Tensor q, int64_t index, int64_t bit, int64_t width, bool enabled);
at::Tensor mask_cuda(int64_t m, int64_t n, int64_t row, int64_t col, int64_t bit, bool enabled, at::Tensor like);
at::Tensor dense_cuda(at::Tensor a, at::Tensor b, at::Tensor mask, at::Tensor bits, int64_t m, int64_t n, int64_t k,
                  int64_t profile, bool instrumented);
at::Tensor mxfp8_cuda(at::Tensor a, at::Tensor sa, at::Tensor ga, at::Tensor b, at::Tensor sb, at::Tensor gb,
                 at::Tensor mask, at::Tensor bits, int64_t m, int64_t n, int64_t k, bool instrumented);
at::Tensor nvfp4_cuda(at::Tensor a, at::Tensor sa, at::Tensor ga, at::Tensor b, at::Tensor sb, at::Tensor gb,
                 at::Tensor mask, at::Tensor bits, int64_t m, int64_t n, int64_t k, bool instrumented);
inline int pad128(int64_t x) { return int((x + 127) / 128 * 128); }
inline void check_shape(int64_t r, int64_t c) {
  TORCH_CHECK(r>0 && c>0 && r<=std::numeric_limits<int>::max()-127 && c<=std::numeric_limits<int>::max()-127,
              "Invalid matrix dimensions");
  TORCH_CHECK(int64_t(pad128(r))*pad128(c)<=std::numeric_limits<int>::max(),"Matrix exceeds 32-bit indexing");
}
inline void check_storage(at::Tensor x, at::ScalarType type, at::Device device, int64_t elements) {
  TORCH_CHECK(x.is_cuda() && x.device()==device && x.is_contiguous() && x.scalar_type()==type && x.numel()==elements,
              "Invalid native tensor storage");
}
inline void check_operand(at::Tensor q, at::Tensor s, at::Tensor g, int64_t r, int64_t c, int64_t profile) {
  check_shape(r,c);
  TORCH_CHECK(profile>=0 && profile<=3,"Unknown precision");
  int width=profile==0 ? 32 : profile==1 ? 16 : profile==2 ? 8 : 4;
  int64_t area=int64_t(pad128(r))*pad128(c);
  check_storage(q,at::kByte,q.device(),area*width/8);
  check_storage(s,at::kByte,q.device(),profile<2 ? 0 : area/(profile==2 ? 32 : 16));
  check_storage(g,at::kFloat,q.device(),1);
}
inline void check_gemm_storage(at::Tensor a, at::Tensor b, at::Tensor mask, at::Tensor bits,
    int64_t m,int64_t n,int64_t k,int width,bool instrumented) {
  check_shape(m,k);check_shape(n,k);check_shape(m,n);
  check_storage(a,at::kByte,a.device(),int64_t(pad128(m))*pad128(k)*width/8);
  check_storage(b,at::kByte,a.device(),int64_t(pad128(n))*pad128(k)*width/8);
  check_storage(bits,at::kLong,a.device(),2);
  if(instrumented) check_storage(mask,at::kFloat,a.device(),int64_t(pad128(m))*pad128(n));
}
