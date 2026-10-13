#include "blockscaled.cuh"
at::Tensor nvfp4_cuda(at::Tensor a, at::Tensor sa, at::Tensor ga, at::Tensor b, at::Tensor sb, at::Tensor gb, at::Tensor mask,
    at::Tensor bits, int64_t m, int64_t n, int64_t k, bool instrumented) {
  check_operand(a,sa,ga,m,k,3);check_operand(b,sb,gb,n,k,3);
  check_gemm_storage(a,b,mask,bits,m,n,k,4,instrumented);
  c10::cuda::CUDAGuard guard(a.device());
  using E=cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  return instrumented ? scaled_impl<E,32,true>(a,sa,ga,b,sb,gb,mask,bits,m,n,k)
                      : scaled_impl<E,32,false>(a,sa,ga,b,sb,gb,mask,bits,m,n,k);
}
