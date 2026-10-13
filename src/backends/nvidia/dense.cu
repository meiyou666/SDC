#include "native.h"
#include "epilogue.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cutlass/gemm/device/gemm.h"

template<bool BFloat, bool Inject> at::Tensor dense_impl(at::Tensor a, at::Tensor b, at::Tensor mask, at::Tensor bits,
    int m, int n, int k) {
  using E = std::conditional_t<BFloat,cutlass::bfloat16_t,float>;
  using Op = std::conditional_t<BFloat,cutlass::arch::OpClassTensorOp,cutlass::arch::OpClassSimt>;
  using Inst = std::conditional_t<BFloat,cutlass::gemm::GemmShape<16,8,16>,cutlass::gemm::GemmShape<1,1,1>>;
  using Block = std::conditional_t<BFloat,cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<128,128,8>>;
  using Warp = std::conditional_t<BFloat,cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<32,64,8>>;
  using Gemm = cutlass::gemm::device::Gemm<E,cutlass::layout::RowMajor,E,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,float,Op,cutlass::arch::Sm80,Block,Warp,Inst,DenseOutput<Inject,BFloat ? 4 : 1>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, BFloat ? 3 : 2, BFloat ? 8 : 1, BFloat ? 8 : 1>;
  int mp=pad128(m), np=pad128(n), kp=pad128(k);
  auto out = at::empty({mp,np},a.options().dtype(at::kFloat));
  typename Gemm::Arguments args({mp,np,kp},
    {reinterpret_cast<E const*>(a.data_ptr()),kp}, {reinterpret_cast<E const*>(b.data_ptr()),kp},
    {Inject ? mask.data_ptr<float>() : nullptr,np}, {out.data_ptr<float>(),np}, {bits.data_ptr<int64_t>()});
  Gemm op;
  TORCH_CHECK(op.can_implement(args)==cutlass::Status::kSuccess,"Dense CUTLASS shape unsupported");
  auto status=op(args,nullptr,at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==cutlass::Status::kSuccess,"Dense CUTLASS launch failed: ",int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
at::Tensor dense_cuda(at::Tensor a, at::Tensor b, at::Tensor mask, at::Tensor bits, int64_t m, int64_t n, int64_t k,
    int64_t profile, bool instrumented) {
  TORCH_CHECK(profile==0 || profile==1,"Expected FP32 or BF16");
  check_gemm_storage(a,b,mask,bits,m,n,k,profile==0 ? 32 : 16,instrumented);
  c10::cuda::CUDAGuard guard(a.device());
  if(profile==0) return instrumented ? dense_impl<false,true>(a,b,mask,bits,m,n,k) : dense_impl<false,false>(a,b,mask,bits,m,n,k);
  TORCH_CHECK(profile==1,"Expected FP32 or BF16");
  return instrumented ? dense_impl<true,true>(a,b,mask,bits,m,n,k) : dense_impl<true,false>(a,b,mask,bits,m,n,k);
}
