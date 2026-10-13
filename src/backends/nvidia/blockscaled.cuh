#pragma once
#include "native.h"
#include "epilogue.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

template<class E, int Alignment, bool Inject> at::Tensor scaled_impl(at::Tensor a, at::Tensor sa, at::Tensor ga,
    at::Tensor b, at::Tensor sb, at::Tensor gb, at::Tensor mask, at::Tensor bits, int m, int n, int k) {
  using namespace cute;
  using Tile=Shape<_128,_128,_128>; using Cluster=Shape<_1,_1,_1>;
  using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassBlockScaledTensorOp,Tile,Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,float,float,
    float,cutlass::layout::RowMajor,4,float,cutlass::layout::RowMajor,4,
    cutlass::epilogue::collective::EpilogueScheduleAuto,typename Fusion<Inject>::Op>::CollectiveOp;
  using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,cutlass::arch::OpClassBlockScaledTensorOp,
    E,cutlass::layout::RowMajor,Alignment,E,cutlass::layout::ColumnMajor,Alignment,float,Tile,Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
  using Gemm=cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Config=typename Mainloop::Sm1xxBlkScaledConfig;
  int mp=pad128(m),np=pad128(n),kp=pad128(k);
  auto shape=make_shape(mp,np,kp,1);
  auto la=Config::tile_atom_to_shape_SFA(shape), lb=Config::tile_atom_to_shape_SFB(shape);
  auto stra=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},make_shape(mp,kp,1));
  auto strb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},make_shape(np,kp,1));
  auto strc=cutlass::make_cute_packed_stride(typename Kernel::StrideC{},make_shape(mp,np,1));
  auto strd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},make_shape(mp,np,1));
  auto out=at::empty({mp,np},a.options().dtype(at::kFloat));
  auto alpha=ga*gb;
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,shape,
    {reinterpret_cast<typename E::DataType const*>(a.data_ptr()),stra,
     reinterpret_cast<typename E::DataType const*>(b.data_ptr()),strb,
     reinterpret_cast<typename E::ScaleFactorType const*>(sa.data_ptr()),la,
     reinterpret_cast<typename E::ScaleFactorType const*>(sb.data_ptr()),lb},
    {{},Inject ? mask.data_ptr<float>() : nullptr,strc,out.data_ptr<float>(),strd}};
  if constexpr(Inject) args.epilogue.thread={{{0.0f},{alpha.data_ptr<float>()}},{{},{},{bits.data_ptr<int64_t>()}},{}};
  else args.epilogue.thread={{{0.0f},{alpha.data_ptr<float>()}},{},{}};
  Gemm op;
  auto workspace=at::empty({int64_t(Gemm::get_workspace_size(args))},a.options());
  TORCH_CHECK(op.can_implement(args)==cutlass::Status::kSuccess,"Blockscaled CUTLASS shape unsupported");
  auto status=op.initialize(args,workspace.data_ptr(),at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==cutlass::Status::kSuccess,"Blockscaled CUTLASS initialization failed: ",int(status));
  status=op.run(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status==cutlass::Status::kSuccess,"Blockscaled CUTLASS launch failed: ",int(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
