/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Official SM120 NVFP4 mainloop with fused FP32 epilogue projections.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <tuple>
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "abft_sm120.cuh"

namespace {
using namespace cute;
// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

using BaselineEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
  >::CollectiveOp;

using Projection = sdc_abft::ProjectionEpilogue<ThreadBlockShape, ElementD>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    typename Projection::Visitor>::CollectiveOp;
static_assert(sizeof(typename CollectiveEpilogue::SharedStorage) ==
              sizeof(typename BaselineEpilogue::SharedStorage),
              "ABFT epilogue must preserve official mainloop stage carveout");

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using Config = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
void check_shape(int64_t m, int64_t n, int64_t k) {
  for (auto dimension : {m, n, k}) {
    TORCH_CHECK(dimension > 0 && dimension <= std::numeric_limits<int>::max() - 128,
                "NVFP4 dimensions must be positive and allow int32 tile padding");
  }
  TORCH_CHECK(k % 32 == 0, "NVFP4 K must be a multiple of 32");
  // The official scale layout stores dynamic extents/strides as int32.
  // Bound padded logical extents before constructing that layout.
  const int64_t padded_k = ((k + 127) / 128) * 128;
  for (auto rows : {m, n}) {
    const int64_t padded_rows = ((rows + 127) / 128) * 128;
    TORCH_CHECK(padded_rows <= std::numeric_limits<int>::max() / padded_k,
                "NVFP4 padded operand extents exceed int32 layout bounds");
  }
}
void check_status(cutlass::Status status, const char* operation) {
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS NVFP4 ", operation, " failed: ", cutlassGetStatusString(status));
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
sdc_cutlass_abft_nvfp4_mm(const at::Tensor& a, const at::Tensor& b,
                               const at::Tensor& sfa, const at::Tensor& sfb,
                               int64_t k_arg, const at::Tensor& alpha,
                               const at::Tensor& weights, int64_t row, int64_t col, int64_t bit) {
  for (const auto* tensor : {&a, &b, &sfa, &sfb}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->device() == a.device(),
                "All NVFP4 tensors must be on the same CUDA device");
    TORCH_CHECK(tensor->layout() == at::kStrided && tensor->is_contiguous(),
                "NVFP4 storage tensors must be contiguous");
    TORCH_CHECK(tensor->scalar_type() == at::kByte, "NVFP4 storage tensors must be uint8");
  }
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Packed A and B_rows must be matrices");
  check_shape(a.size(0), b.size(0), k_arg);
  TORCH_CHECK(a.size(1) == k_arg / 2 && b.size(1) == k_arg / 2,
              "Packed operands must contain K/2 bytes per row");
  TORCH_CHECK(sfa.dim() == 1 && sfb.dim() == 1, "Scale buffers must be one-dimensional");
  TORCH_CHECK(alpha.is_cuda() && alpha.device() == a.device() &&
                  alpha.layout() == at::kStrided && alpha.scalar_type() == at::kFloat &&
                  alpha.dim() == 0,
              "alpha must be a CUDA FP32 scalar tensor on the operand device");
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 12,
              "Official NVFP4 GEMM requires SM120/SM121");
  const int m = a.size(0), n = b.size(0), k = k_arg;
  sdc_abft::check_projection_arguments(a, weights, m, n, row, col, bit);
  const int64_t tiles_m = (int64_t(m) + 127) / 128;
  const int64_t padded_n = ((int64_t(n) + 127) / 128) * 128;
  auto partial = at::zeros({2, tiles_m, padded_n}, a.options().dtype(at::kFloat));
  auto hit = at::zeros({1}, a.options().dtype(at::kInt));
  const auto shape = cute::make_shape(m, n, k, 1);
  const auto layout_a = Config::tile_atom_to_shape_SFA(shape);
  const auto layout_b = Config::tile_atom_to_shape_SFB(shape);
  TORCH_CHECK(sfa.numel() == cute::size(cute::filter_zeros(layout_a)) &&
                  sfb.numel() == cute::size(cute::filter_zeros(layout_b)),
              "Scale buffers must match the official CUTLASS scale layout sizes");
  auto output = at::empty({m, n}, a.options().dtype(at::kBFloat16));
  auto stride_a = cutlass::make_cute_packed_stride(GemmKernel::StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(GemmKernel::StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(GemmKernel::StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(GemmKernel::StrideD{}, {m, n, 1});
  auto* ptr_output = reinterpret_cast<ElementD*>(output.mutable_data_ptr());
  Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
      {reinterpret_cast<const ElementA::DataType*>(a.const_data_ptr()), stride_a,
       reinterpret_cast<const ElementB::DataType*>(b.const_data_ptr()), stride_b,
       reinterpret_cast<const ElementA::ScaleFactorType*>(sfa.const_data_ptr()), layout_a,
       reinterpret_cast<const ElementB::ScaleFactorType*>(sfb.const_data_ptr()), layout_b},
      {Projection::arguments(alpha.const_data_ptr<float>(), weights.const_data_ptr<float>(),
                             m, partial.data_ptr<float>(), tiles_m * padded_n,
                             hit.data_ptr<int>(), row, col, bit), ptr_output, stride_c, ptr_output, stride_d}};
  Gemm gemm;
  check_status(gemm.can_implement(arguments), "can_implement");
  auto workspace = at::empty({static_cast<int64_t>(Gemm::get_workspace_size(arguments))},
                             a.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  check_status(gemm.initialize(arguments, workspace.mutable_data_ptr(), stream.stream()), "initialize");
  check_status(gemm.run(stream.stream()), "run");
  return {output, partial, hit};
}
