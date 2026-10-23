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
// Official SM120 E4M3 data path with checksums consuming its ready shared tiles.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include <tuple>
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "abft/common/sm120_epilogue.cuh"
#include "abft/fp8/fp8_mainloop.cuh"
#include "abft_gemm_universal.hpp"

namespace {
using namespace cute;
template<class Policy>
struct KernelTypes {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  static constexpr int Alignment = 16;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using TileShape = Shape<_128,_64,_64>;
  using ClusterShape = Shape<_1,_1,_1>;

  using BaselineEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using Projection = sdc_abft::ProjectionEpilogue<TileShape, ElementD, Policy>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      typename Projection::Visitor>::CollectiveOp;
  static_assert(sizeof(typename CollectiveEpilogue::SharedStorage) ==
                sizeof(typename BaselineEpilogue::SharedStorage),
                "ABFT epilogue must preserve official mainloop stage carveout");

  using OfficialMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, Alignment,
      ElementB, LayoutB, Alignment,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveMainloop = sdc_abft::Fp8ReferenceMainloop<OfficialMainloop, Policy>;
  using BaselineKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>, OfficialMainloop, CollectiveEpilogue>;
  using GemmKernel = cutlass::gemm::kernel::AbftGemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  static_assert(CollectiveMainloop::DispatchPolicy::Stages == OfficialMainloop::DispatchPolicy::Stages);
  static_assert(sizeof(typename GemmKernel::SharedStorage) <=
                cutlass::gemm::collective::detail::sm120_smem_capacity_bytes,
                "Checksum scratch must fit without changing the official pipeline stages");
};
void check_status(cutlass::Status status, const char* operation) {
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS FP8 ", operation, " failed: ", cutlassGetStatusString(status));
}
}  // namespace

template<class Policy>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fp8_reference(const at::Tensor& a, const at::Tensor& b,
                       const at::Tensor& alpha, const at::Tensor& weights,
                       int64_t row, int64_t col, int64_t bit,
                       const std::optional<at::Tensor>& nonfinite) {
  using Types = KernelTypes<Policy>;
  using Gemm = typename Types::Gemm;
  using GemmKernel = typename Types::GemmKernel;
  using Projection = typename Types::Projection;
  using ElementA = typename Types::ElementA;
  using ElementB = typename Types::ElementB;
  using CollectiveMainloop = typename Types::CollectiveMainloop;
  using OfficialMainloop = typename Types::OfficialMainloop;
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device(),
              "A and B must be on the same CUDA device");
  TORCH_CHECK(a.layout() == at::kStrided && b.layout() == at::kStrided &&
                  a.dim() == 2 && b.dim() == 2 && a.is_contiguous() && b.is_contiguous(),
              "A and B_rows must be contiguous row-major matrices");
  TORCH_CHECK(a.scalar_type() == at::ScalarType::Float8_e4m3fn &&
                  b.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "Official FP8 GEMM requires E4M3FN operands");
  TORCH_CHECK(a.size(1) == b.size(1), "A and B_rows K dimensions must match");
  for (auto dimension : {a.size(0), b.size(0), a.size(1)}) {
    TORCH_CHECK(dimension > 0 && dimension <= std::numeric_limits<int>::max() - 128,
                "Matrix dimensions must be positive and allow int32 tile padding");
  }
  TORCH_CHECK(alpha.is_cuda() && alpha.device() == a.device() &&
                  alpha.layout() == at::kStrided && alpha.scalar_type() == at::kFloat &&
                  alpha.dim() == 0,
              "alpha must be a CUDA FP32 scalar tensor on the operand device");
  const c10::cuda::CUDAGuard guard(a.device());
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major == 12,
              "Official FP8 GEMM requires SM120/SM121");
  const int m = a.size(0), n = b.size(0), k = a.size(1);
  if (nonfinite) {
    at::assert_no_overlap(*nonfinite, b);
    at::assert_no_overlap(*nonfinite, alpha);
  }
  sdc_abft::check_projection_arguments<Policy>(a, weights, m, n, row, col, bit, nonfinite);
  const int64_t tiles_m = (int64_t(m) + 127) / 128;
  const int64_t padded_n = ((int64_t(n) + 63) / 64) * 64;
  at::Tensor partial = at::zeros({Policy::kChecksums, tiles_m, padded_n}, a.options().dtype(at::kFloat));
  at::Tensor predicted = at::zeros_like(partial);
  at::Tensor magnitude = at::zeros_like(partial);
  auto hit = at::zeros({1}, a.options().dtype(at::kInt));
  // Preserve the official column-major FP32 epilogue layout.
  auto output = at::empty_strided({m, n}, {1, m}, a.options().dtype(at::kFloat));
  auto stride_a = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {n, k, 1});
  auto stride_c = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {m, n, 1});
  auto stride_d = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {m, n, 1});
  typename CollectiveMainloop::Arguments mainloop_arguments;
  static_cast<typename OfficialMainloop::Arguments&>(mainloop_arguments) = {
      reinterpret_cast<const ElementA*>(a.const_data_ptr()), stride_a,
      reinterpret_cast<const ElementB*>(b.const_data_ptr()), stride_b};
  mainloop_arguments.reference_weights = weights.const_data_ptr<float>();
  mainloop_arguments.reference_alpha = alpha.const_data_ptr<float>();
  mainloop_arguments.reference_partial = predicted.data_ptr<float>();
  mainloop_arguments.reference_magnitude = magnitude.data_ptr<float>();
  int shared_limit = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit,
      cudaDevAttrMaxSharedMemoryPerBlockOptin, a.get_device()));
  TORCH_CHECK(sizeof(typename GemmKernel::SharedStorage) <= size_t(shared_limit),
              "Fused checksum storage exceeds this GPU's shared-memory capacity");
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
      mainloop_arguments,
      {Projection::arguments(alpha.const_data_ptr<float>(), weights.const_data_ptr<float>(),
                             m, partial.data_ptr<float>(), tiles_m * padded_n,
                             hit.data_ptr<int>(), row, col, bit,
                             nonfinite ? nonfinite->data_ptr<int>() : nullptr),
       output.data_ptr<float>(), stride_c,
       output.data_ptr<float>(), stride_d}};
  Gemm gemm;
  check_status(gemm.can_implement(arguments), "can_implement");
  at::Tensor workspace = at::empty({static_cast<int64_t>(Gemm::get_workspace_size(arguments))},
                             a.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
  check_status(gemm.initialize(arguments, workspace.mutable_data_ptr(), stream.stream()), "initialize");
  check_status(gemm.run(stream.stream()), "run");
  return {output, partial, hit, predicted, magnitude};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
sdc_cutlass_abft_fp8_reference_mm(const at::Tensor& a, const at::Tensor& b,
                       const at::Tensor& alpha, const at::Tensor& weights,
                       int64_t row, int64_t col, int64_t bit, int64_t algorithm,
                       const std::optional<at::Tensor>& nonfinite) {
  return sdc_abft::policy::dispatch(algorithm, [&](auto tag) {
    using Algorithm = typename decltype(tag)::Type;
    return fp8_reference<Algorithm>(a, b, alpha, weights, row, col, bit, nonfinite);
  });
}
