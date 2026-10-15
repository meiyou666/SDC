/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// Adapted host/device wiring from CUTLASS v4.2.1 example 35
// (commit f3fde58372d33e9a5650ba7b80fc48b3b49d40c8).
// The GEMM Mma and accumulator exchange are the original CUTLASS types.
#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/numeric_conversion.h"

namespace sdc_abft {
struct ProjectionParams {
  const float* weights;
  float* partial;
  int* hit;
  int m_tiles;
  int padded_n;
  int fault_row;
  int fault_col;
  int fault_bit;
};

template <class Iterator, int Threads>
class ProjectionVisitor {
 public:
  using OutputTileIterator = Iterator;
  using ElementOutput = typename Iterator::Element;
  using ElementAccumulator = float;
  using ThreadMap = typename Iterator::ThreadMap;
  static constexpr int kIterations = Iterator::kIterations;
  static constexpr int kElementsPerAccess = Iterator::kElementsPerAccess;
  static constexpr int kColumns = ThreadMap::Iterations::kColumn * kElementsPerAccess;
  using AccumulatorFragment = cutlass::Array<float, kElementsPerAccess>;
  using OutputVector = cutlass::Array<ElementOutput, kElementsPerAccess>;
  struct SharedStorage {
    float sums[2][Threads * kColumns];
    int columns[Threads * kColumns];
  };

  CUTLASS_DEVICE ProjectionVisitor(
      ProjectionParams params, SharedStorage& shared, cutlass::MatrixCoord extent,
      int thread, typename Iterator::Params output_params, ElementOutput* output,
      cutlass::MatrixCoord block_offset)
      : params_(params), shared_(shared), extent_(extent), thread_(thread),
        block_offset_(block_offset), iterator_(output_params, output, extent, thread, block_offset) {
    for (int c = 0; c < kColumns; ++c) {
      sums_[0][c] = 0.0f;
      sums_[1][c] = 0.0f;
      columns_[c] = -1;
    }
  }
  CUTLASS_DEVICE void begin_epilogue() {}
  CUTLASS_DEVICE void begin_step(int) { fragment_.clear(); }
  CUTLASS_DEVICE void begin_row(int) {}
  CUTLASS_DEVICE void end_row(int) {}
  CUTLASS_DEVICE void visit(int, int, int column, int frag, AccumulatorFragment const& accum) {
    auto coord = iterator_.thread_start() + ThreadMap::iteration_offset(frag);
    OutputVector& output = reinterpret_cast<OutputVector*>(&fragment_)[frag];
    cutlass::NumericConverter<ElementOutput, float> convert;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < kElementsPerAccess; ++e) {
      const int row = coord.row();
      const int col = coord.column() + e;
      const int slot = column * kElementsPerAccess + e;
      columns_[slot] = col;
      float value = accum[e];  // Legacy GEMM alpha=1, beta=0, before output cast.
      if (row < extent_.row() && col < extent_.column()) {
        if (row == params_.fault_row && col == params_.fault_col) {
          value = __uint_as_float(__float_as_uint(value) ^ (1u << params_.fault_bit));
          *params_.hit = 1;  // Unique output-element owner; no atomic is required.
        }
        sums_[0][slot] += params_.weights[row] * value;
        sums_[1][slot] += params_.weights[int64_t(extent_.row()) + row] * value;
      }
      output[e] = convert(value);
    }
  }
  CUTLASS_DEVICE void end_step(int) {
    iterator_.store(fragment_);
    ++iterator_;
  }
  CUTLASS_DEVICE void end_epilogue() {
    // One writer per thread-local slot. Fixed thread/slot order below makes
    // the CTA reduction deterministic, including when projections are nonfinite.
    for (int c = 0; c < kColumns; ++c) {
      const int index = thread_ * kColumns + c;
      shared_.sums[0][index] = sums_[0][c];
      shared_.sums[1][index] = sums_[1][c];
      shared_.columns[index] = columns_[c];
    }
    __syncthreads();
    for (int local_col = thread_; local_col < 128; local_col += Threads) {
      const int col = block_offset_.column() + local_col;
      if (col < extent_.column()) {
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int index = 0; index < Threads * kColumns; ++index) {
          if (shared_.columns[index] == col) {
            sum0 += shared_.sums[0][index];
            sum1 += shared_.sums[1][index];
          }
        }
        const int64_t index = int64_t(block_offset_.row() / 128) * params_.padded_n + col;
        params_.partial[index] = sum0;
        params_.partial[int64_t(params_.m_tiles) * params_.padded_n + index] = sum1;
      }
    }
  }
 private:
  ProjectionParams params_;
  SharedStorage& shared_;
  cutlass::MatrixCoord extent_;
  int thread_;
  cutlass::MatrixCoord block_offset_;
  Iterator iterator_;
  typename Iterator::Fragment fragment_;
  float sums_[2][kColumns];
  int columns_[kColumns];
};

template <class BaseKernel>
struct ProjectionKernel {
  using Mma = typename BaseKernel::Mma;
  using OriginalEpilogue = typename BaseKernel::Epilogue;
  using Iterator = typename OriginalEpilogue::OutputTileIterator;
  static constexpr int kThreads = BaseKernel::kThreadCount;
  using Visitor = ProjectionVisitor<Iterator, kThreads>;
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
      Visitor, OriginalEpilogue>::Epilogue;
  using ElementA = typename Mma::IteratorA::Element;
  using ElementB = typename Mma::IteratorB::Element;
  using ElementD = typename Iterator::Element;
  struct Params {
    typename Mma::IteratorA::Params a_layout;
    typename Mma::IteratorB::Params b_layout;
    typename Iterator::Params d_layout;
    const ElementA* a;
    const ElementB* b;
    ElementD* d;
    cutlass::gemm::GemmCoord shape;
    ProjectionParams projection;
  };
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    struct {
      typename Epilogue::SharedStorage exchange;
      typename Visitor::SharedStorage projection;
    } epilogue;
  };
  CUTLASS_DEVICE void operator()(Params const& params, SharedStorage& shared) {
    const int thread = threadIdx.x;
    const int warp = __shfl_sync(0xffffffff, thread / 32, 0);
    const int lane = thread % 32;
    const cutlass::MatrixCoord offset(int(blockIdx.x) * 128, int(blockIdx.y) * 128);
    // CUTLASS legacy input iterators take mutable pointers but only load A/B.
    typename Mma::IteratorA a(params.a_layout, const_cast<ElementA*>(params.a), {params.shape.m(), params.shape.k()},
                             thread, {offset.row(), 0});
    typename Mma::IteratorB b(params.b_layout, const_cast<ElementB*>(params.b), {params.shape.k(), params.shape.n()},
                             thread, {0, offset.column()});
    Mma mma(shared.main_loop, thread, warp, lane);
    typename Mma::FragmentC accum;
    accum.clear();
    mma((params.shape.k() + Mma::Shape::kK - 1) / Mma::Shape::kK, accum, a, b, accum);
    __syncthreads();  // Storage union transitions from mainloop to epilogue.
    Visitor visitor(params.projection, shared.epilogue.projection, params.shape.mn(),
                    thread, params.d_layout, params.d, offset);
    Epilogue epilogue(shared.epilogue.exchange, thread, warp, lane);
    epilogue(visitor, accum);
  }
};
}  // namespace sdc_abft
