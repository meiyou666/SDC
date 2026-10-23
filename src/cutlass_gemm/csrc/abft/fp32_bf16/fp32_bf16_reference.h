#pragma once
#include "abft/fp32_bf16/fp32_bf16_observe.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"

namespace sdc_abft {
struct LegacyReferenceParams {
  float* predicted;
  float* magnitude;
};

// Read official logical TensorRefs, including their permuted shared layouts.
// Only q/q_abs live in shared memory; column accumulators remain per-thread.
template<class Base, class Algorithm = policy::Classic>
struct LegacyTileReference {
  using Shape = typename Base::Shape;
  static constexpr int Threads = Base::WarpCount::kCount * 32;
  static constexpr int Slots = (Shape::kN + Threads - 1) / Threads;
  static constexpr int kChecksums = Algorithm::kChecksums;
  struct Storage { float q[kChecksums][Shape::kK], q_abs[kChecksums][Shape::kK]; };
  typename Base::SharedStorage& operands;
  Storage& scratch;
  ProjectionParams projection;
  LegacyReferenceParams destination;
  int rows, columns, inner, tile_m, tile_n, thread, k_tile = 0;
  float predicted[kChecksums][Slots] = {};
  float magnitude[kChecksums][Slots] = {};

  CUTLASS_DEVICE void consume(int stage) {
    auto a = operands.operand_A_ref();
    auto b = operands.operand_B_ref();
    // Both official legacy iterators visit the residue tile first, then all
    // remaining K tiles are full. A tail-last mask would truncate a full tile.
    const int valid_k = k_tile == 0 ? (inner - 1) % Shape::kK + 1 : Shape::kK;
    for (int k = thread; k < Shape::kK; k += Threads) {
      float encoded[kChecksums] = {}, absolute[kChecksums] = {};
      // Explicit K mask also protects predicated non-zfill tail stages.
      if (k < valid_k) {
        for (int m = 0; m < Shape::kM; ++m) {
          int row = tile_m * Shape::kM + m;
          if (row < rows) {
            float value = float(a.at({m, stage * Shape::kK + k}));
            float weights[kChecksums];
            CUTLASS_PRAGMA_UNROLL
            for (int p = 0; p < kChecksums; ++p) {
              weights[p] = projection.weights[int64_t(p) * rows + row];
              encoded[p] = Algorithm::encode(weights[p], value, encoded[p]);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int p = 0; p < kChecksums; ++p) {
              absolute[p] = Algorithm::encode(fabsf(weights[p]), fabsf(value), absolute[p]);
            }
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < kChecksums; ++p) scratch.q[p][k] = encoded[p];
      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < kChecksums; ++p) scratch.q_abs[p][k] = absolute[p];
    }
    __syncthreads();
    for (int slot = 0; slot < Slots; ++slot) {
      int n = thread + slot * Threads;
      if (n < Shape::kN && tile_n * Shape::kN + n < columns) {
        for (int k = 0; k < Shape::kK; ++k) {
          if (k < valid_k) {
            float value = float(b.at({stage * Shape::kK + k, n}));
            for (int p = 0; p < kChecksums; ++p) {
              predicted[p][slot] = Algorithm::propagate(scratch.q[p][k], value, predicted[p][slot]);
              magnitude[p][slot] = Algorithm::propagate(scratch.q_abs[p][k], fabsf(value), magnitude[p][slot]);
            }
          }
        }
      }
    }
    // Complete all stage reads and protect scratch reuse before the mainloop resumes.
    __syncthreads();
    ++k_tile;
  }
  CUTLASS_DEVICE void finish() {
    int64_t plane = int64_t(projection.m_tiles) * projection.padded_n;
    for (int slot = 0; slot < Slots; ++slot) {
      int n = thread + slot * Threads;
      if (n < Shape::kN) {
        int col = tile_n * Shape::kN + n;
        int64_t offset = int64_t(tile_m) * projection.padded_n + col;
        for (int p = 0; p < kChecksums; ++p) {
          destination.predicted[p * plane + offset] = predicted[p][slot];
          destination.magnitude[p * plane + offset] = magnitude[p][slot];
        }
      }
    }
  }
};
} // namespace sdc_abft

#include "abft_mma_multistage.h"
#include "abft_mma_pipelined.h"

namespace sdc_abft {
template<class Mma, class Algorithm = policy::Classic> struct ReferenceMma;
template<class S, class A, class SA, cutlass::arch::CacheOperation::Kind CA,
         class B, class SB, cutlass::arch::CacheOperation::Kind CB,
         class C, class L, class P, int Stages,
         cutlass::gemm::SharedMemoryClearOption Clear, class Enable, class Algorithm>
struct ReferenceMma<cutlass::gemm::threadblock::MmaMultistage<
    S,A,SA,CA,B,SB,CB,C,L,P,Stages,Clear,Enable>, Algorithm> {
  using Type = cutlass::gemm::threadblock::AbftMmaMultistage<
      S,A,SA,CA,B,SB,CB,C,L,P,Stages,Clear,Enable,Algorithm>;
};
template<class S, class A, class SA, class B, class SB, class C, class L,
         class P, class TA, class TB, class Enable, class Algorithm>
struct ReferenceMma<cutlass::gemm::threadblock::MmaPipelined<S,A,SA,B,SB,C,L,P,TA,TB,Enable>, Algorithm> {
  using Type = cutlass::gemm::threadblock::AbftMmaPipelined<S,A,SA,B,SB,C,L,P,TA,TB,Enable,Algorithm>;
};

template<class BaseKernel, class Algorithm = policy::Classic>
struct LegacyReferenceKernel {
  using Mma = typename ReferenceMma<typename BaseKernel::Mma, Algorithm>::Type;
  using OriginalEpilogue = typename BaseKernel::Epilogue;
  using Iterator = typename OriginalEpilogue::OutputTileIterator;
  static constexpr int kThreads = BaseKernel::kThreadCount;
  using Visitor = ProjectionVisitor<Iterator, kThreads, Algorithm>;
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
      Visitor, OriginalEpilogue>::Epilogue;
  using Reference = typename Mma::Reference;
  using ElementA = typename Mma::IteratorA::Element;
  using ElementB = typename Mma::IteratorB::Element;
  using ElementD = typename Iterator::Element;
  struct Params {
    typename Mma::IteratorA::Params a_layout;
    typename Mma::IteratorB::Params b_layout;
    typename Iterator::Params d_layout;
    const ElementA* a; const ElementB* b; ElementD* d;
    cutlass::gemm::GemmCoord shape;
    ProjectionParams projection;
    LegacyReferenceParams reference;
  };
  union SharedStorage {
    struct {
      typename Mma::SharedStorage operands;
      typename Reference::Storage reference;
    } main_loop;
    struct {
      typename Epilogue::SharedStorage exchange;
      typename Visitor::SharedStorage projection;
    } epilogue;
  };
  CUTLASS_DEVICE void operator()(Params const& params, SharedStorage& shared) {
    int thread = threadIdx.x;
    int warp = __shfl_sync(0xffffffff, thread / 32, 0), lane = thread % 32;
    cutlass::MatrixCoord offset(int(blockIdx.x) * 128, int(blockIdx.y) * 128);
    typename Mma::IteratorA a(params.a_layout, const_cast<ElementA*>(params.a), params.shape.mk(),
                             thread, {offset.row(), 0});
    typename Mma::IteratorB b(params.b_layout, const_cast<ElementB*>(params.b), params.shape.kn(),
                             thread, {0, offset.column()});
    Reference reference{shared.main_loop.operands, shared.main_loop.reference,
        params.projection, params.reference, params.shape.m(), params.shape.n(), params.shape.k(),
        int(blockIdx.x), int(blockIdx.y), thread};
    Mma mma(shared.main_loop.operands, thread, warp, lane);
    mma.reference = &reference;
    typename Mma::FragmentC accum;
    accum.clear();
    mma((params.shape.k() + Mma::Shape::kK - 1) / Mma::Shape::kK, accum, a, b, accum);
    reference.finish();
    __syncthreads();
    Visitor visitor(params.projection, shared.epilogue.projection, params.shape.mn(),
                    thread, params.d_layout, params.d, offset);
    Epilogue epilogue(shared.epilogue.exchange, thread, warp, lane);
    epilogue(visitor, accum);
  }
};
} // namespace sdc_abft
