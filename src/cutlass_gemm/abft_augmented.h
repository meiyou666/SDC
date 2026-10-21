// BF16 augmented ABFT using the pinned official CUTLASS SM80 multistage MMA.
// Per CTA, retain all 128 data rows and append two checksum output rows as
// additional official MMA fragments. There is no 130-row hardware instruction.
// Official sources at f3fde58372d33e9a5650ba7b80fc48b3b49d40c8:
// include/cutlass/gemm/threadblock/mma_multistage.h (main MMA and pipeline)
// include/cute/arch/mma_sm80.hpp (BF16 m16n8k16 encoding MMA)
// include/cute/atom/mma_traits_sm80.hpp (official lane/register layouts).
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/numeric_conversion.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm80.hpp"

namespace sdc_abft {

static constexpr int kAugmentedDataRows = 128;

struct AugmentedParams {
  const cutlass::bfloat16_t* weights;
  float* observed;
  float* predicted;
  float* magnitude;
  int* hit;
  int m_tiles;
  int padded_n;
  int rows;
  int fault_row;
  int fault_col;
  int fault_bit;
};

// Encoding is the matrix product W[2,128] * A[128,64], performed by official
// CuTe BF16 Tensor Core MMA atoms. The input A comes exclusively from the
// original mainloop's shared-memory stage. The two checksum rows are cast to
// BF16 and propagated through the same shared B stage by additional official
// BF16 MMA fragments. Data and checksum both use FP32 accumulators. The original
// 128x128 data MMA, its A/B storage, and its three-stage schedule are unchanged.
template<class Base>
struct AugmentedTileEncoding {
  using Shape = typename Base::Shape;
  using Element = cutlass::bfloat16_t;
  static constexpr int Threads = Base::WarpCount::kCount * 32;
  static_assert(Shape::kM == 128 && Shape::kN == 128 && Shape::kK == 64);
  static_assert(Threads == 128, "Encoding partitions 64 A columns over four warps");
  using ChecksumMma = decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{}));
  using ChecksumFragment = decltype(cute::partition_fragment_C(
      ChecksumMma{}, cute::make_shape(cute::_32{}, cute::_8{})));
  struct Storage { Element weights[2][128]; Element q[2][64]; };
  typename Base::SharedStorage& operands;
  Storage& scratch;
  AugmentedParams projection;
  int columns, inner, tile_m, tile_n, thread, k_tile = 0;
  ChecksumFragment predicted;

  CUTLASS_DEVICE void initialize() {
    const int row_base = tile_m * kAugmentedDataRows;
    for (int index = thread; index < 256; index += Threads) {
      const int p = index / 128, local_row = index % 128;
      const int row = row_base + local_row;
      scratch.weights[p][local_row] =
          row < projection.rows
              ? projection.weights[int64_t(p) * projection.rows + row] : Element(0.f);
    }
    cute::clear(predicted);
    __syncthreads();
  }

  CUTLASS_DEVICE void consume(int stage) {
    if (k_tile >= (inner + Shape::kK - 1) / Shape::kK) return;
    const int valid_k = k_tile == 0 ? (inner - 1) % Shape::kK + 1 : Shape::kK;
    auto shared_a = operands.operand_A_ref();
    const int warp = thread / 32, lane = thread % 32;

    auto mma = cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{});
    auto thr_mma = mma.get_slice(lane);
    // Identity tensors let official CuTe define every register's matrix
    // coordinate; no handwritten Tensor Core register mapping or PTX is used.
    // Transpose W*A to A^T*W^T so one 16x8 atom serves 16 K columns and
    // both checksum rows. This uses eight official MMAs per warp/stage.
    auto a_coordinates = thr_mma.partition_A(cute::make_identity_tensor(
        cute::make_shape(cute::_16{}, cute::_16{})));
    auto w_coordinates = thr_mma.partition_B(cute::make_identity_tensor(
        cute::make_shape(cute::_8{}, cute::_16{})));
    auto q_coordinates = thr_mma.partition_C(cute::make_identity_tensor(
        cute::make_shape(cute::_16{}, cute::_8{})));
    auto a_fragment = mma.make_fragment_A(a_coordinates);
    auto w_fragment = mma.make_fragment_B(w_coordinates);
    auto q_fragment = mma.make_fragment_C(q_coordinates);
    cute::clear(q_fragment);

    CUTLASS_PRAGMA_UNROLL
    for (int reduction = 0; reduction < 128; reduction += 16) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(w_fragment); ++i) {
        const int p = cute::get<0>(w_coordinates(i));
        const int local_row = reduction + cute::get<1>(w_coordinates(i));
        w_fragment(i) = p < 2 ? scratch.weights[p][local_row] : Element(0.f);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(a_fragment); ++i) {
        const int k = warp * 16 + cute::get<0>(a_coordinates(i));
        const int local_row = reduction + cute::get<1>(a_coordinates(i));
        const int row = tile_m * kAugmentedDataRows + local_row;
        a_fragment(i) = row < projection.rows && k < valid_k
            ? shared_a.at({local_row, stage * Shape::kK + k}) : Element(0.f);
      }
      cute::gemm(mma, a_fragment, w_fragment, q_fragment);
    }

    cutlass::NumericConverter<Element, float> convert;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(q_fragment); ++i) {
      const int k = warp * 16 + cute::get<0>(q_coordinates(i));
      const int p = cute::get<1>(q_coordinates(i));
      if (p < 2) {
        scratch.q[p][k] = k < valid_k ? convert(q_fragment(i)) : Element(0.f);
      }
    }
    // The two BF16 rows live beside A, not in its original 128 data rows.
    __syncthreads();

    // Propagate q*B as B^T*q^T. Each warp owns 32 B columns and accumulates
    // their two appended results with the official BF16 m16n8k16 operation.
    // Both operands are read on chip; B is never loaded from global again.
    auto shared_b = operands.operand_B_ref();
    auto b_coordinates = thr_mma.partition_A(cute::make_identity_tensor(
        cute::make_shape(cute::_32{}, cute::_16{})));
    auto encoded_coordinates = thr_mma.partition_B(cute::make_identity_tensor(
        cute::make_shape(cute::_8{}, cute::_16{})));
    // The 32x16 identity partition contains a symbolic M-repeat stride.
    // make_fragment_like cannot order that ScaledBasis stride. Allocate the
    // official VMK fragment shape with a numeric compact register layout;
    // coordinate(i) and fragment(i) still use the same logical VMK index.
    auto b_fragment = cute::make_tensor<Element>(cute::shape(b_coordinates));
    auto encoded_fragment = mma.make_fragment_B(encoded_coordinates);
    CUTLASS_PRAGMA_UNROLL
    for (int reduction = 0; reduction < 64; reduction += 16) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(b_fragment); ++i) {
        const int n = warp * 32 + cute::get<0>(b_coordinates(i));
        const int k = reduction + cute::get<1>(b_coordinates(i));
        b_fragment(i) = k < valid_k && tile_n * 128 + n < columns
            ? shared_b.at({stage * Shape::kK + k, n}) : Element(0.f);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < cute::size(encoded_fragment); ++i) {
        const int p = cute::get<0>(encoded_coordinates(i));
        const int k = reduction + cute::get<1>(encoded_coordinates(i));
        encoded_fragment(i) = p < 2 ? scratch.q[p][k] : Element(0.f);
      }
      cute::gemm(mma, b_fragment, encoded_fragment, predicted);
    }
    // Complete all A/B stage reads and protect q reuse before the unchanged
    // mainloop is allowed to advance/recycle its shared stage.
    __syncthreads();
    ++k_tile;
  }

  CUTLASS_DEVICE void finish() {
    ChecksumMma mma;
    auto coordinates = mma.get_slice(thread % 32).partition_C(cute::make_identity_tensor(
        cute::make_shape(cute::_32{}, cute::_8{})));
    const int64_t plane = int64_t(projection.m_tiles) * projection.padded_n;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(predicted); ++i) {
      const int col = tile_n * 128 + (thread / 32) * 32 + cute::get<0>(coordinates(i));
      const int p = cute::get<1>(coordinates(i));
      if (p < 2 && col < columns) {
        const int64_t index = int64_t(tile_m) * projection.padded_n + col;
        projection.predicted[p * plane + index] = predicted(i);
      }
    }
  }
};
}  // namespace sdc_abft

#include "abft_augmented_mma_multistage.h"

namespace sdc_abft {

template<class Mma> struct AugmentedMma;
template<class S, class A, class SA, cutlass::arch::CacheOperation::Kind CA,
         class B, class SB, cutlass::arch::CacheOperation::Kind CB,
         class C, class L, class P, int Stages,
         cutlass::gemm::SharedMemoryClearOption Clear, class Enable>
struct AugmentedMma<cutlass::gemm::threadblock::MmaMultistage<
    S,A,SA,CA,B,SB,CB,C,L,P,Stages,Clear,Enable>> {
  using Type = cutlass::gemm::threadblock::AugmentedMmaMultistage<
      S,A,SA,CA,B,SB,CB,C,L,P,Stages,Clear,Enable>;
};

template<class Iterator, int Threads>
class AugmentedVisitor {
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
    float magnitude[2][Threads * kColumns];
    int columns[Threads * kColumns];
  };

  CUTLASS_DEVICE AugmentedVisitor(
      AugmentedParams params, SharedStorage& shared, cutlass::MatrixCoord extent,
      int thread, typename Iterator::Params output_params, ElementOutput* output,
      cutlass::MatrixCoord offset)
      : params_(params), shared_(shared), extent_(extent), thread_(thread), offset_(offset),
        iterator_(output_params, output, extent, thread, offset) {
    for (int c = 0; c < kColumns; ++c) {
      sums_[0][c] = sums_[1][c] = 0.f;
      magnitude_[0][c] = magnitude_[1][c] = 0.f;
      columns_[c] = -1;
    }
  }
  CUTLASS_DEVICE void begin_epilogue() {}
  CUTLASS_DEVICE void begin_step(int) { fragment_.clear(); }
  CUTLASS_DEVICE void begin_row(int) {}
  CUTLASS_DEVICE void end_row(int) {}
  CUTLASS_DEVICE void visit(int, int, int column, int frag, AccumulatorFragment const& accum) {
    const auto coord = iterator_.thread_start() + ThreadMap::iteration_offset(frag);
    OutputVector& output = reinterpret_cast<OutputVector*>(&fragment_)[frag];
    cutlass::NumericConverter<ElementOutput, float> convert;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < kElementsPerAccess; ++e) {
      const int row = coord.row(), col = coord.column() + e;
      const int local_row = row - offset_.row();
      const int slot = column * kElementsPerAccess + e;
      columns_[slot] = col;
      float value = accum[e];
      if (col < extent_.column()) {
        if (local_row >= 0 && local_row < kAugmentedDataRows && row < extent_.row()) {
          if (row == params_.fault_row && col == params_.fault_col) {
            value = __uint_as_float(__float_as_uint(value) ^ (1u << params_.fault_bit));
            *params_.hit = 1;
          }
          for (int p = 0; p < 2; ++p) {
            const float term = float(params_.weights[int64_t(p) * params_.rows + row]) * value;
            sums_[p][slot] += term;
            // This is sum |w*C_acc|, not the old input absolute-product bound.
            // BF16 encoding rounding is part of the augmented residual and must
            // be calibrated; this scale alone is not a rigorous rounding bound.
            magnitude_[p][slot] += fabsf(term);
          }
        }
      }
      output[e] = convert(value);
    }
  }
  CUTLASS_DEVICE void end_step(int) {
    // This is the original 128-row data tile. Appended checksum fragments are
    // stored independently by Encoding::finish and never change the C layout.
    iterator_.store(fragment_);
    ++iterator_;
  }
  CUTLASS_DEVICE void end_epilogue() {
    for (int c = 0; c < kColumns; ++c) {
      const int index = thread_ * kColumns + c;
      for (int p = 0; p < 2; ++p) {
        shared_.sums[p][index] = sums_[p][c];
        shared_.magnitude[p][index] = magnitude_[p][c];
      }
      shared_.columns[index] = columns_[c];
    }
    __syncthreads();
    for (int local_col = thread_; local_col < 128; local_col += Threads) {
      const int col = offset_.column() + local_col;
      if (col < extent_.column()) {
        const int64_t index = int64_t(blockIdx.x) * params_.padded_n + col;
        const int64_t plane = int64_t(params_.m_tiles) * params_.padded_n;
        for (int p = 0; p < 2; ++p) {
          float sum = 0.f, magnitude = 0.f;
          for (int i = 0; i < Threads * kColumns; ++i) {
            if (shared_.columns[i] == col) {
              sum += shared_.sums[p][i];
              magnitude += shared_.magnitude[p][i];
            }
          }
          params_.observed[p * plane + index] = sum;
          params_.magnitude[p * plane + index] = magnitude;
        }
      }
    }
  }
 private:
  AugmentedParams params_;
  SharedStorage& shared_;
  cutlass::MatrixCoord extent_;
  int thread_;
  cutlass::MatrixCoord offset_;
  Iterator iterator_;
  typename Iterator::Fragment fragment_;
  float sums_[2][kColumns], magnitude_[2][kColumns];
  int columns_[kColumns];
};

template<class BaseKernel>
struct AugmentedKernel {
  using Mma = typename AugmentedMma<typename BaseKernel::Mma>::Type;
  using OriginalEpilogue = typename BaseKernel::Epilogue;
  using Iterator = typename OriginalEpilogue::OutputTileIterator;
  static constexpr int kThreads = BaseKernel::kThreadCount;
  using Visitor = AugmentedVisitor<Iterator, kThreads>;
  using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
      Visitor, OriginalEpilogue>::Epilogue;
  using Encoding = typename Mma::Encoding;
  using Element = cutlass::bfloat16_t;
  struct Params {
    typename Mma::IteratorA::Params a_layout;
    typename Mma::IteratorB::Params b_layout;
    typename Iterator::Params d_layout;
    const Element* a;
    const Element* b;
    Element* d;
    cutlass::gemm::GemmCoord shape;
    AugmentedParams projection;
  };
  union SharedStorage {
    struct {
      typename Mma::SharedStorage operands;
      typename Encoding::Storage encoding;
    } main_loop;
    struct {
      typename Epilogue::SharedStorage exchange;
      typename Visitor::SharedStorage projection;
    } epilogue;
  };
  CUTLASS_DEVICE void operator()(Params const& params, SharedStorage& shared) {
    const int thread = threadIdx.x;
    const int warp = __shfl_sync(0xffffffff, thread / 32, 0), lane = thread % 32;
    const cutlass::MatrixCoord offset(int(blockIdx.x) * kAugmentedDataRows,
                                     int(blockIdx.y) * 128);
    typename Mma::IteratorA a(params.a_layout, const_cast<Element*>(params.a),
                             params.shape.mk(), thread, {offset.row(), 0});
    typename Mma::IteratorB b(params.b_layout, const_cast<Element*>(params.b), params.shape.kn(),
                             thread, {0, offset.column()});
    Encoding encoding{shared.main_loop.operands, shared.main_loop.encoding,
                      params.projection, params.shape.n(), params.shape.k(),
                      int(blockIdx.x), int(blockIdx.y), thread};
    encoding.initialize();
    Mma mma(shared.main_loop.operands, thread, warp, lane);
    mma.encoding = &encoding;
    typename Mma::FragmentC accum;
    accum.clear();
    mma((params.shape.k() + Mma::Shape::kK - 1) / Mma::Shape::kK, accum, a, b, accum);
    encoding.finish();
    __syncthreads();
    Visitor visitor(params.projection, shared.epilogue.projection, params.shape.mn(),
                    thread, params.d_layout, params.d, offset);
    Epilogue epilogue(shared.epilogue.exchange, thread, warp, lane);
    epilogue(visitor, accum);
  }
};
}  // namespace sdc_abft
