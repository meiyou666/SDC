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
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm75.hpp"

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
  // Compact calls own one fully-written status word per CTA. The legacy ABI
  // retains its separately initialized single hit tensor.
  int* tile_flags = nullptr;
};

// Encoding is the matrix product W[2,128] * A[128,64], performed by official
// CuTe BF16 Tensor Core MMA atoms. The input A comes exclusively from the
// original mainloop's shared-memory stage. The two checksum rows are cast to
// BF16 and propagated by additional official BF16 MMA fragments. Compact mode
// reuses the original main MMA's transformed B registers; the reference path
// reads the same shared B stage. Data and checksum use FP32 accumulators. The original
// 128x128 data MMA, its A/B storage, and its three-stage schedule are unchanged.
template<class Base>
struct AugmentedTileEncoding {
  using Shape = typename Base::Shape;
  using Element = cutlass::bfloat16_t;
  static constexpr int Threads = Base::WarpCount::kCount * 32;
  static_assert(Shape::kM == 128 && Shape::kN == 128 && Shape::kK == 64);
  static_assert(Threads == 128, "Encoding partitions 64 A columns over four warps");
  static_assert(Base::kStages == 3 && Base::kWarpGemmIterations == 4,
                "Compact synchronization relies on the pinned three-stage mainloop");
  static_assert(Base::WarpCount::kM == 2 && Base::WarpCount::kN == 2 &&
                    Base::WarpCount::kK == 1,
                "Register reuse requires the official four 64x64 warps");
  using ChecksumMma = decltype(cute::make_tiled_mma(cute::SM80_16x8x16_F32BF16BF16F32_TN{}));
  using ChecksumFragment = decltype(cute::partition_fragment_C(
      ChecksumMma{}, cute::make_shape(cute::_32{}, cute::_8{})));
  // The last warp-K group executes after the official stage barrier. Two q
  // slots let a fast warp encode the next tile without overwriting q still
  // being read by another warp's last group. The next q-publication barrier
  // completes those readers before this slot can be reused two tiles later.
  struct Storage { alignas(4) Element weights[2][128]; Element q[2][2][64]; };
  typename Base::SharedStorage& operands;
  Storage& scratch;
  AugmentedParams projection;
  int columns, inner, tile_m, tile_n, thread, k_tile = 0;
  bool reuse_pipeline_barriers = false;
  // Set only by the host-selected native FullKN kernel instantiation. All
  // K64 stages and N128 tiles then contain valid K/N coordinates; M may tail.
  bool full_kn = false;
  ChecksumFragment predicted;

  // A native B atom is N8xK16; the appended MMA reads B^T as N16xK16.
  // Compose the official CuTe layouts to join adjacent N8 atoms within each
  // lane. Every result is a compile-time register index, not a shared-memory
  // reload, cross-lane shuffle, or runtime search over the fragment.
  using AtomTraits = cute::MMA_Traits<cute::SM80_16x8x16_F32BF16BF16F32_TN>;
  using AtomALayout = typename AtomTraits::ALayout;
  using AtomBLayout = typename AtomTraits::BLayout;
  CUTLASS_HOST_DEVICE static constexpr int source_b_register(int value) {
    const int coordinate = AtomALayout{}(cute::make_coord(0, value % 8));
    const int n = coordinate % 16, k = coordinate / 16;
    const int lane_value = cute::right_inverse(AtomBLayout{})(n % 8 + k * 8);
    return (2 * (value / 8) + n / 8) * 4 + lane_value / 32;
  }
  CUTLASS_HOST_DEVICE static constexpr bool register_mapping_is_lane_local() {
    for (int lane = 0; lane < 32; ++lane) {
      for (int value = 0; value < 16; ++value) {
        const int a = AtomALayout{}(cute::make_coord(lane, value % 8));
        const int source = source_b_register(value);
        const int b = AtomBLayout{}(cute::make_coord(lane, source % 4));
        if (a % 16 + 16 * (value / 8) != b % 8 + 8 * (source / 4) ||
            a / 16 != b / 8) return false;
      }
    }
    return true;
  }
  static_assert(register_mapping_is_lane_local(),
                "Official B-to-A transpose must preserve every lane's coordinates");

  // Official SM80 A/B layouts place consecutive K coordinates in a single
  // packed BF16 register. Check every lane before using a uint32 copy/view;
  // this is a layout-preserving bit copy, never a numeric conversion.
  CUTLASS_HOST_DEVICE static constexpr bool propagation_pairs_are_contiguous() {
    for (int lane = 0; lane < 32; ++lane) {
      for (int value = 0; value < 16; value += 2) {
        const int a0 = AtomALayout{}(cute::make_coord(lane, value % 8));
        const int a1 = AtomALayout{}(cute::make_coord(lane, (value + 1) % 8));
        if (a1 != a0 + 16 || source_b_register(value) % 2 ||
            source_b_register(value + 1) != source_b_register(value) + 1) return false;
      }
      for (int value = 0; value < 4; value += 2) {
        const int b0 = AtomBLayout{}(cute::make_coord(lane, value));
        const int b1 = AtomBLayout{}(cute::make_coord(lane, value + 1));
        if (b1 != b0 + 8 || (b0 / 8) % 2) return false;
      }
    }
    return true;
  }
  static_assert(propagation_pairs_are_contiguous(),
                "Packed qB copies require aligned consecutive BF16 pairs");
  using PropagationPairCopy = cute::Copy_Atom<cute::UniversalCopy<uint32_t>, Element>;

  CUTLASS_HOST_DEVICE static constexpr bool weights_pairs_are_contiguous() {
    for (int lane = 0; lane < 32; ++lane) {
      for (int value = 0; value < 4; value += 2) {
        const int first = AtomBLayout{}(cute::make_coord(lane,value));
        const int second = AtomBLayout{}(cute::make_coord(lane,value+1));
        if (second != first + 8 || (first / 8) % 2 != 0) return false;
      }
    }
    return true;
  }
  static_assert(weights_pairs_are_contiguous());

  // For native WA encoding, the official LDSM copy atom
  // supplies the register fragment required by the official BF16 MMA atom.
  // Global row-major A uses transposed 8x8 shared loads for A^T; global
  // column-major A uses non-transposed loads. Both shared layouts retain
  // contiguous aligned vectors of eight BF16 values.
  using SharedALayout = typename Base::TensorRefA::Layout;
  static constexpr bool kRowMajorA = cutlass::platform::is_same<
      SharedALayout,
      cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16,64>>::value;
  static constexpr bool kColumnMajorA = cutlass::platform::is_same<
      SharedALayout,
      cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<16,64>>::value;
  static_assert(kRowMajorA || kColumnMajorA, "WA LDSM requires pinned shared A layouts");
  using EncodingCopyOp = typename cutlass::platform::conditional<
      kRowMajorA, cute::SM75_U16x8_LDSM_T, cute::SM75_U32x4_LDSM_N>::type;
  using EncodingCopyAtom = cute::Copy_Atom<EncodingCopyOp,Element>;
  using EncodingCopy = decltype(cute::make_tiled_copy_A(EncodingCopyAtom{},ChecksumMma{}));
  CUTLASS_HOST_DEVICE static constexpr bool copy_destination_matches_mma() {
    for (int lane = 0; lane < 32; ++lane) {
      for (int value = 0; value < 8; ++value) {
        if (EncodingCopy::get_layoutD_TV()(cute::make_coord(lane,value)) !=
            ChecksumMma{}.get_layoutA_TV()(cute::make_coord(lane,value))) return false;
      }
    }
    return true;
  }
  static_assert(copy_destination_matches_mma(),
                "Official copy and MMA must agree on all A fragment registers");
  CUTLASS_HOST_DEVICE static constexpr bool copy_source_is_contiguous() {
    for (int lane = 0; lane < 32; ++lane) {
      const int first = EncodingCopy::get_layoutS_TV()(cute::make_coord(lane,0));
      for (int value = 0; value < 8; ++value) {
        const int coordinate = EncodingCopy::get_layoutS_TV()(cute::make_coord(lane,value));
        if (kRowMajorA) {
          if (first % 16 % 8 || coordinate != first + value) return false;
        } else {
          if (first / 16 % 8 || coordinate != first + value * 16) return false;
        }
      }
    }
    return true;
  }
  static_assert(copy_source_is_contiguous(),
                "Official LDSM must read one aligned contiguous BF16 vector per lane");

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
    // The official Mma::operator() calls prologue() then gmem_wait(), whose
    // CTA barrier precedes the first consume. It also publishes these weights.
    if (!reuse_pipeline_barriers) __syncthreads();
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
      if (reuse_pipeline_barriers) {
        cute::for_each(cute::make_seq<2>{}, [&](auto pair) {
          constexpr int i = 2 * decltype(pair)::value;
          const int p = cute::get<0>(w_coordinates(i));
          const int local_row = reduction + cute::get<1>(w_coordinates(i));
          if (p < 2) {
            auto source = cute::make_tensor(cute::make_smem_ptr(&scratch.weights[p][local_row]),
                                            cute::Layout<cute::_2>{});
            auto target = cute::make_tensor(w_fragment.data() + i, cute::Layout<cute::_2>{});
            cute::Copy_Atom<cute::UniversalCopy<uint32_t>, Element>{}.call(source,target);
          } else {
            w_fragment(i) = w_fragment(i+1) = Element(0.f);
          }
        });
      } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(w_fragment); ++i) {
          const int p = cute::get<0>(w_coordinates(i));
          const int local_row = reduction + cute::get<1>(w_coordinates(i));
          w_fragment(i) = p < 2 ? scratch.weights[p][local_row] : Element(0.f);
        }
      }
      if (reuse_pipeline_barriers) {
        // The source coordinate comes from official copy/MMA composition,
        // rather than an independently handwritten ldmatrix register map.
        const int load_coordinate = EncodingCopy::get_layoutS_TV()(cute::make_coord(lane,0));
        auto source = cute::make_tensor(cute::make_smem_ptr(&shared_a.at({
            reduction + load_coordinate / 16,
            stage * Shape::kK + warp * 16 + load_coordinate % 16})),
            cute::Layout<cute::_8>{});
        auto destination = cute::make_tensor(a_fragment.data(), cute::Layout<cute::_8>{});
        EncodingCopyAtom{}.call(source,destination);
        // LDSM reads an allocated shared-memory vector; discard masked M/K
        // elements before the checksum MMA, including the first K residue tile.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(a_fragment); ++i) {
          const int k = warp * 16 + cute::get<0>(a_coordinates(i));
          const int local_row = reduction + cute::get<1>(a_coordinates(i));
          const int row = tile_m * kAugmentedDataRows + local_row;
          if (!(row < projection.rows && (full_kn || k < valid_k))) {
            a_fragment(i) = Element(0.f);
          }
        }
      } else {
        // Keep the reference implementation unchanged for paired comparison.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(a_fragment); ++i) {
          const int k = warp * 16 + cute::get<0>(a_coordinates(i));
          const int local_row = reduction + cute::get<1>(a_coordinates(i));
          const int row = tile_m * kAugmentedDataRows + local_row;
          a_fragment(i) = row < projection.rows && k < valid_k
              ? shared_a.at({local_row, stage * Shape::kK + k}) : Element(0.f);
        }
      }
      cute::gemm(mma, a_fragment, w_fragment, q_fragment);
    }

    const int q_slot = reuse_pipeline_barriers ? k_tile % 2 : 0;
    cutlass::NumericConverter<Element, float> convert;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(q_fragment); ++i) {
      const int k = warp * 16 + cute::get<0>(q_coordinates(i));
      const int p = cute::get<1>(q_coordinates(i));
      if (p < 2) {
        scratch.q[q_slot][p][k] = full_kn || k < valid_k
            ? convert(q_fragment(i)) : Element(0.f);
      }
    }
    // The two BF16 rows live beside A, not in its original 128 data rows.
    __syncthreads();

    if (reuse_pipeline_barriers) {
      // The compact path propagates each K16 group from the main MMA's B
      // registers after the original transform, preserving the K0,16,32,48
      // checksum accumulation order without loading B from shared again.
      ++k_tile;
      return;
    }

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
        encoded_fragment(i) = p < 2 ? scratch.q[0][p][k] : Element(0.f);
      }
      cute::gemm(mma, b_fragment, encoded_fragment, predicted);
    }
    // The reference comparison path retains its original stage-wide barrier.
    __syncthreads();
    ++k_tile;
  }

  template<int Half, class MainFragmentB>
  CUTLASS_DEVICE void propagate_registers(int warp_k, MainFragmentB const& main_b) {
    static_assert(MainFragmentB::kElements == 32,
                  "Expected eight native N8xK16 BF16 B atoms per warp");
    const int lane = thread % 32, warp = thread / 32;
    const int current_tile = k_tile - 1;
    const int valid_k = current_tile == 0 ? (inner - 1) % Shape::kK + 1 : Shape::kK;
    const int q_slot = current_tile % 2;
    ChecksumMma mma;
    auto thr_mma = mma.get_slice(lane);
    auto b_coordinates = thr_mma.partition_A(cute::make_identity_tensor(
        cute::make_shape(cute::_32{}, cute::_16{})));
    auto encoded_coordinates = thr_mma.partition_B(cute::make_identity_tensor(
        cute::make_shape(cute::_8{}, cute::_16{})));
    auto b_fragment = cute::make_tensor<Element>(cute::shape(b_coordinates));
    auto encoded_fragment = mma.make_fragment_B(encoded_coordinates);
    static_assert(decltype(cute::size(b_fragment))::value == 16);
    auto packed_main_b = cute::recast<uint32_t>(cute::make_tensor(
        cute::make_rmem_ptr(main_b.data()), cute::Layout<cute::_32>{}));
    auto packed_b = cute::recast<uint32_t>(cute::make_tensor(
        b_fragment.data(), cute::Layout<cute::_16>{}));
    cute::for_each(cute::make_seq<8>{}, [&](auto pair) {
      constexpr int index = decltype(pair)::value * 2;
      constexpr int source = (Half * 16 + source_b_register(index)) / 2;
      if (full_kn) {
        packed_b(pair) = packed_main_b(cute::Int<source>{});
      } else {
        const int n = warp * 32 + cute::get<0>(b_coordinates(index));
        const int k = warp_k * 16 + cute::get<1>(b_coordinates(index));
        const bool valid_n = tile_n * 128 + n < columns;
        // Keep the two halfword predicates separate for odd K residues. This
        // preserves the scalar mask exactly, including valid nonfinite values.
        const uint32_t mask = (valid_n && k < valid_k ? 0x0000ffffu : 0u) |
                              (valid_n && k + 1 < valid_k ? 0xffff0000u : 0u);
        packed_b(pair) = packed_main_b(cute::Int<source>{}) & mask;
      }
    });
    cute::clear(encoded_fragment);
    cute::for_each(cute::make_seq<2>{}, [&](auto pair) {
      constexpr int index = decltype(pair)::value * 2;
      const int p = cute::get<0>(encoded_coordinates(index));
      const int k = warp_k * 16 + cute::get<1>(encoded_coordinates(index));
      if (p < 2) {
        auto source = cute::make_tensor(cute::make_smem_ptr(&scratch.q[q_slot][p][k]),
                                       cute::Layout<cute::_2>{});
        auto destination = cute::make_tensor(encoded_fragment.data() + index,
                                            cute::Layout<cute::_2>{});
        PropagationPairCopy{}.call(source, destination);
      }
    });
    cute::gemm(mma, b_fragment, encoded_fragment, predicted);
  }

  template<class MainFragmentB>
  CUTLASS_DEVICE void propagate(int warp_k, MainFragmentB const& main_b) {
    if (!reuse_pipeline_barriers) return;
    // The two M warps share the same 64 B columns. Each takes one N32 half;
    // template specialization keeps all source register indices constant.
    if ((thread / 32) % 2) propagate_registers<1>(warp_k, main_b);
    else propagate_registers<0>(warp_k, main_b);
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
      if (p < 2 && (col < columns || projection.tile_flags != nullptr)) {
        const int64_t index = int64_t(tile_m) * projection.padded_n + col;
        projection.predicted[p * plane + index] = col < columns ? predicted(i) : 0.f;
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

template<int Threads, int Columns, bool Compact>
struct AugmentedProjectionStorage {
  float sums[2][Threads * Columns];
  float magnitude[2][Threads * Columns];
  int columns[Threads * Columns];
};

template<int Threads, int Columns>
struct AugmentedProjectionStorage<Threads, Columns, true> {
  float sums[2][Threads * Columns];
  float magnitude[2][Threads * Columns];
  int warp_flags[Threads / 32];
};

template<class Iterator, int Threads, bool Compact = false>
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
  using SharedStorage = AugmentedProjectionStorage<Threads, kColumns, Compact>;

  CUTLASS_DEVICE AugmentedVisitor(
      AugmentedParams params, SharedStorage& shared, cutlass::MatrixCoord extent,
      int thread, typename Iterator::Params output_params, ElementOutput* output,
      cutlass::MatrixCoord offset)
      : params_(params), shared_(shared), extent_(extent), thread_(thread), offset_(offset),
        iterator_(output_params, output, extent, thread, offset) {
    for (int c = 0; c < kColumns; ++c) {
      sums_[0][c] = sums_[1][c] = 0.f;
      magnitude_[0][c] = magnitude_[1][c] = 0.f;
      if constexpr (!Compact) columns_[c] = -1;
    }
  }
  CUTLASS_DEVICE void begin_epilogue() {}
  CUTLASS_DEVICE void begin_step(int) { fragment_.clear(); }
  CUTLASS_DEVICE void begin_row(int) {}
  CUTLASS_DEVICE void end_row(int) {}
  CUTLASS_DEVICE void visit_compact(int column, int frag,
                                   AccumulatorFragment const& accum) {
    const auto coord = iterator_.thread_start() + ThreadMap::iteration_offset(frag);
    const int row = coord.row(), local_row = row - offset_.row();
    const bool valid_row = local_row >= 0 && local_row < kAugmentedDataRows &&
                           row < extent_.row();
    AccumulatorFragment adjusted = accum;
    if (valid_row) {
      // The row is common to all eight values in this vector. Keep each
      // column check explicit so masked columns cannot inject or contribute.
      if (row == params_.fault_row) {
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kElementsPerAccess; ++e) {
          const int col = coord.column() + e;
          if (col < extent_.column() && col == params_.fault_col) {
            adjusted[e] = __uint_as_float(
                __float_as_uint(adjusted[e]) ^ (1u << params_.fault_bit));
            flags_ |= 1;
          }
        }
      }
      const float row_weights[2] = {
          float(params_.weights[row]),
          float(params_.weights[int64_t(params_.rows) + row])};
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < kElementsPerAccess; ++e) {
        const int col = coord.column() + e;
        const int slot = column * kElementsPerAccess + e;
        if (col < extent_.column()) {
          CUTLASS_PRAGMA_UNROLL
          for (int p = 0; p < 2; ++p) {
            const float term = row_weights[p] * adjusted[e];
            sums_[p][slot] += term;
            magnitude_[p][slot] += fabsf(term);
          }
        }
      }
    }
    OutputVector& output = reinterpret_cast<OutputVector*>(&fragment_)[frag];
    output = cutlass::NumericArrayConverter<ElementOutput, float,
        kElementsPerAccess>{}(adjusted);
    if (valid_row) {
      CUTLASS_PRAGMA_UNROLL
      for (int e = 0; e < kElementsPerAccess; ++e) {
        const int col = coord.column() + e;
        if (col < extent_.column() && !cutlass::isfinite(output[e])) flags_ |= 2;
      }
    }
  }
  CUTLASS_DEVICE void visit(int, int, int column, int frag, AccumulatorFragment const& accum) {
    if constexpr (Compact) {
      visit_compact(column, frag, accum);
      return;
    }
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
            const float weight = float(params_.weights[int64_t(p) * params_.rows + row]);
            const float term = weight * value;
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
      if constexpr (!Compact) shared_.columns[index] = columns_[c];
    }
    if constexpr (Compact) {
      const int warp_flags = (__ballot_sync(0xffffffff, flags_ & 1) != 0 ? 1 : 0) |
                             (__ballot_sync(0xffffffff, flags_ & 2) != 0 ? 2 : 0);
      if (thread_ % 32 == 0) shared_.warp_flags[thread_ / 32] = warp_flags;
    }
    __syncthreads();
    if constexpr (Compact) {
      // Fixed official TensorOp output ThreadMap. Columns repeat in the same
      // lane positions across row groups; no runtime coordinate table is
      // needed. These checks reject a changed mapping instead of silently
      // reducing the wrong registers.
      constexpr int access_width = ThreadMap::Detail::kAccessWidth;
      static_assert(Threads % access_width == 0 && 32 % access_width == 0);
      static_assert(ThreadMap::Detail::WarpPartitions::kRow == 1);
      static_assert(ThreadMap::Delta::kColumn == access_width * kElementsPerAccess);
      static_assert(kColumns * access_width == 128);
      if (thread_ == 0) {
        int flags = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int warp = 0; warp < Threads / 32; ++warp) flags |= shared_.warp_flags[warp];
        params_.tile_flags[int64_t(blockIdx.x) * (params_.padded_n / 128) + blockIdx.y] = flags;
      }
      for (int local_col = thread_; local_col < 128; local_col += Threads) {
        const int col = offset_.column() + local_col;
        const int lane_column = (local_col / kElementsPerAccess) % access_width;
        const int slot = (local_col / (access_width * kElementsPerAccess)) *
                             kElementsPerAccess + local_col % kElementsPerAccess;
        const int64_t index = int64_t(blockIdx.x) * params_.padded_n + col;
        const int64_t plane = int64_t(params_.m_tiles) * params_.padded_n;
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < 2; ++p) {
          float sum = 0.f, magnitude = 0.f;
          if (col < extent_.column()) {
            // Match the legacy thread-major/slot-major summation order exactly.
            // For this tile each column reads eight contributing slots, rather
            // than testing all 1024 slots for a matching dynamic column ID.
            CUTLASS_PRAGMA_UNROLL
            for (int contributor = 0; contributor < Threads / access_width; ++contributor) {
              const int source = (lane_column + contributor * access_width) * kColumns + slot;
              sum += shared_.sums[p][source];
              magnitude += shared_.magnitude[p][source];
            }
          }
          params_.observed[p * plane + index] = sum;
          params_.magnitude[p * plane + index] = magnitude;
        }
      }
      return;
    }
    if constexpr (!Compact) {
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
  int columns_[Compact ? 1 : kColumns];
  int flags_ = 0;
};

template<class BaseKernel, bool Compact = false, bool FullKN = false>
struct AugmentedKernel {
  using Mma = typename AugmentedMma<typename BaseKernel::Mma>::Type;
  using OriginalEpilogue = typename BaseKernel::Epilogue;
  using Iterator = typename OriginalEpilogue::OutputTileIterator;
  static constexpr int kThreads = BaseKernel::kThreadCount;
  using Visitor = AugmentedVisitor<Iterator, kThreads, Compact>;
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
    encoding.reuse_pipeline_barriers = Compact;
    encoding.full_kn = Compact && FullKN;
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
