#pragma once

#include <ATen/ATen.h>
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

namespace sdc_abft {
namespace fusion = cutlass::epilogue::fusion;

// A coordinate-aware, register-only visitor. Its input is alpha * accumulator;
// both projection visitors and the D conversion consume the returned fragment.
struct Inject : fusion::Sm90VisitorImpl<> {
  struct Arguments {
    int row = -1;
    int col = -1;
    int bit = -1;
    int* hit = nullptr;
  };
  using Params = Arguments;
  struct SharedStorage {};
  Params params;

  template<class Shape>
  static Params to_underlying_arguments(Shape const&, Arguments const& args, void*) {
    return args;
  }
  template<class Shape>
  static bool can_implement(Shape const&, Arguments const&) { return true; }
  template<class Shape>
  static size_t get_workspace_size(Shape const&, Arguments const&) { return 0; }
  template<class Shape>
  static cutlass::Status initialize_workspace(
      Shape const&, Arguments const&, void*, cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) { return cutlass::Status::kSuccess; }

  CUTLASS_HOST_DEVICE Inject() {}
  CUTLASS_HOST_DEVICE Inject(Params const& p, SharedStorage const&) : params(p) {}

  template<class Coordinates>
  struct Callbacks : fusion::EmptyConsumerStoreCallbacks {
    Coordinates coordinates;
    Params params;
    CUTLASS_DEVICE Callbacks(Coordinates const& c, Params const& p)
        : coordinates(c), params(p) {}

    template<class Acc, class Input, int Count>
    CUTLASS_DEVICE cutlass::Array<Input, Count> visit(
        cutlass::Array<Acc, Count> const&, int epi_v, int epi_m, int epi_n,
        cutlass::Array<Input, Count> const& values) {
      static_assert(cute::is_same_v<Input, float>, "Inject before output conversion");
      auto result = values;
      if (params.bit < 0) return result;
      auto coords = coordinates(cute::_, cute::_, cute::_, epi_m, epi_n);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Count; ++i) {
        auto coord = coords(epi_v * Count + i);
        if (cute::get<0>(coord) == params.row && cute::get<1>(coord) == params.col) {
          result[i] = __uint_as_float(__float_as_uint(result[i]) ^ (uint32_t{1} << params.bit));
          *params.hit = 1;
        }
      }
      return result;
    }
  };

  template<bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      fusion::detail::ConsumerStoreArgs<Args...> const& args) {
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    // Reconstruct global coordinates using the actual epilogue copy mapping.
    // args.tCcD alone is relative to the first element owned by this thread.
    auto all = cute::make_identity_tensor(cute::make_shape(M, N));
    auto tile = cute::local_tile(all, cute::take<0, 2>(args.tile_shape_mnk), cute::make_coord(m, n));
    auto coords = fusion::detail::sm90_partition_for_epilogue<ReferenceSrc>(
        tile, args.epi_tile, args.tiled_copy, args.thread_idx);
    return Callbacks<decltype(coords)>(coords, params);
  }
};

template<class Tile, class Output>
struct ProjectionEpilogue {
  static constexpr auto Round = cutlass::FloatRoundStyle::round_to_nearest;
  using Reduction = fusion::Sm90RowReduction<
      cutlass::homogeneous_multiply_add, cutlass::plus, cutlass::plus,
      0, Tile, float, float, Round, cute::Stride<cute::_0, cute::_1, cute::_0>,
      4, false, false, true, cute::seq<1, 2, 0>>;
  using Weight = fusion::Sm90ColBroadcast<
      0, Tile, float, float, cute::Stride<cute::_1, cute::_0, cute::_0>, 1, false>;
  using Scaled = fusion::Sm90EVT<
      fusion::Sm90Compute<cutlass::multiplies, float, float, Round>,
      fusion::Sm90ScalarBroadcast<float>, fusion::Sm90AccFetch>;
  using Fault = fusion::Sm90EVT<Inject, Scaled>;
  using First = fusion::Sm90EVT<Reduction, Fault, Weight>;
  using Second = fusion::Sm90EVT<Reduction, First, Weight>;
  using Visitor = fusion::Sm90EVT<
      fusion::Sm90Compute<cutlass::epilogue::thread::Identity, Output, float, Round>, Second>;

  static typename Visitor::Arguments arguments(
      float const* alpha, float const* weights, int64_t rows, float* partial,
      int64_t projection_size, int* hit, int64_t row, int64_t col, int64_t bit) {
    typename Visitor::Arguments args{};
    auto& second = args.op_0;
    second.op_1.ptr_col = weights + rows;
    second.op_2.ptr_row = partial + projection_size;
    auto& first = second.op_0;
    first.op_1.ptr_col = weights;
    first.op_2.ptr_row = partial;
    auto& fault = first.op_0;
    fault.op_1 = {int(row), int(col), int(bit), hit};
    fault.op_0.op_0.scalar_ptrs[0] = alpha;
    return args;
  }
};

inline void check_projection_arguments(at::Tensor const& a, at::Tensor const& weights,
                                       int64_t m, int64_t n,
                                       int64_t row, int64_t col, int64_t bit) {
  TORCH_CHECK(weights.is_cuda() && weights.device() == a.device() &&
                  weights.layout() == at::kStrided && weights.scalar_type() == at::kFloat &&
                  weights.is_contiguous() && weights.dim() == 2 &&
                  weights.size(0) == 2 && weights.size(1) == m,
              "ABFT weights must be contiguous CUDA FP32 [2,M] on the operand device");
  TORCH_CHECK((row == -1 && col == -1 && bit == -1) ||
                  (row >= 0 && row < m && col >= 0 && col < n && bit >= 0 && bit < 32),
              "ABFT fault must be all -1, or a valid output row/column and FP32 bit [0,31]");
}
}  // namespace sdc_abft
