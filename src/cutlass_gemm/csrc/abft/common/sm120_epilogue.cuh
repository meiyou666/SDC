#pragma once

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <optional>
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/numeric_conversion.h"
#include "abft/common/policy.cuh"

namespace sdc_abft {
namespace fusion = cutlass::epilogue::fusion;

// A coordinate-aware, register-only visitor. Its input is alpha * accumulator;
// both projection visitors and the D conversion consume the returned fragment.
template<class Output>
struct Inject : fusion::Sm90VisitorImpl<> {
  struct Arguments {
    int row = -1;
    int col = -1;
    int bit = -1;
    int* hit = nullptr;
    int* nonfinite = nullptr;
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
    int rows, columns;
    CUTLASS_DEVICE Callbacks(Coordinates const& c, Params const& p, int m, int n)
        : coordinates(c), params(p), rows(m), columns(n) {}

    template<class Acc, class Input, int Count>
    CUTLASS_DEVICE cutlass::Array<Input, Count> visit(
        cutlass::Array<Acc, Count> const&, int epi_v, int epi_m, int epi_n,
        cutlass::Array<Input, Count> const& values) {
      static_assert(cute::is_same_v<Input, float>, "Inject before output conversion");
      auto result = values;
      if (params.bit < 0 && params.nonfinite == nullptr) return result;
      auto coords = coordinates(cute::_, cute::_, cute::_, epi_m, epi_n);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Count; ++i) {
        auto coord = coords(epi_v * Count + i);
        if (params.bit >= 0 && cute::get<0>(coord) == params.row &&
            cute::get<1>(coord) == params.col) {
          result[i] = __uint_as_float(__float_as_uint(result[i]) ^ (uint32_t{1} << params.bit));
          *params.hit = 1;
        }
      }
      if (params.nonfinite != nullptr) {
        // Inspect the exact post-alpha, post-injection output conversion while
        // the fragment is still in registers. Padding cannot set the flag.
        auto converted = cutlass::NumericArrayConverter<Output, float, Count>{}(result);
        bool nonfinite = false;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Count; ++i) {
          auto coord = coords(epi_v * Count + i);
          if (cute::get<0>(coord) < rows && cute::get<1>(coord) < columns &&
              !isfinite(float(converted[i]))) nonfinite = true;
        }
        if (nonfinite) atomicExch(params.nonfinite, 1);
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
    return Callbacks<decltype(coords)>(coords, params, int(M), int(N));
  }
};

// Nest only the compiled policy's projections. Each row reduction forwards
// its unmodified input fragment to the next projection and final D conversion.
template<int Channels, class Reduction, class Weight, class Input>
struct ProjectionTree {
  using Inner = ProjectionTree<Channels - 1, Reduction, Weight, Input>;
  using Visitor = fusion::Sm90EVT<Reduction, typename Inner::Visitor, Weight>;
  static auto& arguments(typename Visitor::Arguments& args, float const* weights,
                        int64_t rows, float* partial, int64_t projection_size) {
    args.op_1.ptr_col = weights + int64_t(Channels - 1) * rows;
    args.op_2.ptr_row = partial + int64_t(Channels - 1) * projection_size;
    return Inner::arguments(args.op_0, weights, rows, partial, projection_size);
  }
};

template<class Reduction, class Weight, class Input>
struct ProjectionTree<0, Reduction, Weight, Input> {
  using Visitor = Input;
  static auto& arguments(typename Visitor::Arguments& args, float const*,
                        int64_t, float*, int64_t) { return args; }
};

template<class Tile, class Output, class Policy = policy::Classic>
struct ProjectionEpilogue {
  static_assert(Policy::kChecksums > 0, "An ABFT policy needs at least one checksum");
  static constexpr auto Round = cutlass::FloatRoundStyle::round_to_nearest;
  using Reduction = fusion::Sm90RowReduction<
      Policy::template ObserveOp, cutlass::plus, cutlass::plus,
      0, Tile, float, float, Round, cute::Stride<cute::_0, cute::_1, cute::_0>,
      4, false, false, true, cute::seq<1, 2, 0>>;
  using Weight = fusion::Sm90ColBroadcast<
      0, Tile, float, float, cute::Stride<cute::_1, cute::_0, cute::_0>, 1, false>;
  using Scaled = fusion::Sm90EVT<
      fusion::Sm90Compute<cutlass::multiplies, float, float, Round>,
      fusion::Sm90ScalarBroadcast<float>, fusion::Sm90AccFetch>;
  using Fault = fusion::Sm90EVT<Inject<Output>, Scaled>;
  using Tree = ProjectionTree<Policy::kChecksums, Reduction, Weight, Fault>;
  using Visitor = fusion::Sm90EVT<
      fusion::Sm90Compute<cutlass::epilogue::thread::Identity, Output, float, Round>,
      typename Tree::Visitor>;

  static typename Visitor::Arguments arguments(
      float const* alpha, float const* weights, int64_t rows, float* partial,
      int64_t projection_size, int* hit, int64_t row, int64_t col, int64_t bit,
      int* nonfinite = nullptr) {
    typename Visitor::Arguments args{};
    auto& fault = Tree::arguments(args.op_0, weights, rows, partial, projection_size);
    fault.op_1 = {int(row), int(col), int(bit), hit, nonfinite};
    fault.op_0.op_0.scalar_ptrs[0] = alpha;
    return args;
  }
};

template<class Policy = policy::Classic>
inline void check_projection_arguments(at::Tensor const& a, at::Tensor const& weights,
                                       int64_t m, int64_t n,
                                       int64_t row, int64_t col, int64_t bit,
                                       const std::optional<at::Tensor>& nonfinite = std::nullopt) {
  TORCH_CHECK(weights.is_cuda() && weights.device() == a.device() &&
                  weights.layout() == at::kStrided && weights.scalar_type() == at::kFloat &&
                  weights.is_contiguous() && weights.dim() == 2 &&
                  weights.size(0) == Policy::kChecksums && weights.size(1) == m,
              "ABFT weights must be contiguous CUDA FP32 [", Policy::kChecksums,
              ",M] on the operand device");
  TORCH_CHECK((row == -1 && col == -1 && bit == -1) ||
                  (row >= 0 && row < m && col >= 0 && col < n && bit >= 0 && bit < 32),
              "ABFT fault must be all -1, or a valid output row/column and FP32 bit [0,31]");
  if (nonfinite.has_value()) {
    auto const& flag = *nonfinite;
    TORCH_CHECK(flag.is_cuda() && flag.device() == a.device() &&
                    flag.layout() == at::kStrided && flag.scalar_type() == at::kInt &&
                    flag.is_contiguous() && flag.dim() == 1 && flag.size(0) == 1,
                "ABFT nonfinite must be contiguous CUDA int32 [1] on the operand device");
    at::assert_no_overlap(flag, a);
    at::assert_no_overlap(flag, weights);
    flag.zero_();
  }
}
}  // namespace sdc_abft
