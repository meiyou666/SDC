#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cute/algorithm/gemm.hpp"

namespace sdc_abft::policy {

// These IDs select compiled policies once, on the host. Device loops never
// dispatch on an algorithm ID. Input/output scaling belongs to the adapter.
enum class Id : int { Classic = 0, Sum = 1 };

struct Decision {
  float residual;
  float threshold;
  bool nonfinite;
  bool detected;
};

template<int Checksums>
struct Projection {
  static constexpr int kChecksums = Checksums;
  template<class T>
  using ObserveOp = cutlass::homogeneous_multiply_add<T>;

  CUTLASS_DEVICE static float encode(float weight, float value, float accumulator) {
    return fmaf(weight, value, accumulator);
  }
  CUTLASS_DEVICE static float propagate(float encoded, float value, float accumulator) {
    return fmaf(encoded, value, accumulator);
  }
  CUTLASS_DEVICE static float observe(float weight, float value, float accumulator) {
    return accumulator + weight * value;
  }

  // The BF16 adapter supplies the existing official CuTe MMA and fragments.
  // Other algorithms may replace these stages without changing the main MMA.
  template<class Mma, class... Fragments>
  CUTLASS_DEVICE static void encode(Mma const& mma, Fragments&&... fragments) {
    cute::gemm(mma, static_cast<Fragments&&>(fragments)...);
  }
  template<class Mma, class... Fragments>
  CUTLASS_DEVICE static void propagate(Mma const& mma, Fragments&&... fragments) {
    cute::gemm(mma, static_cast<Fragments&&>(fragments)...);
  }

  CUTLASS_DEVICE static float merge(float accumulator, float partial) {
    return __fadd_rn(accumulator, partial);
  }
  CUTLASS_DEVICE static Decision finalize(float observed, float predicted,
                                          float magnitude, float atol, float rtol,
                                          float threshold_override = 0.f,
                                          bool use_override = false) {
    const float residual = __fsub_rn(observed, predicted);
    const float threshold = use_override ? threshold_override : __fadd_rn(atol, __fmul_rn(rtol, magnitude));
    const bool nonfinite = !isfinite(observed) || !isfinite(predicted) ||
        !isfinite(magnitude) || !isfinite(residual) || !isfinite(threshold);
    return {residual, threshold, nonfinite, nonfinite || threshold < 0.f || fabsf(residual) > threshold};
  }
};

struct Classic : Projection<2> {};
// A second compiled algorithm exercises a different number of projections.
// It detects with an all-ones checksum and makes no localization claim.
struct Sum : Projection<1> {};

template<class Algorithm> struct Tag { using Type = Algorithm; };

// The only device-policy registry. A new compiled algorithm is registered
// here; the precision adapters receive its type, not a runtime branch.
template<class Function>
auto dispatch(int64_t algorithm, Function&& function) {
  switch (algorithm) {
    case static_cast<int64_t>(Id::Classic): return function(Tag<Classic>{});
    case static_cast<int64_t>(Id::Sum): return function(Tag<Sum>{});
    default: throw std::runtime_error("Unknown compiled ABFT algorithm ID");
  }
}

} // namespace sdc_abft::policy
