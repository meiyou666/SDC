#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

struct XorArguments { int64_t* bits = nullptr; };
template<class T> struct XorAccumulator;
template<> struct XorAccumulator<float> {
  using Arguments = XorArguments;
  CUTLASS_DEVICE float operator()(float x, float mask, Arguments const& args) const {
    uint32_t old = __float_as_uint(x), flip = __float_as_uint(mask);
    if (flip && args.bits) { args.bits[0] = old; args.bits[1] = old ^ flip; }
    return __uint_as_float(old ^ flip);
  }
};
template<int N> struct XorAccumulator<cutlass::Array<float, N>> {
  using Arguments = XorArguments;
  CUTLASS_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float,N> const& x,
      cutlass::Array<float,N> const& mask, Arguments const& args) const {
    cutlass::Array<float,N> out;
    CUTLASS_PRAGMA_UNROLL
    for (int i=0; i<N; ++i) out[i] = XorAccumulator<float>{}(x[i], mask[i], args);
    return out;
  }
};

// CUTLASS 2.x epilogue: XOR the FP32 accumulator before any conversion/store.
template<bool Instrumented, int Count=4> struct DenseOutput {
  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  static int const kCount = Count;
  using FragmentOutput = cutlass::Array<float,Count>;
  using FragmentAccumulator = FragmentOutput;
  using FragmentCompute = FragmentOutput;
  struct Params { int64_t* bits = nullptr; };
  Params params;
  CUTLASS_HOST_DEVICE DenseOutput(Params p) : params(p) {}
  CUTLASS_HOST_DEVICE bool is_source_needed() const { return Instrumented; }
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
  CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc, FragmentOutput const& src) const {
    if constexpr (Instrumented) return XorAccumulator<FragmentOutput>{}(acc,src,{params.bits});
    return acc;
  }
  CUTLASS_DEVICE FragmentOutput operator()(FragmentAccumulator const& acc) const { return acc; }
};

template<bool Inject> struct Fusion {
  using Acc = cute::conditional_t<Inject,
    cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<XorAccumulator,float,float,cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch,
      cutlass::epilogue::fusion::Sm90SrcFetch<float>>,
    cutlass::epilogue::fusion::Sm90AccFetch>;
  using Op = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,float,float,cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>, Acc>;
};
