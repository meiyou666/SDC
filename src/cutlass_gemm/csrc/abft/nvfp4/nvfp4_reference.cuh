#pragma once
#include "abft/common/context.cuh"
#include "abft/common/policy.cuh"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

namespace sdc_abft {
template<int M, int N, int K, int Threads, class Policy>
struct Nvfp4TileReference {
  static_assert(N <= Threads, "One consumer thread owns each reference column");
  static constexpr int Checksums = Policy::kChecksums;
  // q/q_abs[Checksums,K] retain FP32 encoding of the scaled NVFP4 values.
  // Each participating column owner keeps its accumulators in registers.
  struct Accumulators { float predicted[Checksums]{}; float magnitude[Checksums]{}; };
  struct Storage { float q[Checksums][K]; float q_abs[Checksums][K]; };
  CUTLASS_DEVICE static void sync() {
    cutlass::arch::NamedBarrier::sync(
        Threads, cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
  }
  template<class TensorA, class TensorB, class TensorSFA, class TensorSFB>
  CUTLASS_DEVICE static void consume(Storage& s, Accumulators& registers, TensorA const& a,
      TensorB const& b, TensorSFA const& sfa, TensorSFB const& sfb, int stage, int tid, TileReferenceContext const& c) {
    // NVFP4 shared A/B use packed uint4_t ArrayEngine iterators. Extract
    // the raw nibble, then bitcast to E2M1 (never numerically convert codes).
    // Official SFA/SFB layouts broadcast each scale across 16 K elements.
    // Logical (row,k,stage) indexing preserves that mapping and N permutation.
    // Explicit coordinate reads honor the official swizzled shared layouts.
    // A/B global memory is never revisited. TMA zeros the residual K extent.
    for (int k = tid; k < K; k += Threads) {
      float q[Checksums]{}, q_abs[Checksums]{};
      for (int m = 0; m < M; ++m) {
        int row = c.tile_m * M + m;
        if (row < c.rows) {
          float value = float(cutlass::float_e2m1_t::bitcast(uint8_t(cute::uint4_t(a(m, k, stage))))) * float(sfa(m, k, stage));
          CUTLASS_PRAGMA_UNROLL
          for (int channel = 0; channel < Checksums; ++channel) {
            float weight = c.weights[int64_t(channel) * c.rows + row];
            q[channel] = Policy::encode(weight, value, q[channel]);
            q_abs[channel] = Policy::encode(fabsf(weight), fabsf(value), q_abs[channel]);
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int channel = 0; channel < Checksums; ++channel) {
        s.q[channel][k] = q[channel];
        s.q_abs[channel][k] = q_abs[channel];
      }
    }
    sync();
    for (int n = tid; n < N; n += Threads) {
      if (c.tile_n * N + n < c.columns) {
        for (int k = 0; k < K; ++k) {
          float value = float(cutlass::float_e2m1_t::bitcast(uint8_t(cute::uint4_t(b(n, k, stage))))) * float(sfb(n, k, stage));
          CUTLASS_PRAGMA_UNROLL
          for (int channel = 0; channel < Checksums; ++channel) {
            registers.predicted[channel] = Policy::propagate(
                s.q[channel][k], value, registers.predicted[channel]);
            registers.magnitude[channel] = Policy::propagate(
                s.q_abs[channel][k], fabsf(value), registers.magnitude[channel]);
          }
        }
      }
    }
    // Protect q reuse and ensure all reads complete before stage release.
    sync();
  }
  CUTLASS_DEVICE static void finish(Accumulators const& registers, int tid, TileReferenceContext const& c) {
    float alpha = *c.alpha;
    for (int n = tid; n < N; n += Threads) {
      int column = c.tile_n * N + n;
      if (column < c.padded_n) {
        int64_t offset = int64_t(c.tile_m) * c.padded_n + column;
        int64_t plane = int64_t(c.tiles_m) * c.padded_n;
        CUTLASS_PRAGMA_UNROLL
        for (int channel = 0; channel < Checksums; ++channel) {
          c.partial[channel * plane + offset] = alpha * registers.predicted[channel];
          c.magnitude[channel * plane + offset] = fabsf(alpha) * registers.magnitude[channel];
        }
      }
    }
    sync();
  }
};
} // namespace sdc_abft
