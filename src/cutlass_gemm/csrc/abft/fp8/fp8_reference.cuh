#pragma once
#include <cstdint>
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/float8.h"
#include "abft/common/context.cuh"
#include "abft/common/policy.cuh"

namespace sdc_abft {
template<int M, int N, int K, int Threads, class Policy = policy::Classic>
struct Fp8TileReference {
  static_assert(N <= Threads, "One consumer thread owns each reference column");
  static constexpr int kChecksums = Policy::kChecksums;
  // This existing reference path encodes decoded FP8 values in FP32. It does
  // not quantize checksum rows back to FP8. Each channel retains its original
  // row/K accumulation order and applies alpha only when storing evidence.
  struct Accumulators { float r[kChecksums]{}; float a[kChecksums]{}; };
  struct Storage { float q[kChecksums][K]; float q_abs[kChecksums][K]; };
  CUTLASS_DEVICE static void sync() {
    cutlass::arch::NamedBarrier::sync(
        Threads, cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
  }
  CUTLASS_DEVICE static float decode(uint8_t bits) {
    return float(cutlass::float_e4m3_t::bitcast(bits));
  }
  template<class TensorA, class TensorB>
  CUTLASS_DEVICE static void consume(Storage& s, Accumulators& registers, TensorA const& a,
      TensorB const& b, int stage, int tid, TileReferenceContext const& c) {
    // Explicit coordinate reads honor the official swizzled shared layouts.
    // A/B global memory is never revisited. TMA zeros the residual K extent.
    for (int k = tid; k < K; k += Threads) {
      float encoded[kChecksums]{}, magnitude[kChecksums]{};
      for (int m = 0; m < M; ++m) {
        int row = c.tile_m * M + m;
        if (row < c.rows) {
          float value = decode(a(m, k, stage));
          CUTLASS_PRAGMA_UNROLL
          for (int p = 0; p < kChecksums; ++p) {
            float weight = c.weights[int64_t(p) * c.rows + row];
            encoded[p] = Policy::encode(weight, value, encoded[p]);
            magnitude[p] = Policy::encode(fabsf(weight), fabsf(value), magnitude[p]);
          }
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < kChecksums; ++p) {
        s.q[p][k] = encoded[p];
        s.q_abs[p][k] = magnitude[p];
      }
    }
    sync();
    for (int n = tid; n < N; n += Threads) {
      if (c.tile_n * N + n < c.columns) {
        for (int k = 0; k < K; ++k) {
          float value = decode(b(n, k, stage));
          CUTLASS_PRAGMA_UNROLL
          for (int p = 0; p < kChecksums; ++p) {
            registers.r[p] = Policy::propagate(s.q[p][k], value, registers.r[p]);
            registers.a[p] = Policy::propagate(s.q_abs[p][k], fabsf(value), registers.a[p]);
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
        for (int p = 0; p < kChecksums; ++p) {
          c.partial[int64_t(p) * plane + offset] = alpha * registers.r[p];
          c.magnitude[int64_t(p) * plane + offset] = fabsf(alpha) * registers.a[p];
        }
      }
    }
    sync();
  }
};
} // namespace sdc_abft
