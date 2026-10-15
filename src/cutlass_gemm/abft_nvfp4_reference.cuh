#pragma once
#include "abft_reference.cuh"
#include "cutlass/float_subbyte.h"

namespace sdc_abft {
template<int M, int N, int K, int Threads>
struct Nvfp4TileReference {
  static_assert(N <= Threads, "One consumer thread owns each reference column");
  // q/q_abs[2,K]: 2048 shared bytes for the current 128x128x128 tile.
  // Each participating column owner keeps its four accumulators in registers.
  struct Accumulators { float r0 = 0.f, r1 = 0.f, a0 = 0.f, a1 = 0.f; };
  struct Storage { float q[2][K]; float q_abs[2][K]; };
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
      float q0 = 0.f, q1 = 0.f, a0 = 0.f, a1 = 0.f;
      for (int m = 0; m < M; ++m) {
        int row = c.tile_m * M + m;
        if (row < c.rows) {
          float value = float(cutlass::float_e2m1_t::bitcast(uint8_t(cute::uint4_t(a(m, k, stage))))) * float(sfa(m, k, stage));
          float w0 = c.weights[row], w1 = c.weights[int64_t(c.rows) + row];
          q0 = fmaf(w0, value, q0);
          q1 = fmaf(w1, value, q1);
          a0 = fmaf(fabsf(w0), fabsf(value), a0);
          a1 = fmaf(fabsf(w1), fabsf(value), a1);
        }
      }
      s.q[0][k] = q0; s.q[1][k] = q1;
      s.q_abs[0][k] = a0; s.q_abs[1][k] = a1;
    }
    sync();
    for (int n = tid; n < N; n += Threads) {
      if (c.tile_n * N + n < c.columns) {
        float r0 = registers.r0, r1 = registers.r1;
        float a0 = registers.a0, a1 = registers.a1;
        for (int k = 0; k < K; ++k) {
          float value = float(cutlass::float_e2m1_t::bitcast(uint8_t(cute::uint4_t(b(n, k, stage))))) * float(sfb(n, k, stage));
          r0 = fmaf(s.q[0][k], value, r0);
          r1 = fmaf(s.q[1][k], value, r1);
          a0 = fmaf(s.q_abs[0][k], fabsf(value), a0);
          a1 = fmaf(s.q_abs[1][k], fabsf(value), a1);
        }
        registers.r0 = r0; registers.r1 = r1;
        registers.a0 = a0; registers.a1 = a1;
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
        c.partial[offset] = alpha * registers.r0;
        c.partial[plane + offset] = alpha * registers.r1;
        c.magnitude[offset] = fabsf(alpha) * registers.a0;
        c.magnitude[plane + offset] = fabsf(alpha) * registers.a1;
      }
    }
    sync();
  }
};
} // namespace sdc_abft
