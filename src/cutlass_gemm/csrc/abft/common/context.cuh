#pragma once
#include <cstdint>

namespace sdc_abft {
// Supplied from the scheduler's actual work tile, never from blockIdx.
// This initial interface supports L=1, full-K work tiles only (no split-K).
struct TileReferenceContext {
  float* partial;
  float const* alpha;
  float const* weights;
  float* magnitude;
  int rows, columns, tile_m, tile_n, tiles_m, padded_n;
};
} // namespace sdc_abft
