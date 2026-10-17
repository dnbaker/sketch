# Vendored dependency: simde (SIMD Everywhere)

- Upstream: https://github.com/simd-everywhere/simde
- Version: 0.8.2 (release tarball, vendored 2026-10-07)
- License: MIT (see LICENSE, copied from the release COPYING file)
- Copyright (c) 2017 Evan Nemerson and contributors

`simde/` is the unmodified upstream header tree. The core includes it as
`#include "simde/x86/avx2.h"` with `third_party/simde` on the include path;
simde lowers the 256-bit intrinsics to whatever the compiler targets (AVX2
when compiled with `-mavx2`, SSE2 at the x86-64 baseline, NEON on aarch64,
or its own portable fallbacks elsewhere).

Used by `include/simd_sums.hpp` for the lane-wise Neumaier reductions
behind the SPCA_SIMD_SUMS compile-time axis (default ON; OFF keeps the
scalar reference paths exactly as before).
