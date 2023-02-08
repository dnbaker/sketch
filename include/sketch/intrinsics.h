#pragma once

// For now, we just disable simd for aarch64.
// It would be great to get simde to work, but I have had very little success.

#ifndef __aarch64__
#  if __has_include("x86intrin.h")
#    include <x86intrin.h>
#  endif
#endif
