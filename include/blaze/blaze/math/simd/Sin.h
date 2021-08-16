//=================================================================================================
/*!
//  \file blaze/math/simd/Sin.h
//  \brief Header file for the SIMD sine functionality
//
//  Copyright (C) 2012-2020 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_MATH_SIMD_SIN_H_
#define _BLAZE_MATH_SIMD_SIN_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#if BLAZE_SLEEF_MODE
#  include <sleef.h>
#endif
#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>


namespace blaze {

//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Sine of a vector of single precision floating point values.
// \ingroup simd
//
// \param a The vector of single precision floating point values.
// \return The resulting vector.
//
// This operation is only available via the SVML or SLEEF for SSE, AVX, MIC, and AVX-512.
*/
template< typename T >  // Type of the operand
BLAZE_ALWAYS_INLINE const SIMDfloat sin( const SIMDf32<T>& a ) noexcept
#if BLAZE_SVML_MODE
#  if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return _mm512_sin_ps( (*a).eval().value );
}
#  elif BLAZE_AVX_MODE
{
   return _mm256_sin_ps( (*a).eval().value );
}
#  elif BLAZE_SSE_MODE
{
   return _mm_sin_ps( (*a).eval().value );
}
#  endif
#elif BLAZE_SLEEF_MODE
#  if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return Sleef_sinf16_u10( (*a).eval().value );
}
#  elif BLAZE_AVX_MODE
{
   return Sleef_sinf8_u10( (*a).eval().value );
}
#  elif BLAZE_SSE_MODE
{
   return Sleef_sinf4_u10( (*a).eval().value );
}
#  endif
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Sine of a vector of double precision floating point values.
// \ingroup simd
//
// \param a The vector of double precision floating point values.
// \return The resulting vector.
//
// This operation is only available via the SVML or SLEEF for SSE, AVX, MIC, and AVX-512.
*/
template< typename T >  // Type of the operand
BLAZE_ALWAYS_INLINE const SIMDdouble sin( const SIMDf64<T>& a ) noexcept
#if BLAZE_SVML_MODE
#  if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return _mm512_sin_pd( (*a).eval().value );
}
#  elif BLAZE_AVX_MODE
{
   return _mm256_sin_pd( (*a).eval().value );
}
#  elif BLAZE_SSE_MODE
{
   return _mm_sin_pd( (*a).eval().value );
}
#  endif
#elif BLAZE_SLEEF_MODE
#  if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return Sleef_sind8_u10( (*a).eval().value );
}
#  elif BLAZE_AVX_MODE
{
   return Sleef_sind4_u10( (*a).eval().value );
}
#  elif BLAZE_SSE_MODE
{
   return Sleef_sind2_u10( (*a).eval().value );
}
#  endif
#else
= delete;
#endif
//*************************************************************************************************

} // namespace blaze

#endif
