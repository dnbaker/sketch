//=================================================================================================
/*!
//  \file blaze/math/simd/Atan2.h
//  \brief Header file for the SIMD multi-valued inverse tangent functionality
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

#ifndef _BLAZE_MATH_SIMD_ATAN2_H_
#define _BLAZE_MATH_SIMD_ATAN2_H_


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
/*!\brief Multi-valued inverse tangent of a vector of single precision floating point values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The resulting vector.
//
// This operation is only available via the SVML or SLEEF for SSE, AVX, MIC, and AVX-512.
*/
template< typename T >  // Type of the operand
BLAZE_ALWAYS_INLINE const SIMDfloat atan2( const SIMDf32<T>& a, const SIMDf32<T>& b ) noexcept
#if BLAZE_SVML_MODE
#  if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return _mm512_atan2_ps( (*a).eval().value, (*b).eval().value );
}
#  elif BLAZE_AVX_MODE
{
   return _mm256_atan2_ps( (*a).eval().value, (*b).eval().value );
}
#  elif BLAZE_SSE_MODE
{
   return _mm_atan2_ps( (*a).eval().value, (*b).eval().value );
}
#  endif
#elif BLAZE_SLEEF_MODE
#  if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return Sleef_atan2f16_u10( (*a).eval().value, (*b).eval().value );
}
#  elif BLAZE_AVX_MODE
{
   return Sleef_atan2f8_u10( (*a).eval().value, (*b).eval().value );
}
#  elif BLAZE_SSE_MODE
{
   return Sleef_atan2f4_u10( (*a).eval().value, (*b).eval().value );
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
/*!\brief Multi-valued inverse tangent of a vector of double precision floating point values.
// \ingroup simd
//
// \param a The left-hand side SIMD operand.
// \param b The right-hand side SIMD operand.
// \return The resulting vector.
//
// This operation is only available via the SVML or SLEEF for SSE, AVX, MIC, and AVX-512.
*/
template< typename T >  // Type of the operand
BLAZE_ALWAYS_INLINE const SIMDdouble atan2( const SIMDf64<T>& a, const SIMDf64<T>& b ) noexcept
#if BLAZE_SVML_MODE
#  if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return _mm512_atan2_pd( (*a).eval().value, (*b).eval().value );
}
#  elif BLAZE_AVX_MODE
{
   return _mm256_atan2_pd( (*a).eval().value, (*b).eval().value );
}
#  elif BLAZE_SSE_MODE
{
   return _mm_atan2_pd( (*a).eval().value, (*b).eval().value );
}
#  endif
#elif BLAZE_SLEEF_MODE
#  if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return Sleef_atan2d8_u10( (*a).eval().value, (*b).eval().value );
}
#  elif BLAZE_AVX_MODE
{
   return Sleef_atan2d4_u10( (*a).eval().value, (*b).eval().value );
}
#  elif BLAZE_SSE_MODE
{
   return Sleef_atan2d2_u10( (*a).eval().value, (*b).eval().value );
}
#  endif
#else
= delete;
#endif
//*************************************************************************************************

} // namespace blaze

#endif
