//=================================================================================================
/*!
//  \file blaze/math/simd/Abs.h
//  \brief Header file for the SIMD abs functionality
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

#ifndef _BLAZE_MATH_SIMD_ABS_H_
#define _BLAZE_MATH_SIMD_ABS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Compiler.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>


namespace blaze {

//=================================================================================================
//
//  8-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Absolute value of a vector of 8-bit signed integral values.
// \ingroup simd
//
// \param a The vector of 8-bit unsigned integral values.
// \return The absolute values.
//
// This operation is only available for SSSE3, AVX2, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDint8 abs( const SIMDint8& a ) noexcept
#if BLAZE_AVX512BW_MODE
{
   return _mm512_abs_epi8( a.value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_abs_epi8( a.value );
}
#elif BLAZE_SSSE3_MODE
{
   return _mm_abs_epi8( a.value );
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  16-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Absolute value of a vector of 16-bit signed integral values.
// \ingroup simd
//
// \param a The vector of 16-bit unsigned integral values.
// \return The absolute values.
//
// This operation is only available for SSSE3, AVX2, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDint16 abs( const SIMDint16& a ) noexcept
#if BLAZE_AVX512BW_MODE
{
   return _mm512_abs_epi16( a.value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_abs_epi16( a.value );
}
#elif BLAZE_SSSE3_MODE
{
   return _mm_abs_epi16( a.value );
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Absolute value of a vector of 32-bit signed integral values.
// \ingroup simd
//
// \param a The vector of 32-bit unsigned integral values.
// \return The absolute values.
//
// This operation is only available for SSSE3, AVX2, MIC, and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDint32 abs( const SIMDint32& a ) noexcept
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return _mm512_abs_epi32( a.value );
}
#elif BLAZE_AVX2_MODE
{
   return _mm256_abs_epi32( a.value );
}
#elif BLAZE_SSSE3_MODE
{
   return _mm_abs_epi32( a.value );
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  64-BIT INTEGRAL SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Absolute value of a vector of 64-bit signed integral values.
// \ingroup simd
//
// \param a The vector of 64-bit unsigned integral values.
// \return The absolute values.
//
// This operation is only available for MIC and AVX-512.
*/
BLAZE_ALWAYS_INLINE const SIMDint64 abs( const SIMDint64& a ) noexcept
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return _mm512_abs_epi64( a.value );
}
#else
= delete;
#endif
//*************************************************************************************************




//=================================================================================================
//
//  32-BIT FLOATING POINT SIMD TYPES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Absolute of a vector of single precision floating point values.
// \ingroup simd
//
// \param a The vector of single precision floating point values \f$[-1..1]\f$.
// \return The resulting vector.
//
// This operation is only available for SSE2, AVX, MIC, and AVX-512.
*/
template< typename T >  // Type of the operand
BLAZE_ALWAYS_INLINE const SIMDfloat abs( const SIMDf32<T>& a ) noexcept
#if BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return _mm512_abs_ps( (*a).eval().value );
}
#elif BLAZE_AVX_MODE
{
   const __m256 mask( _mm256_castsi256_ps( _mm256_set1_epi32( 0x80000000 ) ) );
   return _mm256_andnot_ps( mask, (*a).eval().value );
}
#elif BLAZE_SSE2_MODE
{
   const __m128 mask( _mm_castsi128_ps( _mm_set1_epi32( 0x80000000 ) ) );
   return _mm_andnot_ps( mask, (*a).eval().value );
}
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
/*!\brief Absolute of a vector of double precision floating point values.
// \ingroup simd
//
// \param a The vector of double precision floating point values \f$[-1..1]\f$.
// \return The resulting vector.
//
// This operation is only available for SSE2, AVX, MIC, and AVX-512.
*/
template< typename T >  // Type of the operand
BLAZE_ALWAYS_INLINE const SIMDdouble abs( const SIMDf64<T>& a ) noexcept
#if ( BLAZE_AVX512F_MODE || BLAZE_MIC_MODE ) && BLAZE_GNU_COMPILER
= delete;
#elif BLAZE_AVX512F_MODE || BLAZE_MIC_MODE
{
   return _mm512_abs_pd( (*a).eval().value );
}
#elif BLAZE_AVX_MODE
{
   const __m256d mask( _mm256_castsi256_pd(
      _mm256_set_epi32( 0x80000000, 0x0, 0x80000000, 0x0, 0x80000000, 0x0, 0x80000000, 0x0 ) ) );
   return _mm256_andnot_pd( mask, (*a).eval().value );
}
#elif BLAZE_SSE2_MODE
{
   const __m128d mask( _mm_castsi128_pd( _mm_set_epi32( 0x80000000, 0x0, 0x80000000, 0x0 ) ) );
   return _mm_andnot_pd( mask, (*a).eval().value );
}
#else
= delete;
#endif
//*************************************************************************************************

} // namespace blaze

#endif
