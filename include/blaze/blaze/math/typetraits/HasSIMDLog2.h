//=================================================================================================
/*!
//  \file blaze/math/typetraits/HasSIMDLog2.h
//  \brief Header file for the HasSIMDLog2 type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_HASSIMDLOG2_H_
#define _BLAZE_MATH_TYPETRAITS_HASSIMDLOG2_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Vectorization.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/IsDouble.h>
#include <blaze/util/typetraits/IsFloat.h>
#include <blaze/util/typetraits/RemoveCVRef.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary alias declaration for the HasSIMDLog2 type trait.
// \ingroup math_type_traits
*/
template< typename T >  // Type of the operand
using HasSIMDLog2Helper =
   BoolConstant< ( IsFloat_v<T> || IsDouble_v<T> ) &&
                 ( bool( BLAZE_SVML_MODE    ) ||
                   bool( BLAZE_SLEEF_MODE ) ) &&
                 ( bool( BLAZE_SSE_MODE     ) ||
                   bool( BLAZE_AVX_MODE     ) ||
                   bool( BLAZE_MIC_MODE     ) ||
                   bool( BLAZE_AVX512F_MODE ) ) >;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Availability of a SIMD binary logarithm operation for the given data type.
// \ingroup math_type_traits
//
// Depending on the available instruction set (SSE, SSE2, SSE3, SSE4, AVX, AVX2, MIC, ...) and
// the used compiler, this type trait provides the information whether a SIMD binary logarithm
// operation exists for the given data type \a T (ignoring the cv-qualifiers). In case the SIMD
// operation is available, the \a value member constant is set to \a true, the nested type
// definition \a Type is \a TrueType, and the class derives from \a TrueType. Otherwise \a value
// is set to \a false, \a Type is \a FalseType, and the class derives from \a FalseType. The
// following example assumes that the Intel SVML is available:

   \code
   blaze::HasSIMDLog2< float >::value         // Evaluates to 1
   blaze::HasSIMDLog2< double >::Type         // Results in TrueType
   blaze::HasSIMDLog2< const double >         // Is derived from TrueType
   blaze::HasSIMDLog2< unsigned int >::value  // Evaluates to 0
   blaze::HasSIMDLog2< long double >::Type    // Results in FalseType
   blaze::HasSIMDLog2< complex<double> >      // Is derived from FalseType
   \endcode
*/
template< typename T >  // Type of the operand
struct HasSIMDLog2
   : public BoolConstant< HasSIMDLog2Helper< RemoveCVRef_t<T> >::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the HasSIMDLog2 type trait.
// \ingroup math_type_traits
//
// The HasSIMDLog2_v variable template provides a convenient shortcut to access the nested
// \a value of the HasSIMDLog2 class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::HasSIMDLog2<T>::value;
   constexpr bool value2 = blaze::HasSIMDLog2_v<T>;
   \endcode
*/
template< typename T >  // Type of the operand
constexpr bool HasSIMDLog2_v = HasSIMDLog2<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
