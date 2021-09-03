//=================================================================================================
/*!
//  \file blaze/math/typetraits/HasSIMDAtan2.h
//  \brief Header file for the HasSIMDAtan2 type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_HASSIMDATAN2_H_
#define _BLAZE_MATH_TYPETRAITS_HASSIMDATAN2_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Vectorization.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/IsDouble.h>
#include <blaze/util/typetraits/IsFloat.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/RemoveCVRef.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary alias declaration for the HasSIMDAtan2 type trait.
// \ingroup math_type_traits
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
using HasSIMDAtan2Helper =
   BoolConstant< IsSame_v<T1,T2> &&
                 ( IsFloat_v<T1> || IsDouble_v<T1> ) &&
                 ( bool( BLAZE_SVML_MODE    ) ||
                   bool( BLAZE_SLEEF_MODE ) ) &&
                 ( bool( BLAZE_SSE_MODE     ) ||
                   bool( BLAZE_AVX_MODE     ) ||
                   bool( BLAZE_MIC_MODE     ) ||
                   bool( BLAZE_AVX512F_MODE ) ) >;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Availability of a SIMD inverse tangent operation for the given data types.
// \ingroup math_type_traits
//
// Depending on the available instruction set (SSE, SSE2, SSE3, SSE4, AVX, AVX2, MIC, ...) and
// the used compiler, this type trait provides the information whether a SIMD multi-valued
// inverse tangent operation exists for the given data types \a T1 and \a T2 (ignoring the
// cv-qualifiers). In case the SIMD operation is available, the \a value member constant is
// set to \a true, the nested type definition \a Type is \a TrueType, and the class derives
// from \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and the
// class derives from \a FalseType. The following example assumes that the Intel SVML is
// available:

   \code
   blaze::HasSIMDAtan2< float, float >::value               // Evaluates to 1
   blaze::HasSIMDAtan2< double, double >::Type              // Results in TrueType
   blaze::HasSIMDAtan2< const double, volatile double >     // Is derived from TrueType
   blaze::HasSIMDAtan2< int, int >::value                   // Evaluates to 0
   blaze::HasSIMDAtan2< long double, long double >::Type    // Results in FalseType
   blaze::HasSIMDAtan2< complex<double>, complex<double> >  // Is derived from FalseType
   \endcode
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
struct HasSIMDAtan2
   : public BoolConstant< HasSIMDAtan2Helper< RemoveCVRef_t<T1>, RemoveCVRef_t<T2> >::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the HasSIMDAtan2 type trait.
// \ingroup math_type_traits
//
// The HasSIMDAtan2_v variable template provides a convenient shortcut to access the nested
// \a value of the HasSIMDAtan2 class template. For instance, given the types \a T1 and \a T2
// the following two statements are identical:

   \code
   constexpr bool value1 = blaze::HasSIMDAtan2<T1,T2>::value;
   constexpr bool value2 = blaze::HasSIMDAtan2_v<T1,T2>;
   \endcode
*/
template< typename T1    // Type of the left-hand side operand
        , typename T2 >  // Type of the right-hand side operand
constexpr bool HasSIMDAtan2_v = HasSIMDAtan2<T1,T2>::value;
//*************************************************************************************************

} // namespace blaze

#endif
