//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsInvertible.h
//  \brief Header file for the IsInvertible type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISINVERTIBLE_H_
#define _BLAZE_MATH_TYPETRAITS_ISINVERTIBLE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/IsBLASCompatible.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsScalar.h>
#include <blaze/math/typetraits/UnderlyingElement.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for data types.
// \ingroup math_type_traits
//
// This type trait tests whether or not the given template parameter is invertible. The
// type is considered to be invertible if it is a floating point type (\c float, \c double,
// or <tt>long double</tt>), any other scalar type with a floating point element type (e.g.
// \c complex<float>, \c complex<double> or <tt>complex<long double></tt>) or any dense matrix
// type with a BLAS compatible element type. If the given type is invertible, the \a value
// member constant is set to \a true, the nested type definition \a Type is \a TrueType, and
// the class derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is
// \a FalseType, and the class derives from \a FalseType.

   \code
   blaze::IsInvertible< float >::value                  // Evaluates to 1
   blaze::IsInvertible< complex<double> >::Type         // Results in TrueType
   blaze::IsInvertible< blaze::DynamicMatrix<double> >  // Is derived from TrueType
   blaze::IsInvertible< int >::value                    // Evaluates to 0
   blaze::IsInvertible< complex<int> >::Type            // Results in FalseType
   blaze::IsInvertible< blaze::DynamicVector<double> >  // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsInvertible
   : public BoolConstant< ( IsScalar_v<T> && IsFloatingPoint_v< UnderlyingElement_t<T> > ) ||
                          ( IsDenseMatrix_v<T> && IsBLASCompatible_v< UnderlyingElement_t<T> > ) >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsInvertible type trait.
// \ingroup math_type_traits
//
// The IsInvertible_v variable template provides a convenient shortcut to access the nested
// \a value of the IsInvertible class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsInvertible<T>::value;
   constexpr bool value2 = blaze::IsInvertible_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsInvertible_v = IsInvertible<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
