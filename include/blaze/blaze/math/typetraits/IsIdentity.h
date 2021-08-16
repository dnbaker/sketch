//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsIdentity.h
//  \brief Header file for the IsIdentity type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISIDENTITY_H_
#define _BLAZE_MATH_TYPETRAITS_ISIDENTITY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/util/IntegralConstant.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for identity matrices.
// \ingroup math_type_traits
//
// This type trait tests whether or not the given template parameter is an identity matrix type
// (i.e. a matrix type that is guaranteed to be an identity matrix at compile time). In case the
// type is an identity matrix type, the \a value member constant is set to \a true, the nested
// type definition \a Type is \a TrueType, and the class derives from \a TrueType. Otherwise
// \a value is set to \a false, \a Type is \a FalseType, and the class derives from \a FalseType.

   \code
   using DiagonalMatrixType = blaze::DiagonalMatrix< StaticMatrix<float,3UL,3UL> >;

   blaze::IsIdentity< IdentityMatrix<int> >::value         // Evaluates to 1
   blaze::IsIdentity< const IdentityMatrix<float> >::Type  // Results in TrueType
   blaze::IsIdentity< volatile IdentityMatrix<double> >    // Is derived from TrueType
   blaze::IsIdentity< DynamicMatrix<int> >::value          // Evaluates to 0
   blaze::IsIdentity< const DiagonalMatrixType >::Type     // Results in FalseType
   blaze::IsIdentity< volatile CompressedMatrix<double> >  // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsIdentity
   : public BoolConstant< IsUniLower_v<T> && IsUniUpper_v<T> >
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsIdentity type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsIdentity< const T >
   : public IsIdentity<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsIdentity type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsIdentity< volatile T >
   : public IsIdentity<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsIdentity type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsIdentity< const volatile T >
   : public IsIdentity<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsIdentity type trait.
// \ingroup math_type_traits
//
// The IsIdentity_v variable template provides a convenient shortcut to access the nested
// \a value of the IsIdentity class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsIdentity<T>::value;
   constexpr bool value2 = blaze::IsIdentity_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsIdentity_v = IsIdentity<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
