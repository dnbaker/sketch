//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsAdaptor.h
//  \brief Header file for the IsAdaptor type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISADAPTOR_H_
#define _BLAZE_MATH_TYPETRAITS_ISADAPTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/IntegralConstant.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for adaptors.
// \ingroup math_type_traits
//
// This type trait tests whether the given template parameter is an adaptor type (for instance
// \a LowerMatrix, \a UpperMatrix, or \a SymmetricMatrix). In case the type is an adaptor type,
// the \a value member constant is set to \a true, the nested type definition \a Type is
// \a TrueType, and the class derives from \a TrueType. Otherwise \a value is set to \a false,
// \a Type is \a FalseType, and the class derives from \a FalseType. The following example
// demonstrates this by means of the mentioned matrix adaptors:

   \code
   using blaze::rowMajor;

   blaze::IsAdaptor

   using StaticMatrixType     = blaze::StaticMatrix<double,3UL,3UL,rowMajor>;
   using DynamicMatrixType    = blaze::DynamicMatrix<float,rowMajor>;
   using CompressedMatrixType = blaze::CompressedMatrix<int,rowMajor>;

   using LowerStaticType         = blaze::LowerMatrix<StaticMatrixType>;
   using UpperDynamicType        = blaze::UpperMatrix<DynamicMatrixType>;
   using SymmetricCompressedType = blaze::SymmetricMatrix<CompressedMatrixType>;

   blaze::IsAdaptor< LowerStaticType >::value            // Evaluates to 1
   blaze::IsAdaptor< const UpperDynamicType >::Type      // Results in TrueType
   blaze::IsAdaptor< volatile SymmetricCompressedType >  // Is derived from TrueType
   blaze::IsAdaptor< StaticMatrixType >::value           // Evaluates to 0
   blaze::IsAdaptor< const DynamicMatrixType >::Type     // Results in FalseType
   blaze::IsAdaptor< volatile CompressedMatrixType >     // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsAdaptor
   : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsAdaptor type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsAdaptor< const T >
   : public IsAdaptor<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsAdaptor type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsAdaptor< volatile T >
   : public IsAdaptor<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsAdaptor type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsAdaptor< const volatile T >
   : public IsAdaptor<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsAdaptor type trait.
// \ingroup math_type_traits
//
// The IsAdaptor_v variable template provides a convenient shortcut to access the nested
// \a value of the IsAdaptor class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsAdaptor<T>::value;
   constexpr bool value2 = blaze::IsAdaptor_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsAdaptor_v = IsAdaptor<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
