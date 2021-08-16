//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsAligned.h
//  \brief Header file for the IsAligned type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISALIGNED_H_
#define _BLAZE_MATH_TYPETRAITS_ISALIGNED_H_


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
/*!\brief Compile time check for the alignment of data types.
// \ingroup math_type_traits
//
// This type trait tests whether the given data type guarantees to provide aligned data values
// with respect to the requirements of the available instruction set. For instance, vectorizable
// data types such as built-in and complex data types are required to be 16-bit aligned for SSE,
// 32-bit aligned for AVX, and 64-bit aligned for MIC. In case the data type is properly aligned,
// the \a value member constant is set to \a true, the nested type definition \a Type is
// \a TrueType, and the class derives from \a TrueType. Otherwise \a value is set to
// \a false, \a Type is \a FalseType, and the class derives from \a FalseType. Examples:

   \code
   using blaze::StaticVector;
   using blaze::CustomVector;
   using blaze::CompressedVector;
   using blaze::DynamicMatrix;
   using blaze::CustomMatrix;
   using blaze::aligned;
   using blaze::unaligned;
   using blaze::padded;
   using blaze::unpadded;

   blaze::IsAligned< StaticVector<int,3UL> >::value                         // Evaluates to 1
   blaze::IsAligned< const DynamicMatrix<double> >::Type                    // Results in TrueType
   blaze::IsAligned< volatile CustomVector<float,aligned,unpadded> >        // Is derived from TrueType
   blaze::IsAligned< CompressedVector<int> >::value                         // Evaluates to 0
   blaze::IsAligned< const CustomVector<double,unaligned,unpadded> >::Type  // Results in FalseType
   blaze::IsAligned< volatile CustomMatrix<float,unaligned,padded> >        // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsAligned
   : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsAligned type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsAligned< const T >
   : public IsAligned<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsAligned type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsAligned< volatile T >
   : public IsAligned<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsAligned type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsAligned< const volatile T >
   : public IsAligned<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsAligned type trait.
// \ingroup math_type_traits
//
// The IsAligned_v variable template provides a convenient shortcut to access the nested
// \a value of the IsAligned class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsAligned<T>::value;
   constexpr bool value2 = blaze::IsAligned_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsAligned_v = IsAligned<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
