//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsRestricted.h
//  \brief Header file for the IsRestricted type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISRESTRICTED_H_
#define _BLAZE_MATH_TYPETRAITS_ISRESTRICTED_H_


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
/*!\brief Compile time check for data types with restricted data access.
// \ingroup math_type_traits
//
// This type trait tests whether the given data type has a restricted data access. Examples are
// the LowerMatrix and UpperMatrix adaptors that don't allow write access to the elements in the
// upper or lower part of the matrix, respectively. In case the data type has a restricted data
// access, the \a value member constant is set to \a true, the nested type definition \a Type
// is \a TrueType, and the class derives from \a TrueType. Otherwise \a value is set to \a false,
// \a Type is \a FalseType, and the class derives from \a FalseType. Examples:

   \code
   using VectorType = blaze::StaticVector<int,3UL>;
   using MatrixType = blaze::DynamicMatrix<double>;

   using Lower = blaze::LowerMatrix< blaze::DynamicMatrix<double> >;
   using Upper = blaze::LowerMatrix< blaze::CompressedMatrix<int> >;

   blaze::IsRestricted< Lower >::value            // Evaluates to 1
   blaze::IsRestricted< const Upper >::Type       // Results in TrueType
   blaze::IsRestricted< volatile Lower >          // Is derived from TrueType
   blaze::IsRestricted< int >::value              // Evaluates to 0
   blaze::IsRestricted< const VectorType >::Type  // Results in FalseType
   blaze::IsRestricted< volatile MatrixType >     // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsRestricted
   : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsRestricted type trait for const types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsRestricted< const T >
   : public IsRestricted<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsRestricted type trait for volatile types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsRestricted< volatile T >
   : public IsRestricted<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsRestricted type trait for cv qualified types.
// \ingroup math_type_traits
*/
template< typename T >
struct IsRestricted< const volatile T >
   : public IsRestricted<T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsRestricted type trait.
// \ingroup math_type_traits
//
// The IsRestricted_v variable template provides a convenient shortcut to access the nested
// \a value of the IsRestricted class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsRestricted<T>::value;
   constexpr bool value2 = blaze::IsRestricted_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsRestricted_v = IsRestricted<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
