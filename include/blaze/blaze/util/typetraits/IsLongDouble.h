//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsLongDouble.h
//  \brief Header file for the IsLongDouble type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISLONGDOUBLE_H_
#define _BLAZE_UTIL_TYPETRAITS_ISLONGDOUBLE_H_


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
/*!\brief Compile time check for extended precision floating point types.
// \ingroup type_traits
//
// This type trait tests whether or not the given template parameter is of long double type. In
// case the type is long double (ignoring the cv-qualifiers), the \a value member constant is
// set to \a true, the nested type definition \a Type is \a TrueType, and the class derives
// from \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and the
// class derives from \a FalseType.

   \code
   blaze::IsLongDouble<long double>::value          // Evaluates to 'true'
   blaze::IsLongDouble<const long double>::Type     // Results in TrueType
   blaze::IsLongDouble<const volatile long double>  // Is derived from TrueType
   blaze::IsLongDouble<float>::value                // Evaluates to 'false'
   blaze::IsLongDouble<const unsigned int>::Type    // Results in FalseType
   blaze::IsLongDouble<const volatile short>        // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsLongDouble
   : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsLongDouble type trait for the plain 'long double' type.
template<>
struct IsLongDouble<long double>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsLongDouble type trait for 'const long double'.
template<>
struct IsLongDouble<const long double>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsLongDouble type trait for 'volatile long double'.
template<>
struct IsLongDouble<volatile long double>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsLongDouble type trait for 'const volatile long double'.
template<>
struct IsLongDouble<const volatile long double>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsLongDouble type trait.
// \ingroup type_traits
//
// The IsLongDouble_v variable template provides a convenient shortcut to access the nested
// \a value of the IsLongDouble class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsLongDouble<T>::value;
   constexpr bool value2 = blaze::IsLongDouble_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsLongDouble_v = IsLongDouble<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
