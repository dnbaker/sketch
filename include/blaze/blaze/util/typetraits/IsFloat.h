//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsFloat.h
//  \brief Header file for the IsFloat type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISFLOAT_H_
#define _BLAZE_UTIL_TYPETRAITS_ISFLOAT_H_


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
/*!\brief Compile time check for single precision floating point types.
// \ingroup type_traits
//
// This type trait tests whether or not the given template parameter is of float type. In
// case the type is float (ignoring the cv-qualifiers), the \a value member constant is set
// to \a true, the nested type definition \a Type is \a TrueType, and the class derives from
// \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class
// derives from \a FalseType.

   \code
   blaze::IsFloat<float>::value          // Evaluates to 'true'
   blaze::IsFloat<const float>::Type     // Results in TrueType
   blaze::IsFloat<const volatile float>  // Is derived from TrueType
   blaze::IsFloat<double>::value         // Evaluates to 'false'
   blaze::IsFloat<const int>::Type       // Results in FalseType
   blaze::IsFloat<volatile short>        // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsFloat
   : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsFloat type trait for the plain 'float' type.
template<>
struct IsFloat<float>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsFloat type trait for 'const float'.
template<>
struct IsFloat<const float>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsFloat type trait for 'volatile float'.
template<>
struct IsFloat<volatile float>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsFloat type trait for 'const volatile float'.
template<>
struct IsFloat<const volatile float>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsFloat type trait.
// \ingroup type_traits
//
// The IsFloat_v variable template provides a convenient shortcut to access the nested \a value
// of the IsFloat class template. For instance, given the type \a T the following two statements
// are identical:

   \code
   constexpr bool value1 = blaze::IsFloat<T>::value;
   constexpr bool value2 = blaze::IsFloat_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsFloat_v = IsFloat<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
