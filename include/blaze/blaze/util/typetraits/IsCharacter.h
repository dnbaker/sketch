//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsCharacter.h
//  \brief Header file for the IsCharacter type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISCHARACTER_H_
#define _BLAZE_UTIL_TYPETRAITS_ISCHARACTER_H_


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
/*!\brief Compile time check for character types.
// \ingroup type_traits
//
// This type trait tests whether or not the given template parameter is a character type
// (i.e., either char, signed char, unsigned char, or wchar_t, possibly cv-qualified).
// In case the type is a character type (ignoring the cv-qualifiers), the \a value member
// constant is set to \a true, the nested type definition \a Type is \a TrueType, and
// the class derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is
// \a FalseType, and the class derives from \a FalseType.

   \code
   blaze::IsCharacter<char>::value                // Evaluates to 'true'
   blaze::IsCharacter<const unsigned char>::Type  // Results in TrueType
   blaze::IsCharacter<const volatile wchar_t>     // Is derived from TrueType
   blaze::IsCharacter<unsigned short>::value      // Evaluates to 'false'
   blaze::IsCharacter<const int>::Type            // Results in FalseType
   blaze::IsCharacter<volatile long>              // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsCharacter
   : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsCharacter type trait for the plain 'char' type.
template<>
struct IsCharacter<char>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsCharacter type trait for the plain 'signed char' type.
template<>
struct IsCharacter<signed char>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsCharacter type trait for the plain 'unsigned char' type.
template<>
struct IsCharacter<unsigned char>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the IsCharacter type trait for the plain 'wchar_t' type.
template<>
struct IsCharacter<wchar_t>
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Partial specialization of the IsCharacter type trait 'const' qualified types.
template< typename T >
struct IsCharacter<const T>
   : public IsCharacter<T>::Type
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Partial specialization of the IsCharacter type trait 'volatile' qualified types.
template< typename T >
struct IsCharacter<volatile T>
   : public IsCharacter<T>::Type
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Partial specialization of the IsCharacter type trait 'const volatile' types.
template< typename T >
struct IsCharacter<const volatile T>
   : public IsCharacter<T>::Type
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsCharacter type trait.
// \ingroup type_traits
//
// The IsCharacter_v variable template provides a convenient shortcut to access the nested
// \a value of the IsCharacter class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsCharacter<T>::value;
   constexpr bool value2 = blaze::IsCharacter_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsCharacter_v = IsCharacter<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
