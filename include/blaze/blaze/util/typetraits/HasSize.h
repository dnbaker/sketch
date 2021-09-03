//=================================================================================================
/*!
//  \file blaze/util/typetraits/HasSize.h
//  \brief Header file for the HasSize type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_HASSIZE_H_
#define _BLAZE_UTIL_TYPETRAITS_HASSIZE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/IntegralConstant.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS HASSIZE
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time size check.
// \ingroup type_traits
//
// This class offers the possibility to test the size of a type at compile time. If the type
// \a T is exactly \a S bytes large, the \a value member constant is set to \a true, the
// nested type definition \a Type is \a TrueType, and the class derives from \a TrueType.
// Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class derives from
// \a FalseType.

   \code
   blaze::HasSize<int,4>::value              // Evaluates to 'true' (on most architectures)
   blaze::HasSize<float,4>::Type             // Results in TrueType (on most architectures)
   blaze::HasSize<const double,8>            // Is derived from TrueType (on most architectures)
   blaze::HasSize<volatile double,2>::value  // Evaluates to 'false'
   blaze::HasSize<const char,8>::Type        // Results in FalseType
   blaze::HasSize<unsigned char,4>           // Is derived from FalseType
   \endcode
*/
template< typename T, size_t S >
struct HasSize
   : public BoolConstant< sizeof( T ) == S >
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of the compile time size constraint.
// \ingroup type_traits
//
// This class ia a partial specialization of the HasSize template for the type \a void. This
// specialization assumes that an object of type \a void has a size of 0. Therefore \a value
// is set to \a true, \a Type is \a TrueType, and the class derives from \a TrueType only if the
// \a S template argument is 0. Otherwise \a value is set to \a false, \a Type is \a FalseType,
// and the class derives from \a FalseType.
*/
template< size_t S >
struct HasSize<void,S>
   : public BoolConstant< 0 == S >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of the compile time size constraint.
// \ingroup type_traits
//
// This class ia a partial specialization of the HasSize template for constant \a void. This
// specialization assumes that an object of type \a void has a size of 0. Therefore \a value
// is set to \a true, \a Type is \a TrueType, and the class derives from \a TrueType only if
// the \a S template argument is 0. Otherwise \a value is set to \a false, \a Type is
// \a FalseType, and the class derives from \a FalseType.
*/
template< size_t S >
struct HasSize<const void,S>
   : public BoolConstant< 0 == S >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of the compile time size constraint.
// \ingroup type_traits
//
// This class ia a partial specialization of the HasSize template for volatile \a void. This
// specialization assumes that an object of type \a void has a size of 0. Therefore \a value
// is set to \a true, \a Type is \a TrueType, and the class derives from \a TrueType only if
// the \a S template argument is 0. Otherwise \a value is set to \a false, \a Type is
// \a FalseType, and the class derives from \a FalseType.
*/
template< size_t S >
struct HasSize<volatile void,S>
   : public BoolConstant< 0 == S >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Partial specialization of the compile time size constraint.
// \ingroup type_traits
//
// This class ia a partial specialization of the HasSize template for constant volatile \a void.
// This specialization assumes that an object of type \a void has a size of 0. Therefore \a value
// is set to \a true, \a Type is \a TrueType, and the class derives from \a TrueType only if the
// \a S template argument is 0. Otherwise \a value is set to \a false, \a Type is \a FalseType,
// and the class derives from \a FalseType.
*/
template< size_t S >
struct HasSize<const volatile void,S>
   : public BoolConstant< 0 == S >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the HasSize type trait.
// \ingroup type_traits
//
// The HasSize_v variable template provides a convenient shortcut to access the nested
// \a value of the HasSize class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::HasSize<T,8UL>::value;
   constexpr bool value2 = blaze::HasSize_v<T,8UL>;
   \endcode
*/
template< typename T, size_t S >
constexpr bool HasSize_v = HasSize<T,S>::value;
//*************************************************************************************************




//=================================================================================================
//
//  CLASS HAS1BYTE
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time size check.
// \ingroup type_traits
//
// This type trait offers the possibility to test whether a given type has a size of exactly
// one byte. If the type \a T has one byte, the \a value member constant is set to \a true,
// the nested type definition \a Type is \a TrueType, and the class derives from \a TrueType.
// Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class derives from
// \a FalseType.

   \code
   blaze::Has1Byte<const char>::value       // Evaluates to 'true' (on most architectures)
   blaze::Has1Byte<unsigned char>::Type     // Results in TrueType (on most architectures)
   blaze::Has1Byte<signed char>             // Is derived from TrueType (on most architectures)
   blaze::Has1Byte<volatile double>::value  // Evaluates to 'false'
   blaze::Has1Byte<const float>::Type       // Results in FalseType
   blaze::Has1Byte<unsigned short>          // Is derived from FalseType
   \endcode
*/
template< typename T >
struct Has1Byte
   : public HasSize<T,1UL>
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the Has1Byte type trait.
// \ingroup type_traits
//
// The Has1Byte_v variable template provides a convenient shortcut to access the nested
// \a value of the Has1Byte class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::Has1Byte<T>::value;
   constexpr bool value2 = blaze::Has1Byte_v<T>;
   \endcode
*/
template< typename T >
constexpr bool Has1Byte_v = Has1Byte<T>::value;
//*************************************************************************************************




//=================================================================================================
//
//  CLASS HAS2BYTES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time size check.
// \ingroup type_traits
//
// This type trait offers the possibility to test whether a given type has a size of exactly
// two bytes. If the type \a T has two bytes, the \a value member constant is set to \a true,
// the nested type definition \a Type is \a TrueType, and the class derives from \a TrueType.
// Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class derives from
// \a FalseType.

   \code
   blaze::Has2Bytes<const short>::value      // Evaluates to 'true' (on most architectures)
   blaze::Has2Bytes<unsigned short>::Type    // Results in TrueType (on most architectures)
   blaze::Has2Bytes<volatile short>          // Is derived from TrueType (on most architectures)
   blaze::Has2Bytes<volatile double>::value  // Evaluates to 'false'
   blaze::Has2Bytes<const float>::Type       // Results in FalseType
   blaze::Has2Bytes<unsigned int>            // Is derived from FalseType
   \endcode
*/
template< typename T >
struct Has2Bytes
   : public HasSize<T,2UL>
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the Has2Bytes type trait.
// \ingroup type_traits
//
// The Has2Bytes_v variable template provides a convenient shortcut to access the nested
// \a value of the Has2Bytes class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::Has2Bytes<T>::value;
   constexpr bool value2 = blaze::Has2Bytes_v<T>;
   \endcode
*/
template< typename T >
constexpr bool Has2Bytes_v = Has2Bytes<T>::value;
//*************************************************************************************************




//=================================================================================================
//
//  CLASS HAS4BYTES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time size check.
// \ingroup type_traits
//
// This type trait offers the possibility to test whether a given type has a size of exactly
// four bytes. If the type \a T has four bytes, the \a value member constant is set to \a true,
// the nested type definition \a Type is \a TrueType, and the class derives from \a TrueType.
// Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class derives from
// \a FalseType.

   \code
   blaze::Has4Bytes<const int>::value        // Evaluates to 'true' (on most architectures)
   blaze::Has4Bytes<unsigned int>::Type      // Results in TrueType (on most architectures)
   blaze::Has4Bytes<volatile float>          // Is derived from TrueType (on most architectures)
   blaze::Has4Bytes<volatile double>::value  // Evaluates to 'false'
   blaze::Has4Bytes<const float>::Type       // Results in FalseType
   blaze::Has4Bytes<short>                   // Is derived from FalseType
   \endcode
*/
template< typename T >
struct Has4Bytes
   : public HasSize<T,4UL>
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the Has4Bytes type trait.
// \ingroup type_traits
//
// The Has4Bytes_v variable template provides a convenient shortcut to access the nested
// \a value of the Has4Bytes class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::Has4Bytes<T>::value;
   constexpr bool value2 = blaze::Has4Bytes_v<T>;
   \endcode
*/
template< typename T >
constexpr bool Has4Bytes_v = Has4Bytes<T>::value;
//*************************************************************************************************




//=================================================================================================
//
//  CLASS HAS8BYTES
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time size check.
// \ingroup type_traits
//
// This type trait offers the possibility to test whether a given type has a size of exactly
// four bytes. If the type \a T has four bytes, the \a value member constant is set to \a true,
// the nested type definition \a Type is \a TrueType, and the class derives from\a TrueType.
// Otherwise \a value is set to \a false, \a Type is \a FalseType, and the classderives from
// \a FalseType.

   \code
   blaze::Has8Bytes<double>::value        // Evaluates to 'true' (on most architectures)
   blaze::Has8Bytes<const double>::Type   // Results in TrueType (on most architectures)
   blaze::Has8Bytes<volatile double>      // Is derived from TrueType (on most architectures)
   blaze::Has8Bytes<unsigned int>::value  // Evaluates to 'false'
   blaze::Has8Bytes<const float>::Type    // Results in FalseType
   blaze::Has8Bytes<volatile short>       // Is derived from FalseType
   \endcode
*/
template< typename T >
struct Has8Bytes
   : public HasSize<T,8UL>
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the Has8Bytes type trait.
// \ingroup type_traits
//
// The Has8Bytes_v variable template provides a convenient shortcut to access the nested
// \a value of the Has8Bytes class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::Has8Bytes<T>::value;
   constexpr bool value2 = blaze::Has8Bytes_v<T>;
   \endcode
*/
template< typename T >
constexpr bool Has8Bytes_v = Has8Bytes<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
