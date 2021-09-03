//=================================================================================================
/*!
//  \file blaze/util/typetraits/Extent.h
//  \brief Header file for the Extent type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_EXTENT_H_
#define _BLAZE_UTIL_TYPETRAITS_EXTENT_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/IntegralConstant.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for the size of array bounds.
// \ingroup type_traits
//
// Via this type trait it is possible to query at compile time for the size of a particular
// array extent. In case the given template argument is an array type with a rank greater
// than N, the \a value member constant is set to the number of elements of the N'th array
// dimension. In all other cases, and especially in case the N'th array dimension is
// incomplete, \a value is set to 0.

   \code
   blaze::Extent< int[4], 0UL >::value            // Evaluates to 4
   blaze::Extent< int[2][3][4], 0UL >::value      // Evaluates to 2
   blaze::Extent< int[2][3][4], 1UL >::value      // Evaluates to 3
   blaze::Extent< int[2][3][4], 2UL >::value      // Evaluates to 4
   blaze::Extent< int[][2], 0UL >::value          // Evaluates to 0
   blaze::Extent< int[][2], 1UL >::value          // Evaluates to 2
   blaze::Extent< int*, 0UL >::value              // Evaluates to 0
   blaze::Extent< std::vector<int>, 0UL >::value  // Evaluates to 0 (std::vector is NOT an array type)
   \endcode
*/
template< typename T, size_t N >
struct Extent
   : public IntegralConstant<size_t,0UL>
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Partial specialization of the Extent type trait for empty array extents.
template< typename T, size_t N >
struct Extent<T[],N>
   : public IntegralConstant<size_t,Extent<T,N-1UL>::value>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Partial specialization of the Extent type trait for non-empty array extents.
template< typename T, size_t N, size_t E >
struct Extent<T[E],N>
   : public IntegralConstant<size_t,Extent<T,N-1UL>::value>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Terminating partial specialization of the Extent type trait for empty array extents.
template< typename T >
struct Extent<T[],0UL>
   : public IntegralConstant<size_t,0UL>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Terminating partial specialization of the Extent type trait for non-empty array extents.
template< typename T, size_t E >
struct Extent<T[E],0UL>
   : public IntegralConstant<size_t,E>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the Extent type trait.
// \ingroup type_traits
//
// The Extent_v variable template provides a convenient shortcut to access the nested \a value
// of the Extent class template. For instance, given the type \a T and the compile time constant
// \a N the following two statements are identical:

   \code
   constexpr size_t value1 = blaze::Extent<T,N>::value;
   constexpr size_t value2 = blaze::Extent_v<T,N>;
   \endcode
*/
template< typename T, size_t N >
constexpr size_t Extent_v = Extent<T,N>::value;
//*************************************************************************************************

} // namespace blaze

#endif
