//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsSubvector.h
//  \brief Header file for the IsSubvector type trait
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

#ifndef _BLAZE_MATH_TYPETRAITS_ISSUBVECTOR_H_
#define _BLAZE_MATH_TYPETRAITS_ISSUBVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/views/Forward.h>
#include <blaze/util/IntegralConstant.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time check for subvectors.
// \ingroup math_type_traits
//
// This type trait tests whether or not the given template parameter is a subvector (i.e. a view
// on the part of a dense or sparse vector). In case the type is a subvector, the \a value member
// constant is set to \a true, the nested type definition \a Type is \a TrueType, and the class
// derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is \a FalseType, and
// the class derives from \a FalseType.

   \code
   using blaze::aligned;

   using VectorType1 = blaze::StaticVector<int,10UL>;
   using VectorType2 = blaze::DynamicVector<double>;
   using VectorType3 = blaze::CompressedVector<float>;

   VectorType1 a;
   VectorType2 b( 100UL );
   VectorType3 c( 200UL );

   using SubvectorType1 = decltype( blaze::subvector<2UL,4UL>( a ) );
   using SubvectorType2 = decltype( blaze::subvector<aligned>( b, 8UL 24UL ) );
   using SubvectorType3 = decltype( blaze::subvector( c, 5UL, 13UL ) );

   blaze::IsSubvector< SubvectorType1 >::value       // Evaluates to 1
   blaze::IsSubvector< const SubvectorType2 >::Type  // Results in TrueType
   blaze::IsSubvector< volatile SubvectorType3 >     // Is derived from TrueType
   blaze::IsSubvector< VectorType1 >::value          // Evaluates to 0
   blaze::IsSubvector< const VectorType2 >::Type     // Results in FalseType
   blaze::IsSubvector< volatile VectorType3 >        // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsSubvector
   : public FalseType
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsSubvector type trait for 'Subvector'.
// \ingroup math_type_traits
*/
template< typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs >
struct IsSubvector< Subvector<VT,AF,TF,DF,CSAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsSubvector type trait for 'const Subvector'.
// \ingroup math_type_traits
*/
template< typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs >
struct IsSubvector< const Subvector<VT,AF,TF,DF,CSAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsSubvector type trait for 'volatile Subvector'.
// \ingroup math_type_traits
*/
template< typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs >
struct IsSubvector< volatile Subvector<VT,AF,TF,DF,CSAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsSubvector type trait for 'const volatile Subvector'.
// \ingroup math_type_traits
*/
template< typename VT, AlignmentFlag AF, bool TF, bool DF, size_t... CSAs >
struct IsSubvector< const volatile Subvector<VT,AF,TF,DF,CSAs...> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsSubvector type trait.
// \ingroup math_type_traits
//
// The IsSubvector_v variable template provides a convenient shortcut to access the nested
// \a value of the IsSubvector class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsSubvector<T>::value;
   constexpr bool value2 = blaze::IsSubvector_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsSubvector_v = IsSubvector<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
