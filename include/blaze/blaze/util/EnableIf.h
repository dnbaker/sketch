//=================================================================================================
/*!
//  \file blaze/util/EnableIf.h
//  \brief Header file for the EnableIf class template
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

#ifndef _BLAZE_UTIL_ENABLEIF_H_
#define _BLAZE_UTIL_ENABLEIF_H_


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Substitution Failure Is Not An Error (SFINAE) class.
// \ingroup util
//
// The EnableIf class template is an auxiliary tool for an intentional application of the
// Substitution Failure Is Not An Error (SFINAE) principle. It allows a function template
// or a class template specialization to include or exclude itself from a set of matching
// functions or specializations based on properties of its template arguments. For instance,
// it can be used to restrict the selection of a function template to specific data types.
// The following example illustrates this in more detail.

   \code
   template< typename Type >
   void process( Type t ) { ... }
   \endcode

// Due to the general formulation of this function, it will always be a possible candidate for
// every possible argument. However, with the EnableIf class it is for example possible to
// restrict the valid argument types to built-in, numeric data types.

   \code
   template< typename Type >
   typename EnableIf< IsNumeric_v<Type> >::Type process( Type t ) { ... }
   \endcode

// In case the given data type is not a built-in, numeric data type, the access to the nested
// type defintion \a Type of the EnableIf template will fail. However, due to the SFINAE
// principle, this will only result in a compilation error in case the compiler cannot find
// another valid function.\n
// Note that in this application of the EnableIf template the default for the nested type
// definition \a Type is used, which corresponds to \a void. Via the second template argument
// it is possible to explicitly specify the type of \a Type:

   \code
   // Explicity specifying the default
   typename EnableIf< IsNumeric_v<Type>, void >::Type

   // In case the given data type is a boolean data type, the nested type definition
   // 'Type' is set to float
   typename EnableIf< IsBoolean_v<Type>, float >::Type
   \endcode

// For more information on the EnableIf functionality, see the standard library documentation
// of std::enable_if at:
//
//           \a http://en.cppreference.com/w/cpp/types/enable_if.
*/
template< bool Condition     // Compile time condition
        , typename T=void >  // The type to be instantiated
struct EnableIf
{
   //**********************************************************************************************
   using Type = T;  //!< The instantiated type.
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief EnableIf specialization for failed constraints.
// \ingroup util
//
// This specialization of the EnableIf template is selected if the first template parameter (the
// compile time condition) evaluates to \a false. This specialization does not contains a nested
// type definition \a Type and therefore always results in a compilation error in case \a Type
// is accessed. However, due to the SFINAE principle the compilation process is not necessarily
// stopped if another, valid instantiation is found by the compiler.
*/
template< typename T >  // The type to be instantiated
struct EnableIf<false,T>
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary type for the EnableIf class template.
// \ingroup util
//
// The EnableIf_t alias declaration provides a convenient shortcut to access the nested \a Type
// of the EnableIf class template. For instance, given the type \a T the following two type
// definitions are identical:

   \code
   using Type1 = typename EnableIf< IsBuiltin_v<T> >::Type;
   using Type2 = EnableIf_t< IsBuiltin_v<T> >;
   \endcode
*/
template< bool Condition     // Compile time condition
        , typename T=void >  // The type to be instantiated
using EnableIf_t = typename EnableIf<Condition,T>::Type;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary type for the EnableIf class template.
// \ingroup util
//
// The DisableIf alias declaration provides a convenient shortcut for negated SFINAE conditions.
// For instance, given the type \a T the following two type definitions are identical:

   \code
   using Type1 = typename EnableIf< !IsBuiltin_v<T> >::Type;
   using Type2 = typename DisableIf< IsBuiltin_v<T> >::Type;
   \endcode
*/
template< bool Condition     // Compile time condition
        , typename T=void >  // The type to be instantiated
using DisableIf = EnableIf<!Condition,T>;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary type for the EnableIf class template.
// \ingroup util
//
// The DisableIf_t alias declaration provides a convenient shortcut to access the nested \a Type
// of the negated EnableIf class template. For instance, given the type \a T the following two
// type definitions are identical:

   \code
   using Type1 = typename EnableIf< !IsBuiltin_v<T> >::Type;
   using Type2 = DisableIf_t< IsBuiltin_v<T> >;
   \endcode
*/
template< bool Condition     // Compile time condition
        , typename T=void >  // The type to be instantiated
using DisableIf_t = typename EnableIf<!Condition,T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
