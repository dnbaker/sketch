//=================================================================================================
/*!
//  \file blaze/util/typetraits/GetMemberType.h
//  \brief Header file for the GetMemberType type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_GETMEMBERTYPE_H_
#define _BLAZE_UTIL_TYPETRAITS_GETMEMBERTYPE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/typetraits/Void.h>


namespace blaze {

//=================================================================================================
//
//  MACRO DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Macro for the creation of a type trait to acquire member types.
// \ingroup math_type_traits
//
// This macro creates the definition of a type trait \a TYPE_TRAIT_NAME that can determine a
// specified member type of a given type. The first macro parameter \a TYPE_TRAIT_NAME specifies
// the resulting name of the type trait. The second parameter \a MEMBER_NAME specifies the name
// of the member type to be acquired and the third parameter \a FALLBACK_TYPE specifies the type
// to acquire in case the given type doesn't contain the specified member type. The following
// example demonstrates the use of the macro and the resulting type trait:

   \code
   struct MyType1 {
      using ElementType = float;
   };

   struct MyType2 {
      using ElementType = double;
   };

   struct MyType3 {};

   BLAZE_CREATE_GET_TYPE_MEMBER_TYPE_TRAIT( GetElementType, ElementType, int );

   GetElementType<MyType1>::Type  // Results in 'float'
   GetElementType<MyType2>::Type  // Results in 'double'
   GetElementType<MyType3>::Type  // Results in 'int'
   \endcode

// The macro results in the definition of a new class with the specified name \a TYPE_TRAIT_NAME
// and an associated alias template called TYPE_TRAIT_NAME_t within the current namespace. This
// may cause name collisions with any other entity called \a TYPE_TRAIT_NAME in the same namespace.
// Therefore it is advisable to create the type trait as locally as possible to minimize the
// probability of name collisions. Note however that the macro cannot be used within function
// scope since a template declaration cannot appear at block scope.
*/
#define BLAZE_CREATE_GET_TYPE_MEMBER_TYPE_TRAIT( TYPE_TRAIT_NAME, MEMBER_NAME, FALLBACK_TYPE )  \
                                                                                                \
template< typename Type1233, typename = void >                                                  \
struct TYPE_TRAIT_NAME                                                                          \
{                                                                                               \
   using Type = FALLBACK_TYPE;                                                                  \
};                                                                                              \
                                                                                                \
template< typename Type1233 >                                                                   \
struct TYPE_TRAIT_NAME< Type1233, blaze::Void_t< typename Type1233::MEMBER_NAME > >             \
{                                                                                               \
   using Type = typename Type1233::MEMBER_NAME;                                                 \
};                                                                                              \
                                                                                                \
template< typename Type1233 >                                                                   \
using TYPE_TRAIT_NAME##_t = typename TYPE_TRAIT_NAME<Type1233>::Type
//*************************************************************************************************

} // namespace blaze

#endif
