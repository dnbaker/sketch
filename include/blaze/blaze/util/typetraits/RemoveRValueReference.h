//=================================================================================================
/*!
//  \file blaze/util/typetraits/RemoveRValueReference.h
//  \brief Header file for the RemoveRValueReference type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_REMOVERVALUEREFERENCE_H_
#define _BLAZE_UTIL_TYPETRAITS_REMOVERVALUEREFERENCE_H_


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Removal of reference modifiers.
// \ingroup type_traits
//
// The RemoveRValueReference type trait removes an rvalue reference modifiers from the given
// type \a T.

   \code
   blaze::RemoveRValueReference<int>::Type             // Results in 'int'
   blaze::RemoveRValueReference<const int&>::Type      // Results in 'const int&'
   blaze::RemoveRValueReference<volatile int&&>::Type  // Results in 'volatile int'
   blaze::RemoveRValueReference<int*>::Type            // Results in 'int*'
   blaze::RemoveRValueReference<int*&>::Type           // Results in 'int*&'
   blaze::RemoveRValueReference<int*&&>::Type          // Results in 'int*'
   \endcode
*/
template< typename T >
struct RemoveRValueReference
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = T;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//! Specialization of the RemoveRValueReference type trait for rvalue references.
template< typename T >
struct RemoveRValueReference< T&& >
{
 public:
   //**********************************************************************************************
   using Type = T;
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the RemoveRValueReference type trait.
// \ingroup type_traits
//
// The RemoveRValueReference_t alias declaration provides a convenient shortcut to access the
// nested \a Type of the RemoveRValueReference class template. For instance, given the type \a T
// the following two type definitions are identical:

   \code
   using Type1 = typename blaze::RemoveRValueReference<T>::Type;
   using Type2 = blaze::RemoveRValueReference_t<T>;
   \endcode
*/
template< typename T >
using RemoveRValueReference_t = typename RemoveRValueReference<T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
