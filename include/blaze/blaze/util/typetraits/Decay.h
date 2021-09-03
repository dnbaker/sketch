//=================================================================================================
/*!
//  \file blaze/util/typetraits/Decay.h
//  \brief Header file for the Decay type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_DECAY_H_
#define _BLAZE_UTIL_TYPETRAITS_DECAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <type_traits>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Applies the type conversions for by-value function arguments.
// \ingroup type_traits
//
// This type trait applies the type conversions that are used for by-value function arguments.
// This conversions include lvalue-to-rvalue, array-to-pointer, and function-to-pointer implicit
// conversions to the type \c T, and the removal of top level cv-qualifiers.

   \code
   blaze::Decay<int>::Type         // Results in 'int'
   blaze::Decay<int&>::Type        // Results in 'int'
   blaze::Decay<int&&>::Type       // Results in 'int'
   blaze::Decay<const int&>::Type  // Results in 'int'
   blaze::Decay<int[2]>::Type      // Results in 'int*'
   blaze::Decay<int(int)>::Type    // Results in 'int(*)(int)'
   \endcode
*/
template< typename T >
struct Decay
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using Type = typename std::decay<T>::type;
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias declaration for the Decay type trait.
// \ingroup type_traits
//
// The Decay_t alias declaration provides a convenient shortcut to access the nested \a Type of
// the Decay class template. For instance, given the type \a T the following two type definitions
// are identical:

   \code
   using Type1 = typename blaze::Decay<T>::Type;
   using Type2 = blaze::Decay_t<T>;
   \endcode
*/
template< typename T >
using Decay_t = typename Decay<T>::Type;
//*************************************************************************************************

} // namespace blaze

#endif
