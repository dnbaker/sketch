//=================================================================================================
/*!
//  \file blaze/util/typetraits/IsBaseOf.h
//  \brief Header file for the IsBaseOf type trait
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

#ifndef _BLAZE_UTIL_TYPETRAITS_ISBASEOF_H_
#define _BLAZE_UTIL_TYPETRAITS_ISBASEOF_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <type_traits>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/RemoveCV.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time analysis of an inheritance relationship.
// \ingroup type_traits
//
// This type trait tests for an inheritance relationship between the two types \a Base and
// \a Derived. If \a Derived is a type derived from \a Base or the same type as \a Base the
// \a value member contant is set to \a true, the nested type definition \a Type is \a TrueType,
// and the class derives from \a TrueType. Otherwise \a value is set to \a false, \a Type is
// \a FalseType, and the class derives from \a FalseType.

   \code
   class A { ... };
   class B : public A { ... };
   class C { ... };

   blaze::IsBaseOf<A,B>::value  // Evaluates to 'true'
   blaze::IsBaseOf<A,B>::Type   // Results in TrueType
   blaze::IsBaseOf<A,B>         // Is derived from TrueType
   blaze::IsBaseOf<A,C>::value  // Evaluates to 'false'
   blaze::IsBaseOf<B,A>::Type   // Results in FalseType
   blaze::IsBaseOf<B,A>         // Is derived from FalseType
   \endcode
*/
template< typename Base, typename Derived >
class IsBaseOf
   : public BoolConstant< std::is_base_of< RemoveCV_t<Base>, RemoveCV_t<Derived> >::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsBaseOf type trait.
// \ingroup type_traits
//
// The IsBaseOf_v variable template provides a convenient shortcut to access the nested \a value
// of the IsBaseOf class template. For instance, given the types \a T1 and \a T2 the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsBaseOf<T1,T2>::value;
   constexpr bool value2 = blaze::IsBaseOf_v<T1,T2>;
   \endcode
*/
template< typename Base, typename Derived >
constexpr bool IsBaseOf_v = IsBaseOf<Base,Derived>::value;
//*************************************************************************************************

} // namespace blaze

#endif
