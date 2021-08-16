//=================================================================================================
/*!
//  \file blaze/util/mpl/If.h
//  \brief Header file for the If class template
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

#ifndef _BLAZE_UTIL_MPL_IF_H_
#define _BLAZE_UTIL_MPL_IF_H_


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Compile time type selection.
// \ingroup mpl
//
// The If class template selects one of the two given types \a T1 and \a T2 depending on the
// \a Condition template argument. In case the \a Condition compile time constant expression
// evaluates to \a true, the member type definition \a Type is set to \a T1. In case
// \a Condition evaluates to \a false, \a Type is set to \a T2.
*/
template< bool Condition >  // Compile time selection
struct If
{
 public:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   template< typename T1, typename T2 >
   using Type = T1;  //!< The selected type.
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the If class template.
// \ingroup mpl
//
// This specialization of the If template is selected in case the \a Condition compile time
// constant expression evaluates to \a false. The member type definition is set to the second
// given type \a T2.
*/
template<>
struct If<false>
{
 public:
   //**********************************************************************************************
   template< typename T1, typename T2 >
   using Type = T2;  //!< The selected type.
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary alias template for the If class template.
// \ingroup util
//
// The If_t alias template provides a convenient shortcut to access the nested \a Type of
// the If class template. For instance, given the types \a C, \a T1, and \a T2 the following
// two type definitions are identical:

   \code
   using Type1 = typename If< IsBuiltin_v<C>, T1, T2 >::Type;
   using Type2 = If_t< IsBuiltin_v<C>, T1, T2 >;
   \endcode
*/
template< bool Condition  // Compile time selection
        , typename T1     // Type to be selected if Condition=true
        , typename T2 >   // Type to be selected if Condition=false
using If_t = typename If<Condition>::template Type<T1,T2>;
//*************************************************************************************************

} // namespace blaze

#endif
