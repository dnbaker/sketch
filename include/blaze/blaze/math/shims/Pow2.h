//=================================================================================================
/*!
//  \file blaze/math/shims/Pow2.h
//  \brief Header file for the pow2 shim
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

#ifndef _BLAZE_MATH_SHIMS_POW2_H_
#define _BLAZE_MATH_SHIMS_POW2_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/IsScalar.h>
#include <blaze/math/typetraits/IsSIMDPack.h>
#include <blaze/system/Inline.h>
#include <blaze/util/EnableIf.h>


namespace blaze {

//=================================================================================================
//
//  POW2 SHIM
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Squaring the given value/object.
// \ingroup math_shims
//
// \param a The value/object to be squared.
// \return The result of the square operation.
//
// The \a pow2 shim represents an abstract interface for squaring a value/object of any given
// data type. For values of built-in data type this results in a plain multiplication.
*/
template< typename T
        , EnableIf_t< IsScalar_v<T> || IsSIMDPack_v<T> >* = nullptr >
BLAZE_ALWAYS_INLINE constexpr decltype(auto) pow2( const T& a ) noexcept( noexcept( a * a ) )
{
   return ( a * a );
}
//*************************************************************************************************

} // namespace blaze

#endif
