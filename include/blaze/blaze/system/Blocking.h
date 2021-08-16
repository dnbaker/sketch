//=================================================================================================
/*!
//  \file blaze/system/Blocking.h
//  \brief Header file for kernel specific block sizes
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

#ifndef _BLAZE_SYSTEM_BLOCKING_H_
#define _BLAZE_SYSTEM_BLOCKING_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Debugging.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  BLOCKING SETTINGS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
constexpr size_t DEFAULT_BLOCK_SIZE = 256UL;

constexpr size_t MMM_DEFAULT_OUTER_BLOCK_SIZE = 112UL;
constexpr size_t MMM_DEFAULT_INNER_BLOCK_SIZE =  96UL;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
constexpr size_t DEBUG_BLOCK_SIZE = 8UL;

constexpr size_t MMM_DEBUG_OUTER_BLOCK_SIZE = 16UL;
constexpr size_t MMM_DEBUG_INNER_BLOCK_SIZE = 16UL;
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
constexpr size_t BLOCK_SIZE = ( BLAZE_DEBUG_MODE ? DEBUG_BLOCK_SIZE : DEFAULT_BLOCK_SIZE );

constexpr size_t MMM_OUTER_BLOCK_SIZE = ( BLAZE_DEBUG_MODE ? MMM_DEBUG_OUTER_BLOCK_SIZE : MMM_DEFAULT_OUTER_BLOCK_SIZE );
constexpr size_t MMM_INNER_BLOCK_SIZE = ( BLAZE_DEBUG_MODE ? MMM_DEBUG_INNER_BLOCK_SIZE : MMM_DEFAULT_INNER_BLOCK_SIZE );
/*! \endcond */
//*************************************************************************************************

} // namespace blaze




//=================================================================================================
//
//  COMPILE TIME CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

BLAZE_STATIC_ASSERT( blaze::BLOCK_SIZE >= 4UL );

BLAZE_STATIC_ASSERT( blaze::MMM_OUTER_BLOCK_SIZE >= 16UL && blaze::MMM_OUTER_BLOCK_SIZE % 16UL == 0UL );
BLAZE_STATIC_ASSERT( blaze::MMM_INNER_BLOCK_SIZE >= 16UL && blaze::MMM_INNER_BLOCK_SIZE % 16UL == 0UL );

}
/*! \endcond */
//*************************************************************************************************

#endif
