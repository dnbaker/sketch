//=================================================================================================
/*!
//  \file blaze/config/Vectorization.h
//  \brief Configuration of the vectorization policy of the Blaze library
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


//*************************************************************************************************
/*!\brief Compilation switch for (de-)activation of the Blaze vectorization.
// \ingroup config
//
// This compilation switch enables/disables vectorization of mathematical expressions via
// the SSE, AVX, and/or MIC instruction sets. In case the switch is set to 1 (i.e. in case
// vectorization is enabled), the Blaze library attempts to vectorize the linear algebra
// operations by SSE, AVX, and/or MIC intrinsics (depending on which instruction set is
// available on the target platform). In case the switch is set to 0 (i.e. vectorization
// is disabled), the Blaze library chooses default, non-vectorized functionality for the
// operations. Note that deactivating the vectorization may pose a severe performance
// limitation for a large number of operations!
//
// Possible settings for the vectorization switch:
//  - Deactivated: \b 0
//  - Activated  : \b 1 (default)
//
// \note It is possible to (de-)activate vectorization via command line or by defining this
// symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_USE_VECTORIZATION=1 ...
   \endcode

   \code
   #define BLAZE_USE_VECTORIZATION 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_VECTORIZATION
#define BLAZE_USE_VECTORIZATION 1
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for enabling/disabling vectorization by means of the Sleef library.
// \ingroup config
//
// For several complex operations Blaze can make use of the Sleef library for vectorization
// (https://github.com/shibatch/sleef). This compilation switch enables/disables the vectorization
// by means of Sleef. In case the switch is set to 1, Blaze uses Sleef for instance for the
// vectorized computation of trigonometric functions (i.e. \c sin(), \c cos(), \c tan(), etc.)
// and exponential functions (i.e. \c exp(), \c log(), ...).
//
// Possible settings for the Sleef switch:
//  - Deactivated: \b 0 (default)
//  - Activated  : \b 1
//
// \note It is possible to enable/disable Sleef vectorization via command line or by defining
// this symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_USE_SLEEF=1 ...
   \endcode

   \code
   #define BLAZE_USE_SLEEF 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_SLEEF
#define BLAZE_USE_SLEEF 0
#endif
//*************************************************************************************************
