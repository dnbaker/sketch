//=================================================================================================
/*!
//  \file blaze/config/Optimizations.h
//  \brief Configuration of performance optimizations
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
/*!\brief Configuration of the padding of dynamic vectors and matrices.
// \ingroup config
//
// This configuration switch enables/disables the padding of DynamicVector and DynamicMatrix.
// Padding is used by the Blaze library in order to achieve maximum performance for both dense
// vector and matrix operations. Due to padding, the proper alignment of data elements can be
// guaranteed and the need for remainder loops is minimized. In case padding is enabled, it is
// enabled only for the DynamicVector and DynamicMatrix class templates. Other dense vector and
// matrix classes are not affected. If padding is disabled, both DynamicVector and DynamicMatrix
// don't apply padding.
//
// Possible settings for padding:
//  - Disabled: \b 0
//  - Enabled : \b 1
//
// \warning Note that disabling padding can considerably reduce the performance of operations
// with DynamicVector and DynamicMatrix.
//
// \note It is possible to (de-)activate padding via command line or by defining this symbol
// manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_USE_PADDING=1 ...
   \endcode

   \code
   #define BLAZE_USE_PADDING 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_PADDING
#define BLAZE_USE_PADDING 1
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Configuration of the streaming behavior.
// \ingroup config
//
// Via this compilation switch streaming (i.e. non-temporal stores) can be (de-)activated. For
// large vectors and matrices non-temporal stores can provide a significant performance advantage
// of about 20%. However, this advantage is only in effect in case the memory bandwidth of the
// target architecture is maxed out. If the target architecture's memory bandwidth cannot be
// exhausted the use of non-temporal stores can decrease performance instead of increasing it.
//
// Possible settings for streaming:
//  - Disabled: \b 0
//  - Enabled : \b 1
//
// \note It is possible to (de-)activate streaming via command line or by defining this symbol
// manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_USE_STREAMING=1 ...
   \endcode

   \code
   #define BLAZE_USE_STREAMING 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_STREAMING
#define BLAZE_USE_STREAMING 1
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Configuration switch for optimized kernels.
// \ingroup config
//
// This configuration switch enables/disables all optimized compute kernels of the Blaze library,
// including all vectorized and data type depending kernels. In case the switch is set to 1 the
// optimized kernels are used whenever possible. In case the switch is set to 0 all optimized
// kernels are not used, even if it would be possible.
//
// Possible settings for the optimized kernels:
//  - Disabled: \b 0
//  - Enabled : \b 1
//
// \warning Note that disabling the optimized kernels causes a severe performance limitiation
// to nearly all operations!
//
// \note It is possible to (de-)activate the optimized kernels via command line or by defining
// this symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_USE_OPTIMIZED_KERNELS=1 ...
   \endcode

   \code
   #define BLAZE_USE_OPTIMIZED_KERNELS 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_OPTIMIZED_KERNELS
#define BLAZE_USE_OPTIMIZED_KERNELS 1
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Configuration switch for the initialization in default constructors.
// \ingroup config
//
// This configuration switch enables/disables the element initialization in the default
// constructors of the \a StaticVector and \a StaticMatrix class templates. In case the switch
// is set to 1 all elements are initialized to their respective default. In case the switch is
// set to 0 the default initialization is skipped and the elements are not initialized. Please
// note that this switch is only effective in case the elements are of fundamental type (i.e.
// integral or floating point). In case the elements are of class type, this switch has no effect.
//
// Possible settings for the default initialization:
//  - Disabled: \b 0
//  - Enabled : \b 1
//
// \note It is possible to (de-)activate the default initialization via command line or by
// defining this symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_USE_DEFAULT_INITIALIZATION=1 ...
   \endcode

   \code
   #define BLAZE_USE_DEFAULT_INITIALIZATION 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_DEFAULT_INITIALIZATION
#define BLAZE_USE_DEFAULT_INITIALIZATION 1
#endif
//*************************************************************************************************
