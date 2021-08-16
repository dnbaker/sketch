//=================================================================================================
/*!
//  \file blaze/config/BLAS.h
//  \brief Configuration of the BLAS mode
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
/*!\brief Compilation switch for the BLAS mode.
// \ingroup config
//
// This compilation switch enables/disables the BLAS mode. In case the BLAS mode is enabled,
// several basic linear algebra functions (such as for instance matrix-matrix multiplications
// between two dense matrices) are handled by performance optimized BLAS functions. Note that
// in this case it is mandatory to provide the according BLAS header file for the compilation
// of the Blaze library. In case the BLAS mode is disabled, all linear algebra functions use
// the default implementations of the Blaze library and therefore BLAS is not a requirement
// for the compilation process.
//
// Possible settings for the BLAS switch:
//  - Disabled: \b 0
//  - Enabled : \b 1
//
// \warning Changing the setting of the BLAS mode requires a recompilation of all code using
// the Blaze library!
//
// \note It is possible to (de-)activate the BLAS mode via command line or by defining this
// symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_BLAS_MODE=1 ...
   \endcode

   \code
   #define BLAZE_BLAS_MODE 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_BLAS_MODE
#define BLAZE_BLAS_MODE 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the BLAS 64-bit support.
// \ingroup config
//
// This compilation switch enables/disables 64-bit BLAS and LAPACK support. In case the 64-bit
// BLAS mode is enabled, \c blaze::blas_int_t, which is used in the BLAS and LAPACK wrapper
// functions, is a 64-bit signed integral type. In case the 64-bit BLAS mode is disabled,
// \c blaze::blas_int_t is a 32-bit signed integral type.
//
// Possible settings for the switch:
//  - 32-bit BLAS/LAPACK: \b 0 (default)
//  - 64-bit BLAS/LAPACK: \b 1
//
// \warning Changing the setting of the BLAS mode requires a recompilation of all code using the
// Blaze library!
//
// \note It is possible to (de-)activate the 64-bit BLAS mode via command line or by defining
// this symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_BLAS_IS_64BIT=1 ...
   \endcode

   \code
   #define BLAZE_BLAS_IS_64BIT 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_BLAS_IS_64BIT
#define BLAZE_BLAS_IS_64BIT 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the parallel BLAS mode.
// \ingroup config
//
// This compilation switch specifies whether the used BLAS library is itself parallelized or not.
// In case the given BLAS library is itself parallelized, the Blaze library does not perform any
// attempt to parallelize the execution of BLAS kernels. If, however, the given BLAS library is
// not parallelized Blaze will attempt to parallelize the execution of BLAS kernels.
//
// Possible settings for the switch:
//  - BLAS library is not parallelized: \b 0 (default)
//  - BLAS library is parallelized    : \b 1
//
// \warning Changing the setting of the BLAS mode requires a recompilation of all code using the
// Blaze library!
//
// \note It is possible to (de-)activate the parallel BLAS mode via command line or by defining
// this symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_BLAS_IS_PARALLEL=1 ...
   \endcode

   \code
   #define BLAZE_BLAS_IS_PARALLEL 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_BLAS_IS_PARALLEL
#define BLAZE_BLAS_IS_PARALLEL 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the BLAS matrix/vector multiplication kernels (gemv).
// \ingroup config
//
// This compilation switch enables/disables the BLAS matrix/vector multiplication kernels. If the
// switch is enabled, multiplications between dense matrices and dense vectors are computed by
// BLAS kernels, if it is disabled the multiplications are handled by the default Blaze kernels.
//
// Possible settings for the switch:
//  - Disabled: \b 0 (default)
//  - Enabled : \b 1
//
// \warning Changing the setting of this compilation switch requires a recompilation of all code
// using the Blaze library!
//
// \note It is possible to (de-)activate the use of the BLAS matrix/vector multiplication kernels
// via command line or by defining this symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_USE_BLAS_MATRIX_VECTOR_MULTIPLICATION=1 ...
   \endcode

   \code
   #define BLAZE_USE_BLAS_MATRIX_VECTOR_MULTIPLICATION 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_BLAS_MATRIX_VECTOR_MULTIPLICATION
#define BLAZE_USE_BLAS_MATRIX_VECTOR_MULTIPLICATION 0
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the BLAS matrix/matrix multiplication kernels (gemv).
// \ingroup config
//
// This compilation switch enables/disables the BLAS matrix/matrix multiplication kernels. If the
// switch is enabled, multiplications between dense matrices are computed by BLAS kernels, if it
// is disabled the multiplications are handled by the default Blaze kernels.
//
// Possible settings for the switch:
//  - Disabled: \b 0
//  - Enabled : \b 1 (default)
//
// \warning Changing the setting of this compilation switch requires a recompilation of all code
// code using the Blaze library!
//
// \note It is possible to (de-)activate the use of the BLAS matrix/matrix multiplication kernels
// via command line or by defining this symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION=1 ...
   \endcode

   \code
   #define BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION 1
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION
#define BLAZE_USE_BLAS_MATRIX_MATRIX_MULTIPLICATION 1
#endif
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compilation switch for the BLAS include file.
// \ingroup config
//
// This compilation switch specifies the name of the BLAS include file. By default, the header
// \c <cblas.h> is included when the BLAS mode is activated. In case the name of the include file
// differs (as for instance in case of the MKL the file is called \c <mkl_cblas.h>) this switch
// needs to be adapted accordingly.
//
// \warning Changing the name of the BLAS include file requires a recompilation of all code using
// the Blaze library!
//
// \note It is possible to specify the BLAS include file via command line or by defining this
// symbol manually before including any Blaze header file:

   \code
   g++ ... -DBLAZE_BLAS_INCLUDE_FILE="<cblas.h>" ...
   \endcode

   \code
   #define BLAZE_BLAS_INCLUDE_FILE <cblas.h>
   #include <blaze/Blaze.h>
   \endcode
*/
#ifndef BLAZE_BLAS_INCLUDE_FILE
#define BLAZE_BLAS_INCLUDE_FILE <cblas.h>
#endif
//*************************************************************************************************
