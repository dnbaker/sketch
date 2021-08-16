//=================================================================================================
/*!
//  \file blaze/system/Thresholds.h
//  \brief Header file for the thresholds for matrix/vector and matrix/matrix multiplications
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

#ifndef _BLAZE_SYSTEM_THRESHOLDS_H_
#define _BLAZE_SYSTEM_THRESHOLDS_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/system/Debugging.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>




//=================================================================================================
//
//  THRESHOLDS
//
//=================================================================================================

#include <blaze/config/Thresholds.h>




namespace blaze {

//=================================================================================================
//
//  BLAS THRESHOLDS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Row-major dense matrix/dense vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_DMATDVECMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the row-major dense matrix/dense vector multiplication. In case the
// number of elements in the dense matrix is equal or higher than this value, the BLAS kernels
// are preferred over the custom Blaze kernels. In case the number of elements in the dense
// matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t DMATDVECMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major dense matrix/dense vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_TDMATDVECMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the column-major dense matrix/dense vector multiplication. In case
// the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t TDMATDVECMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Dense Vector/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_TDVECDMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the dense vector/row-major dense matrix multiplication. In case the
// number of elements in the dense matrix is equal or higher than this value, the BLAS kernels
// are preferred over the custom Blaze kernels. In case the number of elements in the dense
// matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t TDVECDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Dense Vector/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_TDVECTDMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the dense vector/column-major dense matrix multiplication. In case
// the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t TDVECTDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Row-major dense matrix/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_DMATDMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the row-major dense matrix/row-major dense matrix multiplication. In
// case the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t DMATDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Row-major dense matrix/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_DMATTDMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the row-major dense matrix/column-major dense matrix multiplication.
// In case the number of elements in the dense matrix is equal or higher than this value, the
// BLAS kernels are preferred over the custom Blaze kernels. In case the number of elements in
// the dense matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t DMATTDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major dense matrix/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_TDMATDMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the column-major dense matrix/row-major dense matrix multiplication.
// In case the number of elements in the dense matrix is equal or higher than this value, the
// BLAS kernels are preferred over the custom Blaze kernels. In case the number of elements in
// the dense matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t TDMATDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major dense matrix/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_TDMATTDMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the custom Blaze kernels
// and the BLAS kernels for the column-major dense matrix/column-major dense matrix multiplication.
// In case the number of elements in the dense matrix is equal or higher than this value, the BLAS
// kernels are preferred over the custom Blaze kernels. In case the number of elements in the
// dense matrix is smaller, the Blaze kernels are used.
*/
constexpr size_t TDMATTDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Row-major dense matrix/row-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_DMATSMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the Blaze kernels for
// small and for large row-major dense matrix/row-major sparse matrix multiplications. In case
// the number of elements of the target matrix is equal or higher than this value, the kernel for
// large matrices is preferred over the kernel for small matrices. In case the number of elements
// in the target matrix is smaller, the kernel for small matrices is used.
*/
constexpr size_t DMATSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major dense matrix/row-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_DMATSMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the Blaze kernels for
// small and for large column-major dense matrix/row-major sparse matrix multiplications. In
// case the number of elements of the target matrix is equal or higher than this value, the
// kernel for large matrices is preferred over the kernel for small matrices. In case the number
// of elements in the target matrix is smaller, the kernel for small matrices is used.
*/
constexpr size_t TDMATSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major sparse matrix/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_TSMATDMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the Blaze kernels for
// small and for large column-major sparse matrix/row-major dense matrix multiplications. In
// case the number of elements of the target matrix is equal or higher than this value, the
// kernel for large matrices is preferred over the kernel for small matrices. In case the number
// of elements in the target matrix is smaller, the kernel for small matrices is used.
*/
constexpr size_t TSMATDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Column-major sparse matrix/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_TSMATTDMATMULT_THRESHOLD while the Blaze debug
// mode is active. It specifies the threshold between the application of the Blaze kernels for
// small and for large column-major sparse matrix/column-major dense matrix multiplications. In
// case the number of elements of the target matrix is equal or higher than this value, the
// kernel for large matrices is preferred over the kernel for small matrices. In case the number
// of elements in the target matrix is smaller, the kernel for small matrices is used.
*/
constexpr size_t TSMATTDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
constexpr size_t DMATDVECMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? DMATDVECMULT_DEBUG_THRESHOLD   : BLAZE_DMATDVECMULT_THRESHOLD   );
constexpr size_t TDMATDVECMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? TDMATDVECMULT_DEBUG_THRESHOLD  : BLAZE_TDMATDVECMULT_THRESHOLD  );
constexpr size_t TDVECDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? TDVECDMATMULT_DEBUG_THRESHOLD  : BLAZE_TDVECDMATMULT_THRESHOLD  );
constexpr size_t TDVECTDMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? TDVECTDMATMULT_DEBUG_THRESHOLD : BLAZE_TDVECTDMATMULT_THRESHOLD );
constexpr size_t DMATDMATMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? DMATDMATMULT_DEBUG_THRESHOLD   : BLAZE_DMATDMATMULT_THRESHOLD   );
constexpr size_t DMATTDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? DMATTDMATMULT_DEBUG_THRESHOLD  : BLAZE_DMATTDMATMULT_THRESHOLD  );
constexpr size_t TDMATDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? TDMATDMATMULT_DEBUG_THRESHOLD  : BLAZE_TDMATDMATMULT_THRESHOLD  );
constexpr size_t TDMATTDMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? TDMATTDMATMULT_DEBUG_THRESHOLD : BLAZE_TDMATTDMATMULT_THRESHOLD );
constexpr size_t DMATSMATMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? DMATSMATMULT_DEBUG_THRESHOLD   : BLAZE_DMATSMATMULT_THRESHOLD   );
constexpr size_t TDMATSMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? TDMATSMATMULT_DEBUG_THRESHOLD  : BLAZE_TDMATSMATMULT_THRESHOLD  );
constexpr size_t TSMATDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? TSMATDMATMULT_DEBUG_THRESHOLD  : BLAZE_TSMATDMATMULT_THRESHOLD  );
constexpr size_t TSMATTDMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? TSMATTDMATMULT_DEBUG_THRESHOLD : BLAZE_TSMATTDMATMULT_THRESHOLD );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SMP THRESHOLDS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief SMP dense vector assignment threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DVECASSIGN_THRESHOLD while the Blaze
// debug mode is active. It specifies when an assignment of a simple dense vector can be executed
// in parallel. In case the number of elements of the target vector is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DVECASSIGN_DEBUG_THRESHOLD = 32UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/scalar multiplication/division threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DVECSCALARMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense vector/scalar multiplication/division can be
// executed in parallel. In case the number of elements of the target vector is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DVECSCALARMULT_DEBUG_THRESHOLD = 32UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector addition threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DVECDVECADD_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense vector/dense vector addition can be executed
// in parallel. In case the number of elements of the target vector is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DVECDVECADD_DEBUG_THRESHOLD = 32UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector subtraction threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DVECDVECSUB_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense vector/dense vector subtraction can be executed
// in parallel. In case the number of elements of the target vector is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DVECDVECSUB_DEBUG_THRESHOLD = 32UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DVECDVECMULT_THRESHOLD while the
// Blaze debug mode is active. It specifies when a dense vector/dense vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger
// or equal to this threshold, the operation is executed in parallel. If the number of elements
// is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DVECDVECMULT_DEBUG_THRESHOLD = 32UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector division threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DVECDVECDIV_THRESHOLD while the
// Blaze debug mode is active. It specifies when a dense vector/dense vector division can be
// executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements
// is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DVECDVECDIV_DEBUG_THRESHOLD = 32UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/dense vector outer product threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DVECDVECOUTER_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense vector/dense vector outer product can be executed
// in parallel. In case the number of elements of the target matrix is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DVECDVECOUTER_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/dense vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATDVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/dense vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATDVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/dense vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDMATDVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major dense matrix/dense vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDMATDVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDVECDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense vector/row-major dense matrix multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDVECDMATMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDVECTDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense vector/column-major dense matrix multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDVECTDMATMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/sparse vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATSVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/sparse vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATSVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/sparse vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDMATSVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major dense matrix/sparse vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDMATSVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse vector/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSVECDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a sparse vector/row-major dense matrix multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSVECDMATMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse vector/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSVECTDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a sparse vector/column-major dense matrix multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSVECTDMATMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/dense vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_SMATDVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major sparse matrix/dense vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_SMATDVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/dense vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSMATDVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major sparse matrix/dense vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSMATDVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/row-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDVECSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense vector/row-major sparse matrix multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDVECSMATMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense vector/column-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDVECTSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense vector/column-major sparse matrix multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDVECTSMATMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/sparse vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_SMATSVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major sparse matrix/sparse vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_SMATSVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/sparse vector multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSMATSVECMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major sparse matrix/sparse vector multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSMATSVECMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse vector/row-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSVECSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a sparse vector/row-major sparse matrix multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSVECSMATMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse vector/column-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSVECTSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a sparse vector/column-major sparse matrix multiplication
// can be executed in parallel. In case the number of elements of the target vector is larger or
// equal to this threshold, the operation is executed in parallel. If the number of elements is
// below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSVECTSMATMULT_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense matrix assignment threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATASSIGN_THRESHOLD while the Blaze
// debug mode is active. It specifies when an assignment with a simple dense matrix can be executed
// in parallel. In case the number of elements of the target matrix is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATASSIGN_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense matrix/scalar multiplication/division threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATSCALARMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense matrix/scalar multiplication or division can be
// executed in parallel. In case the number of elements of the target matrix is larger or equal
// to this threshold, the operation is executed in parallel. If the number of elements is below
// this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATSCALARMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix addition threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATDMATADD_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/row-major dense matrix addition
// can be executed in parallel. This threshold affects both additions between two row-major matrices
// or two column-major dense matrices. In case the number of elements of the target matrix is
// larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATDMATADD_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major dense matrix addition threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATTDMATADD_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/column-major dense matrix
// addition can be executed in parallel. This threshold affects both additions between a row-major
// matrix and a column-major matrix and a column-major matrix and a row-major matrix. In case the
// number of elements of the target matrix is larger or equal to this threshold, the operation
// is executed in parallel. If the number of elements is below this threshold the operation is
// executed single-threaded.
*/
constexpr size_t SMP_DMATTDMATADD_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix subtraction threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATDMATSUB_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/row-major dense matrix
// subtraction can be executed in parallel. This threshold affects both subtractions between two
// row-major matrices or two column-major dense matrices. In case the number of elements of
// the target matrix is larger or equal to this threshold, the operation is executed in parallel.
// If the number of elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATDMATSUB_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major dense matrix subtraction threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATTDMATSUB_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/column-major dense matrix
// subtraction can be executed in parallel. This threshold affects both subtractions between a
// row-major matrix and a column-major matrix and a column-major matrix and a row-major matrix.
// In case the number of elements of the target matrix is larger or equal to this threshold, the
// operation is executed in parallel. If the number of elements is below this threshold the
// operation is executed single-threaded.
*/
constexpr size_t SMP_DMATTDMATSUB_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix Schur product threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATDMATSCHUR_THRESHOLD while the
// Blaze debug mode is active. It specifies when a row-major dense matrix/row-major dense matrix
// Schur product can be executed in parallel. This threshold affects both Schur products between
// two row-major matrices or two column-major dense matrices. In case the number of elements of
// the target matrix is larger or equal to this threshold, the operation is executed in parallel.
// If the number of elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATDMATSCHUR_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major dense matrix Schur product threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATTDMATSCHUR_THRESHOLD while the
// Blaze debug mode is active. It specifies when a row-major dense matrix/column-major dense
// matrix Schur product can be executed in parallel. This threshold affects both Schur products
// between a row-major matrix and a column-major matrix and a column-major matrix and a row-major
// matrix. In case the number of elements of the target matrix is larger or equal to this
// threshold, the operation is executed in parallel. If the number of elements is below this
// threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATTDMATSCHUR_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATDMATMULT_THRESHOLD while the
// Blaze debug mode is active. It specifies when a row-major dense matrix/row-major dense matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATTDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/column-major dense matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATTDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDMATDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major dense matrix/row-major dense matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDMATDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDMATTDMATMULT_THRESHOLD while the
// Blaze debug mode is active. It specifies when a column-major dense matrix/column-major dense
// matrix multiplication can be executed in parallel. In case the number of elements of the target
// matrix is larger or equal to this threshold, the operation is executed in parallel. If the
// number of elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDMATTDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/row-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/row-major sparse matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major dense matrix/column-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATTSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major dense matrix/column-major sparse matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_DMATTSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/row-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDMATSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major dense matrix/row-major sparse matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDMATSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major dense matrix/column-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TDMATTSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major dense matrix/column-major sparse matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TDMATTSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_SMATDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major sparse matrix/row-major dense matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_SMATDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_SMATTDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major sparse matrix/column-major dense matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_SMATTDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/row-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSMATDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major sparse matrix/row-major dense matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSMATDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/column-major dense matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSMATTDMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major sparse matrix/column-major dense matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSMATTDMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/row-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_SMATSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major sparse matrix/row-major sparse matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_SMATSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP row-major sparse matrix/column-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_SMATTSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a row-major sparse matrix/column-major sparse matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_SMATTSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/row-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSMATSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major sparse matrix/row-major sparse matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSMATSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP column-major sparse matrix/column-major sparse matrix multiplication threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_TSMATTSMATMULT_THRESHOLD while the Blaze
// debug mode is active. It specifies when a column-major sparse matrix/column-major sparse matrix
// multiplication can be executed in parallel. In case the number of elements of the target matrix
// is larger or equal to this threshold, the operation is executed in parallel. If the number of
// elements is below this threshold the operation is executed single-threaded.
*/
constexpr size_t SMP_TSMATTSMATMULT_DEBUG_THRESHOLD = 256UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP dense matrix reduction threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_DMATREDUCE_THRESHOLD while the Blaze
// debug mode is active. It specifies when a dense matrix reduction can be executed in parallel.
// In case the number of elements of the target vector is larger or equal to this threshold,
// the operation is executed in parallel. If the number of elements is below this threshold the
// operation is executed single-threaded.
*/
constexpr size_t SMP_DMATREDUCE_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*!\brief SMP sparse matrix reduction threshold.
// \ingroup system
//
// This debug value is used instead of the BLAZE_SMP_SMATREDUCE_THRESHOLD while the Blaze
// debug mode is active. It specifies when a sparse matrix reduction can be executed in parallel.
// In case the number of elements of the target vector is larger or equal to this threshold,
// the operation is executed in parallel. If the number of elements is below this threshold the
// operation is executed single-threaded.
*/
constexpr size_t SMP_SMATREDUCE_DEBUG_THRESHOLD = 16UL;
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
constexpr size_t SMP_DVECASSIGN_THRESHOLD     = ( BLAZE_DEBUG_MODE ? SMP_DVECASSIGN_DEBUG_THRESHOLD     : BLAZE_SMP_DVECASSIGN_THRESHOLD     );
constexpr size_t SMP_DVECSCALARMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_DVECSCALARMULT_DEBUG_THRESHOLD : BLAZE_SMP_DVECSCALARMULT_THRESHOLD );
constexpr size_t SMP_DVECDVECADD_THRESHOLD    = ( BLAZE_DEBUG_MODE ? SMP_DVECDVECADD_DEBUG_THRESHOLD    : BLAZE_SMP_DVECDVECADD_THRESHOLD    );
constexpr size_t SMP_DVECDVECSUB_THRESHOLD    = ( BLAZE_DEBUG_MODE ? SMP_DVECDVECSUB_DEBUG_THRESHOLD    : BLAZE_SMP_DVECDVECSUB_THRESHOLD    );
constexpr size_t SMP_DVECDVECMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_DVECDVECMULT_DEBUG_THRESHOLD   : BLAZE_SMP_DVECDVECMULT_THRESHOLD   );
constexpr size_t SMP_DVECDVECDIV_THRESHOLD    = ( BLAZE_DEBUG_MODE ? SMP_DVECDVECDIV_DEBUG_THRESHOLD    : BLAZE_SMP_DVECDVECDIV_THRESHOLD    );
constexpr size_t SMP_DVECDVECOUTER_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_DVECDVECOUTER_DEBUG_THRESHOLD  : BLAZE_SMP_DVECDVECOUTER_THRESHOLD  );
constexpr size_t SMP_DMATDVECMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_DMATDVECMULT_DEBUG_THRESHOLD   : BLAZE_SMP_DMATDVECMULT_THRESHOLD   );
constexpr size_t SMP_TDMATDVECMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TDMATDVECMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TDMATDVECMULT_THRESHOLD  );
constexpr size_t SMP_TDVECDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TDVECDMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TDVECDMATMULT_THRESHOLD  );
constexpr size_t SMP_TDVECTDMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_TDVECTDMATMULT_DEBUG_THRESHOLD : BLAZE_SMP_TDVECTDMATMULT_THRESHOLD );
constexpr size_t SMP_DMATSVECMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_DMATSVECMULT_DEBUG_THRESHOLD   : BLAZE_SMP_DMATSVECMULT_THRESHOLD   );
constexpr size_t SMP_TDMATSVECMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TDMATSVECMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TDMATSVECMULT_THRESHOLD  );
constexpr size_t SMP_TSVECDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TSVECDMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TSVECDMATMULT_THRESHOLD  );
constexpr size_t SMP_TSVECTDMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_TSVECTDMATMULT_DEBUG_THRESHOLD : BLAZE_SMP_TSVECTDMATMULT_THRESHOLD );
constexpr size_t SMP_SMATDVECMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_SMATDVECMULT_DEBUG_THRESHOLD   : BLAZE_SMP_SMATDVECMULT_THRESHOLD   );
constexpr size_t SMP_TSMATDVECMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TSMATDVECMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TSMATDVECMULT_THRESHOLD  );
constexpr size_t SMP_TDVECSMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TDVECSMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TDVECSMATMULT_THRESHOLD  );
constexpr size_t SMP_TDVECTSMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_TDVECTSMATMULT_DEBUG_THRESHOLD : BLAZE_SMP_TDVECTSMATMULT_THRESHOLD );
constexpr size_t SMP_SMATSVECMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_SMATSVECMULT_DEBUG_THRESHOLD   : BLAZE_SMP_SMATSVECMULT_THRESHOLD   );
constexpr size_t SMP_TSMATSVECMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TSMATSVECMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TSMATSVECMULT_THRESHOLD  );
constexpr size_t SMP_TSVECSMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TSVECSMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TSVECSMATMULT_THRESHOLD  );
constexpr size_t SMP_TSVECTSMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_TSVECTSMATMULT_DEBUG_THRESHOLD : BLAZE_SMP_TSVECTSMATMULT_THRESHOLD );
constexpr size_t SMP_DMATASSIGN_THRESHOLD     = ( BLAZE_DEBUG_MODE ? SMP_DMATASSIGN_DEBUG_THRESHOLD     : BLAZE_SMP_DMATASSIGN_THRESHOLD     );
constexpr size_t SMP_DMATSCALARMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_DMATSCALARMULT_DEBUG_THRESHOLD : BLAZE_SMP_DMATSCALARMULT_THRESHOLD );
constexpr size_t SMP_DMATDMATADD_THRESHOLD    = ( BLAZE_DEBUG_MODE ? SMP_DMATDMATADD_DEBUG_THRESHOLD    : BLAZE_SMP_DMATDMATADD_THRESHOLD    );
constexpr size_t SMP_DMATTDMATADD_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_DMATTDMATADD_DEBUG_THRESHOLD   : BLAZE_SMP_DMATTDMATADD_THRESHOLD   );
constexpr size_t SMP_DMATDMATSUB_THRESHOLD    = ( BLAZE_DEBUG_MODE ? SMP_DMATDMATSUB_DEBUG_THRESHOLD    : BLAZE_SMP_DMATDMATSUB_THRESHOLD    );
constexpr size_t SMP_DMATTDMATSUB_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_DMATTDMATSUB_DEBUG_THRESHOLD   : BLAZE_SMP_DMATTDMATSUB_THRESHOLD   );
constexpr size_t SMP_DMATDMATSCHUR_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_DMATDMATSCHUR_DEBUG_THRESHOLD  : BLAZE_SMP_DMATDMATSCHUR_THRESHOLD  );
constexpr size_t SMP_DMATTDMATSCHUR_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_DMATTDMATSCHUR_DEBUG_THRESHOLD : BLAZE_SMP_DMATTDMATSCHUR_THRESHOLD );
constexpr size_t SMP_DMATDMATMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_DMATDMATMULT_DEBUG_THRESHOLD   : BLAZE_SMP_DMATDMATMULT_THRESHOLD   );
constexpr size_t SMP_DMATTDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_DMATTDMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_DMATTDMATMULT_THRESHOLD  );
constexpr size_t SMP_TDMATDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TDMATDMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TDMATDMATMULT_THRESHOLD  );
constexpr size_t SMP_TDMATTDMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_TDMATTDMATMULT_DEBUG_THRESHOLD : BLAZE_SMP_TDMATTDMATMULT_THRESHOLD );
constexpr size_t SMP_DMATSMATMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_DMATSMATMULT_DEBUG_THRESHOLD   : BLAZE_SMP_DMATSMATMULT_THRESHOLD   );
constexpr size_t SMP_DMATTSMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_DMATTSMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_DMATTSMATMULT_THRESHOLD  );
constexpr size_t SMP_TDMATSMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TDMATSMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TDMATSMATMULT_THRESHOLD  );
constexpr size_t SMP_TDMATTSMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_TDMATTSMATMULT_DEBUG_THRESHOLD : BLAZE_SMP_TDMATTSMATMULT_THRESHOLD );
constexpr size_t SMP_SMATDMATMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_SMATDMATMULT_DEBUG_THRESHOLD   : BLAZE_SMP_SMATDMATMULT_THRESHOLD   );
constexpr size_t SMP_SMATTDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_SMATTDMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_SMATTDMATMULT_THRESHOLD  );
constexpr size_t SMP_TSMATDMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TSMATDMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TSMATDMATMULT_THRESHOLD  );
constexpr size_t SMP_TSMATTDMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_TSMATTDMATMULT_DEBUG_THRESHOLD : BLAZE_SMP_TSMATTDMATMULT_THRESHOLD );
constexpr size_t SMP_SMATSMATMULT_THRESHOLD   = ( BLAZE_DEBUG_MODE ? SMP_SMATSMATMULT_DEBUG_THRESHOLD   : BLAZE_SMP_SMATSMATMULT_THRESHOLD   );
constexpr size_t SMP_SMATTSMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_SMATTSMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_SMATTSMATMULT_THRESHOLD  );
constexpr size_t SMP_TSMATSMATMULT_THRESHOLD  = ( BLAZE_DEBUG_MODE ? SMP_TSMATSMATMULT_DEBUG_THRESHOLD  : BLAZE_SMP_TSMATSMATMULT_THRESHOLD  );
constexpr size_t SMP_TSMATTSMATMULT_THRESHOLD = ( BLAZE_DEBUG_MODE ? SMP_TSMATTSMATMULT_DEBUG_THRESHOLD : BLAZE_SMP_TSMATTSMATMULT_THRESHOLD );
constexpr size_t SMP_DMATREDUCE_THRESHOLD     = ( BLAZE_DEBUG_MODE ? SMP_DMATREDUCE_DEBUG_THRESHOLD     : BLAZE_SMP_DMATREDUCE_THRESHOLD     );
constexpr size_t SMP_SMATREDUCE_THRESHOLD     = ( BLAZE_DEBUG_MODE ? SMP_SMATREDUCE_DEBUG_THRESHOLD     : BLAZE_SMP_SMATREDUCE_THRESHOLD     );
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

BLAZE_STATIC_ASSERT( blaze::DMATDVECMULT_THRESHOLD   > 0UL );
BLAZE_STATIC_ASSERT( blaze::TDMATDVECMULT_THRESHOLD  > 0UL );
BLAZE_STATIC_ASSERT( blaze::TDVECDMATMULT_THRESHOLD  > 0UL );
BLAZE_STATIC_ASSERT( blaze::TDVECTDMATMULT_THRESHOLD > 0UL );
BLAZE_STATIC_ASSERT( blaze::DMATDMATMULT_THRESHOLD   > 0UL );
BLAZE_STATIC_ASSERT( blaze::DMATTDMATMULT_THRESHOLD  > 0UL );
BLAZE_STATIC_ASSERT( blaze::TDMATDMATMULT_THRESHOLD  > 0UL );
BLAZE_STATIC_ASSERT( blaze::TDMATTDMATMULT_THRESHOLD > 0UL );

BLAZE_STATIC_ASSERT( blaze::SMP_DVECASSIGN_THRESHOLD     >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DVECSCALARMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DVECDVECADD_THRESHOLD    >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DVECDVECSUB_THRESHOLD    >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DVECDVECMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DVECDVECDIV_THRESHOLD    >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DVECDVECOUTER_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATDVECMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDMATDVECMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDVECDMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDVECTDMATMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATSVECMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDMATSVECMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSVECDMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSVECTDMATMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_SMATDVECMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSMATDVECMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDVECSMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDVECTSMATMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_SMATSVECMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSMATSVECMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSVECSMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSVECTSMATMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATASSIGN_THRESHOLD     >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATSCALARMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATDMATADD_THRESHOLD    >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATTDMATADD_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATDMATSUB_THRESHOLD    >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATTDMATSUB_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATDMATSCHUR_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATTDMATSCHUR_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATDMATMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATTDMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDMATDMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDMATTDMATMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATSMATMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATTSMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDMATSMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TDMATTSMATMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_SMATDMATMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_SMATTDMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSMATDMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSMATTDMATMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_SMATSMATMULT_THRESHOLD   >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_SMATTSMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSMATSMATMULT_THRESHOLD  >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_TSMATTSMATMULT_THRESHOLD >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_DMATREDUCE_THRESHOLD     >= 0UL );
BLAZE_STATIC_ASSERT( blaze::SMP_SMATREDUCE_THRESHOLD     >= 0UL );

}
/*! \endcond */
//*************************************************************************************************

#endif
