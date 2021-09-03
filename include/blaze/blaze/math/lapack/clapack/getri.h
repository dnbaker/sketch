//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/getri.h
//  \brief Header file for the CLAPACK getri wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_GETRI_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_GETRI_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/blas/Types.h>
#include <blaze/util/Complex.h>
#include <blaze/util/StaticAssert.h>


//=================================================================================================
//
//  LAPACK FORWARD DECLARATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
#if !defined(INTEL_MKL_VERSION)
extern "C" {

void sgetri_( blaze::blas_int_t* n, float* A, blaze::blas_int_t* lda, blaze::blas_int_t* ipiv,
              float* work, blaze::blas_int_t* lwork, blaze::blas_int_t* info );
void dgetri_( blaze::blas_int_t* n, double* A, blaze::blas_int_t* lda, blaze::blas_int_t* ipiv,
              double* work, blaze::blas_int_t* lwork, blaze::blas_int_t* info );
void cgetri_( blaze::blas_int_t* n, float* A, blaze::blas_int_t* lda, blaze::blas_int_t* ipiv,
              float* work, blaze::blas_int_t* lwork, blaze::blas_int_t* info );
void zgetri_( blaze::blas_int_t* n, double* A, blaze::blas_int_t* lda, blaze::blas_int_t* ipiv,
              double* work, blaze::blas_int_t* lwork, blaze::blas_int_t* info );

}
#endif
/*! \endcond */
//*************************************************************************************************




namespace blaze {

//=================================================================================================
//
//  LAPACK LU-BASED INVERSION FUNCTIONS (GETRI)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LU-based inversion functions (getri) */
//@{
void getri( blas_int_t n, float* A, blas_int_t lda, const blas_int_t* ipiv,
            float* work, blas_int_t lwork, blas_int_t* info );

void getri( blas_int_t n, double* A, blas_int_t lda, const blas_int_t* ipiv,
            double* work, blas_int_t lwork, blas_int_t* info );

void getri( blas_int_t n, complex<float>* A, blas_int_t lda, const blas_int_t* ipiv,
            complex<float>* work, blas_int_t lwork, blas_int_t* info );

void getri( blas_int_t n, complex<double>* A, blas_int_t lda, const blas_int_t* ipiv,
            complex<double>* work, blas_int_t lwork, blas_int_t* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense general single precision column-major
//        square matrix.
// \ingroup lapack_inversion
//
// \param n The number of rows/columns of the matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK sgetri() function for
// single precision column-major matrices that have already been factorized by the sgetrf()
// function. The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the inversion could not be computed since U(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= N*NB, where NB is the
// optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a workspace
// query is assumed. The function only calculates the optimal size of the \a work array and
// returns this value as the first entry of the \a work array.
//
// For more information on the sgetri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void getri( blas_int_t n, float* A, blas_int_t lda, const blas_int_t* ipiv,
                   float* work, blas_int_t lwork, blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   sgetri_( &n, A, &lda, const_cast<blas_int_t*>( ipiv ), work, &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense general double precision column-major
//        square matrix.
// \ingroup lapack_inversion
//
// \param n The number of rows/columns of the matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK dgetri() function for
// double precision column-major matrices that have already been factorized by the dgetrf()
// function. The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the inversion could not be computed since U(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= N*NB, where NB is the
// optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a workspace
// query is assumed. The function only calculates the optimal size of the \a work array and
// returns this value as the first entry of the \a work array.
//
// For more information on the sgetri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void getri( blas_int_t n, double* A, blas_int_t lda, const blas_int_t* ipiv,
                   double* work, blas_int_t lwork, blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   dgetri_( &n, A, &lda, const_cast<blas_int_t*>( ipiv ), work, &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense general single precision complex
//        column-major square matrix.
// \ingroup lapack_inversion
//
// \param n The number of rows/columns of the matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK cgetri() function for
// single precision complex column-major matrices that have already been factorized by the
// cgetrf() function. The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the inversion could not be computed since U(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= N*NB, where NB is the
// optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a workspace
// query is assumed. The function only calculates the optimal size of the \a work array and
// returns this value as the first entry of the \a work array.
//
// For more information on the sgetri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void getri( blas_int_t n, complex<float>* A, blas_int_t lda, const blas_int_t* ipiv,
                   complex<float>* work, blas_int_t lwork, blas_int_t* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
   BLAZE_STATIC_ASSERT( sizeof( MKL_Complex8 ) == sizeof( complex<float> ) );
   using ET = MKL_Complex8;
#else
   using ET = float;
#endif

   cgetri_( &n, reinterpret_cast<ET*>( A ), &lda, const_cast<blas_int_t*>( ipiv ),
            reinterpret_cast<ET*>( work ), &lwork, info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the inversion of the given dense general double precision complex
//        column-major square matrix.
// \ingroup lapack_inversion
//
// \param n The number of rows/columns of the matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major square matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array for the pivot indices; size >= min( \a m, \a n ).
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix inversion based on the LAPACK cgetri() function for
// double precision complex column-major matrices that have already been factorized by the
// zgetrf() function. The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The inversion finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the inversion could not be computed since U(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= N*NB, where NB is the
// optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a workspace
// query is assumed. The function only calculates the optimal size of the \a work array and
// returns this value as the first entry of the \a work array.
//
// For more information on the sgetri() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void getri( blas_int_t n, complex<double>* A, blas_int_t lda, const blas_int_t* ipiv,
                   complex<double>* work, blas_int_t lwork, blas_int_t* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
   BLAZE_STATIC_ASSERT( sizeof( MKL_Complex16 ) == sizeof( complex<double> ) );
   using ET = MKL_Complex16;
#else
   using ET = double;
#endif

   zgetri_( &n, reinterpret_cast<ET*>( A ), &lda, const_cast<blas_int_t*>( ipiv ),
            reinterpret_cast<ET*>( work ), &lwork, info );
}
//*************************************************************************************************

} // namespace blaze

#endif
