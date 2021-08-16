//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/sysv.h
//  \brief Header file for the CLAPACK sysv wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_SYSV_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_SYSV_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/blas/Types.h>
#include <blaze/util/Complex.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>


//=================================================================================================
//
//  LAPACK FORWARD DECLARATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
#if !defined(INTEL_MKL_VERSION)
extern "C" {

void ssysv_( char* uplo, blaze::blas_int_t* n, blaze::blas_int_t* nrhs, float* A,
             blaze::blas_int_t* lda, blaze::blas_int_t* ipiv, float* b, blaze::blas_int_t* ldb,
             float* work, blaze::blas_int_t* lwork, blaze::blas_int_t* info,
             blaze::fortran_charlen_t nuplo );
void dsysv_( char* uplo, blaze::blas_int_t* n, blaze::blas_int_t* nrhs, double* A,
             blaze::blas_int_t* lda, blaze::blas_int_t* ipiv, double* b, blaze::blas_int_t* ldb,
             double* work, blaze::blas_int_t* lwork, blaze::blas_int_t* info,
             blaze::fortran_charlen_t nuplo );
void csysv_( char* uplo, blaze::blas_int_t* n, blaze::blas_int_t* nrhs, float* A,
             blaze::blas_int_t* lda, blaze::blas_int_t* ipiv, float* b, blaze::blas_int_t* ldb,
             float* work, blaze::blas_int_t* lwork, blaze::blas_int_t* info,
             blaze::fortran_charlen_t nuplo );
void zsysv_( char* uplo, blaze::blas_int_t* n, blaze::blas_int_t* nrhs, double* A,
             blaze::blas_int_t* lda, blaze::blas_int_t* ipiv, double* b, blaze::blas_int_t* ldb,
             double* work, blaze::blas_int_t* lwork, blaze::blas_int_t* info,
             blaze::fortran_charlen_t nuplo );

}
#endif
/*! \endcond */
//*************************************************************************************************




namespace blaze {

//=================================================================================================
//
//  LAPACK SYMMETRIC INDEFINITE LINEAR SYSTEM FUNCTIONS (SYSV)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK symmetric indefinite linear system functions (sysv) */
//@{
void sysv( char uplo, blas_int_t n, blas_int_t nrhs, float* A, blas_int_t lda,
           blas_int_t* ipiv, float* B, blas_int_t ldb, float* work, blas_int_t lwork,
           blas_int_t* info );

void sysv( char uplo, blas_int_t n, blas_int_t nrhs, double* A, blas_int_t lda,
           blas_int_t* ipiv, double* B, blas_int_t ldb, double* work, blas_int_t lwork,
           blas_int_t* info );

void sysv( char uplo, blas_int_t n, blas_int_t nrhs, complex<float>* A, blas_int_t lda,
           blas_int_t* ipiv, complex<float>* B, blas_int_t ldb, complex<float>* work,
           blas_int_t lwork, blas_int_t* info );

void sysv( char uplo, blas_int_t n, blas_int_t nrhs, complex<double>* A, blas_int_t lda,
           blas_int_t* ipiv, complex<double>* B, blas_int_t ldb, complex<double>* work,
           blas_int_t lwork, blas_int_t* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for solving a symmetric indefinite single precision linear system of
//        equations (\f$ A*X=B \f$).
// \ingroup lapack_solver
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major square matrix.
// \param lda The total number of elements between two columns of matrix \a A \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param B Pointer to the first element of the column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK ssysv() function to compute the solution to the symmetric
// indefinite system of linear equations \f$ A*X=B \f$, where \a A is a \a n-by-\a n matrix and
// \a X and \a B are \a n-by-\a nrhs matrices.
//
// The Bunch-Kaufman decomposition is used to factor \a A as

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched. The factored
// form of \a A is then used to solve the system of equations.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but since factor D(i,i) is exactly
//          zero the solution could not be computed.
//
// For more information on the ssysv() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void sysv( char uplo, blas_int_t n, blas_int_t nrhs, float* A, blas_int_t lda,
                  blas_int_t* ipiv, float* B, blas_int_t ldb, float* work, blas_int_t lwork,
                  blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   ssysv_( &uplo, &n, &nrhs, A, &lda, ipiv, B, &ldb, work, &lwork, info
#if !defined(INTEL_MKL_VERSION)
         , blaze::fortran_charlen_t(1)
#endif
         );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for solving a symmetric indefinite double precision linear system of
//        equations (\f$ A*X=B \f$).
// \ingroup lapack_solver
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major square matrix.
// \param lda The total number of elements between two columns of matrix \a A \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param B Pointer to the first element of the column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK dsysv() function to compute the solution to the symmetric
// indefinite system of linear equations \f$ A*X=B \f$, where \a A is a \a n-by-\a n matrix and
// \a X and \a B are \a n-by-\a nrhs matrices.
//
// The Bunch-Kaufman decomposition is used to factor \a A as

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched. The factored
// form of \a A is then used to solve the system of equations.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but since factor D(i,i) is exactly
//          zero the solution could not be computed.
//
// For more information on the dsysv() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void sysv( char uplo, blas_int_t n, blas_int_t nrhs, double* A, blas_int_t lda,
                  blas_int_t* ipiv, double* B, blas_int_t ldb, double* work, blas_int_t lwork,
                  blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   dsysv_( &uplo, &n, &nrhs, A, &lda, ipiv, B, &ldb, work, &lwork, info
#if !defined(INTEL_MKL_VERSION)
         , blaze::fortran_charlen_t(1)
#endif
         );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for solving a symmetric indefinite single precision complex linear system
//        of equations (\f$ A*X=B \f$).
// \ingroup lapack_solver
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major square matrix.
// \param lda The total number of elements between two columns of matrix \a A \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param B Pointer to the first element of the column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK csysv() function to compute the solution to the symmetric
// indefinite system of linear equations \f$ A*X=B \f$, where \a A is a \a n-by-\a n matrix and
// \a X and \a B are \a n-by-\a nrhs matrices.
//
// The Bunch-Kaufman decomposition is used to factor \a A as

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched. The factored
// form of \a A is then used to solve the system of equations.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but since factor D(i,i) is exactly
//          zero the solution could not be computed.
//
// For more information on the csysv() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void sysv( char uplo, blas_int_t n, blas_int_t nrhs, complex<float>* A, blas_int_t lda,
                  blas_int_t* ipiv, complex<float>* B, blas_int_t ldb, complex<float>* work,
                  blas_int_t lwork, blas_int_t* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
   BLAZE_STATIC_ASSERT( sizeof( MKL_Complex8 ) == sizeof( complex<float> ) );
   using ET = MKL_Complex8;
#else
   using ET = float;
#endif

   csysv_( &uplo, &n, &nrhs, reinterpret_cast<ET*>( A ), &lda, ipiv,
           reinterpret_cast<ET*>( B ), &ldb, reinterpret_cast<ET*>( work ), &lwork, info
#if !defined(INTEL_MKL_VERSION)
         , blaze::fortran_charlen_t(1)
#endif
         );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for solving a symmetric indefinite double precision complex linear system
//        of equations (\f$ A*X=B \f$).
// \ingroup lapack_solver
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major square matrix.
// \param lda The total number of elements between two columns of matrix \a A \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param B Pointer to the first element of the column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK zsysv() function to compute the solution to the symmetric
// indefinite system of linear equations \f$ A*X=B \f$, where \a A is a \a n-by-\a n matrix and
// \a X and \a B are \a n-by-\a nrhs matrices.
//
// The Bunch-Kaufman decomposition is used to factor \a A as

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched. The factored
// form of \a A is then used to solve the system of equations.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but since factor D(i,i) is exactly
//          zero the solution could not be computed.
//
// For more information on the zsysv() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void sysv( char uplo, blas_int_t n, blas_int_t nrhs, complex<double>* A, blas_int_t lda,
                  blas_int_t* ipiv, complex<double>* B, blas_int_t ldb, complex<double>* work,
                  blas_int_t lwork, blas_int_t* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
   BLAZE_STATIC_ASSERT( sizeof( MKL_Complex16 ) == sizeof( complex<double> ) );
   using ET = MKL_Complex16;
#else
   using ET = double;
#endif

   zsysv_( &uplo, &n, &nrhs, reinterpret_cast<ET*>( A ), &lda, ipiv,
           reinterpret_cast<ET*>( B ), &ldb, reinterpret_cast<ET*>( work ), &lwork, info
#if !defined(INTEL_MKL_VERSION)
         , blaze::fortran_charlen_t(1)
#endif
         );
}
//*************************************************************************************************

} // namespace blaze

#endif
