//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/potrs.h
//  \brief Header file for the CLAPACK potrs wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_POTRS_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_POTRS_H_


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

void spotrs_( char* uplo, blaze::blas_int_t* n, blaze::blas_int_t* nrhs, float* A,
              blaze::blas_int_t* lda, float* B, blaze::blas_int_t* ldb,
              blaze::blas_int_t* info, blaze::fortran_charlen_t nuplo );
void dpotrs_( char* uplo, blaze::blas_int_t* n, blaze::blas_int_t* nrhs, double* A,
              blaze::blas_int_t* lda, double* B, blaze::blas_int_t* ldb,
              blaze::blas_int_t* info, blaze::fortran_charlen_t nuplo );
void cpotrs_( char* uplo, blaze::blas_int_t* n, blaze::blas_int_t* nrhs, float* A,
              blaze::blas_int_t* lda, float* B, blaze::blas_int_t* ldb,
              blaze::blas_int_t* info, blaze::fortran_charlen_t nuplo );
void zpotrs_( char* uplo, blaze::blas_int_t* n, blaze::blas_int_t* nrhs, double* A,
              blaze::blas_int_t* lda, double* B, blaze::blas_int_t* ldb,
              blaze::blas_int_t* info, blaze::fortran_charlen_t nuplo );

}
#endif
/*! \endcond */
//*************************************************************************************************




namespace blaze {

//=================================================================================================
//
//  LAPACK LLH-BASED SUBSTITUTION FUNCTIONS (POTRS)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LLH-based substitution functions (potrs) */
//@{
void potrs( char uplo, blas_int_t n, blas_int_t nrhs, const float* A, blas_int_t lda,
            float* B, blas_int_t ldb, blas_int_t* info );

void potrs( char uplo, blas_int_t n, blas_int_t nrhs, const double* A, blas_int_t lda,
            double* B, blas_int_t ldb, blas_int_t* info );

void potrs( char uplo, blas_int_t n, blas_int_t nrhs, const complex<float>* A, blas_int_t lda,
            complex<float>* B, blas_int_t ldb, blas_int_t* info );

void potrs( char uplo, blas_int_t n, blas_int_t nrhs, const complex<double>* A, blas_int_t lda,
            complex<double>* B, blas_int_t ldb, blas_int_t* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a positive definite single precision
//        linear system of equations (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of the column-major matrix \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major square matrix.
// \param lda The total number of elements between two columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of the single precision column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK spotrs() function to perform the substitution step to compute
// the solution to the positive definite system of linear equations \f$ A*X=B \f$, where \a A is
// a \a n-by-\a n matrix that has already been factorized by the spotrf() function and \a X and
// \a B are column-major \a n-by-\a nrhs matrices.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//
// For more information on the spotrs() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void potrs( char uplo, blas_int_t n, blas_int_t nrhs, const float* A, blas_int_t lda,
                   float* B, blas_int_t ldb, blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   spotrs_( &uplo, &n, &nrhs, const_cast<float*>( A ), &lda, B, &ldb, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a positive definite double precision
//        linear system of equations (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of the column-major matrix \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major square matrix.
// \param lda The total number of elements between two columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of the double precision column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK dpotrs() function to perform the substitution step to compute
// the solution to the positive definite system of linear equations \f$ A*X=B \f$, where \a A is
// a \a n-by-\a n matrix that has already been factorized by the dpotrf() function and \a X and
// \a B are column-major \a n-by-\a nrhs matrices.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//
// For more information on the dpotrs() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void potrs( char uplo, blas_int_t n, blas_int_t nrhs, const double* A, blas_int_t lda,
                   double* B, blas_int_t ldb, blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   dpotrs_( &uplo, &n, &nrhs, const_cast<double*>( A ), &lda, B, &ldb, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a positive definite single precision
//        complex linear system of equations (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of the column-major matrix \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major square matrix.
// \param lda The total number of elements between two columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of the single precision complex column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK cpotrs() function to perform the substitution step to compute
// the solution to the positive definite system of linear equations \f$ A*X=B \f$, where \a A is
// a \a n-by-\a n matrix that has already been factorized by the cpotrf() function and \a X and
// \a B are column-major \a n-by-\a nrhs matrices.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//
// For more information on the cpotrs() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void potrs( char uplo, blas_int_t n, blas_int_t nrhs, const complex<float>* A, blas_int_t lda,
                   complex<float>* B, blas_int_t ldb, blas_int_t* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
   BLAZE_STATIC_ASSERT( sizeof( MKL_Complex8 ) == sizeof( complex<float> ) );
   using ET = MKL_Complex8;
#else
   using ET = float;
#endif

   cpotrs_( &uplo, &n, &nrhs, const_cast<ET*>( reinterpret_cast<const ET*>( A ) ),
            &lda, reinterpret_cast<ET*>( B ), &ldb, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the substitution step of solving a positive definite double precision
//        complex linear system of equations (\f$ A*X=B \f$).
// \ingroup lapack_substitution
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of the column-major matrix \f$[0..\infty)\f$.
// \param nrhs The number of right-hand side vectors \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major square matrix.
// \param lda The total number of elements between two columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of the double precision complex column-major matrix.
// \param ldb The total number of elements between two columns of matrix \a B \f$[0..\infty)\f$.
// \param info Return code of the function call.
// \return void
//
// This function uses the LAPACK zpotrs() function to perform the substitution step to compute
// the solution to the positive definite system of linear equations \f$ A*X=B \f$, where \a A is
// a \a n-by-\a n matrix that has already been factorized by the zpotrf() function and \a X and
// \a B are column-major \a n-by-\a nrhs matrices.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The function finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//
// For more information on the zpotrs() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void potrs( char uplo, blas_int_t n, blas_int_t nrhs, const complex<double>* A, blas_int_t lda,
                   complex<double>* B, blas_int_t ldb, blas_int_t* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
   BLAZE_STATIC_ASSERT( sizeof( MKL_Complex16 ) == sizeof( complex<double> ) );
   using ET = MKL_Complex16;
#else
   using ET = double;
#endif

   zpotrs_( &uplo, &n, &nrhs, const_cast<ET*>( reinterpret_cast<const ET*>( A ) ),
            &lda, reinterpret_cast<ET*>( B ), &ldb, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************

} // namespace blaze

#endif
