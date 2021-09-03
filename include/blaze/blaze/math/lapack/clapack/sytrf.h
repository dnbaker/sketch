//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/sytrf.h
//  \brief Header file for the CLAPACK sytrf wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_SYTRF_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_SYTRF_H_


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

void ssytrf_( char* uplo, blaze::blas_int_t* n, float* A, blaze::blas_int_t* lda,
              blaze::blas_int_t* ipiv, float* work, blaze::blas_int_t* lwork,
              blaze::blas_int_t* info, blaze::fortran_charlen_t nuplo );
void dsytrf_( char* uplo, blaze::blas_int_t* n, double* A, blaze::blas_int_t* lda,
              blaze::blas_int_t* ipiv, double* work, blaze::blas_int_t* lwork,
              blaze::blas_int_t* info, blaze::fortran_charlen_t nuplo );
void csytrf_( char* uplo, blaze::blas_int_t* n, float* A, blaze::blas_int_t* lda,
              blaze::blas_int_t* ipiv, float* work, blaze::blas_int_t* lwork,
              blaze::blas_int_t* info, blaze::fortran_charlen_t nuplo );
void zsytrf_( char* uplo, blaze::blas_int_t* n, double* A, blaze::blas_int_t* lda,
              blaze::blas_int_t* ipiv, double* work, blaze::blas_int_t* lwork,
              blaze::blas_int_t* info, blaze::fortran_charlen_t nuplo );

}
#endif
/*! \endcond */
//*************************************************************************************************




namespace blaze {

//=================================================================================================
//
//  LAPACK LDLT DECOMPOSITION FUNCTIONS (SYTRF)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LDLT decomposition functions (sytrf) */
//@{
void sytrf( char uplo, blas_int_t n, float* A, blas_int_t lda, blas_int_t* ipiv,
            float* work, blas_int_t lwork, blas_int_t* info );

void sytrf( char uplo, blas_int_t n, double* A, blas_int_t lda, blas_int_t* ipiv,
            double* work, blas_int_t lwork, blas_int_t* info );

void sytrf( char uplo, blas_int_t n, complex<float>* A, blas_int_t lda, blas_int_t* ipiv,
            complex<float>* work, blas_int_t lwork, blas_int_t* info );

void sytrf( char uplo, blas_int_t n, complex<double>* A, blas_int_t lda, blas_int_t* ipiv,
            complex<double>* work, blas_int_t lwork, blas_int_t* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the decomposition of the given dense symmetric indefinite single
//        precision column-major matrix.
// \ingroup lapack_decomposition
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of the symmetric matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix decomposition of a symmetric indefinite single precision
// column-major matrix based on the LAPACK ssytrf() function, which uses the Bunch-Kaufman diagonal
// pivoting method. The decomposition has the form

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but D(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= \a n*NB, where NB
// is the optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a
// workspace query is assumed. The function only calculates the optimal size of the \a work
// array and returns this value as the first entry of the \a work array.
//
// For more information on the ssytrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void sytrf( char uplo, blas_int_t n, float* A, blas_int_t lda, blas_int_t* ipiv,
                   float* work, blas_int_t lwork, blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   ssytrf_( &uplo, &n, A, &lda, ipiv, work, &lwork, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the decomposition of the given dense symmetric indefinite double
//        precision column-major matrix.
// \ingroup lapack_decomposition
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of the symmetric matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix decomposition of a symmetric indefinite double precision
// column-major matrix based on the LAPACK dsytrf() function, which uses the Bunch-Kaufman diagonal
// pivoting method. The decomposition has the form

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but D(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= \a n*NB, where NB
// is the optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a
// workspace query is assumed. The function only calculates the optimal size of the \a work
// array and returns this value as the first entry of the \a work array.
//
// For more information on the dsytrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void sytrf( char uplo, blas_int_t n, double* A, blas_int_t lda, blas_int_t* ipiv,
                   double* work, blas_int_t lwork, blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   dsytrf_( &uplo, &n, A, &lda, ipiv, work, &lwork, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the decomposition of the given dense symmetric indefinite single
//        precision complex column-major matrix.
// \ingroup lapack_decomposition
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of the symmetric matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix decomposition of a symmetric indefinite single precision
// column-major matrix based on the LAPACK csytrf() function, which uses the Bunch-Kaufman diagonal
// pivoting method. The decomposition has the form

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but D(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= \a n*NB, where NB
// is the optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a
// workspace query is assumed. The function only calculates the optimal size of the \a work
// array and returns this value as the first entry of the \a work array.
//
// For more information on the csytrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void sytrf( char uplo, blas_int_t n, complex<float>* A, blas_int_t lda, blas_int_t* ipiv,
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

   csytrf_( &uplo, &n, reinterpret_cast<ET*>( A ), &lda, ipiv,
            reinterpret_cast<ET*>( work ), &lwork, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the decomposition of the given dense symmetric indefinite double
//        precision complex column-major matrix.
// \ingroup lapack_decomposition
//
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows/columns of the symmetric matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision complex column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param ipiv Auxiliary array of size \a n for the pivot indices.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; size >= max( 1, \a n ).
// \param info Return code of the function call.
// \return void
//
// This function performs the dense matrix decomposition of a symmetric indefinite double precision
// column-major matrix based on the LAPACK zsytrf() function, which uses the Bunch-Kaufman diagonal
// pivoting method. The decomposition has the form

                      \f[ A = U D U^{T} \texttt{ (if uplo = 'U'), or }
                          A = L D L^{T} \texttt{ (if uplo = 'L'), } \f]

// where \c U (or \c L) is a product of permutation and unit upper (lower) triangular matrices,
// and \c D is symmetric and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The resulting
// decomposition is stored within \a A: In case \a uplo is set to \c 'L' the result is stored in
// the lower part of the matrix and the upper part remains untouched, in case \a uplo is set to
// \c 'U' the result is stored in the upper part and the lower part remains untouched.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the decomposition has been completed, but D(i,i) is exactly zero.
//
// If the function exits successfully (i.e. \a info = 0) then the first element of the \a work
// array returns the optimal \a lwork. For optimal performance \a lwork >= \a n*NB, where NB
// is the optimal blocksize returned by the LAPACK function ilaenv(). If \a lwork = -1 then a
// workspace query is assumed. The function only calculates the optimal size of the \a work
// array and returns this value as the first entry of the \a work array.
//
// For more information on the zsytrf() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void sytrf( char uplo, blas_int_t n, complex<double>* A, blas_int_t lda, blas_int_t* ipiv,
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

   zsytrf_( &uplo, &n, reinterpret_cast<ET*>( A ), &lda, ipiv,
            reinterpret_cast<ET*>( work ), &lwork, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************

} // namespace blaze

#endif
