//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/syevd.h
//  \brief Header file for the CLAPACK syevd wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_SYEVD_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_SYEVD_H_


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

void ssyevd_( char* jobz, char* uplo, blaze::blas_int_t* n, float* A, blaze::blas_int_t* lda,
              float* w, float* work, blaze::blas_int_t* lwork, blaze::blas_int_t* iwork,
              blaze::blas_int_t* liwork, blaze::blas_int_t* info,
              blaze::fortran_charlen_t njobz, blaze::fortran_charlen_t nuplo );
void dsyevd_( char* jobz, char* uplo, blaze::blas_int_t* n, double* A, blaze::blas_int_t* lda,
              double* w, double* work, blaze::blas_int_t* lwork, blaze::blas_int_t* iwork,
              blaze::blas_int_t* liwork, blaze::blas_int_t* info,
              blaze::fortran_charlen_t njobz, blaze::fortran_charlen_t nuplo );

}
#endif
/*! \endcond */
//*************************************************************************************************




namespace blaze {

//=================================================================================================
//
//  LAPACK SYMMETRIC MATRIX EIGENVALUE FUNCTIONS (SYEVD)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK symmetric matrix eigenvalue functions (syevd) */
//@{
void syevd( char jobz, char uplo, blas_int_t n, float* A, blas_int_t lda,
            float* w, float* work, blas_int_t lwork, blas_int_t* iwork,
            blas_int_t liwork, blas_int_t* info );

void syevd( char jobz, char uplo, blas_int_t n, double* A, blas_int_t lda,
            double* w, double* work, blas_int_t lwork, blas_int_t* iwork,
            blas_int_t liwork, blas_int_t* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for computing the eigenvalues of the given dense symmetric single
//        precision column-major matrix.
// \ingroup lapack_eigenvalue
//
// \param jobz \c 'V' to compute the eigenvectors of \a A, \c 'N' to only compute the eigenvalues.
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows and columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the single precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix A \f$[0..\infty)\f$.
// \param w Pointer to the first element of the vector for the eigenvalues.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; see online reference for details.
// \param iwork Auxiliary array; size >= max( 1, \a liwork ).
// \param liwork The dimension of the array \a iwork; see online reference for details.
// \param info Return code of the function call.
// \return void
//
// This function computes the eigenvalues of a symmetric \a n-by-\a n single precision
// column-major matrix based on the LAPACK ssyevd() function. Optionally, it computes the left
// and right eigenvectors using a divide-and-conquer strategy. The real eigenvalues are returned
// in ascending order in the given \a n-dimensional vector \a w.
//
// The parameter \a jobz specifies the computation of the orthonormal eigenvectors of \a A:
//
//   - \c 'V': The eigenvectors of \a A are computed and returned within the matrix \a A.
//   - \c 'N': The eigenvectors of \a A are not computed.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The computation finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the algorithm failed to converge; i values did not converge to zero.
//
// For more information on the ssyevd() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void syevd( char jobz, char uplo, blas_int_t n, float* A, blas_int_t lda,
                   float* w, float* work, blas_int_t lwork, blas_int_t* iwork,
                   blas_int_t liwork, blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   ssyevd_( &jobz, &uplo, &n, A, &lda, w, work, &lwork, iwork, &liwork, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1), blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for computing the eigenvalues of the given dense symmetric double
//        precision column-major matrix.
// \ingroup lapack_eigenvalue
//
// \param jobz \c 'V' to compute the eigenvectors of \a A, \c 'N' to only compute the eigenvalues.
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \param n The number of rows and columns of the given matrix \f$[0..\infty)\f$.
// \param A Pointer to the first element of the double precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix A \f$[0..\infty)\f$.
// \param w Pointer to the first element of the vector for the eigenvalues.
// \param work Auxiliary array; size >= max( 1, \a lwork ).
// \param lwork The dimension of the array \a work; see online reference for details.
// \param iwork Auxiliary array; size >= max( 1, \a liwork ).
// \param liwork The dimension of the array \a iwork; see online reference for details.
// \param info Return code of the function call.
// \return void
//
// This function computes the eigenvalues of a symmetric \a n-by-\a n double precision
// column-major matrix based on the LAPACK ssyevd() function. Optionally, it computes the left
// and right eigenvectors using a divide-and-conquer strategy. The real eigenvalues are returned
// in ascending order in the given \a n-dimensional vector \a w.
//
// The parameter \a jobz specifies the computation of the orthonormal eigenvectors of \a A:
//
//   - \c 'V': The eigenvectors of \a A are computed and returned within the matrix \a A.
//   - \c 'N': The eigenvectors of \a A are not computed.
//
// The \a info argument provides feedback on the success of the function call:
//
//   - = 0: The computation finished successfully.
//   - < 0: If info = -i, the i-th argument had an illegal value.
//   - > 0: If info = i, the algorithm failed to converge; i values did not converge to zero.
//
// For more information on the ssyevd() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void syevd( char jobz, char uplo, blas_int_t n, double* A, blas_int_t lda,
                   double* w, double* work, blas_int_t lwork, blas_int_t* iwork,
                   blas_int_t liwork, blas_int_t* info )
{
#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
#endif

   dsyevd_( &jobz, &uplo, &n, A, &lda, w, work, &lwork, iwork, &liwork, info
#if !defined(INTEL_MKL_VERSION)
          , blaze::fortran_charlen_t(1), blaze::fortran_charlen_t(1)
#endif
          );
}
//*************************************************************************************************

} // namespace blaze

#endif
