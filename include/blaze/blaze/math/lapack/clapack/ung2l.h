//=================================================================================================
/*!
//  \file blaze/math/lapack/clapack/ung2l.h
//  \brief Header file for the CLAPACK ung2l wrapper functions
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

#ifndef _BLAZE_MATH_LAPACK_CLAPACK_UNG2L_H_
#define _BLAZE_MATH_LAPACK_CLAPACK_UNG2L_H_


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

void cung2l_( blaze::blas_int_t* m, blaze::blas_int_t* n, blaze::blas_int_t* k, float* A,
              blaze::blas_int_t* lda, float* tau, float* work, blaze::blas_int_t* info );
void zung2l_( blaze::blas_int_t* m, blaze::blas_int_t* n, blaze::blas_int_t* k, double* A,
              blaze::blas_int_t* lda, double* tau, double* work, blaze::blas_int_t* info );

}
#endif
/*! \endcond */
//*************************************************************************************************




namespace blaze {

//=================================================================================================
//
//  LAPACK FUNCTIONS TO RECONSTRUCT Q FROM A QL DECOMPOSITION (UNG2L)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK functions to reconstruct Q from a QL decomposition (ung2l) */
//@{
void ung2l( blas_int_t m, blas_int_t n, blas_int_t k, complex<float>* A, blas_int_t lda,
            const complex<float>* tau, complex<float>* work, blas_int_t* info );

void ung2l( blas_int_t m, blas_int_t n, blas_int_t k, complex<double>* A, blas_int_t lda,
            const complex<double>* tau, complex<double>* work, blas_int_t* info );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the reconstruction of the orthogonal matrix Q from a QL decomposition.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..m)\f$.
// \param k The number of elementary reflectors, whose product defines the matrix \f$[0..n)\f$.
// \param A Pointer to the first element of the single precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param tau Array for the scalar factors of the elementary reflectors; size >= min( \a m, \a n ).
// \param work Auxiliary array; size == \a n.
// \param info Return code of the function call.
// \return void
//
// This function generates all or part of the orthogonal matrix Q from a QL decomposition based on
// the LAPACK cung2l() function for single precision column-major matrices that have already been
// factorized by the sgeqlf() function. The \a info argument provides feedback on the success of
// the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: The i-th argument had an illegal value.
//
// For more information on the cung2l() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void ung2l( blas_int_t m, blas_int_t n, blas_int_t k, complex<float>* A, blas_int_t lda,
                   const complex<float>* tau, complex<float>* work, blas_int_t* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
   BLAZE_STATIC_ASSERT( sizeof( MKL_Complex8 ) == sizeof( complex<float> ) );
   using ET = MKL_Complex8;
#else
   using ET = float;
#endif

   cung2l_( &m, &n, &k, reinterpret_cast<ET*>( A ), &lda,
            const_cast<ET*>( reinterpret_cast<const ET*>( tau ) ),
            reinterpret_cast<ET*>( work ), info );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the reconstruction of the orthogonal matrix Q from a QL decomposition.
// \ingroup lapack_decomposition
//
// \param m The number of rows of the given matrix \f$[0..\infty)\f$.
// \param n The number of columns of the given matrix \f$[0..m)\f$.
// \param k The number of elementary reflectors, whose product defines the matrix \f$[0..n)\f$.
// \param A Pointer to the first element of the double precision column-major matrix.
// \param lda The total number of elements between two columns of the matrix \f$[0..\infty)\f$.
// \param tau Array for the scalar factors of the elementary reflectors; size >= min( \a m, \a n ).
// \param work Auxiliary array; size == \a n.
// \param info Return code of the function call.
// \return void
//
// This function generates all or part of the orthogonal matrix Q from a QL decomposition based on
// the LAPACK zung2l() function for double precision column-major matrices that have already been
// factorized by the dgeqlf() function. The \a info argument provides feedback on the success of
// the function call:
//
//   - = 0: The decomposition finished successfully.
//   - < 0: The i-th argument had an illegal value.
//
// For more information on the zung2l() function, see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
inline void ung2l( blas_int_t m, blas_int_t n, blas_int_t k, complex<double>* A, blas_int_t lda,
                   const complex<double>* tau, complex<double>* work, blas_int_t* info )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

#if defined(INTEL_MKL_VERSION)
   BLAZE_STATIC_ASSERT( sizeof( MKL_INT ) == sizeof( blas_int_t ) );
   BLAZE_STATIC_ASSERT( sizeof( MKL_Complex16 ) == sizeof( complex<double> ) );
   using ET = MKL_Complex16;
#else
   using ET = double;
#endif

   zung2l_( &m, &n, &k, reinterpret_cast<ET*>( A ), &lda,
            const_cast<ET*>( reinterpret_cast<const ET*>( tau ) ),
            reinterpret_cast<ET*>( work ), info );
}
//*************************************************************************************************

} // namespace blaze

#endif
