//=================================================================================================
/*!
//  \file blaze/math/lapack/potrf.h
//  \brief Header file for the LAPACK Cholesky decomposition functions (potrf)
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

#ifndef _BLAZE_MATH_LAPACK_POTRF_H_
#define _BLAZE_MATH_LAPACK_POTRF_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Adaptor.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/Contiguous.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/lapack/clapack/potrf.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/NumericCast.h>


namespace blaze {

//=================================================================================================
//
//  LAPACK LLH (CHOLESKY) DECOMPOSITION FUNCTIONS (POTRF)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK LLH (Cholesky) decomposition functions (potrf) */
//@{
template< typename MT, bool SO >
void potrf( DenseMatrix<MT,SO>& A, char uplo );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the Cholesky decomposition of the given dense positive definite matrix.
// \ingroup lapack_decomposition
//
// \param A The matrix to be decomposed.
// \param uplo \c 'L' to use the lower part of the matrix, \c 'U' to use the upper part.
// \return void
// \exception std::invalid_argument Invalid non-square matrix provided.
// \exception std::invalid_argument Invalid uplo argument provided.
// \exception std::runtime_error Decomposition of singular matrix failed.
//
// This function performs the dense matrix Cholesky decomposition of a symmetric positive definite
// matrix based on the LAPACK potrf() functions. Note that the function only works for general,
// non-adapted matrices with \c float, \c double, \c complex<float>, or \c complex<double> element
// type. The attempt to call the function with any adapted matrix or matrices of any other element
// type results in a compile time error!\n
//
// The decomposition has the form

                      \f[ A = U^{H} U \texttt{ (if uplo = 'U'), or }
                          A = L L^{H} \texttt{ (if uplo = 'L'), } \f]

// where \c U is an upper triangular matrix and \c L is a lower triangular matrix. The Cholesky
// decomposition fails if ...
//
//  - ... the given system matrix \a A is not a symmetric positive definite matrix;
//  - ... the given \a uplo argument is neither \c 'L' nor \c 'U'.
//
// In all failure cases an exception is thrown.
//
// For more information on the potrf() functions (i.e. spotrf(), dpotrf(), cpotrf(), and zpotrf())
// see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
//
// \note This function does only provide the basic exception safety guarantee, i.e. in case of an
// exception \a A may already have been modified.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void potrf( DenseMatrix<MT,SO>& A, char uplo )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT );
   BLAZE_CONSTRAINT_MUST_BE_CONTIGUOUS_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT> );

   if( !isSquare( *A ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid non-square matrix provided" );
   }

   if( uplo != 'L' && uplo != 'U' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid uplo argument provided" );
   }

   blas_int_t n   ( numeric_cast<blas_int_t>( (*A).rows()    ) );
   blas_int_t lda ( numeric_cast<blas_int_t>( (*A).spacing() ) );
   blas_int_t info( 0 );

   if( n == 0 ) {
      return;
   }

   if( IsRowMajorMatrix_v<MT> ) {
      ( uplo == 'L' )?( uplo = 'U' ):( uplo = 'L' );
   }

   potrf( uplo, n, (*A).data(), lda, &info );

   BLAZE_INTERNAL_ASSERT( info >= 0, "Invalid argument for Cholesky decomposition" );

   if( info > 0 ) {
      BLAZE_THROW_LAPACK_ERROR( "Decomposition of non-positive-definite matrix failed" );
   }
}
//*************************************************************************************************

} // namespace blaze

#endif
