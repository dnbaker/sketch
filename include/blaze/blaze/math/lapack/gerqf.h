//=================================================================================================
/*!
//  \file blaze/math/lapack/gerqf.h
//  \brief Header file for the LAPACK RQ decomposition functions (gerqf)
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

#ifndef _BLAZE_MATH_LAPACK_GERQF_H_
#define _BLAZE_MATH_LAPACK_GERQF_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <memory>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Adaptor.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/Contiguous.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/lapack/clapack/geqlf.h>
#include <blaze/math/lapack/clapack/gerqf.h>
#include <blaze/util/Assert.h>
#include <blaze/util/NumericCast.h>


namespace blaze {

//=================================================================================================
//
//  LAPACK RQ DECOMPOSITION FUNCTIONS (GERQF)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK RQ decomposition functions (gerqf) */
//@{
template< typename MT, bool SO >
void gerqf( DenseMatrix<MT,SO>& A, ElementType_t<MT>* tau );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the RQ decomposition of the given dense matrix.
// \ingroup lapack_decomposition
//
// \param A The matrix to be decomposed.
// \param tau Array for the scalar factors of the elementary reflectors; size >= min( \a m, \a n ).
// \return void
//
// This function performs the dense matrix RQ decomposition of a general \a m-by-\a n matrix
// based on the LAPACK gerqf() functions. Note that this function can only be used for general,
// non-adapted matrices with \c float, \c double, \c complex<float>, or \c complex<double> element
// type. The attempt to call the function with any adapted matrix or matrices of any other element
// type results in a compile time error!\n
//
// In case of a column-major matrix, the resulting decomposition has the form

                              \f[ A = R \cdot Q, \f]

// where the \c Q is represented as a product of elementary reflectors

               \f[ Q = H(1) H(2) . . . H(k) \texttt{, with k = min(m,n).} \f]

// Each H(i) has the form

                      \f[ H(i) = I - tau \cdot v \cdot v^T, \f]

// where \c tau is a real scalar, and \c v is a real vector with <tt>v(n-k+i+1:n) = 0</tt> and
// <tt>v(n-k+i) = 1</tt>. <tt>v(1:n-k+i-1)</tt> is stored on exit in <tt>A(m-k+i,1:n-k+i-1)</tt>,
// and \c tau in \c tau(i). Thus in case \a m <= \a n, the upper triangle of the subarray
// A(1:m,n-m+1:n) contains the \a m-by-\a m upper triangular matrix \c R and in case \a m >= \a n,
// the elements on and above the (\a m-\a n)-th subdiagonal contain the \a m-by-\a n upper
// trapezoidal matrix \c R; the remaining elements in combination with the array \c tau represent
// the orthogonal matrix \c Q as a product of min(\a m,\a n) elementary reflectors.
//
// For more information on the gerqf() functions (i.e. sgerqf(), dgerqf(), cgerqf(), and zgerqf())
// see the LAPACK online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
template< typename MT  // Type of the dense matrix
        , bool SO >    // Storage order of the dense matrix
inline void gerqf( DenseMatrix<MT,SO>& A, ElementType_t<MT>* tau )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT );
   BLAZE_CONSTRAINT_MUST_BE_CONTIGUOUS_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT> );

   using ET = ElementType_t<MT>;

   blas_int_t m   ( numeric_cast<blas_int_t>( SO ? (*A).rows() : (*A).columns() ) );
   blas_int_t n   ( numeric_cast<blas_int_t>( SO ? (*A).columns() : (*A).rows() ) );
   blas_int_t lda ( numeric_cast<blas_int_t>( (*A).spacing() ) );
   blas_int_t info( 0 );

   if( m == 0 || n == 0 ) {
      return;
   }

   blas_int_t lwork( ( SO ? m : n ) * lda );
   const std::unique_ptr<ET[]> work( new ET[lwork] );

   if( SO ) {
      gerqf( m, n, (*A).data(), lda, tau, work.get(), lwork, &info );
   }
   else {
      geqlf( m, n, (*A).data(), lda, tau, work.get(), lwork, &info );
   }

   BLAZE_INTERNAL_ASSERT( info == 0, "Invalid argument for RQ decomposition" );
}
//*************************************************************************************************

} // namespace blaze

#endif
