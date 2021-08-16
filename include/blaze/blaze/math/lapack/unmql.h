//=================================================================================================
/*!
//  \file blaze/math/lapack/unmql.h
//  \brief Header file for the LAPACK functions to multiply Q from a QL decomposition with a matrix (unmql)
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

#ifndef _BLAZE_MATH_LAPACK_UNMQL_H_
#define _BLAZE_MATH_LAPACK_UNMQL_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <memory>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Adaptor.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/ConstDataAccess.h>
#include <blaze/math/constraints/Contiguous.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/lapack/clapack/unmql.h>
#include <blaze/math/lapack/clapack/unmrq.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Complex.h>
#include <blaze/util/NumericCast.h>


namespace blaze {

//=================================================================================================
//
//  LAPACK FUNCTIONS TO MULTIPLY Q FROM A QL DECOMPOSITION WITH A MATRIX (UNMQL)
//
//=================================================================================================

//*************************************************************************************************
/*!\name LAPACK functions to multiply Q from a QL decomposition with a matrix (unmql) */
//@{
template< typename MT1, bool SO, typename MT2 >
void unmql( DenseMatrix<MT1,SO>& C, const DenseMatrix<MT2,SO>& A,
            char side, char trans, const ElementType_t<MT2>* tau );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief LAPACK kernel for the multiplication of the double precision Q from a QL decomposition
//        with another matrix.
// \ingroup lapack_decomposition
//
// \param C The matrix multiplier.
// \param A The decomposed matrix.
// \param side \c 'L' to apply \f$ Q \f$ or \f$ Q^T \f$ from the left, \c 'R' to apply from the right.
// \param trans \c 'N' for \f$ Q \f$, \c 'T' for \f$ Q^T \f$.
// \param tau Array for the scalar factors of the elementary reflectors.
// \return void
// \exception std::invalid_argument Invalid size of Q matrix.
// \exception std::invalid_argument Invalid side argument provided.
// \exception std::invalid_argument Invalid trans argument provided.
//
// This function multiplies a square \a Q matrix resulting from the QL decomposition of the
// geqlf() functions with the given general \a m-by-\a n matrix \a C. Depending on the settings
// of \a side and \a trans it overwrites \a C with

   \code
                | side = 'L'   | side = 'R'
   -------------|--------------|--------------
   trans = 'N': | Q * C        | C * Q
   trans = 'T': | trans(Q) * C | C * trans(Q)
   \endcode

// Note that the size of matrix \c C is preserved, which means that the function does not work for
// non-square Q matrices. Therefore in case the number of rows of \a A is smaller than the number
// of columns, a \a std::invalid_argument exception is thrown. Also note that this function can
// only be used for general, non-adapted matrices with \c float or \c double element type. The
// attempt to call the function with any adapted matrix or matrices of any other element type
// results in a compile time error!
//
// The following code example demonstrates the use of the unmql() function:

   \code
   using blaze::DynamicMatrix;
   using blaze::columnMajor;

   DynamicMatrix<double,columnMajor> A;
   DynamicMatrix<double,columnMajor> C;
   DynamicVector<double> tau;
   // ... Resizing and initialization

   geqlf( A, tau.data() );               // Performing the QL decomposition
   unmql( C, A, 'R', 'N', tau.data() );  // Computing C = C * Q
   \endcode

// The function fails if ...
//
//  - ... the number of rows of the given \a A matrix is larger than the number of columns;
//  - ... the given \a side argument is neither \c 'L' nor \c 'R';
//  - ... the given \a trans argument is neither \c 'N' nor \c 'T'.
//
// In all failure cases an exception is thrown.
//
// For more information on the unmql() functions (i.e. cunmql() and zunmql()) see the LAPACK
// online documentation browser:
//
//        http://www.netlib.org/lapack/explore-html/
//
// \note This function can only be used if a fitting LAPACK library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
template< typename MT1, bool SO, typename MT2 >
inline void unmql( DenseMatrix<MT1,SO>& C, const DenseMatrix<MT2,SO>& A,
                   char side, char trans, const ElementType_t<MT2>* tau )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_CONTIGUOUS_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_BE_COMPLEX_TYPE( ElementType_t<MT1> );

   BLAZE_CONSTRAINT_MUST_NOT_BE_ADAPTOR_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_CONTIGUOUS_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT2> );
   BLAZE_CONSTRAINT_MUST_BE_COMPLEX_TYPE( ElementType_t<MT2> );

   using ET = ElementType_t<MT1>;

   if( (*A).rows() < (*A).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid size of Q matrix" );
   }

   if( side != 'L' && side != 'R' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid side argument provided" );
   }

   if( trans != 'N' && trans != 'C' ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid trans argument provided" );
   }

   blas_int_t m   ( numeric_cast<blas_int_t>( SO ? (*C).rows() : (*C).columns() ) );
   blas_int_t n   ( numeric_cast<blas_int_t>( SO ? (*C).columns() : (*C).rows() ) );
   blas_int_t k   ( numeric_cast<blas_int_t>( min( (*A).rows(), (*A).columns() ) ) );
   blas_int_t lda ( numeric_cast<blas_int_t>( (*A).spacing() ) );
   blas_int_t ldc ( numeric_cast<blas_int_t>( (*C).spacing() ) );
   blas_int_t info( 0 );

   if( m == 0 || n == 0 || k == 0 ) {
      return;
   }

   if( !SO ) {
      ( side  == 'L' )?( side  = 'R' ):( side  = 'L' );
   }

   blas_int_t lwork( k*ldc );
   const std::unique_ptr<ET[]> work( new ET[lwork] );
   const size_t offset( (*A).rows() - (*A).columns() );

   if( SO ) {
      unmql( side, trans, m, n, k, (*A).data()+offset, lda, tau, (*C).data(), ldc, work.get(), lwork, &info );
   }
   else {
      unmrq( side, trans, m, n, k, (*A).data(offset), lda, tau, (*C).data(), ldc, work.get(), lwork, &info );
   }

   BLAZE_INTERNAL_ASSERT( info == 0, "Invalid argument for Q multiplication" );
}
//*************************************************************************************************

} // namespace blaze

#endif
