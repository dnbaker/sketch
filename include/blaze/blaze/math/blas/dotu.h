//=================================================================================================
/*!
//  \file blaze/math/blas/dotu.h
//  \brief Header file for BLAS dot product (dotu)
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

#ifndef _BLAZE_MATH_BLAS_DOTU_H_
#define _BLAZE_MATH_BLAS_DOTU_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/blas/cblas/dotu.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/ConstDataAccess.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/system/MacroDisable.h>
#include <blaze/util/NumericCast.h>


namespace blaze {

//=================================================================================================
//
//  BLAS DOT PRODUCT (DOTU)
//
//=================================================================================================

//*************************************************************************************************
/*!\name BLAS dot product functions (dotu) */
//@{
template< typename VT1, bool TF1, typename VT2, bool TF2 >
ElementType_t<VT1> dotu( const DenseVector<VT1,TF1>& x, const DenseVector<VT2,TF2>& y );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense vector dot product (\f$ s=\vec{x}*\vec{y} \f$).
// \ingroup blas
//
// \param x The left-hand side dense vector operand.
// \param y The right-hand side dense vector operand.
// \return The result of the dot product computation.
//
// This function performs the dense vector dot product based on the BLAS dotu() functions. Note
// that the function only works for vectors with \c float, \c double, \c complex<float>, or
// \c complex<double> element type. The attempt to call the function with vectors of any other
// element type results in a compile time error.
//
// \note This function can only be used if a fitting BLAS library, which supports this function,
// is available and linked to the executable. Otherwise a call to this function will result in a
// linker error.
*/
template< typename VT1, bool TF1, typename VT2, bool TF2 >
ElementType_t<VT1> dotu( const DenseVector<VT1,TF1>& x, const DenseVector<VT2,TF2>& y )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT2 );

   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS( VT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS( VT2 );

   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<VT1> );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<VT2> );

   const blas_int_t n( numeric_cast<blas_int_t>( (*x).size() ) );

   return dotu( n, (*x).data(), 1, (*y).data(), 1 );
}
//*************************************************************************************************

} // namespace blaze

#endif
