//=================================================================================================
/*!
//  \file blaze/math/expressions/MatScalarDivExpr.h
//  \brief Header file for the MatScalarDivExpr base class
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

#ifndef _BLAZE_MATH_EXPRESSIONS_MATSCALARDIVEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_MATSCALARDIVEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DivExpr.h>
#include <blaze/math/typetraits/IsInvertible.h>
#include <blaze/math/typetraits/IsScalar.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Base class for all matrix/scalar division expression templates.
// \ingroup math
//
// The MatScalarDivExpr class serves as a tag for all expression templates that implement a
// matrix/scalar division. All classes, that represent a matrix/scalar division and that are
// used within the expression template environment of the Blaze library have to derive publicly
// from this class in order to qualify as matrix/scalar division expression template. Only in
// case a class is derived publicly from the MatScalarDivExpr base class, the IsMatScalarDivExpr
// type trait recognizes the class as valid matrix/scalar division expression template.
*/
template< typename MT >  // Matrix base type of the expression
struct MatScalarDivExpr
   : public DivExpr<MT>
{};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a matrix-scalar division expression
//        and a scalar value (\f$ A=(B/s1)*s2 \f$).
// \ingroup math
//
// \param mat The left-hand side matrix-scalar division.
// \param scalar The right-hand side scalar value for the multiplication.
// \return The scaled result matrix.
//
// This operator implements a performance optimized treatment of the multiplication of a
// matrix-scalar division expression and a scalar value.
*/
template< typename MT  // Matrix base type of the expression
        , typename ST  // Type of the right-hand side scalar
        , EnableIf_t< IsScalar_v<ST> &&
                      ( IsInvertible_v<ST> ||
                        IsInvertible_v< RightOperand_t< MatrixType_t<MT> > > ) >* = nullptr >
inline decltype(auto) operator*( const MatScalarDivExpr<MT>& mat, ST scalar )
{
   BLAZE_FUNCTION_TRACE;

   return (*mat).leftOperand() * ( scalar / (*mat).rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication operator for the multiplication of a scalar value and a matrix-scalar
//        division expression (\f$ A=s2*(B/s1) \f$).
// \ingroup math
//
// \param scalar The left-hand side scalar value for the multiplication.
// \param mat The right-hand side matrix-scalar division.
// \return The scaled result matrix.
//
// This operator implements a performance optimized treatment of the multiplication of a
// scalar value and a matrix-scalar division expression.
*/
template< typename ST  // Type of the left-hand side scalar
        , typename MT  // Matrix base type of the expression
        , EnableIf_t< IsScalar_v<ST> &&
                      ( IsInvertible_v<ST> ||
                        IsInvertible_v< RightOperand_t< MatrixType_t<MT> > > ) >* = nullptr >
inline decltype(auto) operator*( ST scalar, const MatScalarDivExpr<MT>& mat )
{
   BLAZE_FUNCTION_TRACE;

   return (*mat).leftOperand() * ( scalar / (*mat).rightOperand() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division operator for the division of a dense matrix-scalar division expression
//        and a scalar value (\f$ A=(B/s1)/s2 \f$).
// \ingroup dense_matrix
//
// \param mat The left-hand side dense matrix-scalar division.
// \param scalar The right-hand side scalar value for the division.
// \return The scaled result matrix.
//
// This operator implements a performance optimized treatment of the division of a dense
// matrix-scalar division expression and a scalar value.
*/
template< typename MT  // Matrix base type of the expression
        , typename ST  // Type of the right-hand side scalar
        , EnableIf_t< IsScalar_v<ST> >* = nullptr >
inline decltype(auto) operator/( const MatScalarDivExpr<MT>& mat, ST scalar )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_USER_ASSERT( scalar != ST(0), "Division by zero detected" );

   return (*mat).leftOperand() / ( (*mat).rightOperand() * scalar );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL RESTRUCTURING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Calculation of the transpose of the given matrix-scalar division.
// \ingroup math
//
// \param matrix The matrix-scalar division expression to be transposed.
// \return The transpose of the expression.
//
// This operator implements the performance optimized treatment of the transpose of a
// matrix-scalar division. It restructures the expression \f$ A=trans(B/s1) \f$ to
// the expression \f$ A=trans(B)/s1 \f$.
*/
template< typename MT >  // Matrix base type of the expression
inline decltype(auto) trans( const MatScalarDivExpr<MT>& matrix )
{
   BLAZE_FUNCTION_TRACE;

   return trans( (*matrix).leftOperand() ) / (*matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Calculation of the complex conjugate of the given matrix-scalar division.
// \ingroup math
//
// \param matrix The matrix-scalar division expression to be conjugated.
// \return The complex conjugate of the expression.
//
// This operator implements the performance optimized treatment of the complex conjugate
// of a matrix-scalar division. It restructures the expression \f$ a=conj(b/s1) \f$ to the
// expression \f$ a=conj(b)/s1 \f$.
*/
template< typename MT >  // Matrix base type of the expression
inline decltype(auto) conj( const MatScalarDivExpr<MT>& matrix )
{
   BLAZE_FUNCTION_TRACE;

   return conj( (*matrix).leftOperand() ) / (*matrix).rightOperand();
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
