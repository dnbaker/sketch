//=================================================================================================
/*!
//  \file blaze/math/expressions/SVecSVecCrossExpr.h
//  \brief Header file for the sparse vector/sparse vector cross product expression
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

#ifndef _BLAZE_MATH_EXPRESSIONS_SVECSVECCROSSEXPR_H_
#define _BLAZE_MATH_EXPRESSIONS_SVECSVECCROSSEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/ColumnVector.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/SparseVector.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/constraints/Zero.h>
#include <blaze/math/dense/Forward.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/Computation.h>
#include <blaze/math/expressions/CrossExpr.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/Forward.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsTemporary.h>
#include <blaze/math/typetraits/IsZero.h>
#include <blaze/system/MacroDisable.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS SVECSVECCROSSEXPR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Expression object for sparse vector-sparse vector cross products.
// \ingroup dense_vector_expression
//
// The SVecSVecCrossExpr class represents the compile time expression for cross products
// between sparse vectors.
*/
template< typename VT1  // Type of the left-hand side sparse vector
        , typename VT2  // Type of the right-hand side sparse vector
        , bool TF >     // Transpose flag
class SVecSVecCrossExpr
   : public CrossExpr< DenseVector< SVecSVecCrossExpr<VT1,VT2,TF>, TF > >
   , private Computation
{
 private:
   //**Type definitions****************************************************************************
   using RT1 = ResultType_t<VT1>;     //!< Result type of the left-hand side sparse vector expression.
   using RT2 = ResultType_t<VT2>;     //!< Result type of the right-hand side sparse vector expression.
   using RN1 = ReturnType_t<VT1>;     //!< Return type of the left-hand side sparse vector expression.
   using RN2 = ReturnType_t<VT2>;     //!< Return type of the right-hand side sparse vector expression.
   using CT1 = CompositeType_t<VT1>;  //!< Composite type of the left-hand side sparse vector expression.
   using CT2 = CompositeType_t<VT2>;  //!< Composite type of the right-hand side sparse vector expression.
   using ET1 = ElementType_t<VT1>;    //!< Element type of the left-hand side sparse vector expression.
   using ET2 = ElementType_t<VT2>;    //!< Element type of the right-hand side sparse vector expression.
   //**********************************************************************************************

   //**Return type evaluation**********************************************************************
   //! Compilation switch for the selection of the subscript operator return type.
   /*! The \a returnExpr compile time constant expression is a compilation switch for the
       selection of the \a ReturnType. If either vector operand returns a temporary vector
       or matrix, \a returnExpr will be set to \a false and the subscript operator will
       return it's result by value. Otherwise \a returnExpr will be set to \a true and
       the subscript operator may return it's result as an expression. */
   static constexpr bool returnExpr = ( !IsTemporary_v<RN1> && !IsTemporary_v<RN2> );

   //! Expression return type for the subscript operator.
   using ExprReturnType = decltype( ( std::declval<RN1>() * std::declval<RN2>() ) -
                                    ( std::declval<RN1>() * std::declval<RN2>() ) );
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this SVecSVecCrossExpr instance.
   using This = SVecSVecCrossExpr<VT1,VT2,TF>;

   //! Base type of this SVecSVecCrossExpr instance.
   using BaseType = CrossExpr< DenseVector<This,TF> >;

   using ResultType    = CrossTrait_t<RT1,RT2>;        //!< Result type for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;  //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<ResultType>;    //!< Resulting element type.

   //! Return type for expression template evaluations.
   using ReturnType = const If_t< returnExpr, ExprReturnType, ElementType >;

   //! Data type for composite expression templates.
   using CompositeType = const ResultType;

   //! Composite type of the left-hand side sparse vector expression.
   using LeftOperand = If_t< IsExpression_v<VT1>, const VT1, const VT1& >;

   //! Composite type of the right-hand side sparse vector expression.
   using RightOperand = If_t< IsExpression_v<VT2>, const VT2, const VT2& >;

   //! Composite type of the left-hand side sparse vector expression.
   using LT = const StaticVector<ET1,3UL,TF>;

   //! Composite type of the right-hand side sparse vector expression.
   using RT = const StaticVector<ET2,3UL,TF>;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = false;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = false;
   //**********************************************************************************************

   //**Constructor*********************************************************************************
   /*!\brief Constructor for the SVecSVecCrossExpr class.
   //
   // \param lhs The left-hand side operand of the cross product expression.
   // \param rhs The right-hand side operand of the cross product expression.
   */
   inline SVecSVecCrossExpr( const VT1& lhs, const VT2& rhs ) noexcept
      : lhs_( lhs )  // Left-hand side sparse vector of the cross product expression
      , rhs_( rhs )  // Right-hand side sparse vector of the cross product expression
   {
      BLAZE_INTERNAL_ASSERT( lhs.size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( rhs.size() == 3UL, "Invalid vector size" );
   }
   //**********************************************************************************************

   //**Subscript operator**************************************************************************
   /*!\brief Subscript operator for the direct access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..2]\f$.
   // \return The resulting value.
   */
   inline ReturnType operator[]( size_t index ) const {
      BLAZE_INTERNAL_ASSERT( index < 3UL, "Invalid vector access index" );

      if( index == 0UL )
         return lhs_[1UL] * rhs_[2UL] - lhs_[2UL] * rhs_[1UL];
      else if( index == 1UL )
         return lhs_[2UL] * rhs_[0UL] - lhs_[0UL] * rhs_[2UL];
      else
         return lhs_[0UL] * rhs_[1UL] - lhs_[1UL] * rhs_[0UL];
   }
   //**********************************************************************************************

   //**At function*********************************************************************************
   /*!\brief Checked access to the vector elements.
   //
   // \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
   // \return The resulting value.
   // \exception std::out_of_range Invalid vector access index.
   */
   inline ReturnType at( size_t index ) const {
      if( index >= 3UL ) {
         BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
      }
      return (*this)[index];
   }
   //**********************************************************************************************

   //**Size function*******************************************************************************
   /*!\brief Returns the current size/dimension of the vector.
   //
   // \return The size of the vector.
   */
   constexpr size_t size() const noexcept {
      return 3UL;
   }
   //**********************************************************************************************

   //**Left operand access*************************************************************************
   /*!\brief Returns the left-hand side sparse vector operand.
   //
   // \return The left-hand side sparse vector operand.
   */
   inline LeftOperand leftOperand() const noexcept {
      return lhs_;
   }
   //**********************************************************************************************

   //**Right operand access************************************************************************
   /*!\brief Returns the right-hand side sparse vector operand.
   //
   // \return The right-hand side sparse vector operand.
   */
   inline RightOperand rightOperand() const noexcept {
      return rhs_;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression can alias with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case the expression can alias, \a false otherwise.
   */
   template< typename T >
   inline bool canAlias( const T* alias ) const noexcept {
      return ( lhs_.canAlias( alias ) || rhs_.canAlias( alias ) );
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Returns whether the expression is aliased with the given address \a alias.
   //
   // \param alias The alias to be checked.
   // \return \a true in case an alias effect is detected, \a false otherwise.
   */
   template< typename T >
   inline bool isAliased( const T* alias ) const noexcept {
      return ( lhs_.isAliased( alias ) || rhs_.isAliased( alias ) );
   }
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   LeftOperand  lhs_;  //!< Left-hand side sparse vector of the cross product expression.
   RightOperand rhs_;  //!< Right-hand side sparse vector of the cross product expression.
   //**********************************************************************************************

   //**Assignment to dense vectors*****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector-sparse vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector-sparse
   // vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void assign( DenseVector<VT,TF>& lhs, const SVecSVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (*rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side sparse vector operand

      (*lhs)[0] = x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (*lhs)[1] = x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (*lhs)[2] = x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Assignment to sparse vectors****************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Assignment of a sparse vector-sparse vector cross product to a sparse vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side sparse vector.
   // \param rhs The right-hand side cross product expression to be assigned.
   // \return void
   //
   // This function implements the performance optimized assignment of a sparse vector-sparse
   // vector cross product expression to a sparse vector.
   */
   template< typename VT >  // Type of the target sparse vector
   friend inline void assign( SparseVector<VT,TF>& lhs, const SVecSVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE ( ResultType );
      BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType, TF );
      BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

      BLAZE_INTERNAL_ASSERT( (*lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (*rhs).size() == 3UL, "Invalid vector size" );

      const ResultType tmp( serial( rhs ) );
      assign( *lhs, tmp );
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Addition assignment of a sparse vector-sparse vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression to be added.
   // \return void
   //
   // This function implements the performance optimized addition assignment of a sparse vector-
   // sparse vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void addAssign( DenseVector<VT,TF>& lhs, const SVecSVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (*rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side sparse vector operand

      (*lhs)[0] += x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (*lhs)[1] += x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (*lhs)[2] += x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Addition assignment to sparse vectors*******************************************************
   // No special implementation for the addition assignment to sparse vectors.
   //**********************************************************************************************

   //**Subtraction assignment to dense vectors*****************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Subtraction assignment of a sparse vector-sparse vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression to be subtracted.
   // \return void
   //
   // This function implements the performance optimized subtraction assignment of a sparse vector-
   // sparse vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void subAssign( DenseVector<VT,TF>& lhs, const SVecSVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (*rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side sparse vector operand

      (*lhs)[0] -= x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (*lhs)[1] -= x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (*lhs)[2] -= x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Subtraction assignment to sparse vectors****************************************************
   // No special implementation for the subtraction assignment to sparse vectors.
   //**********************************************************************************************

   //**Multiplication assignment to dense vectors**************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Multiplication assignment of a sparse vector-sparse vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression to be multiplied.
   // \return void
   //
   // This function implements the performance optimized multiplication assignment of a sparse
   // vector-sparse vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void multAssign( DenseVector<VT,TF>& lhs, const SVecSVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (*rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side sparse vector operand

      (*lhs)[0] *= x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (*lhs)[1] *= x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (*lhs)[2] *= x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Multiplication assignment to sparse vectors*************************************************
   // No special implementation for the multiplication assignment to sparse vectors.
   //**********************************************************************************************

   //**Division assignment to dense vectors********************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Division assignment of a sparse vector-sparse vector cross product to a dense vector.
   // \ingroup dense_vector
   //
   // \param lhs The target left-hand side dense vector.
   // \param rhs The right-hand side cross product expression divisor.
   // \return void
   //
   // This function implements the performance optimized division assignment of a sparse vector-
   // sparse vector cross product expression to a dense vector.
   */
   template< typename VT >  // Type of the target dense vector
   friend inline void divAssign( DenseVector<VT,TF>& lhs, const SVecSVecCrossExpr& rhs )
   {
      BLAZE_FUNCTION_TRACE;

      BLAZE_INTERNAL_ASSERT( (*lhs).size() == 3UL, "Invalid vector size" );
      BLAZE_INTERNAL_ASSERT( (*rhs).size() == 3UL, "Invalid vector size" );

      LT x( serial( rhs.lhs_ ) );  // Evaluation of the left-hand side sparse vector operand
      RT y( serial( rhs.rhs_ ) );  // Evaluation of the right-hand side sparse vector operand

      (*lhs)[0] /= x[1UL]*y[2UL] - x[2UL]*y[1UL];
      (*lhs)[1] /= x[2UL]*y[0UL] - x[0UL]*y[2UL];
      (*lhs)[2] /= x[0UL]*y[1UL] - x[1UL]*y[0UL];
   }
   /*! \endcond */
   //**********************************************************************************************

   //**Division assignment to sparse vectors*******************************************************
   // No special implementation for the division assignment to sparse vectors.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( VT1 );
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( VT2 );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT1, TF );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( VT2, TF );
   BLAZE_CONSTRAINT_MUST_NOT_BE_ZERO_TYPE( VT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_ZERO_TYPE( VT2 );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the cross product of a sparse vector and a sparse vector
//        (\f$ \vec{a}=\vec{b} \times \vec{c} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side sparse vector for the cross product.
// \param rhs The right-hand side sparse vector for the cross product.
// \return The cross product of the two vectors.
//
// This function implements a performance optimized treatment of the cross product of a sparse
// vector and a sparse vector.
*/
template< typename VT1  // Type of the left-hand side sparse vector
        , typename VT2  // Type of the right-hand side sparse vector
        , bool TF       // Transpose flag
        , DisableIf_t< IsZero_v<VT1> || IsZero_v<VT2> >* = nullptr >
inline const SVecSVecCrossExpr<VT1,VT2,TF>
   svecsveccross( const SparseVector<VT1,TF>& lhs, const SparseVector<VT2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).size() == (*rhs).size(), "Invalid vector sizes" );

   return SVecSVecCrossExpr<VT1,VT2,TF>( *lhs, *rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend implementation of the cross product of a (zero) sparse vector and a (zero)
//        sparse vector (\f$ \vec{a}=\vec{b} \times \vec{c} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side sparse vector for the cross product.
// \param rhs The right-hand side sparse vector for the cross product.
// \return The resulting zero vector.
//
// This function implements a performance optimized treatment of the cross product of a (zero)
// sparse vector and a (zero) sparse vector. It returns a zero vector.
*/
template< typename VT1  // Type of the left-hand side sparse vector
        , typename VT2  // Type of the right-hand side sparse vector
        , bool TF       // Transpose flag
        , EnableIf_t< IsZero_v<VT1> || IsZero_v<VT2> >* = nullptr >
inline decltype(auto)
   svecsveccross( const SparseVector<VT1,TF>& lhs, const SparseVector<VT2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   MAYBE_UNUSED( lhs, rhs );

   BLAZE_INTERNAL_ASSERT( (*lhs).size() == (*rhs).size(), "Invalid vector sizes" );

   using ReturnType = const CrossTrait_t< ResultType_t<VT1>, ResultType_t<VT2> >;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ReturnType, TF );
   BLAZE_CONSTRAINT_MUST_BE_ZERO_TYPE( ReturnType );

   return ReturnType( 3UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Operator for the cross product of two sparse vectors (\f$ \vec{a}=\vec{b} \times \vec{c} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side sparse vector for the cross product.
// \param rhs The right-hand side sparse vector for the cross product.
// \return The cross product of the two sparse vectors.
// \exception std::invalid_argument Invalid vector size for cross product.
//
// This operator represents the cross product of two sparse vectors:

   \code
   blaze::CompressedVector<double> a( 3UL ), b( 3UL );
   blaze::StaticVector<double,3UL> c;
   // ... Resizing and initialization
   c = a % b;
   \endcode

// The operator returns a dense vector of the higher-order element type of the two involved
// vector element types \a VT1::ElementType and \a VT2::ElementType. Both vector types \a VT1
// and \a VT2 as well as the two element types \a VT1::ElementType and \a VT2::ElementType
// have to be supported by the CrossTrait class template.\n
// In case the current sizes of the two given vectors don't match, a \a std::invalid_argument
// is thrown.
*/
template< typename VT1  // Type of the left-hand side sparse vector
        , typename VT2  // Type of the right-hand side sparse vector
        , bool TF >     // Transpose flag
inline decltype(auto)
   operator%( const SparseVector<VT1,TF>& lhs, const SparseVector<VT2,TF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   if( (*lhs).size() != 3UL || (*rhs).size() != 3UL ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid vector size for cross product" );
   }

   return svecsveccross( *lhs, *rhs );
}
//*************************************************************************************************

} // namespace blaze

#endif
