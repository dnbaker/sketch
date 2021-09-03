//=================================================================================================
/*!
//  \file blaze/math/adaptors/hermitianmatrix/HermitianValue.h
//  \brief Header file for the HermitianValue class
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

#ifndef _BLAZE_MATH_ADAPTORS_HERMITIANMATRIX_HERMITIANVALUE_H_
#define _BLAZE_MATH_ADAPTORS_HERMITIANMATRIX_HERMITIANVALUE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/Scalar.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/Transformation.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/constraints/View.h>
#include <blaze/math/Exception.h>
#include <blaze/math/proxy/Proxy.h>
#include <blaze/math/RelaxationFlag.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/Invert.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsComplex.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Representation of two synchronized values within a sparse Hermitian matrix.
// \ingroup hermitian_matrix
//
// The HermitianValue class represents two synchronized values within a sparse Hermitian matrix.
// It guarantees that a modification of value \f$ a_{ij} \f$ via iterator is also applied to the
// value \f$ a_{ji} \f$. The following example illustrates this by means of a \f$ 3 \times 3 \f$
// sparse Hermitian matrix:

   \code
   using cplx = std::complex<double>;
   using Hermitian = blaze::HermitianMatrix< blaze::CompressedMatrix<cplx> >;

   // Creating a 3x3 Hermitian dense matrix
   //
   // ( ( 0, 0) (0, 0) (-2,1) )
   // ( ( 0, 0) (3, 0) ( 5,2) )
   // ( (-2,-1) (5,-2) ( 0,0) )
   //
   Hermitian A( 3UL );
   A(0,2) = cplx(-2,1);
   A(1,1) = cplx( 3,0);
   A(1,2) = cplx( 5,2);

   // Modification of the values at position (2,0) and (0,2)
   //
   // ( (0,0) (0, 0) (4,-3) )
   // ( (0,0) (3, 0) (5, 2) )
   // ( (4,3) (5,-2) (0, 0) )
   //
   Hermitian::Iterator it = A.begin( 2UL );
   it->value() = cplx(4,3);
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class HermitianValue
   : public Proxy< HermitianValue<MT> >
{
 private:
   //**Type definitions****************************************************************************
   using IteratorType = typename MT::Iterator;  //!< Type of the underlying sparse matrix iterators.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using RepresentedType = ElementType_t<MT>;  //!< Type of the represented matrix element.

   //! Value type of the represented complex element.
   using ValueType = UnderlyingBuiltin_t<RepresentedType>;

   using value_type = ValueType;  //!< Value type of the represented complex element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline HermitianValue( IteratorType pos, MT* matrix, size_t index );

   HermitianValue( const HermitianValue& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~HermitianValue() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                          inline HermitianValue& operator= ( const HermitianValue& hv );
   template< typename T > inline HermitianValue& operator= ( const T& value );
   template< typename T > inline HermitianValue& operator+=( const T& value );
   template< typename T > inline HermitianValue& operator-=( const T& value );
   template< typename T > inline HermitianValue& operator*=( const T& value );
   template< typename T > inline HermitianValue& operator/=( const T& value );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void invert() const;

   inline RepresentedType get() const noexcept;
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator RepresentedType() const noexcept;
   //@}
   //**********************************************************************************************

   //**Complex data access functions***************************************************************
   /*!\name Complex data access functions */
   //@{
   inline ValueType real() const;
   inline void      real( ValueType value ) const;
   inline ValueType imag() const;
   inline void      imag( ValueType value ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void sync() const;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   IteratorType pos_;     //!< Iterator to the current sparse Hermitian matrix element.
   MT*          matrix_;  //!< The sparse matrix containing the iterator.
   size_t       index_;   //!< The row/column index of the iterator.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE       ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE       ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE         ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST                ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE             ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VIEW_TYPE            ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE     ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSFORMATION_TYPE  ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_HERMITIAN_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_LOWER_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_UPPER_MATRIX_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_BE_SCALAR_TYPE              ( RepresentedType );
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Resetting the Hermitian value to the default initial values.
   // \ingroup hermitian_matrix
   //
   // \param value The given Hermitian value.
   // \return void
   //
   // This function resets the Hermitian value to its default initial value.
   */
   friend inline void reset( const HermitianValue& value )
   {
      using blaze::reset;

      reset( value.pos_->value() );

      if( value.pos_->index() != value.index_ )
      {
         const size_t row   ( ( IsRowMajorMatrix_v<MT> )?( value.pos_->index() ):( value.index_ ) );
         const size_t column( ( IsRowMajorMatrix_v<MT> )?( value.index_ ):( value.pos_->index() ) );
         const IteratorType pos2( value.matrix_->find( row, column ) );

         reset( pos2->value() );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Clearing the Hermitian value.
   // \ingroup hermitian_matrix
   //
   // \param value The given Hermitian value.
   // \return void
   //
   // This function clears the Hermitian value to its default initial state.
   */
   friend inline void clear( const HermitianValue& value )
   {
      using blaze::clear;

      clear( value.pos_->value() );

      if( value.pos_->index() != value.index_ )
      {
         const size_t row   ( ( IsRowMajorMatrix_v<MT> )?( value.pos_->index() ):( value.index_ ) );
         const size_t column( ( IsRowMajorMatrix_v<MT> )?( value.index_ ):( value.pos_->index() ) );
         const IteratorType pos2( value.matrix_->find( row, column ) );

         clear( pos2->value() );
      }
   }
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the HermitianValue class.
//
// \param pos The initial position of the iterator.
// \param matrix The sparse matrix containing the iterator.
// \param index The row/column index of the iterator.
*/
template< typename MT >  // Type of the adapted matrix
inline HermitianValue<MT>::HermitianValue( IteratorType pos, MT* matrix, size_t index )
   : pos_   ( pos    )  // Iterator to the current sparse Hermitian matrix element
   , matrix_( matrix )  // The sparse matrix containing the iterator
   , index_ ( index  )  // The row/column index of the iterator
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for HermitianValue.
//
// \param hv The Hermitian value to be copied.
// \return Reference to the assigned Hermitian value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline HermitianValue<MT>& HermitianValue<MT>::operator=( const HermitianValue& hv )
{
   const bool isDiagonal( pos_->index() == index_ );

   if( IsComplex_v<RepresentedType> && isDiagonal && !isReal( hv.pos_->value() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   pos_->value() = hv.pos_->value();
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the Hermitian value.
//
// \param value The new value of the Hermitian value.
// \return Reference to the assigned Hermitian value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianValue<MT>& HermitianValue<MT>::operator=( const T& value )
{
   const bool isDiagonal( pos_->index() == index_ );

   if( IsComplex_v<RepresentedType> && isDiagonal && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   pos_->value() = value;
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the Hermitian value.
//
// \param value The right-hand side value to be added to the Hermitian value.
// \return Reference to the assigned Hermitian value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianValue<MT>& HermitianValue<MT>::operator+=( const T& value )
{
   const bool isDiagonal( pos_->index() == index_ );

   if( IsComplex_v<RepresentedType> && isDiagonal && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   pos_->value() += value;
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the Hermitian value.
//
// \param value The right-hand side value to be subtracted from the Hermitian value.
// \return Reference to the assigned Hermitian value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianValue<MT>& HermitianValue<MT>::operator-=( const T& value )
{
   const bool isDiagonal( pos_->index() == index_ );

   if( IsComplex_v<RepresentedType> && isDiagonal && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   pos_->value() -= value;
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the Hermitian value.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned Hermitian value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianValue<MT>& HermitianValue<MT>::operator*=( const T& value )
{
   const bool isDiagonal( pos_->index() == index_ );

   if( IsComplex_v<RepresentedType> && isDiagonal && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   pos_->value() *= value;
   sync();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the Hermitian value.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned Hermitian value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianValue<MT>& HermitianValue<MT>::operator/=( const T& value )
{
   const bool isDiagonal( pos_->index() == index_ );

   if( IsComplex_v<RepresentedType> && isDiagonal && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   pos_->value() /= value;
   sync();
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief In-place inversion of the Hermitian value
//
// \return void
*/
template< typename MT >  // Type of the adapted matrix
inline void HermitianValue<MT>::invert() const
{
   using blaze::invert;

   invert( pos_->value() );

   if( pos_->index() != index_ )
   {
      const size_t row   ( ( IsRowMajorMatrix_v<MT> )?( pos_->index() ):( index_ ) );
      const size_t column( ( IsRowMajorMatrix_v<MT> )?( index_ ):( pos_->index() ) );
      const IteratorType pos2( matrix_->find( row, column ) );

      pos2->value() = conj( pos_->value() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Access to the represented value.
//
// \return Copy of the represented value.
*/
template< typename MT >  // Type of the adapted matrix
inline typename HermitianValue<MT>::RepresentedType HermitianValue<MT>::get() const noexcept
{
   return pos_->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Synchronization of the current sparse element to the according paired element.
//
// \return void
*/
template< typename MT >  // Type of the adapted matrix
inline void HermitianValue<MT>::sync() const
{
   if( pos_->index() == index_ || isDefault( pos_->value() ) )
      return;

   const size_t row   ( ( IsRowMajorMatrix_v<MT> )?( pos_->index() ):( index_ ) );
   const size_t column( ( IsRowMajorMatrix_v<MT> )?( index_ ):( pos_->index() ) );

   matrix_->set( row, column, conj( pos_->value() ) );
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to the represented value.
//
// \return Copy of the represented value.
*/
template< typename MT >  // Type of the adapted matrix
inline HermitianValue<MT>::operator RepresentedType() const noexcept
{
   return pos_->value();
}
//*************************************************************************************************




//=================================================================================================
//
//  COMPLEX DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the real part of the represented complex number.
//
// \return The current real part of the represented complex number.
//
// In case the value represents a complex number, this function returns the current value
// of its real part.
*/
template< typename MT >  // Type of the adapted matrix
inline typename HermitianValue<MT>::ValueType HermitianValue<MT>::real() const
{
   return pos_->value().real();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the real part of the represented complex number.
//
// \param value The new value for the real part.
// \return void
//
// In case the value represents a complex number, this function sets a new value to its
// real part.
*/
template< typename MT >  // Type of the adapted matrix
inline void HermitianValue<MT>::real( ValueType value ) const
{
   pos_->value().real() = value;
   sync();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the imaginary part of the represented complex number.
//
// \return The current imaginary part of the represented complex number.
//
// In case the value represents a complex number, this function returns the current value of its
// imaginary part.
*/
template< typename MT >  // Type of the adapted matrix
inline typename HermitianValue<MT>::ValueType HermitianValue<MT>::imag() const
{
   return pos_->value.imag();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the imaginary part of the represented complex number.
//
// \param value The new value for the imaginary part.
// \return void
//
// In case the proxy represents a complex number, this function sets a new value to its
// imaginary part.
*/
template< typename MT >  // Type of the adapted matrix
inline void HermitianValue<MT>::imag( ValueType value ) const
{
   pos_->value().imag( value );
   sync();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name HermitianValue global functions */
//@{
template< typename MT >
void invert( const HermitianValue<MT>& value );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place inversion of the Hermitian value.
// \ingroup hermitian_matrix
//
// \param value The given Hermitian value.
// \return void
*/
template< typename MT >
inline void invert( const HermitianValue<MT>& value )
{
   value.invert();
}
//*************************************************************************************************

} // namespace blaze

#endif
