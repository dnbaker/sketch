//=================================================================================================
/*!
//  \file blaze/math/adaptors/hermitianmatrix/HermitianProxy.h
//  \brief Header file for the HermitianProxy class
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

#ifndef _BLAZE_MATH_ADAPTORS_HERMITIANMATRIX_HERMITIANPROXY_H_
#define _BLAZE_MATH_ADAPTORS_HERMITIANMATRIX_HERMITIANPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/Matrix.h>
#include <blaze/math/constraints/Scalar.h>
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
#include <blaze/math/typetraits/UnderlyingBuiltin.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsComplex.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access proxy for Hermitian matrices.
// \ingroup hermitian_matrix
//
// The HermitianProxy provides controlled access to the elements of a non-const Hermitian matrix.
// It guarantees that a modification of element \f$ a_{ij} \f$ of the accessed matrix is also
// applied to element \f$ a_{ji} \f$. The following example illustrates this by means of a
// \f$ 3 \times 3 \f$ dense Hermitian matrix:

   \code
   // Creating a 3x3 Hermitian dense matrix
   blaze::HermitianMatrix< blaze::DynamicMatrix<int> > A( 3UL );

   A(0,2) = -2;  //        (  0 0 -2 )
   A(1,1) =  3;  // => A = (  0 3  5 )
   A(1,2) =  5;  //        ( -2 5  0 )
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class HermitianProxy
   : public Proxy< HermitianProxy<MT> >
{
 public:
   //**Type definitions****************************************************************************
   using RepresentedType = ElementType_t<MT>;      //!< Type of the represented matrix element.
   using Reference       = Reference_t<MT>;        //!< Reference to the represented element.
   using ConstReference  = ConstReference_t<MT>;   //!< Reference-to-const to the represented element.
   using Pointer         = HermitianProxy*;        //!< Pointer to the represented element.
   using ConstPointer    = const HermitianProxy*;  //!< Pointer-to-const to the represented element.

   //! Value type of the represented complex element.
   using ValueType = UnderlyingBuiltin_t<RepresentedType>;

   //! Value type of the represented complex element.
   using value_type = ValueType;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline HermitianProxy( MT& matrix, size_t row, size_t column );

   HermitianProxy( const HermitianProxy& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~HermitianProxy() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                          inline HermitianProxy& operator= ( const HermitianProxy& hp );
   template< typename T > inline HermitianProxy& operator= ( const T& value );
   template< typename T > inline HermitianProxy& operator+=( const T& value );
   template< typename T > inline HermitianProxy& operator-=( const T& value );
   template< typename T > inline HermitianProxy& operator*=( const T& value );
   template< typename T > inline HermitianProxy& operator/=( const T& value );
   template< typename T > inline HermitianProxy& operator%=( const T& value );
   //@}
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline Pointer      operator->() noexcept;
   inline ConstPointer operator->() const noexcept;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void invert() const;

   inline ConstReference get() const noexcept;
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator ConstReference() const noexcept;
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
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Reference value1_;     //!< Reference to the first accessed matrix element.
   Reference value2_;     //!< Reference to the second accessed matrix element.
   const bool diagonal_;  //!< Flag for the accessed matrix element.
                          /*!< The flag indicates if the accessed element is a diagonal element.
                               It is \a true in case the proxy represents an element on the
                               diagonal. */
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_MATRIX_TYPE              ( MT );
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
   /*!\brief Resetting the represented element to the default initial values.
   // \ingroup hermitian_matrix
   //
   // \param proxy The given access proxy.
   // \return void
   //
   // This function resets the element represented by the access proxy to its default initial
   // value.
   */
   friend inline void reset( const HermitianProxy& proxy )
   {
      using blaze::reset;

      reset( proxy.value1_ );
      if( !proxy.diagonal_ ) {
         reset( proxy.value2_ );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Clearing the represented element.
   //
   // \return void
   //
   // This function clears the element represented by the proxy to its default initial state.
   */
   friend inline void clear( const HermitianProxy& proxy )
   {
      using blaze::clear;

      clear( proxy.value1_ );
      if( !proxy.diagonal_ ) {
         clear( proxy.value2_ );
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
/*!\brief Initialization constructor for a HermitianProxy.
//
// \param matrix Reference to the adapted matrix.
// \param row The row-index of the accessed matrix element.
// \param column The column-index of the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline HermitianProxy<MT>::HermitianProxy( MT& matrix, size_t row, size_t column )
   : value1_  ( matrix(row,column) )  // Reference to the first accessed matrix element
   , value2_  ( matrix(column,row) )  // Reference to the second accessed matrix element
   , diagonal_( row == column )       // Flag for the accessed matrix element
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for HermitianProxy.
//
// \param hp Hermitian proxy to be copied.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
//
// In case the proxy represents a diagonal element and the assigned value does not represent
// a real number, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline HermitianProxy<MT>& HermitianProxy<MT>::operator=( const HermitianProxy& hp )
{
   using ET = ElementType_t<MT>;

   if( IsComplex_v<ET> && diagonal_ && !isReal( hp.value1_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   value1_ = hp.value1_;
   if( !diagonal_ )
      value2_ = conj( value1_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the accessed matrix element.
//
// \param value The new value of the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
//
// In case the proxy represents a diagonal element and the assigned value does not represent
// a real number, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianProxy<MT>& HermitianProxy<MT>::operator=( const T& value )
{
   using ET = ElementType_t<MT>;

   if( IsComplex_v<ET> && diagonal_ && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   value1_ = value;
   if( !diagonal_ )
      value2_ = conj( value1_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the accessed matrix element.
//
// \param value The right-hand side value to be added to the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
//
// In case the proxy represents a diagonal element and the assigned value does not represent
// a real number, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianProxy<MT>& HermitianProxy<MT>::operator+=( const T& value )
{
   using ET = ElementType_t<MT>;

   if( IsComplex_v<ET> && diagonal_ && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   value1_ += value;
   if( !diagonal_ )
      value2_ = conj( value1_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the accessed matrix element.
//
// \param value The right-hand side value to be subtracted from the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
//
// In case the proxy represents a diagonal element and the assigned value does not represent
// a real number, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianProxy<MT>& HermitianProxy<MT>::operator-=( const T& value )
{
   using ET = ElementType_t<MT>;

   if( IsComplex_v<ET> && diagonal_ && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   value1_ -= value;
   if( !diagonal_ )
      value2_ = conj( value1_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the accessed matrix element.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
//
// In case the proxy represents a diagonal element and the assigned value does not represent
// a real number, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianProxy<MT>& HermitianProxy<MT>::operator*=( const T& value )
{
   using ET = ElementType_t<MT>;

   if( IsComplex_v<ET> && diagonal_ && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   value1_ *= value;
   if( !diagonal_ )
      value2_ = conj( value1_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the accessed matrix element.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
//
// In case the proxy represents a diagonal element and the assigned value does not represent
// a real number, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianProxy<MT>& HermitianProxy<MT>::operator/=( const T& value )
{
   using ET = ElementType_t<MT>;

   if( IsComplex_v<ET> && diagonal_ && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   value1_ /= value;
   if( !diagonal_ )
      value2_ = conj( value1_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Modulo assignment to the accessed matrix element.
//
// \param value The right-hand side value for the modulo operation.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
//
// In case the proxy represents a diagonal element and the assigned value does not represent
// a real number, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline HermitianProxy<MT>& HermitianProxy<MT>::operator%=( const T& value )
{
   using ET = ElementType_t<MT>;

   if( IsComplex_v<ET> && diagonal_ && !isReal( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   value1_ %= value;
   if( !diagonal_ )
      value2_ = conj( value1_ );

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the represented matrix element.
//
// \return Pointer to the represented matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename HermitianProxy<MT>::Pointer HermitianProxy<MT>::operator->() noexcept
{
   return this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the represented matrix element.
//
// \return Pointer to the represented matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename HermitianProxy<MT>::ConstPointer HermitianProxy<MT>::operator->() const noexcept
{
   return this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief In-place inversion of the represented element
//
// \return void
*/
template< typename MT >  // Type of the adapted matrix
inline void HermitianProxy<MT>::invert() const
{
   using blaze::invert;

   invert( value1_ );
   if( !diagonal_ )
      value2_ = conj( value1_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returning the value of the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename HermitianProxy<MT>::ConstReference HermitianProxy<MT>::get() const noexcept
{
   return value1_;
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline HermitianProxy<MT>::operator ConstReference() const noexcept
{
   return get();
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
// In case the proxy represents a complex number, this function returns the current value of its
// real part.
*/
template< typename MT >  // Type of the adapted matrix
inline typename HermitianProxy<MT>::ValueType HermitianProxy<MT>::real() const
{
   return value1_.real();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the real part of the represented complex number.
//
// \param value The new value for the real part.
// \return void
//
// In case the proxy represents a complex number, this function sets a new value to its real part.
*/
template< typename MT >  // Type of the adapted matrix
inline void HermitianProxy<MT>::real( ValueType value ) const
{
   value1_.real( value );
   if( !diagonal_ )
      value2_.real( value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the imaginary part of the represented complex number.
//
// \return The current imaginary part of the represented complex number.
//
// In case the proxy represents a complex number, this function returns the current value of its
// imaginary part.
*/
template< typename MT >  // Type of the adapted matrix
inline typename HermitianProxy<MT>::ValueType HermitianProxy<MT>::imag() const
{
   return value1_.imag();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the imaginary part of the represented complex number.
//
// \param value The new value for the imaginary part.
// \return void
// \exception std::invalid_argument Invalid setting for diagonal matrix element.
//
// In case the proxy represents a complex number, this function sets a new value to its imaginary
// part. In case the proxy represents a diagonal element and the given value is not zero, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline void HermitianProxy<MT>::imag( ValueType value ) const
{
   if( diagonal_ && !isZero( value ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal matrix element" );
   }

   value1_.imag( value );
   if( !diagonal_ )
      value2_.imag( -value );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name HermitianProxy global functions */
//@{
template< typename MT >
void invert( const HermitianProxy<MT>& proxy );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place inversion of the represented element.
// \ingroup hermitian_matrix
//
// \param proxy The given proxy instance.
// \return void
*/
template< typename MT >
inline void invert( const HermitianProxy<MT>& proxy )
{
   proxy.invert();
}
//*************************************************************************************************

} // namespace blaze

#endif
