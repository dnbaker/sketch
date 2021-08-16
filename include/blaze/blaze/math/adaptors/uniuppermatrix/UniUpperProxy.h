//=================================================================================================
/*!
//  \file blaze/math/adaptors/uniuppermatrix/UniUpperProxy.h
//  \brief Header file for the UniUpperProxy class
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

#ifndef _BLAZE_MATH_ADAPTORS_UNIUPPERMATRIX_UPPERPROXY_H_
#define _BLAZE_MATH_ADAPTORS_UNIUPPERMATRIX_UPPERPROXY_H_


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
#include <blaze/math/shims/Invert.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/InvalidType.h>
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
/*!\brief Access proxy for upper unitriangular matrices.
// \ingroup uniupper_matrix
//
// The UniUpperProxy provides controlled access to the elements of a non-const upper unitriangular
// matrix. It guarantees that the uniupper matrix invariant is not violated, i.e. that elements
// in the lower part of the matrix remain 0 and the diagonal elements remain 1. The following
// example illustrates this by means of a \f$ 3 \times 3 \f$ dense upper unitriangular matrix:

   \code
   // Creating a 3x3 upper unitriangular dense matrix
   blaze::UniUpperMatrix< blaze::DynamicMatrix<int> > A( 3UL );

   A(0,1) = -2;  //        ( 1 -2  3 )
   A(0,2) =  3;  // => A = ( 0  1  5 )
   A(1,2) =  5;  //        ( 0  0  1 )

   A(1,1) =  4;  // Invalid assignment to diagonal matrix element; results in an exception!
   A(2,0) =  7;  // Invalid assignment to lower matrix element; results in an exception!
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class UniUpperProxy
   : public Proxy< UniUpperProxy<MT> >
{
 private:
   //**Type definitions****************************************************************************
   //! Reference type of the underlying matrix type.
   using ReferenceType = typename MT::Reference;
   //**********************************************************************************************

   //**struct BuiltinType**************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Auxiliary struct to determine the value type of the represented complex element.
   */
   template< typename T >
   struct BuiltinType { using Type = INVALID_TYPE; };
   /*! \endcond */
   //**********************************************************************************************

   //**struct ComplexType**************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Auxiliary struct to determine the value type of the represented complex element.
   */
   template< typename T >
   struct ComplexType { using Type = typename T::value_type; };
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of the represented matrix element.
   using RepresentedType = ElementType_t<MT>;

   //! Value type of the represented complex element.
   using ValueType = typename If_t< IsComplex_v<RepresentedType>
                                  , ComplexType<RepresentedType>
                                  , BuiltinType<RepresentedType> >::Type;

   using value_type = ValueType;  //!< Value type of the represented complex element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline UniUpperProxy( MT& matrix, size_t row, size_t column );

   UniUpperProxy( const UniUpperProxy& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~UniUpperProxy() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                          inline const UniUpperProxy& operator= ( const UniUpperProxy& uup ) const;
   template< typename T > inline const UniUpperProxy& operator= ( const T& value ) const;
   template< typename T > inline const UniUpperProxy& operator+=( const T& value ) const;
   template< typename T > inline const UniUpperProxy& operator-=( const T& value ) const;
   template< typename T > inline const UniUpperProxy& operator*=( const T& value ) const;
   template< typename T > inline const UniUpperProxy& operator/=( const T& value ) const;
   template< typename T > inline const UniUpperProxy& operator%=( const T& value ) const;
   //@}
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline const UniUpperProxy* operator->() const noexcept;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void invert() const;

   inline RepresentedType get()          const noexcept;
   inline bool            isRestricted() const noexcept;
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
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   ReferenceType value_;  //!< Reference to the accessed matrix element.
   size_t row_;           //!< Row index of the accessed matrix element.
   size_t column_;        //!< Column index of the accessed matrix element.
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
   // \ingroup uniupper_matrix
   //
   // \param proxy The given access proxy.
   // \return void
   //
   // This function resets the element represented by the access proxy to its default initial
   // value.
   */
   friend inline void reset( const UniUpperProxy& proxy )
   {
      using blaze::reset;

      if( proxy.row_ < proxy.column_ ) {
         reset( proxy.value_ );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Clearing the represented element.
   // \ingroup uniupper_matrix
   //
   // \param proxy The given access proxy.
   // \return void
   //
   // This function clears the element represented by the access proxy to its default initial
   // state.
   */
   friend inline void clear( const UniUpperProxy& proxy )
   {
      using blaze::clear;

      if( proxy.row_ < proxy.column_ ) {
         clear( proxy.value_ );
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
/*!\brief Initialization constructor for a UniUpperProxy.
//
// \param matrix Reference to the adapted matrix.
// \param row The row-index of the accessed matrix element.
// \param column The column-index of the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline UniUpperProxy<MT>::UniUpperProxy( MT& matrix, size_t row, size_t column )
   : value_ ( matrix( row, column ) )  // Reference to the accessed matrix element
   , row_   ( row    )                 // Row index of the accessed matrix element
   , column_( column )                 // Column index of the accessed matrix element
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for UniUpperProxy.
//
// \param uup Proxy to be copied.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or lower matrix element.
//
// In case the proxy represents an element on the diagonal or in the lower part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline const UniUpperProxy<MT>& UniUpperProxy<MT>::operator=( const UniUpperProxy& uup ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or lower matrix element" );
   }

   value_ = uup.value_;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the accessed matrix element.
//
// \param value The new value of the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or lower matrix element.
//
// In case the proxy represents an element on the diagonal or in the lower part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniUpperProxy<MT>& UniUpperProxy<MT>::operator=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or lower matrix element" );
   }

   value_ = value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the accessed matrix element.
//
// \param value The right-hand side value to be added to the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or lower matrix element.
//
// In case the proxy represents an element on the diagonal or in the lower part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniUpperProxy<MT>& UniUpperProxy<MT>::operator+=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or lower matrix element" );
   }

   value_ += value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the accessed matrix element.
//
// \param value The right-hand side value to be subtracted from the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or lower matrix element.
//
// In case the proxy represents an element on the diagonal or in the lower part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniUpperProxy<MT>& UniUpperProxy<MT>::operator-=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or lower matrix element" );
   }

   value_ -= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the accessed matrix element.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or lower matrix element.
//
// In case the proxy represents an element on the diagonal or in the lower part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniUpperProxy<MT>& UniUpperProxy<MT>::operator*=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or lower matrix element" );
   }

   value_ *= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the accessed matrix element.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or lower matrix element.
//
// In case the proxy represents an element on the diagonal or in the lower part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniUpperProxy<MT>& UniUpperProxy<MT>::operator/=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or lower matrix element" );
   }

   value_ /= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Modulo assignment to the accessed matrix element.
//
// \param value The right-hand side value for the modulo operation.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to diagonal or lower matrix element.
//
// In case the proxy represents an element on the diagonal or in the lower part of the matrix,
// a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UniUpperProxy<MT>& UniUpperProxy<MT>::operator%=( const T& value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal or lower matrix element" );
   }

   value_ %= value;

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  ACCESS OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the accessed matrix element.
//
// \return Pointer to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline const UniUpperProxy<MT>* UniUpperProxy<MT>::operator->() const noexcept
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
// \exception std::invalid_argument Invalid inversion of upper matrix element.
//
// In case the proxy represents a lower element, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniUpperProxy<MT>::invert() const
{
   using blaze::invert;

   if( column_ < row_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid inversion of lower matrix element" );
   }

   if( row_ < column_ )
      invert( value_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returning the value of the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniUpperProxy<MT>::RepresentedType UniUpperProxy<MT>::get() const noexcept
{
   return value_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the proxy represents a restricted matrix element..
//
// \return \a true in case access to the matrix element is restricted, \a false if not.
*/
template< typename MT >  // Type of the adapted matrix
inline bool UniUpperProxy<MT>::isRestricted() const noexcept
{
   return column_ <= row_;
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
inline UniUpperProxy<MT>::operator RepresentedType() const noexcept
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
inline typename UniUpperProxy<MT>::ValueType UniUpperProxy<MT>::real() const
{
   return value_.real();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the real part of the represented complex number.
//
// \param value The new value for the real part.
// \return void
// \exception std::invalid_argument Invalid setting for diagonal or lower matrix element.
//
// In case the proxy represents a complex number, this function sets a new value to its real part.
// In case the represented value is a diagonal element or an element in the lower part of the
// matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniUpperProxy<MT>::real( ValueType value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal or lower matrix element" );
   }

   value_.real( value );
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
inline typename UniUpperProxy<MT>::ValueType UniUpperProxy<MT>::imag() const
{
   return value_.imag();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the imaginary part of the represented complex number.
//
// \param value The new value for the imaginary part.
// \return void
// \exception std::invalid_argument Invalid setting for diagonal or lower matrix element.
//
// In case the proxy represents a complex number, this function sets a new value to its imaginary
// part. In case the represented value is a diagonal element or an element in the lower part of
// the matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniUpperProxy<MT>::imag( ValueType value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal or lower matrix element" );
   }

   value_.imag( value );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name UniUpperProxy global functions */
//@{
template< typename MT >
void invert( const UniUpperProxy<MT>& proxy );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place inversion of the represented element.
// \ingroup uniupper_matrix
//
// \param proxy The given access proxy.
// \return void
*/
template< typename MT >
inline void invert( const UniUpperProxy<MT>& proxy )
{
   proxy.invert();
}
//*************************************************************************************************

} // namespace blaze

#endif
