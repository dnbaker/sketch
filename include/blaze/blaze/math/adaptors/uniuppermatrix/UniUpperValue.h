//=================================================================================================
/*!
//  \file blaze/math/adaptors/uniuppermatrix/UniUpperValue.h
//  \brief Header file for the UniUpperValue class
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

#ifndef _BLAZE_MATH_ADAPTORS_UNIUPPERMATRIX_UNIUPPERVALUE_H_
#define _BLAZE_MATH_ADAPTORS_UNIUPPERMATRIX_UNIUPPERVALUE_H_


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
/*!\brief Representation of a value within a sparse upper unitriangular matrix.
// \ingroup uniupper_matrix
//
// The UniUpperValue class represents a single value within a sparse upper unitriangular matrix.
// It guarantees that the uniupper matrix invariant is not violated, i.e. that elements in the
// upper part of the matrix remain 0 and the diagonal elements remain 1. The following example
// illustrates this by means of a \f$ 3 \times 3 \f$ sparse upper unitriangular matrix:

   \code
   using UniUpper = blaze::UniUpperMatrix< blaze::CompressedMatrix<int> >;

   // Creating a 3x3 upper unitriangular sparse matrix
   UniUpper A( 3UL );

   A(0,1) = -2;  //        ( 1 -2  3 )
   A(0,2) =  3;  // => A = ( 0  1  5 )
   A(1,2) =  5;  //        ( 0  0  1 )

   UniUpper::Iterator it = A.begin( 1UL );
   it->value() = 9;  // Invalid assignment to diagonal matrix element; results in an exception!
   ++it;
   it->value() = 4;  // Modification of matrix element (1,2)
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class UniUpperValue
   : public Proxy< UniUpperValue<MT> >
{
 private:
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
   using RepresentedType = ElementType_t<MT>;   //!< Type of the represented matrix element.

   //! Value type of the represented complex element.
   using ValueType = typename If_t< IsComplex_v<RepresentedType>
                                  , ComplexType<RepresentedType>
                                  , BuiltinType<RepresentedType> >::Type;

   using value_type = ValueType;  //!< Value type of the represented complex element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline UniUpperValue( RepresentedType& value, bool diagonal );

   UniUpperValue( const UniUpperValue& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~UniUpperValue() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
                          inline UniUpperValue& operator= ( const UniUpperValue& uuv );
   template< typename T > inline UniUpperValue& operator= ( const T& value );
   template< typename T > inline UniUpperValue& operator+=( const T& value );
   template< typename T > inline UniUpperValue& operator-=( const T& value );
   template< typename T > inline UniUpperValue& operator*=( const T& value );
   template< typename T > inline UniUpperValue& operator/=( const T& value );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void invert() const;

   inline RepresentedType get() const noexcept;
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
   RepresentedType* value_;     //!< The represented value.
   bool             diagonal_;  //!< \a true in case the element is on the diagonal, \a false if not.
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
   /*!\brief Resetting the uniupper value to the default initial values.
   // \ingroup uniupper_matrix
   //
   // \param value The given uniupper value.
   // \return void
   //
   // This function resets the uniupper value to its default initial value.
   */
   friend inline void reset( const UniUpperValue& value )
   {
      using blaze::reset;

      if( !value.diagonal_ ) {
         reset( *value.value_ );
      }
   }
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Clearing the uniupper value.
   // \ingroup uniupper_matrix
   //
   // \param value The given uniupper value.
   // \return void
   //
   // This function clears the uniupper value to its default initial state.
   */
   friend inline void clear( const UniUpperValue& value )
   {
      using blaze::clear;

      if( !value.diagonal_ ) {
         clear( *value.value_ );
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
/*!\brief Constructor for the UniUpperValue class.
//
// \param value Reference to the represented value.
// \param diagonal \a true in case the element is on the diagonal, \a false if not.
*/
template< typename MT >  // Type of the adapted matrix
inline UniUpperValue<MT>::UniUpperValue( RepresentedType& value, bool diagonal )
   : value_   ( &value   )  // The represented value.
   , diagonal_( diagonal )  // true in case the element is on the diagonal, false if not
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for UniUpperValue.
//
// \param uuv The uniupper value to be copied.
// \return Reference to the assigned uniupper value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline UniUpperValue<MT>& UniUpperValue<MT>::operator=( const UniUpperValue& uuv )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ = *uuv.value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the uniupper value.
//
// \param value The new value of the uniupper value.
// \return Reference to the assigned uniupper value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniUpperValue<MT>& UniUpperValue<MT>::operator=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ = value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the uniupper value.
//
// \param value The right-hand side value to be added to the uniupper value.
// \return Reference to the assigned uniupper value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniUpperValue<MT>& UniUpperValue<MT>::operator+=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ += value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the uniupper value.
//
// \param value The right-hand side value to be subtracted from the uniupper value.
// \return Reference to the assigned uniupper value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniUpperValue<MT>& UniUpperValue<MT>::operator-=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ -= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the uniupper value.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned uniupper value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniUpperValue<MT>& UniUpperValue<MT>::operator*=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ *= value;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the uniupper value.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned uniupper value.
// \exception std::invalid_argument Invalid assignment to diagonal matrix element.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline UniUpperValue<MT>& UniUpperValue<MT>::operator/=( const T& value )
{
   if( diagonal_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to diagonal matrix element" );
   }

   *value_ /= value;

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief In-place inversion of the uniupper value
//
// \return void
*/
template< typename MT >  // Type of the adapted matrix
inline void UniUpperValue<MT>::invert() const
{
   using blaze::invert;

   if( !diagonal_ )
      invert( *value_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Access to the represented value.
//
// \return Copy of the represented value.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UniUpperValue<MT>::RepresentedType UniUpperValue<MT>::get() const noexcept
{
   return *value_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the value represents a restricted matrix element..
//
// \return \a true in case access to the matrix element is restricted, \a false if not.
*/
template< typename MT >  // Type of the adapted matrix
inline bool UniUpperValue<MT>::isRestricted() const noexcept
{
   return diagonal_;
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
inline UniUpperValue<MT>::operator RepresentedType() const noexcept
{
   return *value_;
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
inline typename UniUpperValue<MT>::ValueType UniUpperValue<MT>::real() const
{
   return value_->real();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the real part of the represented complex number.
//
// \param value The new value for the real part.
// \return void
// \exception std::invalid_argument Invalid setting for diagonal matrix element.
//
// In case the proxy represents a complex number, this function sets a new value to its real part.
*/
template< typename MT >  // Type of the adapted matrix
inline void UniUpperValue<MT>::real( ValueType value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal matrix element" );
   }

   value_->real( value );
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
inline typename UniUpperValue<MT>::ValueType UniUpperValue<MT>::imag() const
{
   return value_->imag();
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
inline void UniUpperValue<MT>::imag( ValueType value ) const
{
   if( isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setting for diagonal matrix element" );
   }

   value_->imag( value );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name UniUpperValue global functions */
//@{
template< typename MT >
void invert( const UniUpperValue<MT>& value );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place inversion of the uniupper value.
// \ingroup uniupper_matrix
//
// \param value The given uniupper value.
// \return void
*/
template< typename MT >
inline void invert( const UniUpperValue<MT>& value )
{
   value.invert();
}
//*************************************************************************************************

} // namespace blaze

#endif
