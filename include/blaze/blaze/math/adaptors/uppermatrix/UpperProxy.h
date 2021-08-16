//=================================================================================================
/*!
//  \file blaze/math/adaptors/uppermatrix/UpperProxy.h
//  \brief Header file for the UpperProxy class
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

#ifndef _BLAZE_MATH_ADAPTORS_UPPERMATRIX_UPPERPROXY_H_
#define _BLAZE_MATH_ADAPTORS_UPPERMATRIX_UPPERPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/Matrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/Transformation.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/constraints/View.h>
#include <blaze/math/Exception.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/proxy/Proxy.h>
#include <blaze/math/RelaxationFlag.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/typetraits/AddConst.h>
#include <blaze/util/typetraits/AddLValueReference.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access proxy for upper triangular matrices.
// \ingroup upper_matrix
//
// The UpperProxy provides controlled access to the elements of a non-const upper triangular
// matrix. It guarantees that the upper matrix invariant is not violated, i.e. that elements
// in the lower part of the matrix remain default values. The following example illustrates
// this by means of a \f$ 3 \times 3 \f$ dense upper matrix:

   \code
   // Creating a 3x3 upper dense matrix
   blaze::UpperMatrix< blaze::DynamicMatrix<int> > A( 3UL );

   A(0,0) = -2;  //        ( -2 3 0 )
   A(0,1) =  3;  // => A = (  0 0 5 )
   A(1,2) =  5;  //        (  0 0 0 )

   A(2,0) =  7;  // Invalid assignment to lower matrix element; results in an exception!
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class UpperProxy
   : public Proxy< UpperProxy<MT>, ElementType_t<MT> >
{
 private:
   //**Type definitions****************************************************************************
   //! Reference type of the underlying matrix type.
   using ReferenceType = AddConst_t< typename MT::Reference >;
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using RepresentedType = ElementType_t<MT>;                    //!< Type of the represented matrix element.
   using RawReference    = AddLValueReference_t<ReferenceType>;  //!< Reference-to-non-const to the represented element.
   using ConstReference  = const RepresentedType&;               //!< Reference-to-const to the represented element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline UpperProxy( MT& matrix, size_t row, size_t column );

   UpperProxy( const UpperProxy& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~UpperProxy() = default;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline const UpperProxy& operator=( const UpperProxy& up ) const;

   template< typename T >
   inline const UpperProxy& operator=( initializer_list<T> list ) const;

   template< typename T >
   inline const UpperProxy& operator=( initializer_list< initializer_list<T> > list ) const;

   template< typename T > inline const UpperProxy& operator= ( const T& value ) const;
   template< typename T > inline const UpperProxy& operator+=( const T& value ) const;
   template< typename T > inline const UpperProxy& operator-=( const T& value ) const;
   template< typename T > inline const UpperProxy& operator*=( const T& value ) const;
   template< typename T > inline const UpperProxy& operator/=( const T& value ) const;
   template< typename T > inline const UpperProxy& operator%=( const T& value ) const;
   //@}
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline const UpperProxy* operator->() const noexcept;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline RawReference get()          const noexcept;
   inline bool         isRestricted() const noexcept;
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator ConstReference() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   ReferenceType value_;    //!< Reference to the accessed matrix element.
   const bool restricted_;  //!< Access flag for the accessed matrix element.
                            /*!< The flag indicates if access to the matrix element is
                                 restricted. It is \a true in case the proxy represents
                                 an element in the lower part of the matrix. */
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
/*!\brief Initialization constructor for a UpperProxy.
//
// \param matrix Reference to the adapted matrix.
// \param row The row-index of the accessed matrix element.
// \param column The column-index of the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline UpperProxy<MT>::UpperProxy( MT& matrix, size_t row, size_t column )
   : value_     ( matrix( row, column ) )  // Reference to the accessed matrix element
   , restricted_( column < row )           // Access flag for the accessed matrix element
{}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for UpperProxy.
//
// \param up Upper proxy to be copied.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
inline const UpperProxy<MT>& UpperProxy<MT>::operator=( const UpperProxy& up ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
   }

   value_ = up.value_;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the accessed matrix element.
//
// \param list The list to be assigned to the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UpperProxy<MT>& UpperProxy<MT>::operator=( initializer_list<T> list ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
   }

   value_ = list;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the accessed matrix element.
//
// \param list The list to be assigned to the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UpperProxy<MT>& UpperProxy<MT>::operator=( initializer_list< initializer_list<T> > list ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
   }

   value_ = list;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the accessed matrix element.
//
// \param value The new value of the matrix element.
// \return Reference to the assigned proxy.
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UpperProxy<MT>& UpperProxy<MT>::operator=( const T& value ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
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
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UpperProxy<MT>& UpperProxy<MT>::operator+=( const T& value ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
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
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UpperProxy<MT>& UpperProxy<MT>::operator-=( const T& value ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
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
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UpperProxy<MT>& UpperProxy<MT>::operator*=( const T& value ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
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
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UpperProxy<MT>& UpperProxy<MT>::operator/=( const T& value ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
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
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// In case the proxy represents an element in the lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline const UpperProxy<MT>& UpperProxy<MT>::operator%=( const T& value ) const
{
   if( restricted_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to lower matrix element" );
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
inline const UpperProxy<MT>* UpperProxy<MT>::operator->() const noexcept
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
/*!\brief Returning the value of the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline typename UpperProxy<MT>::RawReference UpperProxy<MT>::get() const noexcept
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
inline bool UpperProxy<MT>::isRestricted() const noexcept
{
   return restricted_;
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
// \return Reference-to-const to the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline UpperProxy<MT>::operator ConstReference() const noexcept
{
   return static_cast<ConstReference>( value_ );
}
//*************************************************************************************************

} // namespace blaze

#endif
