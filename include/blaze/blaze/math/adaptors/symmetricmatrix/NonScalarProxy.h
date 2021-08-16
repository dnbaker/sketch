//=================================================================================================
/*!
//  \file blaze/math/adaptors/symmetricmatrix/NonScalarProxy.h
//  \brief Header file for the NonScalarProxy class
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

#ifndef _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_NONSCALARPROXY_H_
#define _BLAZE_MATH_ADAPTORS_SYMMETRICMATRIX_NONSCALARPROXY_H_


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
#include <blaze/math/InitializerList.h>
#include <blaze/math/proxy/Proxy.h>
#include <blaze/math/RelaxationFlag.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access proxy for symmetric, square matrices with non-scalar element types.
// \ingroup symmetric_matrix
//
// The NonScalarProxy provides controlled access to the elements of a non-const symmetric matrix
// with non-scalar element type (e.g. vectors or matrices). It guarantees that a modification of
// element \f$ a_{ij} \f$ of the accessed matrix is also applied to element \f$ a_{ji} \f$. The
// following example illustrates this by means of a \f$ 3 \times 3 \f$ sparse symmetric matrix
// with StaticVector elements:

   \code
   using blaze::CompressedMatrix;
   using blaze::StaticVector;
   using blaze::SymmetricMatrix;

   using Vector = StaticVector<int,3UL>;

   // Creating a 3x3 symmetric sparses matrix
   SymmetricMatrix< CompressedMatrix< Vector > > A( 3UL );

   A(0,2) = Vector( -2,  1 );  //        ( (  0 0 ) ( 0  0 ) ( -2  1 ) )
   A(1,1) = Vector(  3,  4 );  // => A = ( (  0 0 ) ( 3  4 ) (  5 -1 ) )
   A(1,2) = Vector(  5, -1 );  //        ( ( -2 1 ) ( 5 -1 ) (  0  0 ) )
   \endcode
*/
template< typename MT >  // Type of the adapted matrix
class NonScalarProxy
   : public Proxy< NonScalarProxy<MT>, ValueType_t< ElementType_t<MT> > >
{
 private:
   //**Enumerations********************************************************************************
   //! Compile time flag indicating whether the given matrix type is a row-major matrix.
   static constexpr bool rmm = IsRowMajorMatrix_v<MT>;
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   /*! \cond BLAZE_INTERNAL */
   using ET = ElementType_t<MT>;  //!< Element type of the adapted matrix.
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using RepresentedType = ValueType_t<ET>;  //!< Type of the represented matrix element.
   using RawReference    = Reference_t<ET>;  //!< Raw reference to the represented element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline NonScalarProxy( MT& sm, size_t i, size_t j );

   NonScalarProxy( const NonScalarProxy& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~NonScalarProxy();
   //@}
   //**********************************************************************************************

   //**Operators***********************************************************************************
   /*!\name Operators */
   //@{
   inline NonScalarProxy& operator= ( const NonScalarProxy& nsp );

   template< typename T >
   inline NonScalarProxy& operator=( initializer_list<T> list );

   template< typename T >
   inline NonScalarProxy& operator=( initializer_list< initializer_list<T> > list );

   template< typename T > inline NonScalarProxy& operator= ( const T& value );
   template< typename T > inline NonScalarProxy& operator+=( const T& value );
   template< typename T > inline NonScalarProxy& operator-=( const T& value );
   template< typename T > inline NonScalarProxy& operator*=( const T& value );
   template< typename T > inline NonScalarProxy& operator/=( const T& value );
   template< typename T > inline NonScalarProxy& operator%=( const T& value );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline RawReference get() const noexcept;
   //@}
   //**********************************************************************************************

   //**Conversion operator*************************************************************************
   /*!\name Conversion operator */
   //@{
   inline operator RawReference() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   MT&    matrix_;  //!< Reference to the adapted matrix.
   size_t i_;       //!< Row-index of the accessed matrix element.
   size_t j_;       //!< Column-index of the accessed matrix element.
   //@}
   //**********************************************************************************************

   //**Forbidden operations************************************************************************
   /*!\name Forbidden operations */
   //@{
   void* operator&() const;  //!< Address operator (private & undefined)
   //@}
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
   BLAZE_CONSTRAINT_MUST_NOT_BE_SCALAR_TYPE          ( RepresentedType );
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
/*!\brief Initialization constructor for a NonScalarProxy.
//
// \param matrix Reference to the adapted matrix.
// \param i The row-index of the accessed matrix element.
// \param j The column-index of the accessed matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline NonScalarProxy<MT>::NonScalarProxy( MT& matrix, size_t i, size_t j )
   : matrix_( matrix )  // Reference to the adapted matrix
   , i_     ( i )       // Row-index of the accessed matrix element
   , j_     ( j )       // Column-index of the accessed matrix element
{
   const typename MT::Iterator pos( matrix_.find( i_, j_ ) );
   const size_t index( rmm ? i_ : j_ );

   if( pos == matrix_.end(index) )
   {
      const ElementType_t<MT> element( ( RepresentedType() ) );
      matrix_.insert( i_, j_, element );
      if( i_ != j_ )
         matrix_.insert( j_, i_, element );
   }

   BLAZE_INTERNAL_ASSERT( matrix_.find(i_,j_)->value() == matrix_.find(j_,i_)->value(), "Unbalance detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for NonScalarProxy.
*/
template< typename MT >  // Type of the adapted matrix
inline NonScalarProxy<MT>::~NonScalarProxy()
{
   const typename MT::Iterator pos( matrix_.find( i_, j_ ) );
   const size_t index( rmm ? i_ : j_ );

   if( pos != matrix_.end( index ) && isDefault( *pos->value() ) )
   {
      matrix_.erase( index, pos );
      if( i_ != j_ )
         matrix_.erase( ( rmm ? j_ : i_ ), matrix_.find( j_, i_ ) );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for NonScalarProxy.
//
// \param nsp Non-scalar access proxy to be copied.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator=( const NonScalarProxy& nsp )
{
   get() = nsp.get();
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the represented matrix element.
//
// \param list The list to be assigned to the matrix element.
// \return Reference to the assigned proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator=( initializer_list<T> list )
{
   get() = list;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the represented matrix element.
//
// \param list The list to be assigned to the matrix element.
// \return Reference to the assigned proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator=( initializer_list< initializer_list<T> > list )
{
   get() = list;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the represented matrix element.
//
// \param value The new value of the matrix element.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator=( const T& value )
{
   get() = value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the represented matrix element.
//
// \param value The right-hand side value to be added to the matrix element.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator+=( const T& value )
{
   get() += value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the represented matrix element.
//
// \param value The right-hand side value to be subtracted from the matrix element.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator-=( const T& value )
{
   get() -= value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the represented matrix element.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator*=( const T& value )
{
   get() *= value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the represented matrix element.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator/=( const T& value )
{
   get() /= value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Modulo assignment to the represented matrix element.
//
// \param value The right-hand side value for the modulo operation.
// \return Reference to the assigned access proxy.
*/
template< typename MT >  // Type of the adapted matrix
template< typename T >   // Type of the right-hand side value
inline NonScalarProxy<MT>& NonScalarProxy<MT>::operator%=( const T& value )
{
   get() %= value;
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returning a reference to the accessed matrix element.
//
// \return Direct/raw reference to the accessed matrix element.
*/
template< typename MT >  // Type of the sparse matrix
inline typename NonScalarProxy<MT>::RawReference NonScalarProxy<MT>::get() const noexcept
{
   const typename MT::Iterator pos( matrix_.find( i_, j_ ) );
   BLAZE_INTERNAL_ASSERT( pos != matrix_.end( rmm ? i_ : j_ ), "Missing matrix element detected" );
   return *pos->value();
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to the represented matrix element.
//
// \return Direct/raw reference to the represented matrix element.
*/
template< typename MT >  // Type of the adapted matrix
inline NonScalarProxy<MT>::operator RawReference() const noexcept
{
   return get();
}
//*************************************************************************************************

} // namespace blaze

#endif
