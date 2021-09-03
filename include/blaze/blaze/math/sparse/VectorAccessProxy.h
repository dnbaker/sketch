//=================================================================================================
/*!
//  \file blaze/math/sparse/VectorAccessProxy.h
//  \brief Header file for the VectorAccessProxy class
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

#ifndef _BLAZE_MATH_SPARSE_VECTORACCESSPROXY_H_
#define _BLAZE_MATH_SPARSE_VECTORACCESSPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/SparseVector.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/proxy/Proxy.h>
#include <blaze/math/RelaxationFlag.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsOne.h>
#include <blaze/math/shims/IsReal.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Access proxy for sparse, N-dimensional vectors.
// \ingroup sparse_vector
//
// The VectorAccessProxy provides safe access to the elements of a non-const sparse vector.\n
// The proxied access to the elements of a sparse vector is necessary since it may be possible
// that several insertion operations happen in the same statement. The following code illustrates
// this with two examples by means of the CompressedVector class:

   \code
   blaze::CompressedVector<double> a( 5 );

   // Standard usage of the subscript operator to initialize a vector element.
   // Only a single sparse vector element is accessed!
   a[0] = 1.0;

   // Initialization of a vector element via another vector element.
   // Two sparse vector accesses in one statement!
   a[1] = a[0];

   // Multiple accesses to elements of the sparse vector in one statement!
   const double result = a[0] + a[2] + a[4];
   \endcode

// The problem (especially with the last statement) is that several insertion operations might
// take place due to the access via the subscript operator. If the subscript operator would
// return a direct reference to one of the accessed elements, this reference might be invalidated
// during the evaluation of a subsequent subscript operator, which results in undefined behavior.
// This class provides the necessary functionality to guarantee a safe access to the sparse vector
// elements while preserving the intuitive use of the subscript operator.
//
*/
template< typename VT >  // Type of the sparse vector
class VectorAccessProxy
   : public Proxy< VectorAccessProxy<VT>, ElementType_t<VT> >
{
 public:
   //**Type definitions****************************************************************************
   using RepresentedType = ElementType_t<VT>;  //!< Type of the represented sparse vector element.
   using RawReference    = RepresentedType&;   //!< Raw reference to the represented element.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline VectorAccessProxy( VT& sv, size_t i );

   VectorAccessProxy( const VectorAccessProxy& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~VectorAccessProxy();
   //@}
   //**********************************************************************************************

   //**Operators***********************************************************************************
   /*!\name Operators */
   //@{
   inline const VectorAccessProxy& operator=( const VectorAccessProxy& vap ) const;

   template< typename T >
   inline const VectorAccessProxy& operator=( initializer_list<T> list ) const;

   template< typename T >
   inline const VectorAccessProxy& operator=( initializer_list< initializer_list<T> > list ) const;

   template< typename T > inline const VectorAccessProxy& operator= ( const T& value ) const;
   template< typename T > inline const VectorAccessProxy& operator+=( const T& value ) const;
   template< typename T > inline const VectorAccessProxy& operator-=( const T& value ) const;
   template< typename T > inline const VectorAccessProxy& operator*=( const T& value ) const;
   template< typename T > inline const VectorAccessProxy& operator/=( const T& value ) const;
   template< typename T > inline const VectorAccessProxy& operator%=( const T& value ) const;
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
   inline operator RawReference() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   VT&    sv_;  //!< Reference to the accessed sparse vector.
   size_t i_;   //!< Index of the accessed sparse vector element.
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
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_VECTOR_TYPE( VT );
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
/*!\brief Initialization constructor for a VectorAccessProxy.
//
// \param sv Reference to the accessed sparse vector.
// \param i The index of the accessed sparse vector element.
*/
template< typename VT >  // Type of the sparse vector
inline VectorAccessProxy<VT>::VectorAccessProxy( VT& sv, size_t i )
   : sv_( sv )  // Reference to the accessed sparse vector
   , i_ ( i  )  // Index of the accessed sparse vector element
{
   const Iterator_t<VT> element( sv_.find( i_ ) );
   if( element == sv_.end() )
      sv_.insert( i_, RepresentedType{} );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for VectorAccessProxy.
*/
template< typename VT >  // Type of the sparse vector
inline VectorAccessProxy<VT>::~VectorAccessProxy()
{
   const Iterator_t<VT> element( sv_.find( i_ ) );
   if( element != sv_.end() && isDefault<strict>( element->value() ) )
      sv_.erase( element );
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Copy assignment operator for VectorAccessProxy.
//
// \param vap Sparse vector access proxy to be copied.
// \return Reference to the assigned access proxy.
*/
template< typename VT >  // Type of the sparse vector
inline const VectorAccessProxy<VT>& VectorAccessProxy<VT>::operator=( const VectorAccessProxy& vap ) const
{
   const Iterator_t<VT> element( vap.sv_.find( vap.i_ ) );
   const auto& source( element != vap.sv_.end() ? element->value() : RepresentedType{} );

   get() = source;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the accessed sparse vector element.
//
// \param list The list to be assigned to the sparse vector element.
// \return Reference to the assigned access proxy.
*/
template< typename VT >  // Type of the sparse vector
template< typename T >   // Type of the right-hand side elements
inline const VectorAccessProxy<VT>&
   VectorAccessProxy<VT>::operator=( initializer_list<T> list ) const
{
   get() = list;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initializer list assignment to the accessed sparse vector element.
//
// \param list The list to be assigned to the sparse vector element.
// \return Reference to the assigned access proxy.
*/
template< typename VT >  // Type of the sparse vector
template< typename T >   // Type of the right-hand side elements
inline const VectorAccessProxy<VT>&
   VectorAccessProxy<VT>::operator=( initializer_list< initializer_list<T> > list ) const
{
   get() = list;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to the accessed sparse vector element.
//
// \param value The new value of the sparse vector element.
// \return Reference to the assigned access proxy.
*/
template< typename VT >  // Type of the sparse vector
template< typename T >   // Type of the right-hand side value
inline const VectorAccessProxy<VT>& VectorAccessProxy<VT>::operator=( const T& value ) const
{
   get() = value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment to the accessed sparse vector element.
//
// \param value The right-hand side value to be added to the sparse vector element.
// \return Reference to the assigned access proxy.
*/
template< typename VT >  // Type of the sparse vector
template< typename T >   // Type of the right-hand side value
inline const VectorAccessProxy<VT>& VectorAccessProxy<VT>::operator+=( const T& value ) const
{
   get() += value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment to the accessed sparse vector element.
//
// \param value The right-hand side value to be subtracted from the sparse vector element.
// \return Reference to the assigned access proxy.
*/
template< typename VT >  // Type of the sparse vector
template< typename T >   // Type of the right-hand side value
inline const VectorAccessProxy<VT>& VectorAccessProxy<VT>::operator-=( const T& value ) const
{
   get() -= value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment to the accessed sparse vector element.
//
// \param value The right-hand side value for the multiplication.
// \return Reference to the assigned access proxy.
*/
template< typename VT >  // Type of the sparse vector
template< typename T >   // Type of the right-hand side value
inline const VectorAccessProxy<VT>& VectorAccessProxy<VT>::operator*=( const T& value ) const
{
   get() *= value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment to the accessed sparse vector element.
//
// \param value The right-hand side value for the division.
// \return Reference to the assigned access proxy.
*/
template< typename VT >  // Type of the sparse vector
template< typename T >   // Type of the right-hand side value
inline const VectorAccessProxy<VT>& VectorAccessProxy<VT>::operator/=( const T& value ) const
{
   get() /= value;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Modulo assignment to the accessed sparse vector element.
//
// \param value The right-hand side value for the modulo operation.
// \return Reference to the assigned access proxy.
//
// If the access proxy represents an element of numeric type, this function performs a modulo
// assignment, if the proxy represents a dense or sparse vector, a cross product is computed,
// and if the proxy represents a dense or sparse matrix, a Schur product is computed.
*/
template< typename VT >  // Type of the sparse vector
template< typename T >   // Type of the right-hand side value
inline const VectorAccessProxy<VT>& VectorAccessProxy<VT>::operator%=( const T& value ) const
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
/*!\brief Returning the value of the accessed sparse vector element.
//
// \return Direct/raw reference to the accessed sparse vector element.
*/
template< typename VT >  // Type of the sparse vector
inline typename VectorAccessProxy<VT>::RawReference VectorAccessProxy<VT>::get() const noexcept
{
   Iterator_t<VT> element( sv_.find( i_ ) );
   if( element == sv_.end() )
      element = sv_.insert( i_, RepresentedType{} );
   BLAZE_INTERNAL_ASSERT( element != sv_.end(), "Missing vector element detected" );
   return element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the proxy represents a restricted sparse vector element..
//
// \return \a true in case access to the sparse vector element is restricted, \a false if not.
*/
template< typename VT >  // Type of the sparse vector
inline bool VectorAccessProxy<VT>::isRestricted() const noexcept
{
   return false;
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion to the accessed sparse vector element.
//
// \return Direct/raw reference to the accessed sparse vector element.
*/
template< typename VT >  // Type of the sparse vector
inline VectorAccessProxy<VT>::operator RawReference() const noexcept
{
   return get();
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name VectorAccessProxy global functions */
//@{
template< typename VT >
void swap( const VectorAccessProxy<VT>& a, const VectorAccessProxy<VT>& b ) noexcept;

template< typename VT, typename T >
void swap( const VectorAccessProxy<VT>& a, T& b ) noexcept;

template< typename T, typename VT >
void swap( T& a, const VectorAccessProxy<VT>& v ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two access proxies.
// \ingroup sparse_vector
//
// \param a The first access proxy to be swapped.
// \param b The second access proxy to be swapped.
// \return void
*/
template< typename VT >
inline void swap( const VectorAccessProxy<VT>& a, const VectorAccessProxy<VT>& b ) noexcept
{
   using std::swap;

   swap( a.get(), b.get() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of an access proxy with another element.
// \ingroup sparse_vector
//
// \param a The access proxy to be swapped.
// \param b The other element to be swapped.
// \return void
*/
template< typename VT, typename T >
inline void swap( const VectorAccessProxy<VT>& a, T& b ) noexcept
{
   using std::swap;

   swap( a.get(), b );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of an access proxy with another element.
// \ingroup sparse_vector
//
// \param a The other element to be swapped.
// \param b The access proxy to be swapped.
// \return void
*/
template< typename T, typename VT >
inline void swap( T& a, const VectorAccessProxy<VT>& b ) noexcept
{
   using std::swap;

   swap( a, b.get() );
}
//*************************************************************************************************

} // namespace blaze

#endif
