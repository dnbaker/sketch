//=================================================================================================
/*!
//  \file blaze/util/AlignedArray.h
//  \brief Header file for the AlignedArray implementation
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

#ifndef _BLAZE_UTIL_ALIGNEDARRAY_H_
#define _BLAZE_UTIL_ALIGNEDARRAY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <array>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/Exception.h>
#include <blaze/util/IntegerSequence.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/typetraits/RemoveCV.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of a static array with a fixed alignment.
// \ingroup util
//
// The AlignedArray class template represents a static array with a guaranteed, fixed alignment.
// The type of the array elements, the number of elements and the alignment of the array can be
// specified via the three template parameters:

   \code
   template< typename Type, size_t N, size_t Alignment >
   class AlignedArray;
   \endcode

// The alignment of the array, which must be a power of two (i.e. 1, 2, 4, 8, ...), can either be
// specified explicitly via the template parameter \a Alignment or it is evaluated automatically
// based on the alignment requirements of the given data type \a Type. In the latter case, if
// \a T is a built-in, vectorizable data type, AlignedArray enforces an alignment of 16 or 32
// bytes, depending on the active SSE/AVX level. In all other cases, no specific alignment is
// enforced.
//
// AlignedArray can be used exactly like any built-in static array. It is possible to access the
// individual element via the subscript operator and the array can be used wherever a pointer is
// expected:

   \code
   void func( const int* );

   blaze::AlignedArray<int,100UL> array;

   array[10] = 2;  // Accessing and assigning the 10th array element
   func( array );  // Passing the aligned array to a function expecting a pointer

   blaze::AlignedArray<int,3UL> array2{ 1, 2, 3 };  // Directly initialized array
   blaze::AlignedArray<int,3UL> array3( 1, 2, 3 );  // Same effect as above
   \endcode
*/
template< typename Type                             // Data type of the elements
        , size_t N                                  // Number of elements
        , size_t Alignment = AlignmentOf_v<Type> >  // Array alignment
class AlignedArray
{
 public:
   //**Type definitions****************************************************************************
   using ElementType    = Type;         //!< Type of the array elements.
   using Pointer        = Type*;        //!< Pointer to a non-constant array element.
   using ConstPointer   = const Type*;  //!< Pointer to a constant array element.
   using Reference      = Type&;        //!< Reference to a non-constant array element.
   using ConstReference = const Type&;  //!< Reference to a constant array element.
   using Iterator       = Type*;        //!< Iterator over non-constant elements.
   using ConstIterator  = const Type*;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   AlignedArray() = default;
   AlignedArray( const AlignedArray& ) = default;
   AlignedArray( AlignedArray&& ) = default;

   template< typename... Ts >
   constexpr AlignedArray( const Ts&... args );

   template< typename T, size_t M >
   constexpr AlignedArray( const T (&array)[M] );

   template< typename T, size_t M >
   constexpr AlignedArray( const std::array<T,M>& array );

   template< typename T, size_t M >
   constexpr AlignedArray( const AlignedArray<T,M>& array );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~AlignedArray() = default;
   //@}
   //**********************************************************************************************

   //**Conversion operators************************************************************************
   /*!\name Conversion operators */
   //@{
   constexpr operator Pointer     () noexcept;
   constexpr operator ConstPointer() const noexcept;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   constexpr Reference      operator[]( size_t index ) noexcept;
   constexpr ConstReference operator[]( size_t index ) const noexcept;
   inline    Reference      at( size_t index );
   inline    ConstReference at( size_t index ) const;
   constexpr Pointer        data() noexcept;
   constexpr ConstPointer   data() const noexcept;
   constexpr Iterator       begin () noexcept;
   constexpr ConstIterator  begin () const noexcept;
   constexpr ConstIterator  cbegin() const noexcept;
   constexpr Iterator       end   () noexcept;
   constexpr ConstIterator  end   () const noexcept;
   constexpr ConstIterator  cend  () const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   AlignedArray& operator=( const AlignedArray& ) = default;
   AlignedArray& operator=( AlignedArray&& ) = default;

   template< typename T, size_t M >
   constexpr AlignedArray& operator=( const T (&array)[M] );

   template< typename T, size_t M >
   constexpr AlignedArray& operator=( const std::array<T,M>& array );

   template< typename T, size_t M >
   constexpr AlignedArray& operator=( const AlignedArray<T,M>& array );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   constexpr size_t size() const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\name Member variables */
   //@{
   alignas( Alignment ) Type v_[ N>0UL ? N : 1UL ];  //!< The aligned array of size max(N,1).
                                                     /*!< This data member must not be accessed
                                                          directly. It is declared public in order
                                                          to provide an array-like distinction
                                                          between default and value initialization. */
   //@}
   /*! \endcond */
   //**********************************************************************************************

 private:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename T, size_t... Is >
   constexpr AlignedArray( const T& array, std::index_sequence<Is...> );
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST   ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE( Type );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DEDUCTION GUIDES
//
//=================================================================================================

//*************************************************************************************************
#if BLAZE_CPP17_MODE

template< typename Type, typename... Ts >
AlignedArray( Type, Ts... ) -> AlignedArray<Type,1+sizeof...(Ts)>;

template< typename Type, size_t N >
AlignedArray( Type (&)[N] ) -> AlignedArray< RemoveCV_t<Type>, N >;

template< typename Type, size_t N >
AlignedArray( std::array<Type,N> ) -> AlignedArray<Type,N>;

#endif
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initialization constructor for AlignedArray.
//
// \param args Pack of initialization values.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
template< typename... Ts >    // Types of the array initializers
constexpr AlignedArray<Type,N,Alignment>::AlignedArray( const Ts&... args )
   : v_{ args... }  // The aligned array
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initialization of all aligned array elements from the given static array.
//
// \param array The given static array for the initialization.
//
// The aligned array is initialized with the values from the given static array. Missing values
// are initialized with default values.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
template< typename T          // Data type of the static array
        , size_t M >          // Number of elements of the static array
constexpr AlignedArray<Type,N,Alignment>::AlignedArray( const T (&array)[M] )
   : AlignedArray( array, make_index_sequence<M>{} )
{
   BLAZE_STATIC_ASSERT( M <= N );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initialization of all aligned array elements from the given std::array.
//
// \param array The given std::array for the initialization.
//
// The aligned array is initialized with the values from the given std::array. Missing values are
// initialized with default values.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
template< typename T          // Data type of the std::array
        , size_t M >          // Number of elements of the std::array
constexpr AlignedArray<Type,N,Alignment>::AlignedArray( const std::array<T,M>& array )
   : AlignedArray( array, make_index_sequence<M>{} )
{
   BLAZE_STATIC_ASSERT( M <= N );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initialization of all aligned array elements from another aligned  array.
//
// \param array The given aligned array for the initialization.
//
// The aligned array is initialized with the values from the another aligned array. Missing
// values are initialized with default values.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
template< typename T          // Data type of the aligned array
        , size_t M >          // Number of elements of the aligned array
constexpr AlignedArray<Type,N,Alignment>::AlignedArray( const AlignedArray<T,M>& array )
   : AlignedArray( array, make_index_sequence<M>{} )
{
   BLAZE_STATIC_ASSERT( M <= N );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Initialization of all aligned array elements from the given array.
//
// \param array The given array for the initialization.
//
// The aligned array is initialized with the values from the given array. Missing values are
// initialized with default values.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
template< typename T          // Data type of the array
        , size_t... Is >      // Sequence of indices for the given array
constexpr AlignedArray<Type,N,Alignment>::AlignedArray( const T& array, std::index_sequence<Is...> )
   : v_{ array[Is]... }
{}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator to a pointer.
//
// \return The raw pointer of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr AlignedArray<Type,N,Alignment>::operator Pointer() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator to a pointer-to-const.
//
// \return The raw pointer of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr AlignedArray<Type,N,Alignment>::operator ConstPointer() const noexcept
{
   return v_;
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the array elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// \note This operator does not perform any kind of index check!
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::Reference
   AlignedArray<Type,N,Alignment>::operator[]( size_t index ) noexcept
{
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the array elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference-to-const to the accessed value.
//
// \note This operator does not perform any kind of index check!
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::ConstReference
   AlignedArray<Type,N,Alignment>::operator[]( size_t index ) const noexcept
{
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the array elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid array access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
inline typename AlignedArray<Type,N,Alignment>::Reference
   AlignedArray<Type,N,Alignment>::at( size_t index )
{
   if( index >= N ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid array access index" );
   }
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the array elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid array access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
inline typename AlignedArray<Type,N,Alignment>::ConstReference
   AlignedArray<Type,N,Alignment>::at( size_t index ) const
{
   if( index >= N ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid array access index" );
   }
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the array elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::Pointer
   AlignedArray<Type,N,Alignment>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the array elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::ConstPointer
   AlignedArray<Type,N,Alignment>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the aligned array.
//
// \return Iterator to the first element of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::Iterator
   AlignedArray<Type,N,Alignment>::begin() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the aligned array.
//
// \return Iterator to the first element of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::ConstIterator
   AlignedArray<Type,N,Alignment>::begin() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the aligned array.
//
// \return Iterator to the first element of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::ConstIterator
   AlignedArray<Type,N,Alignment>::cbegin() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the aligned array.
//
// \return Iterator just past the last element of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::Iterator
   AlignedArray<Type,N,Alignment>::end() noexcept
{
   return v_ + N;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the aligned array.
//
// \return Iterator just past the last element of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::ConstIterator
   AlignedArray<Type,N,Alignment>::end() const noexcept
{
   return v_ + N;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the aligned array.
//
// \return Iterator just past the last element of the aligned array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr typename AlignedArray<Type,N,Alignment>::ConstIterator
   AlignedArray<Type,N,Alignment>::cend() const noexcept
{
   return v_ + N;
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Assignment to all array elements from the given static array.
//
// \param array The given static array for the assignment.
// \return Reference to the assigned array.
//
// The elements of the aligned array are assigned the values from the given static array.
// Missing values are assigned default values.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
template< typename T          // Data type of the static array
        , size_t M >          // Number of elements of the static array
constexpr AlignedArray<Type,N,Alignment>&
   AlignedArray<Type,N,Alignment>::operator=( const T (&array)[M] )
{
   BLAZE_STATIC_ASSERT( M <= N );

   for( size_t i=0UL; i<M; ++i )
      v_[i] = array[i];

   for( size_t i=M; i<N; ++i )
      v_[i] = Type{};

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to all array elements from the given std::array.
//
// \param array The given std::array for the assignment.
// \return Reference to the assigned array.
//
// The elements of the aligned array are assigned the values from the given std::array. Missing
// values are assigned default values.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
template< typename T          // Data type of the std::array
        , size_t M >          // Number of elements of the std::array
constexpr AlignedArray<Type,N,Alignment>&
   AlignedArray<Type,N,Alignment>::operator=( const std::array<T,M>& array )
{
   BLAZE_STATIC_ASSERT( M <= N );

   for( size_t i=0UL; i<M; ++i )
      v_[i] = array[i];

   for( size_t i=M; i<N; ++i )
      v_[i] = Type{};

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment to all array elements from another aligned array.
//
// \param array The given aligned array for the assignment.
// \return Reference to the assigned array.
//
// The elements of the aligned array are assigned the values from another aligned array.
// Missing values are assigned default values.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
template< typename T          // Data type of the std::array
        , size_t M >          // Number of elements of the std::array
constexpr AlignedArray<Type,N,Alignment>&
   AlignedArray<Type,N,Alignment>::operator=( const AlignedArray<T,M>& array )
{
   BLAZE_STATIC_ASSERT( M <= N );

   for( size_t i=0UL; i<M; ++i )
      v_[i] = array[i];

   for( size_t i=M; i<N; ++i )
      v_[i] = Type{};

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current size/dimension of the aligned array.
//
// \return The size of the array.
*/
template< typename Type       // Data type of the elements
        , size_t N            // Number of elements
        , size_t Alignment >  // Array alignment
constexpr size_t AlignedArray<Type,N,Alignment>::size() const noexcept
{
   return N;
}
//*************************************************************************************************

} // namespace blaze

#endif
