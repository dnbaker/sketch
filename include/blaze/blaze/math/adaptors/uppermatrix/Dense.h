//=================================================================================================
/*!
//  \file blaze/math/adaptors/uppermatrix/Dense.h
//  \brief UpperMatrix specialization for dense matrices
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

#ifndef _BLAZE_MATH_ADAPTORS_UPPERMATRIX_DENSE_H_
#define _BLAZE_MATH_ADAPTORS_UPPERMATRIX_DENSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <utility>
#include <blaze/math/adaptors/Forward.h>
#include <blaze/math/adaptors/uppermatrix/BaseTemplate.h>
#include <blaze/math/adaptors/uppermatrix/UpperProxy.h>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/Resizable.h>
#include <blaze/math/constraints/Square.h>
#include <blaze/math/constraints/Static.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/Transformation.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/constraints/View.h>
#include <blaze/math/dense/DenseMatrix.h>
#include <blaze/math/dense/InitializerMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/IsZero.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsScalar.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/views/Submatrix.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR DENSE MATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of UpperMatrix for dense matrices.
// \ingroup upper_matrix
//
// This specialization of UpperMatrix adapts the class template to the requirements of dense
// matrices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
class UpperMatrix<MT,SO,true>
   : public DenseMatrix< UpperMatrix<MT,SO,true>, SO >
{
 private:
   //**Type definitions****************************************************************************
   using OT = OppositeType_t<MT>;   //!< Opposite type of the dense matrix.
   using TT = TransposeType_t<MT>;  //!< Transpose type of the dense matrix.
   using ET = ElementType_t<MT>;    //!< Element type of the dense matrix.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This           = UpperMatrix<MT,SO,true>;   //!< Type of this UpperMatrix instance.
   using BaseType       = DenseMatrix<This,SO>;      //!< Base type of this UpperMatrix instance.
   using ResultType     = This;                      //!< Result type for expression template evaluations.
   using OppositeType   = UpperMatrix<OT,!SO,true>;  //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType  = LowerMatrix<TT,!SO,true>;  //!< Transpose type for expression template evaluations.
   using ElementType    = ET;                        //!< Type of the matrix elements.
   using SIMDType       = SIMDType_t<MT>;            //!< SIMD type of the matrix elements.
   using TagType        = TagType_t<MT>;             //!< Tag type of this UpperMatrix instance.
   using ReturnType     = ReturnType_t<MT>;          //!< Return type for expression template evaluations.
   using CompositeType  = const This&;               //!< Data type for composite expression templates.
   using Reference      = UpperProxy<MT>;            //!< Reference to a non-constant matrix value.
   using ConstReference = ConstReference_t<MT>;      //!< Reference to a constant matrix value.
   using Pointer        = Pointer_t<MT>;             //!< Pointer to a non-constant matrix value.
   using ConstPointer   = ConstPointer_t<MT>;        //!< Pointer to a constant matrix value.
   using ConstIterator  = ConstIterator_t<MT>;       //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain an UpperMatrix with different data/element type.
   */
   template< typename NewType >  // Data type of the other matrix
   struct Rebind {
      //! The type of the other UpperMatrix.
      using Other = UpperMatrix< typename MT::template Rebind<NewType>::Other >;
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a UpperMatrix with different fixed dimensions.
   */
   template< size_t NewM    // Number of rows of the other matrix
           , size_t NewN >  // Number of columns of the other matrix
   struct Resize {
      //! The type of the other UpperMatrix.
      using Other = UpperMatrix< typename MT::template Resize<NewM,NewN>::Other >;
   };
   //**********************************************************************************************

   //**Iterator class definition*******************************************************************
   /*!\brief Iterator over the non-constant elements of the dense upper matrix.
   */
   class Iterator
   {
    public:
      //**Type definitions*************************************************************************
      using IteratorCategory = std::random_access_iterator_tag;  //!< The iterator category.
      using ValueType        = ElementType_t<MT>;                //!< Type of the underlying elements.
      using PointerType      = UpperProxy<MT>;                   //!< Pointer return type.
      using ReferenceType    = UpperProxy<MT>;                   //!< Reference return type.
      using DifferenceType   = ptrdiff_t;                        //!< Difference between two iterators.

      // STL iterator requirements
      using iterator_category = IteratorCategory;  //!< The iterator category.
      using value_type        = ValueType;         //!< Type of the underlying elements.
      using pointer           = PointerType;       //!< Pointer return type.
      using reference         = ReferenceType;     //!< Reference return type.
      using difference_type   = DifferenceType;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Default constructor of the Iterator class.
      */
      inline Iterator() noexcept
         : matrix_( nullptr )  // Reference to the adapted dense matrix
         , row_   ( 0UL )      // The current row index of the iterator
         , column_( 0UL )      // The current column index of the iterator
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the Iterator class.
      //
      // \param matrix The adapted matrix.
      // \param row Initial row index of the iterator.
      // \param column Initial column index of the iterator.
      */
      inline Iterator( MT& matrix, size_t row, size_t column ) noexcept
         : matrix_( &matrix )  // Reference to the adapted dense matrix
         , row_   ( row     )  // The current row-index of the iterator
         , column_( column  )  // The current column-index of the iterator
      {}
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment operator.
      //
      // \param inc The increment of the iterator.
      // \return The incremented iterator.
      */
      inline Iterator& operator+=( size_t inc ) noexcept {
         ( SO )?( row_ += inc ):( column_ += inc );
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment operator.
      //
      // \param dec The decrement of the iterator.
      // \return The decremented iterator.
      */
      inline Iterator& operator-=( size_t dec ) noexcept {
         ( SO )?( row_ -= dec ):( column_ -= dec );
         return *this;
      }
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline Iterator& operator++() noexcept {
         ( SO )?( ++row_ ):( ++column_ );
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const Iterator operator++( int ) noexcept {
         const Iterator tmp( *this );
         ++(*this);
         return tmp;
      }
      //*******************************************************************************************

      //**Prefix decrement operator****************************************************************
      /*!\brief Pre-decrement operator.
      //
      // \return Reference to the decremented iterator.
      */
      inline Iterator& operator--() noexcept {
         ( SO )?( --row_ ):( --column_ );
         return *this;
      }
      //*******************************************************************************************

      //**Postfix decrement operator***************************************************************
      /*!\brief Post-decrement operator.
      //
      // \return The previous position of the iterator.
      */
      inline const Iterator operator--( int ) noexcept {
         const Iterator tmp( *this );
         --(*this);
         return tmp;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline ReferenceType operator*() const {
         return ReferenceType( *matrix_, row_, column_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the element at the current iterator position.
      //
      // \return The resulting value.
      */
      inline PointerType operator->() const {
         return PointerType( *matrix_, row_, column_ );
      }
      //*******************************************************************************************

      //**Load function****************************************************************************
      /*!\brief Load of a SIMD element at the current iterator position.
      //
      // \return The loaded SIMD element.
      //
      // This function performs a load of the current SIMD element at the current iterator
      // position. This function must \b NOT be called explicitly! It is used internally for
      // the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType load() const {
         return (*matrix_).load(row_,column_);
      }
      //*******************************************************************************************

      //**Loada function***************************************************************************
      /*!\brief Aligned load of a SIMD element at the current iterator position.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an aligned load of the current SIMD element at the current
      // iterator position. This function must \b NOT be called explicitly! It is used internally
      // for the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loada() const {
         return (*matrix_).loada(row_,column_);
      }
      //*******************************************************************************************

      //**Loadu function***************************************************************************
      /*!\brief Unaligned load of a SIMD element at the current iterator position.
      //
      // \return The loaded SIMD element.
      //
      // This function performs an unaligned load of the current SIMD element at the current
      // iterator position. This function must \b NOT be called explicitly! It is used internally
      // for the performance optimized evaluation of expression templates. Calling this function
      // explicitly might result in erroneous results and/or in compilation errors.
      */
      inline SIMDType loadu() const {
         return (*matrix_).loadu(row_,column_);
      }
      //*******************************************************************************************

      //**Conversion operator**********************************************************************
      /*!\brief Conversion to an iterator over constant elements.
      //
      // \return An iterator over constant elements.
      */
      inline operator ConstIterator() const {
         if( SO )
            return matrix_->begin( column_ ) + row_;
         else
            return matrix_->begin( row_ ) + column_;
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two Iterator objects.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      friend inline bool operator==( const Iterator& lhs, const Iterator& rhs ) noexcept {
         return ( SO )?( lhs.row_ == rhs.row_ ):( lhs.column_ == rhs.column_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between a Iterator and a ConstIterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      friend inline bool operator==( const Iterator& lhs, const ConstIterator& rhs ) {
         return ( ConstIterator( lhs ) == rhs );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between a ConstIterator and a Iterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      friend inline bool operator==( const ConstIterator& lhs, const Iterator& rhs ) {
         return ( lhs == ConstIterator( rhs ) );
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two Iterator objects.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      friend inline bool operator!=( const Iterator& lhs, const Iterator& rhs ) noexcept {
         return ( SO )?( lhs.row_ != rhs.row_ ):( lhs.column_ != rhs.column_ );
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between a Iterator and ConstIterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      friend inline bool operator!=( const Iterator& lhs, const ConstIterator& rhs ) {
         return ( ConstIterator( lhs ) != rhs );
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between a ConstIterator and Iterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      friend inline bool operator!=( const ConstIterator& lhs, const Iterator& rhs ) {
         return ( lhs != ConstIterator( rhs ) );
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between two Iterator objects.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      friend inline bool operator<( const Iterator& lhs, const Iterator& rhs ) noexcept {
         return ( SO )?( lhs.row_ < rhs.row_ ):( lhs.column_ < rhs.column_ );
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between a Iterator and a ConstIterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      friend inline bool operator<( const Iterator& lhs, const ConstIterator& rhs ) {
         return ( ConstIterator( lhs ) < rhs );
      }
      //*******************************************************************************************

      //**Less-than operator***********************************************************************
      /*!\brief Less-than comparison between a ConstIterator and a Iterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller, \a false if not.
      */
      friend inline bool operator<( const ConstIterator& lhs, const Iterator& rhs ) {
         return ( lhs < ConstIterator( rhs ) );
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between two Iterator objects.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      friend inline bool operator>( const Iterator& lhs, const Iterator& rhs ) noexcept {
         return ( SO )?( lhs.row_ > rhs.row_ ):( lhs.column_ > rhs.column_ );
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between a Iterator and a ConstIterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      friend inline bool operator>( const Iterator& lhs, const ConstIterator& rhs ) {
         return ( ConstIterator( lhs ) > rhs );
      }
      //*******************************************************************************************

      //**Greater-than operator********************************************************************
      /*!\brief Greater-than comparison between a ConstIterator and a Iterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater, \a false if not.
      */
      friend inline bool operator>( const ConstIterator& lhs, const Iterator& rhs ) {
         return ( lhs > ConstIterator( rhs ) );
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between two Iterator objects.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      friend inline bool operator<=( const Iterator& lhs, const Iterator& rhs ) noexcept {
         return ( SO )?( lhs.row_ <= rhs.row_ ):( lhs.column_ <= rhs.column_ );
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between a Iterator and a ConstIterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      friend inline bool operator<=( const Iterator& lhs, const ConstIterator& rhs ) {
         return ( ConstIterator( lhs ) <= rhs );
      }
      //*******************************************************************************************

      //**Less-or-equal-than operator**************************************************************
      /*!\brief Less-than comparison between a ConstIterator and a Iterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is smaller or equal, \a false if not.
      */
      friend inline bool operator<=( const ConstIterator& lhs, const Iterator& rhs ) {
         return ( lhs <= ConstIterator( rhs ) );
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two Iterator objects.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      friend inline bool operator>=( const Iterator& lhs, const Iterator& rhs ) noexcept {
         return ( SO )?( lhs.row_ >= rhs.row_ ):( lhs.column_ >= rhs.column_ );
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between a Iterator and a ConstIterator.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      friend inline bool operator>=( const Iterator& lhs, const ConstIterator& rhs ) {
         return ( ConstIterator( lhs ) >= rhs );
      }
      //*******************************************************************************************

      //**Greater-or-equal-than operator***********************************************************
      /*!\brief Greater-than comparison between two Iterator objects.
      //
      // \param lhs The left-hand side iterator.
      // \param rhs The right-hand side iterator.
      // \return \a true if the left-hand side iterator is greater or equal, \a false if not.
      */
      friend inline bool operator>=( const ConstIterator& lhs, const Iterator& rhs ) {
         return ( lhs >= ConstIterator( rhs ) );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two iterators.
      //
      // \param rhs The right-hand side iterator.
      // \return The number of elements between the two iterators.
      */
      inline DifferenceType operator-( const Iterator& rhs ) const noexcept {
         return ( SO )?( row_ - rhs.row_ ):( column_ - rhs.column_ );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between a Iterator and an integral value.
      //
      // \param it The iterator to be incremented.
      // \param inc The number of elements the iterator is incremented.
      // \return The incremented iterator.
      */
      friend inline const Iterator operator+( const Iterator& it, size_t inc ) noexcept {
         if( SO )
            return Iterator( *it.matrix_, it.row_ + inc, it.column_ );
         else
            return Iterator( *it.matrix_, it.row_, it.column_ + inc );
      }
      //*******************************************************************************************

      //**Addition operator************************************************************************
      /*!\brief Addition between an integral value and a Iterator.
      //
      // \param inc The number of elements the iterator is incremented.
      // \param it The iterator to be incremented.
      // \return The incremented iterator.
      */
      friend inline const Iterator operator+( size_t inc, const Iterator& it ) noexcept {
         if( SO )
            return Iterator( *it.matrix_, it.row_ + inc, it.column_ );
         else
            return Iterator( *it.matrix_, it.row_, it.column_ + inc );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Subtraction between a Iterator and an integral value.
      //
      // \param it The iterator to be decremented.
      // \param dec The number of elements the iterator is decremented.
      // \return The decremented iterator.
      */
      friend inline const Iterator operator-( const Iterator& it, size_t dec ) noexcept {
         if( SO )
            return Iterator( *it.matrix_, it.row_ - dec, it.column_ );
         else
            return Iterator( *it.matrix_, it.row_, it.column_ - dec );
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      MT*    matrix_;  //!< Reference to the adapted dense matrix.
      size_t row_;     //!< The current row-index of the iterator.
      size_t column_;  //!< The current column-index of the iterator.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool simdEnabled = MT::simdEnabled;

   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
                                    inline UpperMatrix();
   template< typename A1 > explicit inline UpperMatrix( const A1& a1 );
                                    inline UpperMatrix( size_t n, const ElementType& init );

   inline UpperMatrix( initializer_list< initializer_list<ElementType> > list );

   template< typename Other >
   inline UpperMatrix( size_t n, const Other* array );

   template< typename Other, size_t N >
   inline UpperMatrix( const Other (&array)[N][N] );

   inline UpperMatrix( ElementType* ptr, size_t n );
   inline UpperMatrix( ElementType* ptr, size_t n, size_t nn );

   inline UpperMatrix( const UpperMatrix& m );
   inline UpperMatrix( UpperMatrix&& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~UpperMatrix() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j );
   inline ConstReference operator()( size_t i, size_t j ) const;
   inline Reference      at( size_t i, size_t j );
   inline ConstReference at( size_t i, size_t j ) const;
   inline ConstPointer   data  () const noexcept;
   inline ConstPointer   data  ( size_t i ) const noexcept;
   inline Iterator       begin ( size_t i );
   inline ConstIterator  begin ( size_t i ) const;
   inline ConstIterator  cbegin( size_t i ) const;
   inline Iterator       end   ( size_t i );
   inline ConstIterator  end   ( size_t i ) const;
   inline ConstIterator  cend  ( size_t i ) const;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline UpperMatrix& operator=( const ElementType& rhs );
   inline UpperMatrix& operator=( initializer_list< initializer_list<ElementType> > list );

   template< typename Other, size_t N >
   inline UpperMatrix& operator=( const Other (&array)[N][N] );

   inline UpperMatrix& operator=( const UpperMatrix& rhs );
   inline UpperMatrix& operator=( UpperMatrix&& rhs ) noexcept;

   template< typename MT2, bool SO2 >
   inline auto operator=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< IsComputation_v<MT2>, UpperMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< IsComputation_v<MT2>, UpperMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator+=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< IsComputation_v<MT2>, UpperMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator+=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< IsComputation_v<MT2>, UpperMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator-=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< IsComputation_v<MT2>, UpperMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator-=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< IsComputation_v<MT2>, UpperMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator%=( const Matrix<MT2,SO2>& rhs ) -> UpperMatrix&;

   template< typename ST >
   inline auto operator*=( ST rhs ) -> EnableIf_t< IsScalar_v<ST>, UpperMatrix& >;

   template< typename ST >
   inline auto operator/=( ST rhs ) -> EnableIf_t< IsScalar_v<ST>, UpperMatrix& >;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t rows() const noexcept;
   inline size_t columns() const noexcept;
   inline size_t spacing() const noexcept;
   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t i ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i ) const;
   inline void   reset();
   inline void   reset( size_t i );
   inline void   clear();
          void   resize ( size_t n, bool preserve=true );
   inline void   extend ( size_t n, bool preserve=true );
   inline void   reserve( size_t elements );
   inline void   shrinkToFit();
   inline void   swap( UpperMatrix& m ) noexcept;

   static constexpr size_t maxNonZeros() noexcept;
   static constexpr size_t maxNonZeros( size_t n ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   template< typename Other > inline UpperMatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

   //**Debugging functions*************************************************************************
   /*!\name Debugging functions */
   //@{
   inline bool isIntact() const noexcept;
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   BLAZE_ALWAYS_INLINE SIMDType load ( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loada( size_t i, size_t j ) const noexcept;
   BLAZE_ALWAYS_INLINE SIMDType loadu( size_t i, size_t j ) const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Construction functions**********************************************************************
   /*!\name Construction functions */
   //@{
   inline const MT construct( size_t n                , TrueType  );
   inline const MT construct( const ElementType& value, FalseType );

   template< typename MT2, bool SO2, typename T >
   inline const MT construct( const Matrix<MT2,SO2>& m, T );
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   MT matrix_;  //!< The adapted dense matrix.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, bool SO2, bool DF2 >
   friend MT2& derestrict( UpperMatrix<MT2,SO2,DF2>& m );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE        ( MT );
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
   BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( OT, !SO );
   BLAZE_CONSTRAINT_MUST_BE_MATRIX_WITH_STORAGE_ORDER( TT, !SO );
   BLAZE_STATIC_ASSERT( ( Size_v<MT,0UL> == Size_v<MT,1UL> ) );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The default constructor for UpperMatrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>::UpperMatrix()
   : matrix_()  // The adapted dense matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Single argument constructor for an upper matrix.
//
// \param a1 The single constructor argument.
// \exception std::invalid_argument Invalid setup of upper matrix.
//
// This constructor constructs the upper matrix based on the given argument and the type of
// the underlying matrix \a MT:
//  - in case the given argument is a matrix, the upper matrix is initialized as a copy of
//    the given matrix.
//  - in case the given argument is not a matrix and the underlying matrix of type \a MT is
//    resizable, the given argument \a a1 specifies the number of rows and columns of the
//    upper matrix.
//  - in case the given argument is not a matrix and the underlying matrix of type \a MT is
//    a matrix with fixed size, the given argument \a a1 specifies the initial value of the
//    upper and diagonal elements.
*/
template< typename MT    // Type of the adapted dense matrix
        , bool SO >      // Storage order of the adapted dense matrix
template< typename A1 >  // Type of the constructor argument
inline UpperMatrix<MT,SO,true>::UpperMatrix( const A1& a1 )
   : matrix_( construct( a1, IsResizable<MT>() ) )  // The adapted dense matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a \f$ n \times n \f$ matrix with initialized upper and diagonal elements.
//
// \param n The number of rows and columns of the matrix.
// \param init The initial value of the upper and diagonal matrix elements.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>::UpperMatrix( size_t n, const ElementType& init )
   : matrix_( n, n, ElementType() )  // The adapted dense matrix
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE_TYPE( MT );

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) = init;
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=i; j<columns(); ++j )
            matrix_(i,j) = init;
   }

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List initialization of all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid setup of upper matrix.
//
// This constructor provides the option to explicitly initialize the elements of the upper
// matrix by means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::UpperMatrix< blaze::StaticMatrix<int,3,3,rowMajor> > A{ { 1, 2, 3 },
                                                                  { 0, 4 },
                                                                  { 0, 0, 6 } };
   \endcode

// The matrix is sized according to the size of the initializer list and all matrix elements are
// initialized with the values from the given list. Missing values are initialized with default
// values. In case the matrix cannot be resized and the dimensions of the initializer list don't
// match or if the given list does not represent an upper matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>::UpperMatrix( initializer_list< initializer_list<ElementType> > list )
   : matrix_( list )  // The adapted dense matrix
{
   if( !isUpper( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of upper matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array initialization of all upper matrix elements.
//
// \param n The number of rows and columns of the matrix.
// \param array Dynamic array for the initialization.
// \exception std::invalid_argument Invalid setup of upper matrix.
//
// This constructor offers the option to directly initialize the elements of the upper matrix
// with a dynamic array:

   \code
   using blaze::rowMajor;

   int* array = new int[16];
   // ... Initialization of the dynamic array
   blaze::UpperMatrix< blaze::DynamicMatrix<int,rowMajor> > v( 4UL, array );
   delete[] array;
   \endcode

// The matrix is sized accoring to the given size of the array and initialized with the values
// from the given array. Note that it is expected that the given \a array has at least \a n by
// \a n elements. Providing an array with less elements results in undefined behavior! Also, in
// case the given array does not represent a upper triangular matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the initialization array
inline UpperMatrix<MT,SO,true>::UpperMatrix( size_t n, const Other* array )
   : matrix_( n, n, array )  // The adapted dense matrix
{
   if( !isUpper( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of upper matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array initialization of all upper matrix elements.
//
// \param array \f$ N \times N \f$ dimensional array for the initialization.
// \exception std::invalid_argument Invalid setup of upper matrix.
//
// This constructor offers the option to directly initialize the elements of the upper matrix
// with a static array:

   \code
   using blaze::rowMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 0, 4 },
                            { 0, 0, 6 } };
   blaze::UpperMatrix< blaze::StaticMatrix<int,3,3,rowMajor> > A( init );
   \endcode

// The matrix is initialized with the values from the given array. Missing values are initialized
// with default values. In case the given array does not represent a lower triangular matrix, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename Other  // Data type of the initialization array
        , size_t N >      // Number of rows and columns of the initialization array
inline UpperMatrix<MT,SO,true>::UpperMatrix( const Other (&array)[N][N] )
   : matrix_( array )  // The adapted dense matrix
{
   if( !isUpper( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of upper matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for an upper custom matrix of size \f$ n \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param n The number of rows and columns of the array of elements.
// \exception std::invalid_argument Invalid setup of upper custom matrix.
//
// This constructor creates an unpadded upper custom matrix of size \f$ n \times n \f$:

   \code
   using blaze::UpperMatrix;
   using blaze::CustomMatrix;
   using blaze::unaligned;
   using blaze::unpadded;

   std::vector<int> memory( 9UL );
   UpperMatrix< CustomMatrix<int,unaligned,unpadded> > A( memory.data(), 3UL );
   \endcode

// The construction fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the values in the given array do not represent an upper triangular matrix.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note This constructor is \b NOT available for padded upper custom matrices!
// \note The matrix does \b NOT take responsibility for the given array of elements!
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>::UpperMatrix( ElementType* ptr, size_t n )
   : matrix_( ptr, n, n )  // The adapted dense matrix
{
   if( !isUpper( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of upper matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for an upper custom matrix of size \f$ n \times n \f$.
//
// \param ptr The array of elements to be used by the matrix.
// \param n The number of rows and columns of the array of elements.
// \param nn The total number of elements between two rows/columns.
// \exception std::invalid_argument Invalid setup of upper custom matrix.
//
// This constructor creates an upper custom matrix of size \f$ n \times n \f$:

   \code
   using blaze::UpperMatrix;
   using blaze::CustomMatrix;
   using blaze::unaligned;
   using blaze::padded;

   std::vector<int> memory( 24UL );
   UpperMatrix< CustomMatrix<int,unaligned,padded> > A( memory.data(), 3UL, 8UL );
   \endcode

// The construction fails if ...
//
//  - ... the passed pointer is \c nullptr;
//  - ... the alignment flag \a AF is set to \a aligned, but the passed pointer is not properly
//    aligned according to the available instruction set (SSE, AVX, ...);
//  - ... the specified spacing \a nn is insufficient for the given data type \a Type and the
//    available instruction set;
//  - ... the values in the given array do not represent an upper triangular matrix.
//
// In all failure cases a \a std::invalid_argument exception is thrown.
//
// \note The matrix does \b NOT take responsibility for the given array of elements!
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>::UpperMatrix( ElementType* ptr, size_t n, size_t nn )
   : matrix_( ptr, n, n, nn )  // The adapted dense matrix
{
   if( !isUpper( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of upper matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The copy constructor for UpperMatrix.
//
// \param m The upper matrix to be copied.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>::UpperMatrix( const UpperMatrix& m )
   : matrix_( m.matrix_ )  // The adapted dense matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The move constructor for UpperMatrix.
//
// \param m The upper matrix to be moved into this instance.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>::UpperMatrix( UpperMatrix&& m ) noexcept
   : matrix_( std::move( m.matrix_ ) )  // The adapted dense matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..N-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// The function call operator provides access to the elements at position (i,j). The attempt to
// assign to an element in the lower part of the matrix (i.e. below the diagonal) will result in
// a \a std::invalid_argument exception.
//
// Note that this function only performs an index check in case BLAZE_USER_ASSERT() is active. In
// contrast, the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::Reference
   UpperMatrix<MT,SO,true>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i<rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<columns(), "Invalid column access index" );

   return Reference( matrix_, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..N-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// The function call operator provides access to the elements at position (i,j). The attempt to
// assign to an element in the lower part of the matrix (i.e. below the diagonal) will result in
// a \a std::invalid_argument exception.
//
// Note that this function only performs an index check in case BLAZE_USER_ASSERT() is active. In
// contrast, the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::ConstReference
   UpperMatrix<MT,SO,true>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i<rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<columns(), "Invalid column access index" );

   return matrix_(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..N-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// The function call operator provides access to the elements at position (i,j). The attempt to
// assign to an element in the lower part of the matrix (i.e. below the diagonal) will result in
// a \a std::invalid_argument exception.
//
// Note that in contrast to the subscript operator this function always performs a check of the
// given access indices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::Reference
   UpperMatrix<MT,SO,true>::at( size_t i, size_t j )
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..N-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
// \exception std::invalid_argument Invalid assignment to lower matrix element.
//
// The function call operator provides access to the elements at position (i,j). The attempt to
// assign to an element in the lower part of the matrix (i.e. below the diagonal) will result in
// a \a std::invalid_argument exception.
//
// Note that in contrast to the subscript operator this function always performs a check of the
// given access indices.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::ConstReference
   UpperMatrix<MT,SO,true>::at( size_t i, size_t j ) const
{
   if( i >= rows() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= columns() ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the upper matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The upper matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::ConstPointer
   UpperMatrix<MT,SO,true>::data() const noexcept
{
   return matrix_.data();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::ConstPointer
   UpperMatrix<MT,SO,true>::data( size_t i ) const noexcept
{
   return matrix_.data(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the upper matrix adapts a \a rowMajor dense matrix the function returns an iterator to the
// first element of row \a i, in case it adapts a \a columnMajor dense matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::Iterator
   UpperMatrix<MT,SO,true>::begin( size_t i )
{
   if( SO )
      return Iterator( matrix_, 0UL, i );
   else
      return Iterator( matrix_, i, 0UL );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the upper matrix adapts a \a rowMajor dense matrix the function returns an iterator to the
// first element of row \a i, in case it adapts a \a columnMajor dense matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::ConstIterator
   UpperMatrix<MT,SO,true>::begin( size_t i ) const
{
   return matrix_.begin(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the upper matrix adapts a \a rowMajor dense matrix the function returns an iterator to the
// first element of row \a i, in case it adapts a \a columnMajor dense matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::ConstIterator
   UpperMatrix<MT,SO,true>::cbegin( size_t i ) const
{
   return matrix_.cbegin(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the upper matrix adapts a \a rowMajor dense matrix the function returns an iterator
// just past the last element of row \a i, in case it adapts a \a columnMajor dense matrix the
// function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::Iterator
   UpperMatrix<MT,SO,true>::end( size_t i )
{
   if( SO )
      return Iterator( matrix_, rows(), i );
   else
      return Iterator( matrix_, i, columns() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the upper matrix adapts a \a rowMajor dense matrix the function returns an iterator
// just past the last element of row \a i, in case it adapts a \a columnMajor dense matrix the
// function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::ConstIterator
   UpperMatrix<MT,SO,true>::end( size_t i ) const
{
   return matrix_.end(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the upper matrix adapts a \a rowMajor dense matrix the function returns an iterator
// just past the last element of row \a i, in case it adapts a \a columnMajor dense matrix the
// function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline typename UpperMatrix<MT,SO,true>::ConstIterator
   UpperMatrix<MT,SO,true>::cend( size_t i ) const
{
   return matrix_.cend(i);
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Homogenous assignment to all upper and diagonal matrix elements.
//
// \param rhs Scalar value to be assigned to the upper and diagonal matrix elements.
// \return Reference to the assigned matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>&
   UpperMatrix<MT,SO,true>::operator=( const ElementType& rhs )
{
   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            matrix_(i,j) = rhs;
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=i; j<columns(); ++j )
            matrix_(i,j) = rhs;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// This assignment operator offers the option to directly assign to all elements of the upper
// matrix by means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::UpperMatrix< blaze::StaticMatrix<int,3UL,3UL,rowMajor> > A;
   A = { { 1, 2, 3 },
         { 0, 4 },
         { 0, 0, 6 } };
   \endcode

// The matrix is resized according to the size of the initializer list and all matrix elements
// are assigned the values from the given list. Missing values are assigned default values. In
// case the matrix cannot be resized and the dimensions of the initializer list don't match or
// if the given list does not represent an upper matrix, a \a std::invalid_argument exception
// is thrown.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>&
   UpperMatrix<MT,SO,true>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   const InitializerMatrix<ElementType> tmp( list, list.size() );

   if( !isUpper( tmp ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   matrix_ = list;

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array assignment to all upper matrix elements.
//
// \param array \f$ N \times N \f$ dimensional array for the assignment.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// This assignment operator offers the option to directly set all elements of the upper matrix:

   \code
   using blaze::rowMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 0, 4 },
                            { 0, 0, 6 } };
   blaze::UpperMatrix< blaze::StaticMatrix<int,3UL,3UL,rowMajor> > A;
   A = init;
   \endcode

// The matrix is assigned the values from the given array. Missing values are initialized with
// default values. In case the given array does not represent a upper triangular matrix, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT     // Type of the adapted dense matrix
        , bool SO >       // Storage order of the adapted dense matrix
template< typename Other  // Data type of the initialization array
        , size_t N >      // Number of rows and columns of the initialization array
inline UpperMatrix<MT,SO,true>&
   UpperMatrix<MT,SO,true>::operator=( const Other (&array)[N][N] )
{
   MT tmp( array );

   if( !isUpper( tmp ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   matrix_ = std::move( tmp );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for UpperMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>&
   UpperMatrix<MT,SO,true>::operator=( const UpperMatrix& rhs )
{
   matrix_ = rhs.matrix_;

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Move assignment operator for UpperMatrix.
//
// \param rhs The matrix to be moved into this instance.
// \return Reference to the assigned matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline UpperMatrix<MT,SO,true>&
   UpperMatrix<MT,SO,true>::operator=( UpperMatrix&& rhs ) noexcept
{
   matrix_ = std::move( rhs.matrix_ );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for general matrices.
//
// \param rhs The general matrix to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix. If the matrix cannot be resized accordingly,
// a \a std::invalid_argument exception is thrown. Also note that the given matrix must be an
// upper matrix. Otherwise, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto UpperMatrix<MT,SO,true>::operator=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< IsComputation_v<MT2>, UpperMatrix& >
{
   if( !IsUpper_v<MT2> && !isUpper( *rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   matrix_ = declupp( *rhs );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for matrix computations.
//
// \param rhs The matrix computation to be copied.
// \return Reference to the assigned matrix.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix. If the matrix cannot be resized accordingly,
// a \a std::invalid_argument exception is thrown. Also note that the given matrix must be an
// upper matrix. Otherwise, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto UpperMatrix<MT,SO,true>::operator=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsComputation_v<MT2>, UpperMatrix& >
{
   if( !IsSquare_v<MT2> && !isSquare( *rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   if( IsUpper_v<MT2> ) {
      matrix_ = *rhs;
   }
   else {
      MT tmp( *rhs );

      if( !isUpper( tmp ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
      }

      matrix_ = std::move( tmp );
   }

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a general matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side general matrix to be added.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument
// exception is thrown. Also note that the result of the addition operation must be an upper
// matrix, i.e. the given matrix must be an upper matrix. In case the result is not an upper
// matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto UpperMatrix<MT,SO,true>::operator+=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< IsComputation_v<MT2>, UpperMatrix& >
{
   if( !IsUpper_v<MT2> && !isUpper( *rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   matrix_ += declupp( *rhs );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix computation (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix computation to be added.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument
// exception is thrown. Also note that the result of the addition operation must be an upper
// matrix, i.e. the given matrix must be an upper matrix. In case the result is not an upper
// matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto UpperMatrix<MT,SO,true>::operator+=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsComputation_v<MT2>, UpperMatrix& >
{
   if( !IsSquare_v<MT2> && !isSquare( *rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   if( IsUpper_v<MT2> ) {
      matrix_ += *rhs;
   }
   else {
      const ResultType_t<MT2> tmp( *rhs );

      if( !isUpper( tmp ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
      }

      matrix_ += declupp( tmp );
   }

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a general matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side general matrix to be subtracted.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument
// exception is thrown. Also note that the result of the subtraction operation must be an
// upper matrix, i.e. the given matrix must be an upper matrix. In case the result is not
// an upper matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto UpperMatrix<MT,SO,true>::operator-=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< IsComputation_v<MT2>, UpperMatrix& >
{
   if( !IsUpper_v<MT2> && !isUpper( *rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   matrix_ -= declupp( *rhs );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix computation (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix computation to be subtracted.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument
// exception is thrown. Also note that the result of the subtraction operation must be an
// upper matrix, i.e. the given matrix must be an upper matrix. In case the result is not
// an upper matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto UpperMatrix<MT,SO,true>::operator-=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsComputation_v<MT2>, UpperMatrix& >
{
   if( !IsSquare_v<MT2> && !isSquare( *rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   if( IsUpper_v<MT2> ) {
      matrix_ -= *rhs;
   }
   else {
      const ResultType_t<MT2> tmp( *rhs );

      if( !isUpper( tmp ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
      }

      matrix_ -= declupp( tmp );
   }

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a matrix (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side matrix for the Schur product.
// \return Reference to the matrix.
// \exception std::invalid_argument Invalid assignment to upper matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto UpperMatrix<MT,SO,true>::operator%=( const Matrix<MT2,SO2>& rhs )
   -> UpperMatrix&
{
   if( !IsSquare_v<MT2> && !isSquare( *rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to upper matrix" );
   }

   matrix_ %= *rhs;

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Multiplication assignment operator for the multiplication between a matrix and
//        a scalar value (\f$ A*=s \f$).
//
// \param rhs The right-hand side scalar value for the multiplication.
// \return Reference to the matrix.
*/
template< typename MT    // Type of the adapted dense matrix
        , bool SO >      // Storage order of the adapted dense matrix
template< typename ST >  // Data type of the right-hand side scalar
inline auto UpperMatrix<MT,SO,true>::operator*=( ST rhs )
   -> EnableIf_t< IsScalar_v<ST>, UpperMatrix& >
{
   matrix_ *= rhs;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Division assignment operator for the division of a matrix by a scalar value
//        (\f$ A/=s \f$).
//
// \param rhs The right-hand side scalar value for the division.
// \return Reference to the matrix.
*/
template< typename MT    // Type of the adapted dense matrix
        , bool SO >      // Storage order of the adapted dense matrix
template< typename ST >  // Data type of the right-hand side scalar
inline auto UpperMatrix<MT,SO,true>::operator/=( ST rhs )
   -> EnableIf_t< IsScalar_v<ST>, UpperMatrix& >
{
   BLAZE_USER_ASSERT( !isZero( rhs ), "Division by zero detected" );

   matrix_ /= rhs;
   return *this;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current number of rows of the matrix.
//
// \return The number of rows of the matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t UpperMatrix<MT,SO,true>::rows() const noexcept
{
   return matrix_.rows();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current number of columns of the matrix.
//
// \return The number of columns of the matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t UpperMatrix<MT,SO,true>::columns() const noexcept
{
   return matrix_.columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the spacing between the beginning of two rows/columns.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e. the total
// number of elements of a row/column. In case the upper matrix adapts a \a rowMajor dense
// matrix the function returns the spacing between two rows, in case it adapts a \a columnMajor
// dense matrix the function returns the spacing between two columns.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t UpperMatrix<MT,SO,true>::spacing() const noexcept
{
   return matrix_.spacing();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the matrix.
//
// \return The capacity of the matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t UpperMatrix<MT,SO,true>::capacity() const noexcept
{
   return matrix_.capacity();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current capacity of the specified row/column.
//
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the upper
// matrix adapts a \a rowMajor dense matrix the function returns the capacity of row \a i, in
// case it adapts a \a columnMajor dense matrix the function returns the capacity of column
// \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t UpperMatrix<MT,SO,true>::capacity( size_t i ) const noexcept
{
   return matrix_.capacity(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the total number of non-zero elements in the matrix
//
// \return The number of non-zero elements in the upper matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t UpperMatrix<MT,SO,true>::nonZeros() const
{
   return matrix_.nonZeros();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the specified row/column.
//
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column.
// In case the upper matrix adapts a \a rowMajor dense matrix the function returns the number
// of non-zero elements in row \a i, in case it adapts a to \a columnMajor dense matrix the
// function returns the number of non-zero elements in column \a i.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline size_t UpperMatrix<MT,SO,true>::nonZeros( size_t i ) const
{
   return matrix_.nonZeros(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void UpperMatrix<MT,SO,true>::reset()
{
   using blaze::clear;

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            clear( matrix_(i,j) );
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=i; j<columns(); ++j )
            clear( matrix_(i,j) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified row/column to the default initial values.
//
// \param i The index of the row/column.
// \return void
// \exception std::invalid_argument Invalid row/column access index.
//
// This function resets the values in the specified row/column to their default value. In case
// the storage order is set to \a rowMajor the function resets the values in row \a i, in case
// the storage order is set to \a columnMajor the function resets the values in column \a i.
// Note that the reset() function has no impact on the capacity of the matrix or row/column.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void UpperMatrix<MT,SO,true>::reset( size_t i )
{
   using blaze::clear;

   if( SO ) {
      for( size_t j=0UL; j<=i; ++j )
         clear( matrix_(j,i) );
   }
   else {
      for( size_t j=i; j<columns(); ++j )
         clear( matrix_(i,j) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the upper matrix.
//
// \return void
//
// This function clears the upper matrix and returns it to its default state. The function has
// the same effect as calling clear() on the adapted matrix of type \a MT: In case of a resizable
// matrix (for instance DynamicMatrix or HybridMatrix) the number of rows and columns will be set
// to 0, whereas in case of a fixed-size matrix (for instance StaticMatrix) only the elements will
// be reset to their default state.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void UpperMatrix<MT,SO,true>::clear()
{
   matrix_.clear();

   BLAZE_INTERNAL_ASSERT( matrix_.rows()    == 0UL, "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( matrix_.columns() == 0UL, "Invalid number of columns" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Changing the size of the upper matrix.
//
// \param n The new number of rows and columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// In case the upper matrix adapts a resizable matrix, this function resizes the matrix using
// the given size to \f$ n \times n \f$. During this operation, new dynamic memory may be allocated
// in case the capacity of the matrix is too small. Note that this function may invalidate all
// existing views (submatrices, rows, columns, ...) on the matrix if it is used to shrink the
// matrix. Additionally, the resize operation potentially changes all matrix elements. In order
// to preserve the old matrix values, the \a preserve flag can be set to \a true. Also note that
// in case the size of the matrix is increased, only the new elements in the lower part of the
// matrix are default initialized.\n
// The following example illustrates the resize operation of a \f$ 3 \times 3 \f$ matrix to a
// \f$ 4 \times 4 \f$ matrix. The new, uninitialized elements are marked with \a x:

                              \f[
                              \left(\begin{array}{*{3}{c}}
                              1 & 2 & 3 \\
                              0 & 4 & 5 \\
                              0 & 0 & 6 \\
                              \end{array}\right)

                              \Longrightarrow

                              \left(\begin{array}{*{4}{c}}
                              1 & 2 & 3 & x \\
                              0 & 4 & 5 & x \\
                              0 & 0 & 6 & x \\
                              0 & 0 & 0 & x \\
                              \end{array}\right)
                              \f]
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
void UpperMatrix<MT,SO,true>::resize( size_t n, bool preserve )
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE_TYPE( MT );

   MAYBE_UNUSED( preserve );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square upper matrix detected" );

   const size_t oldsize( matrix_.rows() );

   matrix_.resize( n, n, true );

   if( n > oldsize ) {
      const size_t increment( n - oldsize );
      submatrix( matrix_, oldsize, 0UL, increment, n-1 ).reset();
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Extending the size of the matrix.
//
// \param n Number of additional rows and columns.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function increases the matrix size by \a n rows and \a n columns. During this operation,
// new dynamic memory may be allocated in case the capacity of the matrix is too small. Therefore
// this function potentially changes all matrix elements. In order to preserve the old matrix
// values, the \a preserve flag can be set to \a true. The new elements are default initialized.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void UpperMatrix<MT,SO,true>::extend( size_t n, bool preserve )
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE_TYPE( MT );

   MAYBE_UNUSED( preserve );

   resize( rows() + n, true );
}
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of the matrix.
//
// \param elements The new minimum capacity of the upper matrix.
// \return void
//
// This function increases the capacity of the upper matrix to at least \a elements elements.
// The current values of the matrix elements are preserved.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void UpperMatrix<MT,SO,true>::reserve( size_t elements )
{
   matrix_.reserve( elements );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Requesting the removal of unused capacity.
//
// \return void
//
// This function minimizes the capacity of the matrix by removing unused capacity. Please note
// that in case a reallocation occurs, all iterators (including end() iterators), all pointers
// and references to elements of this matrix are invalidated.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void UpperMatrix<MT,SO,true>::shrinkToFit()
{
   matrix_.shrinkToFit();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Swapping the contents of two matrices.
//
// \param m The matrix to be swapped.
// \return void
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void UpperMatrix<MT,SO,true>::swap( UpperMatrix& m ) noexcept
{
   using std::swap;

   swap( matrix_, m.matrix_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum number of non-zero values for an upper triangular matrix.
//
// \return The maximum number of non-zero values.
//
// This function returns the maximum possible number of non-zero values for an upper triangular
// matrix with fixed-size adapted matrix of type \a MT. Note that this function can only be
// called in case the adapted dense matrix is a fixed-size matrix (as for instance StaticMatrix).
// The attempt to call this function in case the adapted matrix is resizable matrix will result
// in a compile time error.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
constexpr size_t UpperMatrix<MT,SO,true>::maxNonZeros() noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_STATIC_TYPE( MT );

   return maxNonZeros( Size_v<MT,0UL> );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum number of non-zero values for an upper triangular matrix.
//
// \param n The number of rows and columns of the matrix.
// \return The maximum number of non-zero values.
//
// This function returns the maximum possible number of non-zero values for an upper triangular
// matrix of the given number of rows and columns.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
constexpr size_t UpperMatrix<MT,SO,true>::maxNonZeros( size_t n ) noexcept
{
   return ( ( n + 1UL ) * n ) / 2UL;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  NUMERIC FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the matrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return Reference to the matrix.
//
// This function scales the matrix by applying the given scalar value \a scalar to each element
// of the matrix. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator:

   \code
   blaze::UpperMatrix< blaze::DynamicMatrix<int> > A;
   // ... Resizing and initialization
   A *= 4;        // Scaling of the matrix
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the scalar value
inline UpperMatrix<MT,SO,true>& UpperMatrix<MT,SO,true>::scale( const Other& scalar )
{
   matrix_.scale( scalar );
   return *this;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the upper matrix are intact.
//
// \return \a true in case the upper matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the upper matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline bool UpperMatrix<MT,SO,true>::isIntact() const noexcept
{
   using blaze::isIntact;

   return ( isIntact( matrix_ ) && isUpper( matrix_ ) );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address can alias with the matrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool UpperMatrix<MT,SO,true>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.canAlias( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address is aliased with the matrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename MT       // Type of the adapted dense matrix
        , bool SO >         // Storage order of the adapted dense matrix
template< typename Other >  // Data type of the foreign expression
inline bool UpperMatrix<MT,SO,true>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix is properly aligned in memory.
//
// \return \a true in case the matrix is aligned, \a false if not.
//
// This function returns whether the matrix is guaranteed to be properly aligned in memory, i.e.
// whether the beginning and the end of each row/column of the matrix are guaranteed to conform
// to the alignment restrictions of the element type \a Type.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline bool UpperMatrix<MT,SO,true>::isAligned() const noexcept
{
   return matrix_.isAligned();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix can be used in SMP assignments.
//
// \return \a true in case the matrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the matrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the matrix).
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline bool UpperMatrix<MT,SO,true>::canSMPAssign() const noexcept
{
   return matrix_.canSMPAssign();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs a load of a specific SIMD element of the upper matrix. The row index
// must be smaller than the number of rows and the column index must be smaller than the number
// of columns. Additionally, the column index (in case of a row-major matrix) or the row index
// (in case of a column-major matrix) must be a multiple of the number of values inside the
// SIMD element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
BLAZE_ALWAYS_INLINE typename UpperMatrix<MT,SO,true>::SIMDType
   UpperMatrix<MT,SO,true>::load( size_t i, size_t j ) const noexcept
{
   return matrix_.load( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Aligned load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an aligned load of a specific SIMD element of the upper matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
BLAZE_ALWAYS_INLINE typename UpperMatrix<MT,SO,true>::SIMDType
   UpperMatrix<MT,SO,true>::loada( size_t i, size_t j ) const noexcept
{
   return matrix_.loada( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Unaligned load of a SIMD element of the matrix.
//
// \param i Access index for the row. The index has to be in the range [0..M-1].
// \param j Access index for the column. The index has to be in the range [0..N-1].
// \return The loaded SIMD element.
//
// This function performs an unaligned load of a specific SIMD element of the upper matrix.
// The row index must be smaller than the number of rows and the column index must be smaller
// than the number of columns. Additionally, the column index (in case of a row-major matrix)
// or the row index (in case of a column-major matrix) must be a multiple of the number of
// values inside the SIMD element. This function must \b NOT be called explicitly! It is used
// internally for the performance optimized evaluation of expression templates. Calling this
// function explicitly might result in erroneous results and/or in compilation errors.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
BLAZE_ALWAYS_INLINE typename UpperMatrix<MT,SO,true>::SIMDType
   UpperMatrix<MT,SO,true>::loadu( size_t i, size_t j ) const noexcept
{
   return matrix_.loadu( i, j );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructing a resizable matrix of size \f$ n \times n \f$.
//
// \param n The number of rows and columns of the matrix.
// \return The newly constructed matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline const MT UpperMatrix<MT,SO,true>::construct( size_t n, TrueType )
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE_TYPE( MT );

   return MT( n, n, ElementType() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructing a fixed-size matrix with homogeneously initialized upper and diagonal elements.
//
// \param init The initial value of the upper and diagonal matrix elements.
// \return The newly constructed matrix.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline const MT UpperMatrix<MT,SO,true>::construct( const ElementType& init, FalseType )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_RESIZABLE_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_BE_SQUARE_MATRIX_TYPE( MT );

   MT tmp;

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j )
         for( size_t i=0UL; i<=j; ++i )
            tmp(i,j) = init;
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         for( size_t j=i; j<columns(); ++j )
            tmp(i,j) = init;
   }

   return tmp;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructing a matrix as a copy from another matrix.
//
// \param m Matrix to be copied.
// \exception std::invalid_argument Invalid setup of upper matrix.
// \return The newly constructed matrix.
//
// In case the given matrix is not an upper matrix, a \a std::invalid_argument exception is
// thrown.
*/
template< typename MT   // Type of the adapted dense matrix
        , bool SO >     // Storage order of the adapted dense matrix
template< typename MT2  // Type of the foreign matrix
        , bool SO2      // Storage order of the foreign matrix
        , typename T >  // Type of the third argument
inline const MT UpperMatrix<MT,SO,true>::construct( const Matrix<MT2,SO2>& m, T )
{
   const MT tmp( *m );

   if( !IsUpper_v<MT2> && !isUpper( tmp ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of upper matrix" );
   }

   return tmp;
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
