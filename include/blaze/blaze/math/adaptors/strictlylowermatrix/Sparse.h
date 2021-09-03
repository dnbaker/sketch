//=================================================================================================
/*!
//  \file blaze/math/adaptors/strictlylowermatrix/Sparse.h
//  \brief StrictlyLowerMatrix specialization for sparse matrices
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

#ifndef _BLAZE_MATH_ADAPTORS_STRICTLYLOWERMATRIX_SPARSE_H_
#define _BLAZE_MATH_ADAPTORS_STRICTLYLOWERMATRIX_SPARSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <vector>
#include <blaze/math/adaptors/Forward.h>
#include <blaze/math/adaptors/strictlylowermatrix/BaseTemplate.h>
#include <blaze/math/adaptors/strictlylowermatrix/StrictlyLowerProxy.h>
#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/Hermitian.h>
#include <blaze/math/constraints/Lower.h>
#include <blaze/math/constraints/Resizable.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/Static.h>
#include <blaze/math/constraints/StorageOrder.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/Transformation.h>
#include <blaze/math/constraints/Upper.h>
#include <blaze/math/constraints/View.h>
#include <blaze/math/dense/InitializerMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/sparse/SparseElement.h>
#include <blaze/math/sparse/SparseMatrix.h>
#include <blaze/math/typetraits/IsComputation.h>
#include <blaze/math/typetraits/IsSquare.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsUniTriangular.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/util/algorithms/Max.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR SPARSE MATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of StrictlyLowerMatrix for sparse matrices.
// \ingroup strictly_lower_matrix
//
// This specialization of StrictlyLowerMatrix adapts the class template to the requirements of sparse
// matrices.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
class StrictlyLowerMatrix<MT,SO,false>
   : public SparseMatrix< StrictlyLowerMatrix<MT,SO,false>, SO >
{
 private:
   //**Type definitions****************************************************************************
   using OT = OppositeType_t<MT>;   //!< Opposite type of the sparse matrix.
   using TT = TransposeType_t<MT>;  //!< Transpose type of the sparse matrix.
   using ET = ElementType_t<MT>;    //!< Element type of the sparse matrix.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   using This           = StrictlyLowerMatrix<MT,SO,false>;   //!< Type of this StrictlyLowerMatrix instance.
   using BaseType       = SparseMatrix<This,SO>;              //!< Base type of this StrictlyLowerMatrix instance.
   using ResultType     = This;                               //!< Result type for expression template evaluations.
   using OppositeType   = StrictlyLowerMatrix<OT,!SO,false>;  //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType  = StrictlyUpperMatrix<TT,!SO,false>;  //!< Transpose type for expression template evaluations.
   using ElementType    = ET;                                 //!< Type of the matrix elements.
   using TagType        = TagType_t<MT>;                      //!< Tag type of this StrictlyLowerMatrix instance.
   using ReturnType     = ReturnType_t<MT>;                   //!< Return type for expression template evaluations.
   using CompositeType  = const This&;                        //!< Data type for composite expression templates.
   using Reference      = StrictlyLowerProxy<MT>;             //!< Reference to a non-constant matrix value.
   using ConstReference = ConstReference_t<MT>;               //!< Reference to a constant matrix value.
   using Iterator       = Iterator_t<MT>;                     //!< Iterator over non-constant elements.
   using ConstIterator  = ConstIterator_t<MT>;                //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a StrictlyLowerMatrix with different data/element type.
   */
   template< typename NewType >  // Data type of the other matrix
   struct Rebind {
      //! The type of the other StrictlyLowerMatrix.
      using Other = StrictlyLowerMatrix< typename MT::template Rebind<NewType>::Other >;
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a StrictlyLowerMatrix with different fixed dimensions.
   */
   template< size_t NewM    // Number of rows of the other matrix
           , size_t NewN >  // Number of columns of the other matrix
   struct Resize {
      //! The type of the other StrictlyLowerMatrix.
      using Other = StrictlyLowerMatrix< typename MT::template Resize<NewM,NewN>::Other >;
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = false;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
            inline StrictlyLowerMatrix();
   explicit inline StrictlyLowerMatrix( size_t n );
            inline StrictlyLowerMatrix( size_t n, size_t nonzeros );
            inline StrictlyLowerMatrix( size_t n, const std::vector<size_t>& nonzeros );
            inline StrictlyLowerMatrix( initializer_list< initializer_list<ElementType> > list );

   inline StrictlyLowerMatrix( const StrictlyLowerMatrix& m );
   inline StrictlyLowerMatrix( StrictlyLowerMatrix&& m ) noexcept;

   template< typename MT2, bool SO2 >
   inline StrictlyLowerMatrix( const Matrix<MT2,SO2>& m );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~StrictlyLowerMatrix() = default;
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j );
   inline ConstReference operator()( size_t i, size_t j ) const;
   inline Reference      at( size_t i, size_t j );
   inline ConstReference at( size_t i, size_t j ) const;
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
   inline StrictlyLowerMatrix& operator=( initializer_list< initializer_list<ElementType> > list );

   inline StrictlyLowerMatrix& operator=( const StrictlyLowerMatrix& rhs );
   inline StrictlyLowerMatrix& operator=( StrictlyLowerMatrix&& rhs ) noexcept;

   template< typename MT2, bool SO2 >
   inline auto operator=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator+=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator+=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator-=( const Matrix<MT2,SO2>& rhs )
      -> DisableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator-=( const Matrix<MT2,SO2>& rhs )
      -> EnableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >;

   template< typename MT2, bool SO2 >
   inline auto operator%=( const Matrix<MT2,SO2>& rhs ) -> StrictlyLowerMatrix&;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t rows() const noexcept;
   inline size_t columns() const noexcept;
   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t i ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i ) const;
   inline void   reset();
   inline void   reset( size_t i );
   inline void   clear();
   inline void   resize ( size_t n, bool preserve=true );
   inline void   reserve( size_t nonzeros );
   inline void   reserve( size_t i, size_t nonzeros );
   inline void   trim();
   inline void   trim( size_t i );
   inline void   shrinkToFit();
   inline void   swap( StrictlyLowerMatrix& m ) noexcept;

   static constexpr size_t maxNonZeros() noexcept;
   static constexpr size_t maxNonZeros( size_t n ) noexcept;
   //@}
   //**********************************************************************************************

   //**Insertion functions*************************************************************************
   /*!\name Insertion functions */
   //@{
   inline Iterator set     ( size_t i, size_t j, const ElementType& value );
   inline Iterator insert  ( size_t i, size_t j, const ElementType& value );
   inline void     append  ( size_t i, size_t j, const ElementType& value, bool check=false );
   inline void     finalize( size_t i );
   //@}
   //**********************************************************************************************

   //**Erase functions*****************************************************************************
   /*!\name Erase functions */
   //@{
   inline void     erase( size_t i, size_t j );
   inline Iterator erase( size_t i, Iterator pos );
   inline Iterator erase( size_t i, Iterator first, Iterator last );

   template< typename Pred >
   inline void erase( Pred predicate );

   template< typename Pred >
   inline void erase( size_t i, Iterator first, Iterator last, Pred predicate );
   //@}
   //**********************************************************************************************

   //**Lookup functions****************************************************************************
   /*!\name Lookup functions */
   //@{
   inline Iterator      find      ( size_t i, size_t j );
   inline ConstIterator find      ( size_t i, size_t j ) const;
   inline Iterator      lowerBound( size_t i, size_t j );
   inline ConstIterator lowerBound( size_t i, size_t j ) const;
   inline Iterator      upperBound( size_t i, size_t j );
   inline ConstIterator upperBound( size_t i, size_t j ) const;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   template< typename Other > inline StrictlyLowerMatrix& scale( const Other& scalar );
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

   inline bool canSMPAssign() const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline void resetUpper();
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   MT matrix_;  //!< The adapted sparse matrix.
   //@}
   //**********************************************************************************************

   //**Friend declarations*************************************************************************
   template< typename MT2, bool SO2, bool DF2 >
   friend MT2& derestrict( StrictlyLowerMatrix<MT2,SO2,DF2>& m );
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
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
/*!\brief The default constructor for StrictlyLowerMatrix.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>::StrictlyLowerMatrix()
   : matrix_()  // The adapted sparse matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ n \times n \f$.
//
// \param n The number of rows and columns of the matrix.
//
// The matrix is initialized as empty \f$ n \times n \f$ matrix without free capacity.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>::StrictlyLowerMatrix( size_t n )
   : matrix_( n, n )  // The adapted sparse matrix
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ n \times n \f$.
//
// \param n The number of rows and columns of the matrix.
// \param nonzeros The number of expected non-zero elements.
//
// The matrix is initialized as identity matrix and will have at least the capacity for
// \a nonzeros non-zero elements.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>::StrictlyLowerMatrix( size_t n, size_t nonzeros )
   : matrix_( n, n, max( nonzeros, n ) )  // The adapted sparse matrix
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ n \times n \f$.
//
// \param n The number of rows and columns of the matrix.
// \param nonzeros The expected number of non-zero elements in each row/column.
// \exception std::invalid_argument Invalid capacity specification.
//
// The matrix is initialized as identity matrix and will have the specified capacity in each
// row/column. Note that since the matrix is initialized as \f$ n \times n \f$ identity matrix
// the given vector must have at least \a n elements, all of which must not be 0. If the number
// of non-zero elements of any row/column is specified as 0, a \a std::invalid_argument exception
// is thrown.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>::StrictlyLowerMatrix( size_t n, const std::vector<size_t>& nonzeros )
   : matrix_( n, n, nonzeros )  // The adapted sparse matrix
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List initialization of all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid setup of strictly lower matrix.
//
// This constructor provides the option to explicitly initialize the elements of the strictly
// lower matrix by means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::StrictlyLowerMatrix< blaze::CompressedMatrix<int,rowMajor> > A{ { },
                                                                          { 2 },
                                                                          { 4, 5 } };
   \endcode

// The matrix is sized according to the size of the initializer list and all matrix elements are
// initialized with the values from the given list. Missing values are initialized with default
// values. In case the matrix cannot be resized and the dimensions of the initializer list don't
// match or if the given list does not represent a strictly lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>::StrictlyLowerMatrix( initializer_list< initializer_list<ElementType> > list )
   : matrix_( list )  // The adapted sparse matrix
{
   if( !isStrictlyLower( matrix_ ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of strictly lower matrix" );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The copy constructor for StrictlyLowerMatrix.
//
// \param m The strictly lower matrix to be copied.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>::StrictlyLowerMatrix( const StrictlyLowerMatrix& m )
   : matrix_( m.matrix_ )  // The adapted sparse matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The move constructor for StrictlyLowerMatrix.
//
// \param m The strictly lower matrix to be moved into this instance.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>::StrictlyLowerMatrix( StrictlyLowerMatrix&& m ) noexcept
   : matrix_( std::move( m.matrix_ ) )  // The adapted sparse matrix
{
   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Conversion constructor from different matrices.
//
// \param m Matrix to be copied.
// \exception std::invalid_argument Invalid setup of strictly lower matrix.
//
// This constructor initializes the strictly lower matrix as a copy of the given matrix. In
// case the given matrix is not an strictly lower matrix, a \a std::invalid_argument exception
// is thrown.
*/
template< typename MT   // Type of the adapted sparse matrix
        , bool SO >     // Storage order of the adapted sparse matrix
template< typename MT2  // Type of the foreign matrix
        , bool SO2 >    // Storage order of the foreign matrix
inline StrictlyLowerMatrix<MT,SO,false>::StrictlyLowerMatrix( const Matrix<MT2,SO2>& m )
   : matrix_( *m )  // The adapted sparse matrix
{
   if( IsUniTriangular_v<MT2> ||
       ( !IsStrictlyLower_v<MT2> && !isStrictlyLower( matrix_ ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid setup of strictly lower matrix" );
   }

   if( !IsStrictlyLower_v<MT2> )
      resetUpper();

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
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
// \exception std::invalid_argument Invalid assignment to upper matrix element.
//
// The function call operator provides access to the elements at position (i,j). The attempt
// to assign to an element on the diagonal or in the upper part of the matrix (i.e. above the
// diagonal) will result in a \a std::invalid_argument exception.
//
// Note that this function only performs an index check in case BLAZE_USER_ASSERT() is active. In
// contrast, the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Reference
   StrictlyLowerMatrix<MT,SO,false>::operator()( size_t i, size_t j )
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
// \exception std::invalid_argument Invalid assignment to upper matrix element.
//
// The function call operator provides access to the elements at position (i,j). The attempt
// to assign to an element on the diagonal or in the upper part of the matrix (i.e. above the
// diagonal) will result in a \a std::invalid_argument exception.
//
// Note that this function only performs an index check in case BLAZE_USER_ASSERT() is active. In
// contrast, the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstReference
   StrictlyLowerMatrix<MT,SO,false>::operator()( size_t i, size_t j ) const
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
// \exception std::invalid_argument Invalid assignment to upper matrix element.
//
// The function call operator provides access to the elements at position (i,j). The attempt
// to assign to an element on the diagonal or in the upper part of the matrix (i.e. above the
// diagonal) will result in a \a std::invalid_argument exception.
//
// Note that in contrast to the subscript operator this function always performs a check of the
// given access indices.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Reference
   StrictlyLowerMatrix<MT,SO,false>::at( size_t i, size_t j )
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
// \exception std::invalid_argument Invalid assignment to upper matrix element.
//
// The function call operator provides access to the elements at position (i,j). The attempt
// to assign to an element on the diagonal or in the upper part of the matrix (i.e. above the
// diagonal) will result in a \a std::invalid_argument exception.
//
// Note that in contrast to the subscript operator this function always performs a check of the
// given access indices.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstReference
   StrictlyLowerMatrix<MT,SO,false>::at( size_t i, size_t j ) const
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
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the strictly lower matrix adapts a \a rowMajor sparse matrix the function returns an iterator
// to the first element of row \a i, in case it adapts a \a columnMajor sparse matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::begin( size_t i )
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
// the strictly lower matrix adapts a \a rowMajor sparse matrix the function returns an iterator
// to the first element of row \a i, in case it adapts a \a columnMajor sparse matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstIterator
   StrictlyLowerMatrix<MT,SO,false>::begin( size_t i ) const
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
// the strictly lower matrix adapts a \a rowMajor sparse matrix the function returns an iterator
// to the first element of row \a i, in case it adapts a \a columnMajor sparse matrix the function
// returns an iterator to the first element of column \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstIterator
   StrictlyLowerMatrix<MT,SO,false>::cbegin( size_t i ) const
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
// In case the strictly lower matrix adapts a \a rowMajor sparse matrix the function returns an
// iterator just past the last element of row \a i, in case it adapts a \a columnMajor sparse
// matrix the function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::end( size_t i )
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
// In case the strictly lower matrix adapts a \a rowMajor sparse matrix the function returns an
// iterator just past the last element of row \a i, in case it adapts a \a columnMajor sparse
// matrix the function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstIterator
   StrictlyLowerMatrix<MT,SO,false>::end( size_t i ) const
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
// In case the strictly lower matrix adapts a \a rowMajor sparse matrix the function returns an
// iterator just past the last element of row \a i, in case it adapts a \a columnMajor sparse
// matrix the function returns an iterator just past the last element of column \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstIterator
   StrictlyLowerMatrix<MT,SO,false>::cend( size_t i ) const
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
/*!\brief List assignment to all matrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to strictly lower matrix.
//
// This assignment operator offers the option to directly assign to all elements of the strictly
// lower matrix by means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::StrictlyLowerMatrix< blaze::CompressedMatrix<int,rowMajor> > A;
   A = { { },
         { 2 },
         { 4, 5 } };
   \endcode

// The matrix is resized according to the size of the initializer list and all matrix elements
// are assigned the values from the given list. Missing values are assigned default values. In
// case the matrix cannot be resized and the dimensions of the initializer list don't match or
// if the given list does not represent a strictly lower matrix, a \a std::invalid_argument
// exception is thrown.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>&
   StrictlyLowerMatrix<MT,SO,false>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   const InitializerMatrix<ElementType> tmp( list, list.size() );

   if( !isStrictlyLower( tmp ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
   }

   matrix_ = list;

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for StrictlyLowerMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>&
   StrictlyLowerMatrix<MT,SO,false>::operator=( const StrictlyLowerMatrix& rhs )
{
   matrix_ = rhs.matrix_;

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Move assignment operator for StrictlyLowerMatrix.
//
// \param rhs The matrix to be moved into this instance.
// \return Reference to the assigned matrix.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline StrictlyLowerMatrix<MT,SO,false>&
   StrictlyLowerMatrix<MT,SO,false>::operator=( StrictlyLowerMatrix&& rhs ) noexcept
{
   matrix_ = std::move( rhs.matrix_ );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
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
// \exception std::invalid_argument Invalid assignment to strictly lower matrix.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix. If the matrix cannot be resized accordingly,
// a \a std::invalid_argument exception is thrown. Also note that the given matrix must be a
// strictly lower matrix. Otherwise, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted sparse matrix
        , bool SO >     // Storage order of the adapted sparse matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto StrictlyLowerMatrix<MT,SO,false>::operator=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >
{
   if( IsUniTriangular_v<MT2> ||
       ( !IsStrictlyLower_v<MT2> && !isStrictlyLower( *rhs ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
   }

   matrix_ = decllow( *rhs );

   if( !IsStrictlyLower_v<MT2> )
      resetUpper();

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
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
// \exception std::invalid_argument Invalid assignment to strictly lower matrix.
//
// If possible and necessary, the matrix is resized according to the given \f$ N \times N \f$
// matrix and initialized as a copy of this matrix. If the matrix cannot be resized accordingly,
// a \a std::invalid_argument exception is thrown. Also note that the given matrix must be a
// strictly lower matrix. Otherwise, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted sparse matrix
        , bool SO >     // Storage order of the adapted sparse matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto StrictlyLowerMatrix<MT,SO,false>::operator=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >
{
   if( IsUniTriangular_v<MT2> || ( !IsSquare_v<MT2> && !isSquare( *rhs ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
   }

   if( IsStrictlyLower_v<MT2> ) {
      matrix_ = *rhs;
   }
   else {
      MT tmp( *rhs );

      if( !isStrictlyLower( tmp ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
      }

      matrix_ = std::move( tmp );
   }

   if( !IsStrictlyLower_v<MT2> )
      resetUpper();

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
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
// \exception std::invalid_argument Invalid assignment to strictly lower matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument
// exception is thrown. Also note that the result of the addition operation must be a strictly
// lower matrix, i.e. the given matrix must be a strictly lower matrix. In case the result is
// not a strictly lower matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted sparse matrix
        , bool SO >     // Storage order of the adapted sparse matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto StrictlyLowerMatrix<MT,SO,false>::operator+=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >
{
   if( IsUniTriangular_v<MT2> ||
       ( !IsStrictlyLower_v<MT2> && !isStrictlyLower( *rhs ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
   }

   matrix_ += decllow( *rhs );

   if( !IsStrictlyLower_v<MT2> )
      resetUpper();

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
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
// \exception std::invalid_argument Invalid assignment to strictly lower matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument
// exception is thrown. Also note that the result of the addition operation must be a strictly
// lower matrix, i.e. the given matrix must be a strictly lower matrix. In case the result is
// not a strictly lower matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted sparse matrix
        , bool SO >     // Storage order of the adapted sparse matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto StrictlyLowerMatrix<MT,SO,false>::operator+=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >
{
   if( IsUniTriangular_v<MT2> || ( !IsSquare_v<MT2> && !isSquare( *rhs ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
   }

   if( IsStrictlyLower_v<MT2> ) {
      matrix_ += *rhs;
   }
   else {
      const ResultType_t<MT2> tmp( *rhs );

      if( !isStrictlyLower( tmp ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
      }

      matrix_ += decllow( tmp );
   }

   if( !IsStrictlyLower_v<MT2> )
      resetUpper();

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
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
// \exception std::invalid_argument Invalid assignment to strictly lower matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument
// exception is thrown. Also note that the result of the subtraction operation must be a strictly
// lower matrix, i.e. the given matrix must be a strictly lower matrix. In case the the result is
// not a strictly lower matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted sparse matrix
        , bool SO >     // Storage order of the adapted sparse matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto StrictlyLowerMatrix<MT,SO,false>::operator-=( const Matrix<MT2,SO2>& rhs )
   -> DisableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >
{
   if( IsUniTriangular_v<MT2> ||
       ( !IsStrictlyLower_v<MT2> && !isStrictlyLower( *rhs ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
   }

   matrix_ -= decllow( *rhs );

   if( !IsStrictlyLower_v<MT2> )
      resetUpper();

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
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
// \exception std::invalid_argument Invalid assignment to strictly lower matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument
// exception is thrown. Also note that the result of the subtraction operation must be a strictly
// lower matrix, i.e. the given matrix must be a strictly lower matrix. In case the result is not
// a strictly lower matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted sparse matrix
        , bool SO >     // Storage order of the adapted sparse matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto StrictlyLowerMatrix<MT,SO,false>::operator-=( const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsComputation_v<MT2>, StrictlyLowerMatrix& >
{
   if( IsUniTriangular_v<MT2> || ( !IsSquare_v<MT2> && !isSquare( *rhs ) ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
   }

   if( IsStrictlyLower_v<MT2> ) {
      matrix_ -= *rhs;
   }
   else {
      const ResultType_t<MT2> tmp( *rhs );

      if( !isStrictlyLower( tmp ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
      }

      matrix_ -= decllow( tmp );
   }

   if( !IsStrictlyLower_v<MT2> )
      resetUpper();

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
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
// \exception std::invalid_argument Invalid assignment to strictly lower matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also note that the result of the Schur product operation must be a strictly lower
// matrix. In case it is not, a \a std::invalid_argument exception is thrown.
*/
template< typename MT   // Type of the adapted sparse matrix
        , bool SO >     // Storage order of the adapted sparse matrix
template< typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto StrictlyLowerMatrix<MT,SO,false>::operator%=( const Matrix<MT2,SO2>& rhs )
   -> StrictlyLowerMatrix&
{
   if( !IsSquare_v<MT2> && !isSquare( *rhs ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to strictly lower matrix" );
   }

   matrix_ %= *rhs;

   if( !IsStrictlyLower_v<MT2> )
      resetUpper();

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );

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
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline size_t StrictlyLowerMatrix<MT,SO,false>::rows() const noexcept
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
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline size_t StrictlyLowerMatrix<MT,SO,false>::columns() const noexcept
{
   return matrix_.columns();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the matrix.
//
// \return The capacity of the matrix.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline size_t StrictlyLowerMatrix<MT,SO,false>::capacity() const noexcept
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
// This function returns the current capacity of the specified row/column. In case the strictly
// lower matrix adapts a \a rowMajor sparse matrix the function returns the capacity of row \a i,
// in case it adapts a \a columnMajor sparse matrix the function returns the capacity of column
// \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline size_t StrictlyLowerMatrix<MT,SO,false>::capacity( size_t i ) const noexcept
{
   return matrix_.capacity(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the total number of non-zero elements in the matrix
//
// \return The number of non-zero elements in the strictly lower matrix.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline size_t StrictlyLowerMatrix<MT,SO,false>::nonZeros() const
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
// In case the strictly lower matrix adapts a \a rowMajor sparse matrix the function returns
// the number of non-zero elements in row \a i, in case it adapts a to \a columnMajor sparse
// matrix the function returns the number of non-zero elements in column \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline size_t StrictlyLowerMatrix<MT,SO,false>::nonZeros( size_t i ) const
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
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::reset()
{
   using blaze::erase;

   if( SO ) {
      for( size_t j=0UL; j<columns(); ++j ) {
         erase( matrix_, j, matrix_.lowerBound(j+1UL,j), matrix_.end(j) );
      }
   }
   else {
      for( size_t i=1UL; i<rows(); ++i ) {
         erase( matrix_, i, matrix_.begin(i), matrix_.lowerBound(i,i) );
      }
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
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::reset( size_t i )
{
   using blaze::erase;

   if( SO ) {
      erase( matrix_, i, matrix_.lowerBound(i+1UL,i), matrix_.end(i) );
   }
   else {
      erase( matrix_, i, matrix_.begin(i), matrix_.lowerBound(i,i) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the strictly lower matrix.
//
// \return void
//
// This function clears the strictly lower matrix and returns it to its default state.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::clear()
{
   matrix_.clear();

   BLAZE_INTERNAL_ASSERT( matrix_.rows()    == 0UL, "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( matrix_.columns() == 0UL, "Invalid number of columns" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Changing the size of the strictly lower matrix.
//
// \param n The new number of rows and columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function resizes the matrix using the given size to \f$ m \times n \f$. During this
// operation, new dynamic memory may be allocated in case the capacity of the matrix is too
// small. Note that this function may invalidate all existing views (submatrices, rows, columns,
// ...) on the matrix if it is used to shrink the matrix. Additionally, the resize operation
// potentially changes all matrix elements. In order to preserve the old matrix values, the
// \a preserve flag can be set to \a true.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
void StrictlyLowerMatrix<MT,SO,false>::resize( size_t n, bool preserve )
{
   BLAZE_CONSTRAINT_MUST_BE_RESIZABLE_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( isSquare( matrix_ ), "Non-square strictly lower matrix detected" );

   matrix_.resize( n, n, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of the strictly lower matrix.
//
// \param nonzeros The new minimum capacity of the strictly lower matrix.
// \return void
//
// This function increases the capacity of the strictly lower matrix to at least \a nonzeros
// elements. The current values of the matrix elements and the individual capacities of the
// matrix rows are preserved.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::reserve( size_t nonzeros )
{
   matrix_.reserve( nonzeros );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of a specific row/column of the strictly lower matrix.
//
// \param i The row/column index \f$[0..N-1]\f$.
// \param nonzeros The new minimum capacity of the specified row/column.
// \return void
//
// This function increases the capacity of row/column \a i of the strictly lower matrix to at
// least \a nonzeros elements. The current values of the strictly lower matrix and all other
// individual row/column capacities are preserved. In case the strictly lower matrix adapts a
// \a rowMajor sparse matrix the function reserves capacity for row \a i. In case it adapts a
// \a columnMajor the function reserves capacity for column \a i. The index has to be in the
// range \f$[0..N-1]\f$.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::reserve( size_t i, size_t nonzeros )
{
   matrix_.reserve( i, nonzeros );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removing all excessive capacity from all rows/columns.
//
// \return void
//
// The trim() function can be used to reverse the effect of all row/column-specific reserve()
// calls. The function removes all excessive capacity from all rows (in case of a rowMajor
// matrix) or columns (in case of a columnMajor matrix). Note that this function does not
// remove the overall capacity but only reduces the capacity per row/column.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::trim()
{
   matrix_.trim();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removing all excessive capacity of a specific row/column of the strictly lower matrix.
//
// \param i The index of the row/column to be trimmed \f$[0..N-1]\f$.
// \return void
//
// This function can be used to reverse the effect of a row/column-specific reserve() call.
// It removes all excessive capacity from the specified row (in case of a rowMajor matrix)
// or column (in case of a columnMajor matrix). The excessive capacity is assigned to the
// subsequent row/column.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::trim( size_t i )
{
   matrix_.trim( i );
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
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::shrinkToFit()
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
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::swap( StrictlyLowerMatrix& m ) noexcept
{
   using std::swap;

   swap( matrix_, m.matrix_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum number of non-zero values for a strictly lower triangular matrix.
//
// \return The maximum number of non-zero values.
//
// This function returns the maximum possible number of non-zero values for a strictly lower
// triangular matrix with fixed-size adapted matrix of type \a MT. Note that this function can
// only be called in case the adapted dense matrix is a fixed-size matrix. The attempt to call
// this function in case the adapted matrix is resizable matrix will result in a compile time
// error.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
constexpr size_t StrictlyLowerMatrix<MT,SO,false>::maxNonZeros() noexcept
{
   BLAZE_CONSTRAINT_MUST_BE_STATIC_TYPE( MT );

   return maxNonZeros( Size_v<MT,0UL> );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum number of non-zero values for a strictly lower triangular matrix.
//
// \param n The number of rows and columns of the matrix.
// \return The maximum number of non-zero values.
//
// This function returns the maximum possible number of non-zero values for a strictly lower
// triangular matrix of the given number of rows and columns.
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
constexpr size_t StrictlyLowerMatrix<MT,SO,false>::maxNonZeros( size_t n ) noexcept
{
   return ( ( n - 1UL ) * n ) / 2UL;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the complete upper part of the matrix to the default initial values.
//
// \return void
*/
template< typename MT  // Type of the adapted dense matrix
        , bool SO >    // Storage order of the adapted dense matrix
inline void StrictlyLowerMatrix<MT,SO,false>::resetUpper()
{
   using blaze::erase;

   if( SO ) {
      for( size_t j=1UL; j<columns(); ++j )
         erase( matrix_, j, matrix_.begin( j ), matrix_.lowerBound( j, j ) );
   }
   else {
      for( size_t i=0UL; i<rows(); ++i )
         erase( matrix_, i, matrix_.upperBound( i, i ), matrix_.end( i ) );
   }
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  INSERTION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting elements of the strictly lower matrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be set.
// \return Iterator to the set element.
// \exception std::invalid_argument Invalid access to diagonal or upper matrix element.
//
// This function sets the value of an element of the strictly lower matrix. In case the matrix
// already contains an element with row index \a i and column index \a j its value is modified,
// else a new element with the given \a value is inserted. The attempt to set an element on the
// diagonal or in the upper part of the matrix (i.e. above the diagonal) will result in a
// \a std::invalid_argument exception.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::set( size_t i, size_t j, const ElementType& value )
{
   if( i <= j ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to diagonal or upper matrix element" );
   }

   return matrix_.set( i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Inserting elements into the strictly lower matrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be inserted.
// \return Iterator to the newly inserted element.
// \exception std::invalid_argument Invalid sparse matrix access index.
// \exception std::invalid_argument Invalid access to diagonal or upper matrix element.
//
// This function inserts a new element into the strictly lower matrix. However, duplicate elements
// are not allowed. In case the matrix already contains an element with row index \a i and column
// index \a j, a \a std::invalid_argument exception is thrown. Also, the attempt to insert an
// element on the diagonal or in the upper part of the matrix (i.e. above the diagonal) will
// result in a \a std::invalid_argument exception.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::insert( size_t i, size_t j, const ElementType& value )
{
   if( i <= j ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to diagonal or upper matrix element" );
   }

   return matrix_.insert( i, j, value );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Appending elements to the specified row/column of the strictly lower matrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be appended.
// \param check \a true if the new value should be checked for default values, \a false if not.
// \return void
// \exception std::invalid_argument Invalid access to diagonal or upper matrix element.
//
// This function provides a very efficient way to fill a sparse matrix with elements. It appends
// a new element to the end of the specified row/column without any additional memory allocation.
// Therefore it is strictly necessary to keep the following preconditions in mind:
//
//  - the index of the new element must be strictly larger than the largest index of non-zero
//    elements in the specified row/column of the sparse matrix
//  - the current number of non-zero elements in the matrix must be smaller than the capacity
//    of the matrix
//
// Ignoring these preconditions might result in undefined behavior! The optional \a check
// parameter specifies whether the new value should be tested for a default value. If the new
// value is a default value (for instance 0 in case of an integral element type) the value is
// not appended. Per default the values are not tested.
//
// In combination with the reserve() and the finalize() function, append() provides the most
// efficient way to add new elements to a (newly created) sparse matrix:

   \code
   using blaze::CompressedMatrix;
   using blaze::StrictlyLowerMatrix;
   using blaze::columnMajor;

   StrictlyLowerMatrix< CompressedMatrix<double,columnMajor> > A( 4 );

   A.reserve( 3 );         // Reserving enough capacity for 3 non-zero elements
   A.append( 1, 0, 1.0 );  // Appending the value 1 in column 0 with row index 1
   A.finalize( 0 );        // Finalizing column 0
   A.append( 2, 1, 2.0 );  // Appending the value 2 in column 1 with row index 2
   A.finalize( 1 );        // Finalizing column 1
   A.append( 3, 2, 3.0 );  // Appending the value 3 in column 2 with row index 3
   A.finalize( 2 );        // Finalizing column 2
   A.finalize( 3 );        // Finalizing the empty column 3
   \endcode

// Note that although append() does not allocate new memory it still invalidates all iterators
// returned by the end() functions! Also note that the attempt to append an element within the
// upper part of the matrix (i.e. above the diagonal) will result in a \a std::invalid_argument
// exception.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::append( size_t i, size_t j, const ElementType& value, bool check )
{
   if( i <= j ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to diagonal or upper matrix element" );
   }

   matrix_.append( i, j, value, check );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Finalizing the element insertion of a row/column.
//
// \param i The index of the row/column to be finalized \f$[0..N-1]\f$.
// \return void
//
// This function is part of the low-level interface to efficiently fill a matrix with elements.
// After completion of row/column \a i via the append() function, this function can be called to
// finalize row/column \a i and prepare the next row/column for insertion process via append().
//
// \note Although finalize() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::finalize( size_t i )
{
   matrix_.finalize( i );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ERASE FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing elements from the strictly lower matrix.
//
// \param i The row index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \return void
//
// This function erases an element from the strictly lower matrix.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline void StrictlyLowerMatrix<MT,SO,false>::erase( size_t i, size_t j )
{
   using blaze::erase;

   erase( matrix_, i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing elements from the strictly lower matrix.
//
// \param i The row/column index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \param pos Iterator to the element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases an element from the strictly lower matrix. In case the strictly lower
// matrix adapts a \a rowMajor sparse matrix the function erases an element from row \a i, in
// case it adapts a \a columnMajor sparse matrix the function erases an element from column \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::erase( size_t i, Iterator pos )
{
   using blaze::erase;

   return erase( matrix_, i, pos );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing a range of elements from the strictly lower matrix.
//
// \param i The row/column index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \param first Iterator to first element to be erased.
// \param last Iterator just past the last element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases a range of elements from the strictly lower matrix. In case the matrix
// adapts a \a rowMajor sparse matrix the function erases a range of elements from row \a i, in
// case it adapts a \a columnMajor matrix the function erases a range of elements from column
// \a i.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::erase( size_t i, Iterator first, Iterator last )
{
   using blaze::erase;

   return erase( matrix_, i, first, last );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing specific elements from the strictly lower matrix.
//
// \param predicate The unary predicate for the element selection.
// \return void.
//
// This function erases specific elements from the strictly lower matrix. The elements are
// selected by the given unary predicate \a predicate, which is expected to accept a single
// argument of the type of the elements and to be pure. The following example demonstrates
// how to remove all elements that are smaller than a certain threshold value:

   \code
   blaze::StrictlyLowerMatrix< CompressedMatrix<double,blaze::rowMajor> > A;
   // ... Resizing and initialization

   A.erase( []( double value ){ return value < 1E-8; } );
   \endcode

// \note The predicate is required to be pure, i.e. to produce deterministic results for elements
// with the same value. The attempt to use an impure predicate leads to undefined behavior!
*/
template< typename MT      // Type of the adapted sparse matrix
        , bool SO >        // Storage order of the adapted sparse matrix
template< typename Pred >  // Type of the unary predicate
inline void StrictlyLowerMatrix<MT,SO,false>::erase( Pred predicate )
{
   using blaze::erase;

   erase( matrix_, predicate );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing specific elements from a range of the strictly lower matrix.
//
// \param i The row/column index of the elements to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param first Iterator to first element of the range.
// \param last Iterator just past the last element of the range.
// \param predicate The unary predicate for the element selection.
// \return void
//
// This function erases specific elements from a range of elements of the strictly lower matrix.
// The elements are selected by the given unary predicate \a predicate, which is expected to
// accept a single argument of the type of the elements and to be pure. In case the storage
// order is set to \a rowMajor the function erases a range of elements from row \a i, in case
// the storage flag is set to \a columnMajor the function erases a range of elements from column
// \a i. The following example demonstrates how to remove all elements that are smaller than a
// certain threshold value:

   \code
   blaze::StrictlyLowerMatrix< CompressedMatrix<double,blaze::rowMajor> > A;
   // ... Resizing and initialization

   A.erase( 2UL, A.begin(2UL), A.end(2UL), []( double value ){ return value < 1E-8; } );
   \endcode

// \note The predicate is required to be pure, i.e. to produce deterministic results for elements
// with the same value. The attempt to use an impure predicate leads to undefined behavior!
*/
template< typename MT      // Type of the adapted sparse matrix
        , bool SO >        // Storage order of the adapted sparse matrix
template< typename Pred >  // Type of the unary predicate
inline void StrictlyLowerMatrix<MT,SO,false>::erase( size_t i, Iterator first, Iterator last, Pred predicate )
{
   using blaze::erase;

   erase( matrix_, i, first, last, predicate );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Broken invariant detected" );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LOOKUP FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Searches for a specific matrix element.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the strictly
// lower matrix. It specifically searches for the element with row index \a i and column index
// \a j. In case the element is found, the function returns an row/column iterator to the
// element. Otherwise an iterator just past the last non-zero element of row \a i or column
// \a j (the end() iterator) is returned. Note that the returned unilower matrix iterator is
// subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::find( size_t i, size_t j )
{
   return matrix_.find( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Searches for a specific matrix element.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the strictly
// lower matrix. It specifically searches for the element with row index \a i and column index
// \a j. In case the element is found, the function returns an row/column iterator to the
// element. Otherwise an iterator just past the last non-zero element of row \a i or column
// \a j (the end() iterator) is returned. Note that the returned strictly lower matrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstIterator
   StrictlyLowerMatrix<MT,SO,false>::find( size_t i, size_t j ) const
{
   return matrix_.find( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// In case of a row-major matrix, this function returns a row iterator to the first element with
// an index not less then the given column index. In case of a column-major matrix, the function
// returns a column iterator to the first element with an index not less then the given row
// index. In combination with the upperBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned strictly lower
// matrix iterator is subject to invalidation due to inserting operations via the function
// call operator, the set() function or the insert() function!
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::lowerBound( size_t i, size_t j )
{
   return matrix_.lowerBound( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// In case of a row-major matrix, this function returns a row iterator to the first element with
// an index not less then the given column index. In case of a column-major matrix, the function
// returns a column iterator to the first element with an index not less then the given row
// index. In combination with the upperBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned strictly lower
// matrix iterator is subject to invalidation due to inserting operations via the function
// call operator, the set() function or the insert() function!
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstIterator
   StrictlyLowerMatrix<MT,SO,false>::lowerBound( size_t i, size_t j ) const
{
   return matrix_.lowerBound( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// In case of a row-major matrix, this function returns a row iterator to the first element with
// an index greater then the given column index. In case of a column-major matrix, the function
// returns a column iterator to the first element with an index greater then the given row
// index. In combination with the lowerBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned strictly lower
// matrix iterator is subject to invalidation due to inserting operations via the function
// call operator, the set() function or the insert() function!
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::Iterator
   StrictlyLowerMatrix<MT,SO,false>::upperBound( size_t i, size_t j )
{
   return matrix_.upperBound( i, j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// In case of a row-major matrix, this function returns a row iterator to the first element with
// an index greater then the given column index. In case of a column-major matrix, the function
// returns a column iterator to the first element with an index greater then the given row
// index. In combination with the lowerBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned strictly lower
// matrix iterator is subject to invalidation due to inserting operations via the function
// call operator, the set() function or the insert() function!
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline typename StrictlyLowerMatrix<MT,SO,false>::ConstIterator
   StrictlyLowerMatrix<MT,SO,false>::upperBound( size_t i, size_t j ) const
{
   return matrix_.upperBound( i, j );
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
   blaze::StrictlyLowerMatrix< blaze::CompressedMatrix<int> > A;
   // ... Resizing and initialization
   A *= 4;        // Scaling of the matrix
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename MT       // Type of the adapted sparse matrix
        , bool SO >         // Storage order of the adapted sparse matrix
template< typename Other >  // Data type of the scalar value
inline StrictlyLowerMatrix<MT,SO,false>&
   StrictlyLowerMatrix<MT,SO,false>::scale( const Other& scalar )
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
/*!\brief Returns whether the invariants of the strictly lower matrix are intact.
//
// \return \a true in case the strictly lower matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the strictly lower matrix are intact, i.e. if
// its state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline bool StrictlyLowerMatrix<MT,SO,false>::isIntact() const noexcept
{
   using blaze::isIntact;

   return ( isIntact( matrix_ ) && isStrictlyLower( matrix_ ) );
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
template< typename MT       // Type of the adapted sparse matrix
        , bool SO >         // Storage order of the adapted sparse matrix
template< typename Other >  // Data type of the foreign expression
inline bool StrictlyLowerMatrix<MT,SO,false>::canAlias( const Other* alias ) const noexcept
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
template< typename MT       // Type of the adapted sparse matrix
        , bool SO >         // Storage order of the adapted sparse matrix
template< typename Other >  // Data type of the foreign expression
inline bool StrictlyLowerMatrix<MT,SO,false>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( alias );
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
template< typename MT  // Type of the adapted sparse matrix
        , bool SO >    // Storage order of the adapted sparse matrix
inline bool StrictlyLowerMatrix<MT,SO,false>::canSMPAssign() const noexcept
{
   return matrix_.canSMPAssign();
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
