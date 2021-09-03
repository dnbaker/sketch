//=================================================================================================
/*!
//  \file blaze/math/views/submatrix/Sparse.h
//  \brief Submatrix specialization for sparse matrices
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

#ifndef _BLAZE_MATH_VIEWS_SUBMATRIX_SPARSE_H_
#define _BLAZE_MATH_VIEWS_SUBMATRIX_SPARSE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <vector>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/Matrix.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/constraints/SparseMatrix.h>
#include <blaze/math/constraints/Submatrix.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/constraints/TransExpr.h>
#include <blaze/math/constraints/UniTriangular.h>
#include <blaze/math/dense/InitializerMatrix.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/expressions/View.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/sparse/SparseElement.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/SchurTrait.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/IsExpression.h>
#include <blaze/math/typetraits/IsHermitian.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsUniLower.h>
#include <blaze/math/typetraits/IsUniUpper.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/views/Check.h>
#include <blaze/math/views/submatrix/BaseTemplate.h>
#include <blaze/math/views/submatrix/SubmatrixData.h>
#include <blaze/util/algorithms/Max.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/mpl/If.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsConst.h>
#include <blaze/util/typetraits/IsReference.h>


namespace blaze {

//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR ROW-MAJOR SPARSE MATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Submatrix for row-major sparse submatrices.
// \ingroup submatrix
//
// This specialization of Submatrix adapts the class template to the requirements of row-major
// sparse submatrices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
class Submatrix<MT,AF,false,false,CSAs...>
   : public View< SparseMatrix< Submatrix<MT,AF,false,false,CSAs...>, false > >
   , private SubmatrixData<CSAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = SubmatrixData<CSAs...>;               //!< The type of the SubmatrixData base class.
   using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the matrix expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this Submatrix instance.
   using This = Submatrix<MT,AF,false,false,CSAs...>;

   //! Base type of this Submatrix instance.
   using BaseType = View< SparseMatrix<This,false> >;

   using ViewedType    = MT;                            //!< The type viewed by this Submatrix instance.
   using ResultType    = SubmatrixTrait_t<MT,CSAs...>;  //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;   //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;             //!< Type of the submatrix elements.
   using ReturnType    = ReturnType_t<MT>;              //!< Return type for expression template evaluations
   using CompositeType = const Submatrix&;              //!< Data type for composite expression templates.

   //! Reference to a constant submatrix value.
   using ConstReference = ConstReference_t<MT>;

   //! Reference to a non-constant submatrix value.
   using Reference = If_t< IsConst_v<MT>, ConstReference, Reference_t<MT> >;
   //**********************************************************************************************

   //**SubmatrixElement class definition***********************************************************
   /*!\brief Access proxy for a specific element of the sparse submatrix.
   */
   template< typename MatrixType      // Type of the sparse matrix
           , typename IteratorType >  // Type of the sparse matrix iterator
   class SubmatrixElement
      : private SparseElement
   {
    public:
      //**Constructor******************************************************************************
      /*!\brief Constructor for the SubmatrixElement class.
      //
      // \param pos Iterator to the current position within the sparse submatrix.
      // \param offset The offset within the according row/column of the sparse matrix.
      */
      inline SubmatrixElement( IteratorType pos, size_t offset )
         : pos_   ( pos    )  // Iterator to the current position within the sparse submatrix
         , offset_( offset )  // Row offset within the according sparse matrix
      {}
      //*******************************************************************************************

      //**Assignment operator**********************************************************************
      /*!\brief Assignment to the accessed sparse submatrix element.
      //
      // \param v The new value of the sparse submatrix element.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator=( const T& v ) {
         *pos_ = v;
         return *this;
      }
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment to the accessed sparse submatrix element.
      //
      // \param v The right-hand side value for the addition.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator+=( const T& v ) {
         *pos_ += v;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment to the accessed sparse submatrix element.
      //
      // \param v The right-hand side value for the subtraction.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator-=( const T& v ) {
         *pos_ -= v;
         return *this;
      }
      //*******************************************************************************************

      //**Multiplication assignment operator*******************************************************
      /*!\brief Multiplication assignment to the accessed sparse submatrix element.
      //
      // \param v The right-hand side value for the multiplication.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator*=( const T& v ) {
         *pos_ *= v;
         return *this;
      }
      //*******************************************************************************************

      //**Division assignment operator*************************************************************
      /*!\brief Division assignment to the accessed sparse submatrix element.
      //
      // \param v The right-hand side value for the division.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator/=( const T& v ) {
         *pos_ /= v;
         return *this;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the sparse submatrix element at the current iterator position.
      //
      // \return Reference to the sparse submatrix element at the current iterator position.
      */
      inline const SubmatrixElement* operator->() const {
         return this;
      }
      //*******************************************************************************************

      //**Value function***************************************************************************
      /*!\brief Access to the current value of the sparse submatrix element.
      //
      // \return The current value of the sparse submatrix element.
      */
      inline decltype(auto) value() const {
         return pos_->value();
      }
      //*******************************************************************************************

      //**Index function***************************************************************************
      /*!\brief Access to the current index of the sparse element.
      //
      // \return The current index of the sparse element.
      */
      inline size_t index() const {
         return pos_->index() - offset_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType pos_;  //!< Iterator to the current position within the sparse submatrix.
      size_t offset_;     //!< Offset within the according row/column of the sparse matrix.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**SubmatrixIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the sparse submatrix.
   */
   template< typename MatrixType      // Type of the sparse matrix
           , typename IteratorType >  // Type of the sparse matrix iterator
   class SubmatrixIterator
   {
    public:
      //**Type definitions*************************************************************************
      using IteratorCategory = std::forward_iterator_tag;                  //!< The iterator category.
      using ValueType        = SubmatrixElement<MatrixType,IteratorType>;  //!< Type of the underlying elements.
      using PointerType      = ValueType;                                  //!< Pointer return type.
      using ReferenceType    = ValueType;                                  //!< Reference return type.
      using DifferenceType   = ptrdiff_t;                                  //!< Difference between two iterators.

      // STL iterator requirements
      using iterator_category = IteratorCategory;  //!< The iterator category.
      using value_type        = ValueType;         //!< Type of the underlying elements.
      using pointer           = PointerType;       //!< Pointer return type.
      using reference         = ReferenceType;     //!< Reference return type.
      using difference_type   = DifferenceType;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Default constructor**********************************************************************
      /*!\brief Default constructor for the SubmatrixIterator class.
      */
      inline SubmatrixIterator()
         : pos_   ()  // Iterator to the current sparse element
         , offset_()  // The offset of the according row/column of the sparse matrix
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the SubmatrixIterator class.
      //
      // \param iterator Iterator to the current sparse element.
      // \param index The starting index within the according row/column of the sparse matrix.
      */
      inline SubmatrixIterator( IteratorType iterator, size_t index )
         : pos_   ( iterator )  // Iterator to the current sparse element
         , offset_( index    )  // The offset of the according row/column of the sparse matrix
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different SubmatrixIterator instances.
      //
      // \param it The submatrix iterator to be copied.
      */
      template< typename MatrixType2, typename IteratorType2 >
      inline SubmatrixIterator( const SubmatrixIterator<MatrixType2,IteratorType2>& it )
         : pos_   ( it.base()   )  // Iterator to the current sparse element.
         , offset_( it.offset() )  // The offset of the according row/column of the sparse matrix
      {}
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline SubmatrixIterator& operator++() {
         ++pos_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubmatrixIterator operator++( int ) {
         const SubmatrixIterator tmp( *this );
         ++(*this);
         return tmp;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the current sparse submatrix element.
      //
      // \return Reference to the current sparse submatrix element.
      */
      inline ReferenceType operator*() const {
         return ReferenceType( pos_, offset_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the current sparse submatrix element.
      //
      // \return Pointer to the current sparse submatrix element.
      */
      inline PointerType operator->() const {
         return PointerType( pos_, offset_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side submatrix iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      template< typename MatrixType2, typename IteratorType2 >
      inline bool operator==( const SubmatrixIterator<MatrixType2,IteratorType2>& rhs ) const {
         return base() == rhs.base();
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side submatrix iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      template< typename MatrixType2, typename IteratorType2 >
      inline bool operator!=( const SubmatrixIterator<MatrixType2,IteratorType2>& rhs ) const {
         return !( *this == rhs );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two submatrix iterators.
      //
      // \param rhs The right-hand side submatrix iterator.
      // \return The number of elements between the two submatrix iterators.
      */
      inline DifferenceType operator-( const SubmatrixIterator& rhs ) const {
         return pos_ - rhs.pos_;
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the submatrix iterator.
      //
      // \return The current position of the submatrix iterator.
      */
      inline IteratorType base() const {
         return pos_;
      }
      //*******************************************************************************************

      //**Offset function**************************************************************************
      /*!\brief Access to the offset of the submatrix iterator.
      //
      // \return The offset of the submatrix iterator.
      */
      inline size_t offset() const noexcept {
         return offset_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType pos_;     //!< Iterator to the current sparse element.
      size_t       offset_;  //!< The offset of the according row/column of the sparse matrix.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   using ConstIterator = SubmatrixIterator< const MT, ConstIterator_t<MT> >;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<MT>, ConstIterator, SubmatrixIterator< MT, Iterator_t<MT> > >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;

   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool compileTimeArgs = DataType::compileTimeArgs;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline Submatrix( MT& matrix, RSAs... args );

   Submatrix( const Submatrix& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~Submatrix() = default;
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
   inline Submatrix& operator=( initializer_list< initializer_list<ElementType> > list );
   inline Submatrix& operator=( const Submatrix& rhs );

   template< typename MT2, bool SO > inline Submatrix& operator= ( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline Submatrix& operator+=( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline Submatrix& operator-=( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline Submatrix& operator%=( const Matrix<MT2,SO>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::row;
   using DataType::column;
   using DataType::rows;
   using DataType::columns;

   inline MT&       operand() noexcept;
   inline const MT& operand() const noexcept;

   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t i ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i ) const;
   inline void   reset();
   inline void   reset( size_t i );
   inline void   reserve( size_t nonzeros );
          void   reserve( size_t i, size_t nonzeros );
   inline void   trim();
   inline void   trim( size_t i );
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
   inline Submatrix& transpose();
   inline Submatrix& ctranspose();

   template< typename Other > inline Submatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   inline bool canSMPAssign() const noexcept;

   template< typename MT2, bool SO > inline void assign     ( const DenseMatrix<MT2,SO>& rhs );
   template< typename MT2 >          inline void assign     ( const SparseMatrix<MT2,false>& rhs );
   template< typename MT2 >          inline void assign     ( const SparseMatrix<MT2,true>& rhs );
   template< typename MT2, bool SO > inline void addAssign  ( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline void subAssign  ( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline void schurAssign( const Matrix<MT2,SO>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand matrix_;  //!< The matrix containing the submatrix.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE   ( MT );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE     ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE   ( MT );
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
/*!\brief Constructor for row-major sparse submatrices.
//
// \param matrix The sparse matrix containing the submatrix.
// \param args The runtime submatrix arguments.
// \exception std::invalid_argument Invalid submatrix specification.
//
// By default, the provided submatrix arguments are checked at runtime. In case the submatrix is
// not properly specified (i.e. if the specified submatrix is not contained in the given sparse
// matrix) a \a std::invalid_argument exception is thrown. The checks can be skipped by providing
// the optional \a blaze::unchecked argument.
*/
template< typename MT         // Type of the sparse matrix
        , AlignmentFlag AF    // Alignment flag
        , size_t... CSAs >    // Compile time submatrix arguments
template< typename... RSAs >  // Runtime submatrix arguments
inline Submatrix<MT,AF,false,false,CSAs...>::Submatrix( MT& matrix, RSAs... args )
   : DataType( args... )  // Base class initialization
   , matrix_ ( matrix  )  // The matrix containing the submatrix
{
   if( isChecked( args... ) ) {
      if( ( row() + rows() > matrix_.rows() ) || ( column() + columns() > matrix_.columns() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( row()    + rows()    <= matrix_.rows()   , "Invalid submatrix specification" );
      BLAZE_USER_ASSERT( column() + columns() <= matrix_.columns(), "Invalid submatrix specification" );
   }
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
/*!\brief 2D-access to the sparse submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Reference
   Submatrix<MT,AF,false,false,CSAs...>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_(row()+i,column()+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the sparse submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstReference
   Submatrix<MT,AF,false,false,CSAs...>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return const_cast<const MT&>( matrix_ )(row()+i,column()+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Reference
   Submatrix<MT,AF,false,false,CSAs...>::at( size_t i, size_t j )
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
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstReference
   Submatrix<MT,AF,false,false,CSAs...>::at( size_t i, size_t j ) const
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
/*!\brief Returns an iterator to the first non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first non-zero element of row/column \a i.
//
// This function returns a row/column iterator to the first non-zero element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator to the first
// non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator to the first non-zero element of column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::begin( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid sparse submatrix row access index" );

   if( column() == 0UL )
      return Iterator( matrix_.begin( i + row() ), column() );
   else
      return Iterator( matrix_.lowerBound( i + row(), column() ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first non-zero element of row/column \a i.
//
// This function returns a row/column iterator to the first non-zero element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator to the first
// non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator to the first non-zero element of column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,false,false,CSAs...>::begin( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid sparse submatrix row access index" );

   if( column() == 0UL )
      return ConstIterator( matrix_.cbegin( i + row() ), column() );
   else
      return ConstIterator( matrix_.lowerBound( i + row(), column() ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first non-zero element of row/column \a i.
//
// This function returns a row/column iterator to the first non-zero element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator to the first
// non-zero element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator to the first non-zero element of column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,false,false,CSAs...>::cbegin( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid sparse submatrix row access index" );

   if( column() == 0UL )
      return ConstIterator( matrix_.cbegin( i + row() ), column() );
   else
      return ConstIterator( matrix_.lowerBound( i + row(), column() ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last non-zero element of row/column \a i.
//
// This function returns an row/column iterator just past the last non-zero element of row/column
// \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
// past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
// the function returns an iterator just past the last non-zero element of column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::end( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid sparse submatrix row access index" );

   if( matrix_.columns() == column() + columns() )
      return Iterator( matrix_.end( i + row() ), column() );
   else
      return Iterator( matrix_.lowerBound( i + row(), column() + columns() ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last non-zero element of row/column \a i.
//
// This function returns an row/column iterator just past the last non-zero element of row/column
// \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
// past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
// the function returns an iterator just past the last non-zero element of column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,false,false,CSAs...>::end( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid sparse submatrix row access index" );

   if( matrix_.columns() == column() + columns() )
      return ConstIterator( matrix_.cend( i + row() ), column() );
   else
      return ConstIterator( matrix_.lowerBound( i + row(), column() + columns() ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last non-zero element of row/column \a i.
//
// This function returns an row/column iterator just past the last non-zero element of row/column
// \a i. In case the storage order is set to \a rowMajor the function returns an iterator just
// past the last non-zero element of row \a i, in case the storage flag is set to \a columnMajor
// the function returns an iterator just past the last non-zero element of column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,false,false,CSAs...>::cend( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid sparse submatrix row access index" );

   if( matrix_.columns() == column() + columns() )
      return ConstIterator( matrix_.cend( i + row() ), column() );
   else
      return ConstIterator( matrix_.lowerBound( i + row(), column() + columns() ), column() );
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
/*!\brief List assignment to all submatrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to submatrix.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// This assignment operator offers the option to directly assign to all elements of the submatrix
// by means of an initializer list. The submatrix elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size of the
// top-level initializer list does not match the number of rows of the submatrix or the size of
// any nested list exceeds the number of columns, a \a std::invalid_argument exception is thrown.
// Also, if the underlying matrix \a MT is restricted and the assignment would violate an
// invariant of the matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   using blaze::assign;

   if( list.size() != rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to submatrix" );
   }

   const InitializerMatrix<ElementType> tmp( list, columns() );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   left.reset();
   assign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for Submatrix.
//
// \param rhs Sparse submatrix to be copied.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Submatrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The sparse submatrix is initialized as a copy of the given sparse submatrix. In case the
// current sizes of the two submatrices don't match, a \a std::invalid_argument exception is
// thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::operator=( const Submatrix& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &matrix_ == &rhs.matrix_ && row() == rhs.row() && column() == rhs.column() ) )
      return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Submatrix sizes do not match" );
   }

   if( !tryAssign( matrix_, rhs, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( rhs.canAlias( this ) ) {
      const ResultType tmp( rhs );
      left.reset();
      assign( left, tmp );
   }
   else {
      left.reset();
      assign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be assigned.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The sparse submatrix is initialized as a copy of the given matrix. In case the current sizes
// of the two matrices don't match, a \a std::invalid_argument exception is thrown. Also, if
// the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::operator=( const Matrix<MT2,SO>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   if( rows() != (*rhs).rows() || columns() != (*rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = CompositeType_t<MT2>;
   Right right( *rhs );

   if( !tryAssign( matrix_, right, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( this ) ) {
      const ResultType_t<MT2> tmp( right );
      left.reset();
      assign( left, transIf< IsSymmetric_v<This> >( tmp ) );
   }
   else {
      left.reset();
      assign( left, transIf< IsSymmetric_v<This> >( right ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the sparse submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::operator+=( const Matrix<MT2,SO>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using AddType = AddTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (*rhs).rows() || columns() != (*rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const AddType tmp( *this + (*rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   left.reset();
   assign( left, transIf< IsSymmetric_v<This> >( tmp ) );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the sparse submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::operator-=( const Matrix<MT2,SO>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SubType = SubTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (*rhs).rows() || columns() != (*rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SubType tmp( *this - (*rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   left.reset();
   assign( left, transIf< IsSymmetric_v<This> >( tmp ) );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a matrix (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side matrix to be for the Schur product.
// \return Reference to the sparse submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::operator%=( const Matrix<MT2,SO>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   if( rows() != (*rhs).rows() || columns() != (*rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SchurType tmp( *this % (*rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   left.reset();
   assign( left, transIf< IsSymmetric_v<This> >( tmp ) );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

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
/*!\brief Returns the matrix containing the submatrix.
//
// \return The matrix containing the submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline MT& Submatrix<MT,AF,false,false,CSAs...>::operand() noexcept
{
   return matrix_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the matrix containing the submatrix.
//
// \return The matrix containing the submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline const MT& Submatrix<MT,AF,false,false,CSAs...>::operand() const noexcept
{
   return matrix_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the sparse submatrix.
//
// \return The capacity of the sparse submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline size_t Submatrix<MT,AF,false,false,CSAs...>::capacity() const noexcept
{
   return nonZeros() + matrix_.capacity() - matrix_.nonZeros();
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
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline size_t Submatrix<MT,AF,false,false,CSAs...>::capacity( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   return nonZeros( i ) + matrix_.capacity( row()+i ) - matrix_.nonZeros( row()+i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the sparse submatrix
//
// \return The number of non-zero elements in the sparse submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline size_t Submatrix<MT,AF,false,false,CSAs...>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t i=0UL; i<rows(); ++i )
      nonzeros += nonZeros( i );

   return nonzeros;
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
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline size_t Submatrix<MT,AF,false,false,CSAs...>::nonZeros( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   return end(i) - begin(i);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,false,false,CSAs...>::reset()
{
   for( size_t i=row(); i<row()+rows(); ++i )
   {
      const size_t jbegin( ( IsUpper_v<MT> )
                           ?( ( IsUniUpper_v<MT> || IsStrictlyUpper_v<MT> )
                              ?( max( i+1UL, column() ) )
                              :( max( i, column() ) ) )
                           :( column() ) );
      const size_t jend  ( ( IsLower_v<MT> )
                           ?( ( IsUniLower_v<MT> || IsStrictlyLower_v<MT> )
                              ?( min( i, column()+columns() ) )
                              :( min( i+1UL, column()+columns() ) ) )
                           :( column()+columns() ) );

      matrix_.erase( i, matrix_.lowerBound( i, jbegin ), matrix_.lowerBound( i, jend ) );
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
//
// This function resets the values in the specified row/column to their default value. In case
// the storage order is set to \a rowMajor the function resets the values in row \a i, in case
// the storage order is set to \a columnMajor the function resets the values in column \a i.
// Note that the capacity of the row/column remains unchanged.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,false,false,CSAs...>::reset( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t index( row() + i );

   const size_t jbegin( ( IsUpper_v<MT> )
                        ?( ( IsUniUpper_v<MT> || IsStrictlyUpper_v<MT> )
                           ?( max( i+1UL, column() ) )
                           :( max( i, column() ) ) )
                        :( column() ) );
   const size_t jend  ( ( IsLower_v<MT> )
                        ?( ( IsUniLower_v<MT> || IsStrictlyLower_v<MT> )
                           ?( min( i, column()+columns() ) )
                           :( min( i+1UL, column()+columns() ) ) )
                        :( column()+columns() ) );

   matrix_.erase( index, matrix_.lowerBound( index, jbegin ), matrix_.lowerBound( index, jend ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of the sparse submatrix.
//
// \param nonzeros The new minimum capacity of the sparse submatrix.
// \return void
//
// This function increases the capacity of the sparse submatrix to at least \a nonzeros elements.
// The current values of the submatrix elements and the individual capacities of the submatrix
// rows are preserved.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,false,false,CSAs...>::reserve( size_t nonzeros )
{
   const size_t current( capacity() );

   if( nonzeros > current ) {
      matrix_.reserve( matrix_.capacity() + nonzeros - current );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of a specific row/column of the sparse submatrix.
//
// \param i The row/column index of the new element \f$[0..M-1]\f$ or \f$[0..N-1]\f$.
// \param nonzeros The new minimum capacity of the specified row/column.
// \return void
//
// This function increases the capacity of row/column \a i of the sparse submatrix to at least
// \a nonzeros elements, but not beyond the current number of columns/rows, respectively. The
// current values of the sparse submatrix and all other individual row/column capacities are
// preserved. In case the storage order is set to \a rowMajor, the function reserves capacity
// for row \a i and the index has to be in the range \f$[0..M-1]\f$. In case the storage order
// is set to \a columnMajor, the function reserves capacity for column \a i and the index has
// to be in the range \f$[0..N-1]\f$.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
void Submatrix<MT,AF,false,false,CSAs...>::reserve( size_t i, size_t nonzeros )
{
   const size_t current( capacity( i ) );
   const size_t index  ( row() + i );

   if( nonzeros > current ) {
      matrix_.reserve( index, matrix_.capacity( index ) + nonzeros - current );
   }
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
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
void Submatrix<MT,AF,false,false,CSAs...>::trim()
{
   for( size_t i=0UL; i<rows(); ++i )
      trim( i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removing all excessive capacity of a specific row/column of the sparse matrix.
//
// \param i The index of the row/column to be trimmed (\f$[0..M-1]\f$ or \f$[0..N-1]\f$).
// \return void
//
// This function can be used to reverse the effect of a row/column-specific reserve() call.
// It removes all excessive capacity from the specified row (in case of a rowMajor matrix)
// or column (in case of a columnMajor matrix). The excessive capacity is assigned to the
// subsequent row/column.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
void Submatrix<MT,AF,false,false,CSAs...>::trim( size_t i )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   matrix_.trim( row() + i );
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
/*!\brief Setting an element of the sparse submatrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be set.
// \return Iterator to the set element.
//
// This function sets the value of an element of the sparse submatrix. In case the sparse matrix
// already contains an element with row index \a i and column index \a j its value is modified,
// else a new element with the given \a value is inserted.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::set( size_t i, size_t j, const ElementType& value )
{
   return Iterator( matrix_.set( row()+i, column()+j, value ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Inserting an element into the sparse submatrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be inserted.
// \return Iterator to the newly inserted element.
// \exception std::invalid_argument Invalid sparse submatrix access index.
//
// This function inserts a new element into the sparse submatrix. However, duplicate elements are
// not allowed. In case the sparse submatrix already contains an element with row index \a i and
// column index \a j, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::insert( size_t i, size_t j, const ElementType& value )
{
   return Iterator( matrix_.insert( row()+i, column()+j, value ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Appending an element to the specified row/column of the sparse submatrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be appended.
// \param check \a true if the new value should be checked for default values, \a false if not.
// \return void
//
// This function provides a very efficient way to fill a sparse submatrix with elements. It appends
// a new element to the end of the specified row/column without any additional memory allocation.
// Therefore it is strictly necessary to keep the following preconditions in mind:
//
//  - the index of the new element must be strictly larger than the largest index of non-zero
//    elements in the specified row/column of the sparse submatrix
//  - the current number of non-zero elements in the submatrix must be smaller than the capacity
//    of the matrix
//
// Ignoring these preconditions might result in undefined behavior! The optional \a check
// parameter specifies whether the new value should be tested for a default value. If the new
// value is a default value (for instance 0 in case of an integral element type) the value is
// not appended. Per default the values are not tested.
//
// In combination with the reserve() and the finalize() function, append() provides the most
// efficient way to add new elements to a sparse submatrix:

   \code
   blaze::CompressedMatrix<double,blaze::rowMajor> A( 42, 54 );
   auto B = submatrix( A, 10, 10, 4, 3 );

   B.reserve( 3 );         // Reserving enough capacity for 3 non-zero elements
   B.append( 0, 1, 1.0 );  // Appending the value 1 in row 0 with column index 1
   B.finalize( 0 );        // Finalizing row 0
   B.append( 1, 1, 2.0 );  // Appending the value 2 in row 1 with column index 1
   B.finalize( 1 );        // Finalizing row 1
   B.finalize( 2 );        // Finalizing the empty row 2 to prepare row 3
   B.append( 3, 0, 3.0 );  // Appending the value 3 in row 3 with column index 0
   B.finalize( 3 );        // Finalizing row 3
   \endcode

// \note Although append() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,false,false,CSAs...>::append( size_t i, size_t j, const ElementType& value, bool check )
{
   if( column() + columns() == matrix_.columns() ) {
      matrix_.append( row() + i, column() + j, value, check );
   }
   else if( !check || !isDefault<strict>( value ) ) {
      matrix_.insert( row() + i, column() + j, value );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Finalizing the element insertion of a row/column.
//
// \param i The index of the row/column to be finalized \f$[0..M-1]\f$.
// \return void
//
// This function is part of the low-level interface to efficiently fill a submatrix with elements.
// After completion of row/column \a i via the append() function, this function can be called to
// finalize row/column \a i and prepare the next row/column for insertion process via append().
//
// \note Although finalize() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,false,false,CSAs...>::finalize( size_t i )
{
   matrix_.trim( row() + i );
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
/*!\brief Erasing an element from the sparse submatrix.
//
// \param i The row index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \return void
//
// This function erases an element from the sparse submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,false,false,CSAs...>::erase( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   matrix_.erase( row() + i, column() + j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing an element from the sparse submatrix.
//
// \param i The row/column index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param pos Iterator to the element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases an element from the sparse submatrix. In case the storage order is set
// to \a rowMajor the function erases an element from row \a i, in case the storage flag is set
// to \a columnMajor the function erases an element from column \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::erase( size_t i, Iterator pos )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   return Iterator( matrix_.erase( row()+i, pos.base() ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing a range of elements from the sparse submatrix.
//
// \param i The row/column index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param first Iterator to first element to be erased.
// \param last Iterator just past the last element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases a range of elements from the sparse submatrix. In case the storage order
// is set to \a rowMajor the function erases a range of elements element from row \a i, in case
// the storage flag is set to \a columnMajor the function erases a range of elements from column
// \a i.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::erase( size_t i, Iterator first, Iterator last )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   return Iterator( matrix_.erase( row()+i, first.base(), last.base() ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing specific elements from the sparse submatrix.
//
// \param predicate The unary predicate for the element selection.
// \return void.
//
// This function erases specific elements from the sparse submatrix. The elements are selected
// by the given unary predicate \a predicate, which is expected to accept a single argument of
// the type of the elements and to be pure. The following example demonstrates how to remove
// all elements that are smaller than a certain threshold value:

   \code
   blaze::CompressedMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   auto sm = submatrix( A, 4UL, 3UL, 5UL, 7UL );
   sm.erase( []( double value ){ return value < 1E-8; } );
   \endcode

// \note The predicate is required to be pure, i.e. to produce deterministic results for elements
// with the same value. The attempt to use an impure predicate leads to undefined behavior!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Pred >   // Type of the unary predicate
inline void Submatrix<MT,AF,false,false,CSAs...>::erase( Pred predicate )
{
   for( size_t i=0UL; i<rows(); ++i ) {
      matrix_.erase( row()+i, begin(i).base(), end(i).base(), predicate );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing specific elements from a range of the sparse submatrix.
//
// \param i The row index of the elements to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param first Iterator to first element of the range.
// \param last Iterator just past the last element of the range.
// \param predicate The unary predicate for the element selection.
// \return void
//
// This function erases specific elements from a range of elements of the sparse submatrix. The
// elements are selected by the given unary predicate \a predicate, which is expected to accept
// a single argument of the type of the elements and to be pure. In case the storage order is
// set to \a rowMajor the function erases a range of elements from row \a i, in case the storage
// flag is set to \a columnMajor the function erases a range of elements from column \a i. The
// following example demonstrates how to remove all elements that are smaller than a certain
// threshold value:

   \code
   blaze::CompressedMatrix<double,blaze::rowMajor> A;
   // ... Resizing and initialization

   auto sm = submatrix( A, 4UL, 3UL, 5UL, 7UL );
   sm.erase( 2UL, sm.begin(2UL), sm.end(2UL), []( double value ){ return value < 1E-8; } );
   \endcode

// \note The predicate is required to be pure, i.e. to produce deterministic results for elements
// with the same value. The attempt to use an impure predicate leads to undefined behavior!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Pred >   // Type of the unary predicate
inline void Submatrix<MT,AF,false,false,CSAs...>::erase( size_t i, Iterator first, Iterator last, Pred predicate )
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   matrix_.erase( row()+i, first.base(), last.base(), predicate );
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
/*!\brief Searches for a specific submatrix element.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// submatrix. It specifically searches for the element with row index \a i and column index
// \a j. In case the element is found, the function returns an row/column iterator to the
// element. Otherwise an iterator just past the last non-zero element of row \a i or column
// \a j (the end() iterator) is returned. Note that the returned sparse submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::find( size_t i, size_t j )
{
   const Iterator_t<MT> pos( matrix_.find( row() + i, column() + j ) );

   if( pos != matrix_.end( row() + i ) )
      return Iterator( pos, column() );
   else
      return end( i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Searches for a specific submatrix element.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// submatrix. It specifically searches for the element with row index \a i and column index
// \a j. In case the element is found, the function returns an row/column iterator to the
// element. Otherwise an iterator just past the last non-zero element of row \a i or column
// \a j (the end() iterator) is returned. Note that the returned sparse submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,false,false,CSAs...>::find( size_t i, size_t j ) const
{
   const ConstIterator_t<MT> pos( matrix_.find( row() + i, column() + j ) );

   if( pos != matrix_.end( row() + i ) )
      return ConstIterator( pos, column() );
   else
      return end( i );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// In case of a row-major submatrix, this function returns a row iterator to the first element
// with an index not less then the given column index. In case of a column-major submatrix, the
// function returns a column iterator to the first element with an index not less then the given
// row index. In combination with the upperBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::lowerBound( size_t i, size_t j )
{
   return Iterator( matrix_.lowerBound( row() + i, column() + j ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// In case of a row-major submatrix, this function returns a row iterator to the first element
// with an index not less then the given column index. In case of a column-major submatrix, the
// function returns a column iterator to the first element with an index not less then the given
// row index. In combination with the upperBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,false,false,CSAs...>::lowerBound( size_t i, size_t j ) const
{
   return ConstIterator( matrix_.lowerBound( row() + i, column() + j ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// In case of a row-major submatrix, this function returns a row iterator to the first element
// with an index greater then the given column index. In case of a column-major submatrix, the
// function returns a column iterator to the first element with an index greater then the given
// row index. In combination with the lowerBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::Iterator
   Submatrix<MT,AF,false,false,CSAs...>::upperBound( size_t i, size_t j )
{
   return Iterator( matrix_.upperBound( row() + i, column() + j ), column() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// In case of a row-major submatrix, this function returns a row iterator to the first element
// with an index greater then the given column index. In case of a column-major submatrix, the
// function returns a column iterator to the first element with an index greater then the given
// row index. In combination with the lowerBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,false,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,false,false,CSAs...>::upperBound( size_t i, size_t j ) const
{
   return ConstIterator( matrix_.upperBound( row() + i, column() + j ), column() );
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
/*!\brief In-place transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the sparse submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::transpose()
{
   using blaze::assign;

   if( rows() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, trans( *this ), row(), column() ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   decltype(auto) left( derestrict( *this ) );
   const ResultType tmp( trans( *this ) );

   reset();
   assign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the sparse submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::ctranspose()
{
   using blaze::assign;

   if( rows() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, trans( *this ), row(), column() ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   decltype(auto) left( derestrict( *this ) );
   const ResultType tmp( ctrans( *this ) );

   reset();
   assign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the sparse submatrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the submatrix scaling.
// \return Reference to the sparse submatrix.
//
// This function scales the submatrix by applying the given scalar value \a scalar to each
// element of the submatrix. For built-in and \c complex data types it has the same effect
// as using the multiplication assignment operator. Note that the function cannot be used
// to scale a submatrix on a lower or upper unitriangular matrix. The attempt to scale
// such a submatrix results in a compile time error!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Other >  // Data type of the scalar value
inline Submatrix<MT,AF,false,false,CSAs...>&
   Submatrix<MT,AF,false,false,CSAs...>::scale( const Other& scalar )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   for( size_t i=0UL; i<rows(); ++i ) {
      const Iterator last( end(i) );
      for( Iterator element=begin(i); element!=last; ++element )
         element->value() *= scalar;
   }

   return *this;
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
/*!\brief Returns whether the submatrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,AF,false,false,CSAs...>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.isAliased( &unview( *alias ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,AF,false,false,CSAs...>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( &unview( *alias ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can be used in SMP assignments.
//
// \return \a true in case the submatrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the submatrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the matrix).
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline bool Submatrix<MT,AF,false,false,CSAs...>::canSMPAssign() const noexcept
{
   return false;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side dense matrix
        , bool SO >         // Storage order of the right-hand side dense matrix
inline void Submatrix<MT,AF,false,false,CSAs...>::assign( const DenseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   reserve( 0UL, rows() * columns() );

   for( size_t i=0UL; i<rows(); ++i ) {
      for( size_t j=0UL; j<columns(); ++j ) {
         if( IsSymmetric_v<MT> || IsHermitian_v<MT> ) {
            const ElementType& value( (*rhs)(i,j) );
            if( !isDefault<strict>( value ) )
               set( i, j, value );
         }
         else {
            append( i, j, (*rhs)(i,j), true );
         }
      }
      finalize( i );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2 >    // Type of the right-hand side sparse matrix
inline void Submatrix<MT,AF,false,false,CSAs...>::assign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   reserve( 0UL, (*rhs).nonZeros() );

   for( size_t i=0UL; i<(*rhs).rows(); ++i ) {
      for( ConstIterator_t<MT2> element=(*rhs).begin(i); element!=(*rhs).end(i); ++element ) {
         if( IsSymmetric_v<MT> || IsHermitian_v<MT> ) {
            const ElementType& value( element->value() );
            if( !isDefault<strict>( value ) )
               set( i, element->index(), value );
         }
         else {
            append( i, element->index(), element->value(), true );
         }
      }
      finalize( i );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2 >    // Type of the right-hand side sparse matrix
inline void Submatrix<MT,AF,false,false,CSAs...>::assign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   // Counting the number of elements per row
   std::vector<size_t> rowLengths( rows(), 0UL );
   for( size_t j=0UL; j<columns(); ++j ) {
      for( auto element=(*rhs).begin(j); element!=(*rhs).end(j); ++element )
         ++rowLengths[element->index()];
   }

   // Resizing the sparse matrix
   for( size_t i=0UL; i<rows(); ++i ) {
      reserve( i, rowLengths[i] );
   }

   // Appending the elements to the rows of the sparse submatrix
   for( size_t j=0UL; j<columns(); ++j ) {
      for( auto element=(*rhs).begin(j); element!=(*rhs).end(j); ++element )
         if( IsSymmetric_v<MT> || IsHermitian_v<MT> ) {
            const ElementType& value( element->value() );
            if( !isDefault<strict>( value ) )
               set( element->index(), j, value );
         }
         else {
            append( element->index(), j, element->value(), true );
         }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a matrix.
//
// \param rhs The right-hand side matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline void Submatrix<MT,AF,false,false,CSAs...>::addAssign( const Matrix<MT2,SO>& rhs )
{
   using AddType = AddTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   const AddType tmp( serial( *this + (*rhs) ) );
   reset();
   assign( tmp );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a matrix.
//
// \param rhs The right-hand side matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline void Submatrix<MT,AF,false,false,CSAs...>::subAssign( const Matrix<MT2,SO>& rhs )
{
   using SubType = SubTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   const SubType tmp( serial( *this - (*rhs) ) );
   reset();
   assign( tmp );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a matrix.
//
// \param rhs The right-hand side matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline void Submatrix<MT,AF,false,false,CSAs...>::schurAssign( const Matrix<MT2,SO>& rhs )
{
   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( SchurType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   const SchurType tmp( serial( *this % (*rhs) ) );
   reset();
   assign( tmp );
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR COLUMN-MAJOR SPARSE MATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of Submatrix for column-major sparse submatrices.
// \ingroup submatrix
//
// This specialization of Submatrix adapts the class template to the requirements of column-major
// sparse submatrices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
class Submatrix<MT,AF,true,false,CSAs...>
   : public View< SparseMatrix< Submatrix<MT,AF,true,false,CSAs...>, true > >
   , private SubmatrixData<CSAs...>
{
 private:
   //**Type definitions****************************************************************************
   using DataType = SubmatrixData<CSAs...>;               //!< The type of the SubmatrixData base class.
   using Operand  = If_t< IsExpression_v<MT>, MT, MT& >;  //!< Composite data type of the matrix expression.
   //**********************************************************************************************

 public:
   //**Type definitions****************************************************************************
   //! Type of this Submatrix instance.
   using This = Submatrix<MT,AF,true,false,CSAs...>;

   //! Base type of this Submatrix instance.
   using BaseType = View< SparseMatrix<This,true> >;

   using ViewedType    = MT;                            //!< The type viewed by this Submatrix instance.
   using ResultType    = SubmatrixTrait_t<MT,CSAs...>;  //!< Result type for expression template evaluations.
   using OppositeType  = OppositeType_t<ResultType>;    //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = TransposeType_t<ResultType>;   //!< Transpose type for expression template evaluations.
   using ElementType   = ElementType_t<MT>;             //!< Type of the submatrix elements.
   using ReturnType    = ReturnType_t<MT>;              //!< Return type for expression template evaluations
   using CompositeType = const Submatrix&;              //!< Data type for composite expression templates.

   //! Reference to a constant submatrix value.
   using ConstReference = ConstReference_t<MT>;

   //! Reference to a non-constant submatrix value.
   using Reference = If_t< IsConst_v<MT>, ConstReference, Reference_t<MT> >;
   //**********************************************************************************************

   //**SubmatrixElement class definition***********************************************************
   /*!\brief Access proxy for a specific element of the sparse submatrix.
   */
   template< typename MatrixType      // Type of the sparse matrix
           , typename IteratorType >  // Type of the sparse matrix iterator
   class SubmatrixElement
      : private SparseElement
   {
    public:
      //**Constructor******************************************************************************
      /*!\brief Constructor for the SubmatrixElement class.
      //
      // \param pos Iterator to the current position within the sparse submatrix.
      // \param offset The offset within the according row/column of the sparse matrix.
      */
      inline SubmatrixElement( IteratorType pos, size_t offset )
         : pos_   ( pos    )  // Iterator to the current position within the sparse submatrix
         , offset_( offset )  // Row offset within the according sparse matrix
      {}
      //*******************************************************************************************

      //**Assignment operator**********************************************************************
      /*!\brief Assignment to the accessed sparse submatrix element.
      //
      // \param v The new value of the sparse submatrix element.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator=( const T& v ) {
         *pos_ = v;
         return *this;
      }
      //*******************************************************************************************

      //**Addition assignment operator*************************************************************
      /*!\brief Addition assignment to the accessed sparse submatrix element.
      //
      // \param v The right-hand side value for the addition.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator+=( const T& v ) {
         *pos_ += v;
         return *this;
      }
      //*******************************************************************************************

      //**Subtraction assignment operator**********************************************************
      /*!\brief Subtraction assignment to the accessed sparse submatrix element.
      //
      // \param v The right-hand side value for the subtraction.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator-=( const T& v ) {
         *pos_ -= v;
         return *this;
      }
      //*******************************************************************************************

      //**Multiplication assignment operator*******************************************************
      /*!\brief Multiplication assignment to the accessed sparse submatrix element.
      //
      // \param v The right-hand side value for the multiplication.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator*=( const T& v ) {
         *pos_ *= v;
         return *this;
      }
      //*******************************************************************************************

      //**Division assignment operator*************************************************************
      /*!\brief Division assignment to the accessed sparse submatrix element.
      //
      // \param v The right-hand side value for the division.
      // \return Reference to the sparse submatrix element.
      */
      template< typename T > inline SubmatrixElement& operator/=( const T& v ) {
         *pos_ /= v;
         return *this;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the sparse submatrix element at the current iterator position.
      //
      // \return Reference to the sparse submatrix element at the current iterator position.
      */
      inline const SubmatrixElement* operator->() const {
         return this;
      }
      //*******************************************************************************************

      //**Value function***************************************************************************
      /*!\brief Access to the current value of the sparse submatrix element.
      //
      // \return The current value of the sparse submatrix element.
      */
      inline decltype(auto) value() const {
         return pos_->value();
      }
      //*******************************************************************************************

      //**Index function***************************************************************************
      /*!\brief Access to the current index of the sparse element.
      //
      // \return The current index of the sparse element.
      */
      inline size_t index() const {
         return pos_->index() - offset_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType pos_;  //!< Iterator to the current position within the sparse submatrix.
      size_t offset_;     //!< Offset within the according row/column of the sparse matrix.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**SubmatrixIterator class definition**********************************************************
   /*!\brief Iterator over the elements of the sparse submatrix.
   */
   template< typename MatrixType      // Type of the sparse matrix
           , typename IteratorType >  // Type of the sparse matrix iterator
   class SubmatrixIterator
   {
    public:
      //**Type definitions*************************************************************************
      using IteratorCategory = std::forward_iterator_tag;                  //!< The iterator category.
      using ValueType        = SubmatrixElement<MatrixType,IteratorType>;  //!< Type of the underlying elements.
      using PointerType      = ValueType;                                  //!< Pointer return type.
      using ReferenceType    = ValueType;                                  //!< Reference return type.
      using DifferenceType   = ptrdiff_t;                                  //!< Difference between two iterators.

      // STL iterator requirements
      using iterator_category = IteratorCategory;  //!< The iterator category.
      using value_type        = ValueType;         //!< Type of the underlying elements.
      using pointer           = PointerType;       //!< Pointer return type.
      using reference         = ReferenceType;     //!< Reference return type.
      using difference_type   = DifferenceType;    //!< Difference between two iterators.
      //*******************************************************************************************

      //**Default constructor**********************************************************************
      /*!\brief Default constructor for the SubmatrixIterator class.
      */
      inline SubmatrixIterator()
         : pos_   ()  // Iterator to the current sparse element
         , offset_()  // The offset of the according row/column of the sparse matrix
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Constructor for the SubmatrixIterator class.
      //
      // \param iterator Iterator to the current sparse element.
      // \param index The starting index within the according row/column of the sparse matrix.
      */
      inline SubmatrixIterator( IteratorType iterator, size_t index )
         : pos_   ( iterator )  // Iterator to the current sparse element
         , offset_( index    )  // The offset of the according row/column of the sparse matrix
      {}
      //*******************************************************************************************

      //**Constructor******************************************************************************
      /*!\brief Conversion constructor from different SubmatrixIterator instances.
      //
      // \param it The submatrix iterator to be copied.
      */
      template< typename MatrixType2, typename IteratorType2 >
      inline SubmatrixIterator( const SubmatrixIterator<MatrixType2,IteratorType2>& it )
         : pos_   ( it.base()   )  // Iterator to the current sparse element.
         , offset_( it.offset() )  // The offset of the according row/column of the sparse matrix
      {}
      //*******************************************************************************************

      //**Prefix increment operator****************************************************************
      /*!\brief Pre-increment operator.
      //
      // \return Reference to the incremented iterator.
      */
      inline SubmatrixIterator& operator++() {
         ++pos_;
         return *this;
      }
      //*******************************************************************************************

      //**Postfix increment operator***************************************************************
      /*!\brief Post-increment operator.
      //
      // \return The previous position of the iterator.
      */
      inline const SubmatrixIterator operator++( int ) {
         const SubmatrixIterator tmp( *this );
         ++(*this);
         return tmp;
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the current sparse submatrix element.
      //
      // \return Reference to the current sparse submatrix element.
      */
      inline ReferenceType operator*() const {
         return ReferenceType( pos_, offset_ );
      }
      //*******************************************************************************************

      //**Element access operator******************************************************************
      /*!\brief Direct access to the current sparse submatrix element.
      //
      // \return Pointer to the current sparse submatrix element.
      */
      inline PointerType operator->() const {
         return PointerType( pos_, offset_ );
      }
      //*******************************************************************************************

      //**Equality operator************************************************************************
      /*!\brief Equality comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side submatrix iterator.
      // \return \a true if the iterators refer to the same element, \a false if not.
      */
      template< typename MatrixType2, typename IteratorType2 >
      inline bool operator==( const SubmatrixIterator<MatrixType2,IteratorType2>& rhs ) const {
         return base() == rhs.base();
      }
      //*******************************************************************************************

      //**Inequality operator**********************************************************************
      /*!\brief Inequality comparison between two SubmatrixIterator objects.
      //
      // \param rhs The right-hand side submatrix iterator.
      // \return \a true if the iterators don't refer to the same element, \a false if they do.
      */
      template< typename MatrixType2, typename IteratorType2 >
      inline bool operator!=( const SubmatrixIterator<MatrixType2,IteratorType2>& rhs ) const {
         return !( *this == rhs );
      }
      //*******************************************************************************************

      //**Subtraction operator*********************************************************************
      /*!\brief Calculating the number of elements between two submatrix iterators.
      //
      // \param rhs The right-hand side submatrix iterator.
      // \return The number of elements between the two submatrix iterators.
      */
      inline DifferenceType operator-( const SubmatrixIterator& rhs ) const {
         return pos_ - rhs.pos_;
      }
      //*******************************************************************************************

      //**Base function****************************************************************************
      /*!\brief Access to the current position of the submatrix iterator.
      //
      // \return The current position of the submatrix iterator.
      */
      inline IteratorType base() const {
         return pos_;
      }
      //*******************************************************************************************

      //**Offset function**************************************************************************
      /*!\brief Access to the offset of the submatrix iterator.
      //
      // \return The offset of the submatrix iterator.
      */
      inline size_t offset() const noexcept {
         return offset_;
      }
      //*******************************************************************************************

    private:
      //**Member variables*************************************************************************
      IteratorType pos_;     //!< Iterator to the current sparse element.
      size_t       offset_;  //!< The offset of the according row/column of the sparse matrix.
      //*******************************************************************************************
   };
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   //! Iterator over constant elements.
   using ConstIterator = SubmatrixIterator< const MT, ConstIterator_t<MT> >;

   //! Iterator over non-constant elements.
   using Iterator = If_t< IsConst_v<MT>, ConstIterator, SubmatrixIterator< MT, Iterator_t<MT> > >;
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation switch for the expression template assignment strategy.
   static constexpr bool smpAssignable = MT::smpAssignable;

   //! Compilation switch for the expression template evaluation strategy.
   static constexpr bool compileTimeArgs = DataType::compileTimeArgs;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   template< typename... RSAs >
   explicit inline Submatrix( MT& matrix, RSAs... args );

   Submatrix( const Submatrix& ) = default;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   ~Submatrix() = default;
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
   inline Submatrix& operator=( initializer_list< initializer_list<ElementType> > list );
   inline Submatrix& operator=( const Submatrix& rhs );

   template< typename MT2, bool SO > inline Submatrix& operator= ( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline Submatrix& operator+=( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline Submatrix& operator-=( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline Submatrix& operator%=( const Matrix<MT2,SO>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   using DataType::row;
   using DataType::column;
   using DataType::rows;
   using DataType::columns;

   inline MT&       operand() noexcept;
   inline const MT& operand() const noexcept;

   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t i ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i ) const;
   inline void   reset();
   inline void   reset( size_t i );
   inline void   reserve( size_t nonzeros );
          void   reserve( size_t i, size_t nonzeros );
   inline void   trim();
   inline void   trim( size_t j );
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
   inline Submatrix& transpose();
   inline Submatrix& ctranspose();

   template< typename Other > inline Submatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   inline bool canSMPAssign() const noexcept;

   template< typename MT2, bool SO > inline void assign     ( const DenseMatrix<MT2,SO>& rhs );
   template< typename MT2 >          inline void assign     ( const SparseMatrix<MT2,true>& rhs );
   template< typename MT2 >          inline void assign     ( const SparseMatrix<MT2,false>& rhs );
   template< typename MT2, bool SO > inline void addAssign  ( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline void subAssign  ( const Matrix<MT2,SO>& rhs );
   template< typename MT2, bool SO > inline void schurAssign( const Matrix<MT2,SO>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   Operand matrix_;  //!< The matrix containing the submatrix.
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE    ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_TRANSEXPR_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SUBMATRIX_TYPE      ( MT );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE        ( MT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE      ( MT );
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
/*!\brief Constructor for column-major sparse submatrices.
//
// \param matrix The sparse matrix containing the submatrix.
// \param args The runtime submatrix arguments.
// \exception std::invalid_argument Invalid submatrix specification.
//
// By default, the provided submatrix arguments are checked at runtime. In case the submatrix is
// not properly specified (i.e. if the specified submatrix is not contained in the given sparse
// matrix) a \a std::invalid_argument exception is thrown. The checks can be skipped by providing
// the optional \a blaze::unchecked argument.
*/
template< typename MT         // Type of the sparse matrix
        , AlignmentFlag AF    // Alignment flag
        , size_t... CSAs >    // Compile time submatrix arguments
template< typename... RSAs >  // Runtime submatrix arguments
inline Submatrix<MT,AF,true,false,CSAs...>::Submatrix( MT& matrix, RSAs... args )
   : DataType( args... )  // Base class initialization
   , matrix_ ( matrix  )  // The matrix containing the submatrix
{
   if( isChecked( args... ) ) {
      if( ( row() + rows() > matrix_.rows() ) || ( column() + columns() > matrix_.columns() ) ) {
         BLAZE_THROW_INVALID_ARGUMENT( "Invalid submatrix specification" );
      }
   }
   else {
      BLAZE_USER_ASSERT( row()    + rows()    <= matrix_.rows()   , "Invalid submatrix specification" );
      BLAZE_USER_ASSERT( column() + columns() <= matrix_.columns(), "Invalid submatrix specification" );
   }
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
/*!\brief 2D-access to the sparse submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Reference
   Submatrix<MT,AF,true,false,CSAs...>::operator()( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return matrix_(row()+i,column()+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the sparse submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstReference
   Submatrix<MT,AF,true,false,CSAs...>::operator()( size_t i, size_t j ) const
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   return const_cast<const MT&>( matrix_ )(row()+i,column()+j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Reference
   Submatrix<MT,AF,true,false,CSAs...>::at( size_t i, size_t j )
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
/*!\brief Checked access to the submatrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the function call operator this function always performs a check of the given
// access indices.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstReference
   Submatrix<MT,AF,true,false,CSAs...>::at( size_t i, size_t j ) const
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
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
   Submatrix<MT,AF,true,false,CSAs...>::begin( size_t j )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid sparse submatrix column access index" );

   if( row() == 0UL )
      return Iterator( matrix_.begin( j + column() ), row() );
   else
      return Iterator( matrix_.lowerBound( row(), j + column() ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,true,false,CSAs...>::begin( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid sparse submatrix column access index" );

   if( row() == 0UL )
      return ConstIterator( matrix_.cbegin( j + column() ), row() );
   else
      return ConstIterator( matrix_.lowerBound( row(), j + column() ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator to the first non-zero element of column \a j.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,true,false,CSAs...>::cbegin( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid sparse submatrix column access index" );

   if( row() == 0UL )
      return ConstIterator( matrix_.cbegin( j + column() ), row() );
   else
      return ConstIterator( matrix_.lowerBound( row(), j + column() ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
  Submatrix<MT,AF,true,false,CSAs...>::end( size_t j )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid sparse submatrix column access index" );

   if( matrix_.rows() == row() + rows() )
      return Iterator( matrix_.end( j + column() ), row() );
   else
      return Iterator( matrix_.lowerBound( row() + rows(), j + column() ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,true,false,CSAs...>::end( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid sparse submatrix column access index" );

   if( matrix_.rows() == row() + rows() )
      return ConstIterator( matrix_.cend( j + column() ), row() );
   else
      return ConstIterator( matrix_.lowerBound( row() + rows(), j + column() ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last non-zero element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last non-zero element of column \a j.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,true,false,CSAs...>::cend( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid sparse submatrix column access index" );

   if( matrix_.rows() == row() + rows() )
      return ConstIterator( matrix_.cend( j + column() ), row() );
   else
      return ConstIterator( matrix_.lowerBound( row() + rows(), j + column() ), row() );
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
/*!\brief List assignment to all submatrix elements.
//
// \param list The initializer list.
// \exception std::invalid_argument Invalid assignment to submatrix.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// This assignment operator offers the option to directly assign to all elements of the submatrix
// by means of an initializer list. The submatrix elements are assigned the values from the given
// initializer list. Missing values are initialized as default. Note that in case the size of the
// top-level initializer list does not match the number of rows of the submatrix or the size of
// any nested list exceeds the number of columns, a \a std::invalid_argument exception is thrown.
// Also, if the underlying matrix \a MT is restricted and the assignment would violate an
// invariant of the matrix, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::operator=( initializer_list< initializer_list<ElementType> > list )
{
   using blaze::assign;

   if( list.size() != rows() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to submatrix" );
   }

   const InitializerMatrix<ElementType> tmp( list, columns() );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   left.reset();
   assign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for Submatrix.
//
// \param rhs Sparse submatrix to be copied.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Submatrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The sparse submatrix is initialized as a copy of the given sparse submatrix. In case the
// current sizes of the two submatrices don't match, a \a std::invalid_argument exception is
// thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::operator=( const Submatrix& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );

   if( this == &rhs || ( &matrix_ == &rhs.matrix_ && row() == rhs.row() && column() == rhs.column() ) )
      return *this;

   if( rows() != rhs.rows() || columns() != rhs.columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Submatrix sizes do not match" );
   }

   if( !tryAssign( matrix_, rhs, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( rhs.canAlias( this ) ) {
      const ResultType tmp( rhs );
      left.reset();
      assign( left, tmp );
   }
   else {
      left.reset();
      assign( left, rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be assigned.
// \return Reference to the assigned submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// The sparse submatrix is initialized as a copy of the given matrix. In case the current sizes
// of the two matrices don't match, a \a std::invalid_argument exception is thrown. Also, if
// the underlying matrix \a MT is a lower triangular, upper triangular, or symmetric matrix
// and the assignment would violate its lower, upper, or symmetry property, respectively, a
// \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::operator=( const Matrix<MT2,SO>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   if( rows() != (*rhs).rows() || columns() != (*rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   using Right = CompositeType_t<MT2>;
   Right right( *rhs );

   if( !tryAssign( matrix_, right, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   if( IsReference_v<Right> && right.canAlias( this ) ) {
      const ResultType_t<MT2> tmp( right );
      left.reset();
      assign( left, transIf< IsSymmetric_v<This> >( tmp ) );
   }
   else {
      left.reset();
      assign( left, transIf< IsSymmetric_v<This> >( right ) );
   }

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the submatrix.
// \return Reference to the sparse submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::operator+=( const Matrix<MT2,SO>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using AddType = AddTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   if( rows() != (*rhs).rows() || columns() != (*rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const AddType tmp( *this + (*rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   left.reset();
   assign( left, transIf< IsSymmetric_v<This> >( tmp ) );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the submatrix.
// \return Reference to the sparse submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::operator-=( const Matrix<MT2,SO>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SubType = SubTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   if( rows() != (*rhs).rows() || columns() != (*rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SubType tmp( *this - (*rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   left.reset();
   assign( left, transIf< IsSymmetric_v<This> >( tmp ) );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a matrix (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side matrix for the Schur product.
// \return Reference to the sparse submatrix.
// \exception std::invalid_argument Matrix sizes do not match.
// \exception std::invalid_argument Invalid assignment to restricted matrix.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown. Also, if the underlying matrix \a MT is a lower triangular, upper triangular, or
// symmetric matrix and the assignment would violate its lower, upper, or symmetry property,
// respectively, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::operator%=( const Matrix<MT2,SO>& rhs )
{
   using blaze::assign;

   BLAZE_CONSTRAINT_MUST_BE_SPARSE_MATRIX_TYPE ( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<MT2> );

   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   if( rows() != (*rhs).rows() || columns() != (*rhs).columns() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   const SchurType tmp( *this % (*rhs) );

   if( !tryAssign( matrix_, tmp, row(), column() ) ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid assignment to restricted matrix" );
   }

   decltype(auto) left( derestrict( *this ) );

   left.reset();
   assign( left, transIf< IsSymmetric_v<This> >( tmp ) );

   BLAZE_INTERNAL_ASSERT( isIntact( matrix_ ), "Invariant violation detected" );

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
/*!\brief Returns the matrix containing the submatrix.
//
// \return The matrix containing the submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline MT& Submatrix<MT,AF,true,false,CSAs...>::operand() noexcept
{
   return matrix_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the matrix containing the submatrix.
//
// \return The matrix containing the submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline const MT& Submatrix<MT,AF,true,false,CSAs...>::operand() const noexcept
{
   return matrix_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the sparse submatrix.
//
// \return The capacity of the sparse submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline size_t Submatrix<MT,AF,true,false,CSAs...>::capacity() const noexcept
{
   return nonZeros() + matrix_.capacity() - matrix_.nonZeros();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current capacity of the specified column.
//
// \param j The index of the column.
// \return The current capacity of column \a j.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline size_t Submatrix<MT,AF,true,false,CSAs...>::capacity( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   return nonZeros( j ) + matrix_.capacity( column()+j ) - matrix_.nonZeros( column()+j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the sparse submatrix
//
// \return The number of non-zero elements in the sparse submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline size_t Submatrix<MT,AF,true,false,CSAs...>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t i=0UL; i<columns(); ++i )
      nonzeros += nonZeros( i );

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the specified column.
//
// \param j The index of the column.
// \return The number of non-zero elements of column \a j.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline size_t Submatrix<MT,AF,true,false,CSAs...>::nonZeros( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   return end(j) - begin(j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,true,false,CSAs...>::reset()
{
   for( size_t j=column(); j<column()+columns(); ++j )
   {
      const size_t ibegin( ( IsLower_v<MT> )
                           ?( ( IsUniLower_v<MT> || IsStrictlyLower_v<MT> )
                              ?( max( j+1UL, row() ) )
                              :( max( j, row() ) ) )
                           :( row() ) );
      const size_t iend  ( ( IsUpper_v<MT> )
                           ?( ( IsUniUpper_v<MT> || IsStrictlyUpper_v<MT> )
                              ?( min( j, row()+rows() ) )
                              :( min( j+1UL, row()+rows() ) ) )
                           :( row()+rows() ) );

      matrix_.erase( j, matrix_.lowerBound( ibegin, j ), matrix_.lowerBound( iend, j ) );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified column to the default initial values.
//
// \param j The index of the column.
// \return void
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,true,false,CSAs...>::reset( size_t j )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   const size_t index( column() + j );

   const size_t ibegin( ( IsLower_v<MT> )
                        ?( ( IsUniLower_v<MT> || IsStrictlyLower_v<MT> )
                           ?( max( j+1UL, row() ) )
                           :( max( j, row() ) ) )
                        :( row() ) );
   const size_t iend  ( ( IsUpper_v<MT> )
                        ?( ( IsUniUpper_v<MT> || IsStrictlyUpper_v<MT> )
                           ?( min( j, row()+rows() ) )
                           :( min( j+1UL, row()+rows() ) ) )
                        :( row()+rows() ) );

   matrix_.erase( index, matrix_.lowerBound( ibegin, index ), matrix_.lowerBound( iend, index ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of the sparse submatrix.
//
// \param nonzeros The new minimum capacity of the sparse submatrix.
// \return void
//
// This function increases the capacity of the sparse submatrix to at least \a nonzeros elements.
// The current values of the submatrix elements and the individual capacities of the submatrix
// rows are preserved.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,true,false,CSAs...>::reserve( size_t nonzeros )
{
   const size_t current( capacity() );

   if( nonzeros > current ) {
      matrix_.reserve( matrix_.capacity() + nonzeros - current );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of a specific column of the sparse submatrix.
//
// \param j The column index of the new element \f$[0..M-1]\f$ or \f$[0..N-1]\f$.
// \param nonzeros The new minimum capacity of the specified column.
// \return void
//
// This function increases the capacity of column \a i of the sparse submatrix to at least
// \a nonzeros elements, but not beyond the current number of rows. The current values of
// the sparse submatrix and all other individual row/column capacities are preserved.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
void Submatrix<MT,AF,true,false,CSAs...>::reserve( size_t j, size_t nonzeros )
{
   const size_t current( capacity( j ) );
   const size_t index  ( column() + j );

   if( nonzeros > current ) {
      matrix_.reserve( index, matrix_.capacity( index ) + nonzeros - current );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removing all excessive capacity from all columns.
//
// \return void
//
// The trim() function can be used to reverse the effect of all column-specific reserve() calls
// It removes all excessive capacity from all columns. Note that this function does not remove
// the overall capacity but only reduces the capacity per column.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
void Submatrix<MT,AF,true,false,CSAs...>::trim()
{
   for( size_t j=0UL; j<columns(); ++j )
      trim( j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Removing all excessive capacity of a specific column of the sparse matrix.
//
// \param j The index of the column to be trimmed (\f$[0..N-1]\f$).
// \return void
//
// This function can be used to reverse the effect of a column-specific reserve() call. It
// removes all excessive capacity from the specified column. The excessive capacity is assigned
// to the subsequent column.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
void Submatrix<MT,AF,true,false,CSAs...>::trim( size_t j )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   matrix_.trim( column() + j );
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
/*!\brief Setting an element of the sparse submatrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be set.
// \return Iterator to the set element.
//
// This function sets the value of an element of the sparse submatrix. In case the sparse matrix
// already contains an element with row index \a i and column index \a j its value is modified,
// else a new element with the given \a value is inserted.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
   Submatrix<MT,AF,true,false,CSAs...>::set( size_t i, size_t j, const ElementType& value )
{
   return Iterator( matrix_.set( row()+i, column()+j, value ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Inserting an element into the sparse submatrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be inserted.
// \return Iterator to the newly inserted element.
// \exception std::invalid_argument Invalid sparse submatrix access index.
//
// This function inserts a new element into the sparse submatrix. However, duplicate elements are
// not allowed. In case the sparse submatrix already contains an element with row index \a i and
// column index \a j, a \a std::invalid_argument exception is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
   Submatrix<MT,AF,true,false,CSAs...>::insert( size_t i, size_t j, const ElementType& value )
{
   return Iterator( matrix_.insert( row()+i, column()+j, value ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Appending an element to the specified row/column of the sparse submatrix.
//
// \param i The row index of the new element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the new element. The index has to be in the range \f$[0..N-1]\f$.
// \param value The value of the element to be appended.
// \param check \a true if the new value should be checked for default values, \a false if not.
// \return void
//
// This function provides a very efficient way to fill a sparse submatrix with elements. It appends
// a new element to the end of the specified row/column without any additional memory allocation.
// Therefore it is strictly necessary to keep the following preconditions in mind:
//
//  - the index of the new element must be strictly larger than the largest index of non-zero
//    elements in the specified row/column of the sparse submatrix
//  - the current number of non-zero elements in the submatrix must be smaller than the capacity
//    of the matrix
//
// Ignoring these preconditions might result in undefined behavior! The optional \a check
// parameter specifies whether the new value should be tested for a default value. If the new
// value is a default value (for instance 0 in case of an integral element type) the value is
// not appended. Per default the values are not tested.
//
// In combination with the reserve() and the finalize() function, append() provides the most
// efficient way to add new elements to a sparse submatrix:

   \code
   blaze::CompressedMatrix<double,blaze::rowMajor> A( 42, 54 );
   auto B = submatrix( A, 10, 10, 4, 3 );

   B.reserve( 3 );         // Reserving enough capacity for 3 non-zero elements
   B.append( 0, 1, 1.0 );  // Appending the value 1 in row 0 with column index 1
   B.finalize( 0 );        // Finalizing row 0
   B.append( 1, 1, 2.0 );  // Appending the value 2 in row 1 with column index 1
   B.finalize( 1 );        // Finalizing row 1
   B.finalize( 2 );        // Finalizing the empty row 2 to prepare row 3
   B.append( 3, 0, 3.0 );  // Appending the value 3 in row 3 with column index 0
   B.finalize( 3 );        // Finalizing row 3
   \endcode

// \note Although append() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,true,false,CSAs...>::append( size_t i, size_t j, const ElementType& value, bool check )
{
   if( row() + rows() == matrix_.rows() ) {
      matrix_.append( row() + i, column() + j, value, check );
   }
   else if( !check || !isDefault<strict>( value ) ) {
      matrix_.insert( row() + i, column() + j, value );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Finalizing the element insertion of a column.
//
// \param j The index of the column to be finalized \f$[0..M-1]\f$.
// \return void
//
// This function is part of the low-level interface to efficiently fill a submatrix with elements.
// After completion of column \a j via the append() function, this function can be called to
// finalize column \a j and prepare the next column for insertion process via append().
//
// \note Although finalize() does not allocate new memory, it still invalidates all iterators
// returned by the end() functions!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,true,false,CSAs...>::finalize( size_t j )
{
   matrix_.trim( column() + j );
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
/*!\brief Erasing an element from the sparse submatrix.
//
// \param i The row index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the element to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \return void
//
// This function erases an element from the sparse submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline void Submatrix<MT,AF,true,false,CSAs...>::erase( size_t i, size_t j )
{
   BLAZE_USER_ASSERT( i < rows()   , "Invalid row access index"    );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   matrix_.erase( row() + i, column() + j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing an element from the sparse submatrix.
//
// \param j The column index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param pos Iterator to the element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases an element from column \a j of the sparse submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
   Submatrix<MT,AF,true,false,CSAs...>::erase( size_t j, Iterator pos )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   return Iterator( matrix_.erase( column()+j, pos.base() ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing a range of elements from the sparse submatrix.
//
// \param j The column index of the element to be erased. The index has to be in the range \f$[0..M-1]\f$.
// \param first Iterator to first element to be erased.
// \param last Iterator just past the last element to be erased.
// \return Iterator to the element after the erased element.
//
// This function erases a range of elements from column \a j of the sparse submatrix.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
   Submatrix<MT,AF,true,false,CSAs...>::erase( size_t j, Iterator first, Iterator last )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   return Iterator( matrix_.erase( column()+j, first.base(), last.base() ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing specific elements from the sparse submatrix.
//
// \param predicate The unary predicate for the element selection.
// \return void.
//
// This function erases specific elements from the sparse submatrix. The elements are selected
// by the given unary predicate \a predicate, which is expected to accept a single argument of
// the type of the elements and to be pure. The following example demonstrates how to remove
// all elements that are smaller than a certain threshold value:

   \code
   blaze::CompressedMatrix<double,blaze::columnMajor> A;
   // ... Resizing and initialization

   auto sm = submatrix( A, 4UL, 3UL, 5UL, 7UL );
   sm.erase( []( double value ){ return value < 1E-8; } );
   \endcode

// \note The predicate is required to be pure, i.e. to produce deterministic results for elements
// with the same value. The attempt to use an impure predicate leads to undefined behavior!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Pred >   // Type of the unary predicate
inline void Submatrix<MT,AF,true,false,CSAs...>::erase( Pred predicate )
{
   for( size_t j=0UL; j<columns(); ++j ) {
      matrix_.erase( column()+j, begin(j).base(), end(j).base(), predicate );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Erasing specific elements from a range of the sparse submatrix.
//
// \param j The column index of the elements to be erased. The index has to be in the range \f$[0..N-1]\f$.
// \param first Iterator to first element of the range.
// \param last Iterator just past the last element of the range.
// \param predicate The unary predicate for the element selection.
// \return void
//
// This function erases specific elements from a range of elements of the sparse submatrix.
// The elements are selected by the given unary predicate \a predicate, which is expected to
// accept a single argument of the type of the elements and to be pure. The following example
// demonstrates how to remove all elements that are smaller than a certain threshold value:

   \code
   blaze::CompressedMatrix<double,blaze::columnMajor> A;
   // ... Resizing and initialization

   auto sm = submatrix( A, 4UL, 3UL, 5UL, 7UL );
   sm.erase( 2UL, sm.begin(2UL), sm.end(2UL), []( double value ){ return value < 1E-8; } );
   \endcode

// \note The predicate is required to be pure, i.e. to produce deterministic results for elements
// with the same value. The attempt to use an impure predicate leads to undefined behavior!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Pred >   // Type of the unary predicate
inline void Submatrix<MT,AF,true,false,CSAs...>::erase( size_t j, Iterator first, Iterator last, Pred predicate )
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   matrix_.erase( column()+j, first.base(), last.base(), predicate );
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
/*!\brief Searches for a specific submatrix element.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// submatrix. It specifically searches for the element with row index \a i and column index
// \a j. In case the element is found, the function returns an row/column iterator to the
// element. Otherwise an iterator just past the last non-zero element of row \a i or column
// \a j (the end() iterator) is returned. Note that the returned sparse submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator,
// the set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
   Submatrix<MT,AF,true,false,CSAs...>::find( size_t i, size_t j )
{
   const Iterator_t<MT> pos( matrix_.find( row() + i, column() + j ) );

   if( pos != matrix_.end( column() + j ) )
      return Iterator( pos, row() );
   else
      return end( j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Searches for a specific submatrix element.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the element in case the index is found, end() iterator otherwise.
//
// This function can be used to check whether a specific element is contained in the sparse
// submatrix. It specifically searches for the element with row index \a i and column index
// \a j. In case the element is found, the function returns an row/column iterator to the
// element. Otherwise an iterator just past the last non-zero element of row \a i or column
// \a j (the end() iterator) is returned. Note that the returned sparse submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator,
// the set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,true,false,CSAs...>::find( size_t i, size_t j ) const
{
   const ConstIterator_t<MT> pos( matrix_.find( row() + i, column() + j ) );

   if( pos != matrix_.end( column() + j ) )
      return ConstIterator( pos, row() );
   else
      return end( j );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// In case of a row-major submatrix, this function returns a row iterator to the first element
// with an index not less then the given column index. In case of a column-major submatrix, the
// function returns a column iterator to the first element with an index not less then the given
// row index. In combination with the upperBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
   Submatrix<MT,AF,true,false,CSAs...>::lowerBound( size_t i, size_t j )
{
   return Iterator( matrix_.lowerBound( row() + i, column() + j ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index not less then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index not less then the given index, end() iterator otherwise.
//
// In case of a row-major submatrix, this function returns a row iterator to the first element
// with an index not less then the given column index. In case of a column-major submatrix, the
// function returns a column iterator to the first element with an index not less then the given
// row index. In combination with the upperBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,true,false,CSAs...>::lowerBound( size_t i, size_t j ) const
{
   return ConstIterator( matrix_.lowerBound( row() + i, column() + j ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// In case of a row-major submatrix, this function returns a row iterator to the first element
// with an index greater then the given column index. In case of a column-major submatrix, the
// function returns a column iterator to the first element with an index greater then the given
// row index. In combination with the lowerBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::Iterator
   Submatrix<MT,AF,true,false,CSAs...>::upperBound( size_t i, size_t j )
{
   return Iterator( matrix_.upperBound( row() + i, column() + j ), row() );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first index greater then the given index.
//
// \param i The row index of the search element. The index has to be in the range \f$[0..M-1]\f$.
// \param j The column index of the search element. The index has to be in the range \f$[0..N-1]\f$.
// \return Iterator to the first index greater then the given index, end() iterator otherwise.
//
// In case of a row-major submatrix, this function returns a row iterator to the first element
// with an index greater then the given column index. In case of a column-major submatrix, the
// function returns a column iterator to the first element with an index greater then the given
// row index. In combination with the lowerBound() function this function can be used to create
// a pair of iterators specifying a range of indices. Note that the returned submatrix iterator
// is subject to invalidation due to inserting operations via the function call operator, the
// set() function or the insert() function!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline typename Submatrix<MT,AF,true,false,CSAs...>::ConstIterator
   Submatrix<MT,AF,true,false,CSAs...>::upperBound( size_t i, size_t j ) const
{
   return ConstIterator( matrix_.upperBound( row() + i, column() + j ), row() );
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
/*!\brief In-place transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the sparse submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::transpose()
{
   using blaze::assign;

   if( rows() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, trans( *this ), row(), column() ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   decltype(auto) left( derestrict( *this ) );
   const ResultType tmp( trans( *this ) );

   reset();
   assign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the submatrix.
//
// \return Reference to the transposed submatrix.
// \exception std::logic_error Invalid transpose of a non-quadratic submatrix.
// \exception std::logic_error Invalid transpose operation.
//
// This function transposes the sparse submatrix in-place. Note that this function can only be used
// for quadratic submatrices, i.e. if the number of rows is equal to the number of columns. Also,
// the function fails if ...
//
//  - ... the submatrix contains elements from the upper part of the underlying lower matrix;
//  - ... the submatrix contains elements from the lower part of the underlying upper matrix;
//  - ... the result would be non-deterministic in case of a symmetric or Hermitian matrix.
//
// In all cases, a \a std::logic_error is thrown.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::ctranspose()
{
   using blaze::assign;

   if( rows() != columns() ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose of a non-quadratic submatrix" );
   }

   if( !tryAssign( matrix_, ctrans( *this ), row(), column() ) ) {
      BLAZE_THROW_LOGIC_ERROR( "Invalid transpose operation" );
   }

   decltype(auto) left( derestrict( *this ) );
   const ResultType tmp( ctrans(*this) );

   reset();
   assign( left, tmp );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the sparse submatrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the submatrix scaling.
// \return Reference to the sparse submatrix.
//
// This function scales the submatrix by applying the given scalar value \a scalar to each
// element of the submatrix. For built-in and \c complex data types it has the same effect
// as using the multiplication assignment operator. Note that the function cannot be used
// to scale a submatrix on a lower or upper unitriangular matrix. The attempt to scale
// such a submatrix results in a compile time error!
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Other >  // Data type of the scalar value
inline Submatrix<MT,AF,true,false,CSAs...>&
   Submatrix<MT,AF,true,false,CSAs...>::scale( const Other& scalar )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_UNITRIANGULAR_MATRIX_TYPE( MT );

   for( size_t i=0UL; i<columns(); ++i ) {
      const Iterator last( end(i) );
      for( Iterator element=begin(i); element!=last; ++element )
         element->value() *= scalar;
   }

   return *this;
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
/*!\brief Returns whether the submatrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address can alias with the submatrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,AF,true,false,CSAs...>::canAlias( const Other* alias ) const noexcept
{
   return matrix_.isAliased( &unview( *alias ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this submatrix, \a false if not.
//
// This function returns whether the given address is aliased with the submatrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions to
// optimize the evaluation.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename Other >  // Data type of the foreign expression
inline bool Submatrix<MT,AF,true,false,CSAs...>::isAliased( const Other* alias ) const noexcept
{
   return matrix_.isAliased( &unview( *alias ) );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the submatrix can be used in SMP assignments.
//
// \return \a true in case the submatrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the submatrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the matrix).
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
inline bool Submatrix<MT,AF,true,false,CSAs...>::canSMPAssign() const noexcept
{
   return false;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side dense matrix
        , bool SO >         // Storage order of the right-hand side dense matrix
inline void Submatrix<MT,AF,true,false,CSAs...>::assign( const DenseMatrix<MT2,SO>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   reserve( 0UL, rows() * columns() );

   for( size_t j=0UL; j<columns(); ++j ) {
      for( size_t i=0UL; i<rows(); ++i ) {
         if( IsSymmetric_v<MT> || IsHermitian_v<MT> ) {
            const ElementType& value( (*rhs)(i,j) );
            if( !isDefault<strict>( value ) )
               set( i, j, value );
         }
         else {
            append( i, j, (*rhs)(i,j), true );
         }
      }
      finalize( j );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2 >    // Type of the right-hand side sparse matrix
inline void Submatrix<MT,AF,true,false,CSAs...>::assign( const SparseMatrix<MT2,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   reserve( 0UL, (*rhs).nonZeros() );

   for( size_t j=0UL; j<(*rhs).columns(); ++j ) {
      for( ConstIterator_t<MT2> element=(*rhs).begin(j); element!=(*rhs).end(j); ++element ) {
         if( IsSymmetric_v<MT> || IsHermitian_v<MT> ) {
            const ElementType& value( element->value() );
            if( !isDefault<strict>( value ) )
               set( element->index(), j, value );
         }
         else
            append( element->index(), j, element->value(), true );
      }
      finalize( j );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2 >    // Type of the right-hand side sparse matrix
inline void Submatrix<MT,AF,true,false,CSAs...>::assign( const SparseMatrix<MT2,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT2 );

   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   // Counting the number of elements per column
   std::vector<size_t> columnLengths( columns(), 0UL );
   for( size_t i=0UL; i<rows(); ++i ) {
      for( auto element=(*rhs).begin(i); element!=(*rhs).end(i); ++element )
         ++columnLengths[element->index()];
   }

   // Resizing the sparse matrix
   for( size_t j=0UL; j<columns(); ++j ) {
      reserve( j, columnLengths[j] );
   }

   // Appending the elements to the columns of the sparse matrix
   for( size_t i=0UL; i<rows(); ++i ) {
      for( auto element=(*rhs).begin(i); element!=(*rhs).end(i); ++element )
         if( IsSymmetric_v<MT> || IsHermitian_v<MT> ) {
            const ElementType& value( element->value() );
            if( !isDefault<strict>( value ) )
               set( i, element->index(), value );
         }
         else {
            append( i, element->index(), element->value(), true );
         }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a matrix.
//
// \param rhs The right-hand side matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline void Submatrix<MT,AF,true,false,CSAs...>::addAssign( const Matrix<MT2,SO>& rhs )
{
   using AddType = AddTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( AddType );

   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   const AddType tmp( serial( *this + (*rhs) ) );
   reset();
   assign( tmp );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a matrix.
//
// \param rhs The right-hand side matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline void Submatrix<MT,AF,true,false,CSAs...>::subAssign( const Matrix<MT2,SO>& rhs )
{
   using SubType = SubTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SubType );

   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   const SubType tmp( serial( *this - (*rhs) ) );
   reset();
   assign( tmp );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a matrix.
//
// \param rhs The right-hand side matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT       // Type of the sparse matrix
        , AlignmentFlag AF  // Alignment flag
        , size_t... CSAs >  // Compile time submatrix arguments
template< typename MT2      // Type of the right-hand side matrix
        , bool SO >         // Storage order of the right-hand side matrix
inline void Submatrix<MT,AF,true,false,CSAs...>::schurAssign( const Matrix<MT2,SO>& rhs )
{
   using SchurType = SchurTrait_t< ResultType, ResultType_t<MT2> >;

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( SchurType );

   BLAZE_INTERNAL_ASSERT( rows()    == (*rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( columns() == (*rhs).columns(), "Invalid number of columns" );

   const SchurType tmp( serial( *this % (*rhs) ) );
   reset();
   assign( tmp );
}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
