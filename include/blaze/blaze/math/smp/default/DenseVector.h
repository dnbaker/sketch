//=================================================================================================
/*!
//  \file blaze/math/smp/default/DenseVector.h
//  \brief Header file for the default dense vector SMP implementation
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

#ifndef _BLAZE_MATH_SMP_DEFAULT_DENSEVECTOR_H_
#define _BLAZE_MATH_SMP_DEFAULT_DENSEVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/Vector.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/system/MacroDisable.h>
#include <blaze/system/SMP.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/StaticAssert.h>


namespace blaze {

//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name Dense vector SMP functions */
//@{
template< typename VT1, bool TF1, typename VT2, bool TF2 >
auto smpAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >;

template< typename VT1, bool TF1, typename VT2, bool TF2 >
auto smpAddAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >;

template< typename VT1, bool TF1, typename VT2, bool TF2 >
auto smpSubAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >;

template< typename VT1, bool TF1, typename VT2, bool TF2 >
auto smpMultAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >;

template< typename VT1, bool TF1, typename VT2, bool TF2 >
auto smpDivAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP assignment of a vector to a dense vector.
// \ingroup smp
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side vector to be assigned.
// \return void
//
// This function implements the default SMP assignment of a vector to a dense vector.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).size() == (*rhs).size(), "Invalid vector sizes" );
   assign( *lhs, *rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP addition assignment of a vector to a dense vector.
// \ingroup smp
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side vector to be added.
// \return void
//
// This function implements the default SMP addition assignment of a vector to a dense vector.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpAddAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).size() == (*rhs).size(), "Invalid vector sizes" );
   addAssign( *lhs, *rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP subtraction assignment of a vector to a dense vector.
// \ingroup smp
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side vector to be subtracted.
// \return void
//
// This function implements the default SMP subtraction assignment of a vector to a dense vector.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpSubAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).size() == (*rhs).size(), "Invalid vector sizes" );
   subAssign( *lhs, *rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP multiplication assignment of a vector to a dense vector.
// \ingroup smp
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side vector to be multiplied.
// \return void
//
// This function implements the default SMP multiplication assignment of a vector to a dense vector.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpMultAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).size() == (*rhs).size(), "Invalid vector sizes" );
   multAssign( *lhs, *rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the SMP division assignment of a vector to a dense vector.
// \ingroup smp
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side vector divisor.
// \return void
//
// This function implements the default SMP division assignment of a vector to a dense vector.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpDivAssign( Vector<VT1,TF1>& lhs, const Vector<VT2,TF2>& rhs )
   -> EnableIf_t< IsDenseVector_v<VT1> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (*lhs).size() == (*rhs).size(), "Invalid vector sizes" );
   divAssign( *lhs, *rhs );
}
//*************************************************************************************************




//=================================================================================================
//
//  COMPILE TIME CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

BLAZE_STATIC_ASSERT( !BLAZE_HPX_PARALLEL_MODE           );
BLAZE_STATIC_ASSERT( !BLAZE_CPP_THREADS_PARALLEL_MODE   );
BLAZE_STATIC_ASSERT( !BLAZE_BOOST_THREADS_PARALLEL_MODE );
BLAZE_STATIC_ASSERT( !BLAZE_OPENMP_PARALLEL_MODE        );

}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
