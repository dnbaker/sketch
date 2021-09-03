//=================================================================================================
/*!
//  \file blaze/math/StaticVector.h
//  \brief Header file for the complete StaticVector implementation
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

#ifndef _BLAZE_MATH_STATICVECTOR_H_
#define _BLAZE_MATH_STATICVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/dense/StaticVector.h>
#include <blaze/math/DenseVector.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/StaticMatrix.h>
#include <blaze/math/ZeroVector.h>
#include <blaze/util/Random.h>


namespace blaze {

//=================================================================================================
//
//  RAND SPECIALIZATION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the Rand class template for StaticVector.
// \ingroup random
//
// This specialization of the Rand class creates random instances of StaticVector.
*/
template< typename Type     // Data type of the vector
        , size_t N          // Number of elements
        , bool TF           // Transpose flag
        , AlignmentFlag AF  // Alignment flag
        , PaddingFlag PF    // Padding flag
        , typename Tag >    // Type tag
class Rand< StaticVector<Type,N,TF,AF,PF,Tag> >
{
 public:
   //**********************************************************************************************
   /*!\brief Generation of a random StaticVector.
   //
   // \return The generated random vector.
   */
   inline const StaticVector<Type,N,TF,AF,PF,Tag> generate() const
   {
      StaticVector<Type,N,TF,AF,PF,Tag> vector;
      randomize( vector );
      return vector;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Generation of a random StaticVector.
   //
   // \param min The smallest possible value for a vector element.
   // \param max The largest possible value for a vector element.
   // \return The generated random vector.
   */
   template< typename Arg >  // Min/max argument type
   inline const StaticVector<Type,N,TF,AF,PF,Tag> generate( const Arg& min, const Arg& max ) const
   {
      StaticVector<Type,N,TF,AF,PF,Tag> vector;
      randomize( vector, min, max );
      return vector;
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Randomization of a StaticVector.
   //
   // \param vector The vector to be randomized.
   // \return void
   */
   inline void randomize( StaticVector<Type,N,TF,AF,PF,Tag>& vector ) const
   {
      using blaze::randomize;

      for( size_t i=0UL; i<N; ++i ) {
         randomize( vector[i] );
      }
   }
   //**********************************************************************************************

   //**********************************************************************************************
   /*!\brief Randomization of a StaticVector.
   //
   // \param vector The vector to be randomized.
   // \param min The smallest possible value for a vector element.
   // \param max The largest possible value for a vector element.
   // \return void
   */
   template< typename Arg >  // Min/max argument type
   inline void randomize( StaticVector<Type,N,TF,AF,PF,Tag>& vector,
                          const Arg& min, const Arg& max ) const
   {
      using blaze::randomize;

      for( size_t i=0UL; i<N; ++i ) {
         randomize( vector[i], min, max );
      }
   }
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
