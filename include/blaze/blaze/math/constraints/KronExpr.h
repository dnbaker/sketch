//=================================================================================================
/*!
//  \file blaze/math/constraints/KronExpr.h
//  \brief Constraint on the data type
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

#ifndef _BLAZE_MATH_CONSTRAINTS_KRONEXPR_H_
#define _BLAZE_MATH_CONSTRAINTS_KRONEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/IsKronExpr.h>


namespace blaze {

//=================================================================================================
//
//  MUST_BE_KRONEXPR_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup math_constraints
//
// In case the given data type \a T is not a Kronecker product expression (i.e. a type derived
// from the KronExpr base class), a compilation error is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_KRONEXPR_TYPE(T) \
   static_assert( ::blaze::IsKronExpr_v<T>, "Non-Kronecker product expression type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_KRONEXPR_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup math_constraints
//
// In case the given data type \a T is a Kronecker product expression (i.e. a type derived from
// the KronExpr base class), a compilation error is created.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_KRONEXPR_TYPE(T) \
   static_assert( !::blaze::IsKronExpr_v<T>, "Kronecker product expression type detected" )
//*************************************************************************************************

} // namespace blaze

#endif
