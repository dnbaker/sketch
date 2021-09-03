//=================================================================================================
/*!
//  \file blaze/util/constraints/Constructible.h
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

#ifndef _BLAZE_UTIL_CONSTRAINTS_CONSTRUCTIBLE_H_
#define _BLAZE_UTIL_CONSTRAINTS_CONSTRUCTIBLE_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/typetraits/IsConstructible.h>


namespace blaze {

//=================================================================================================
//
//  MUST_BE_DEFAULT_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does not provide a default constructor, a compilation error
// is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_DEFAULT_CONSTRUCTIBLE_TYPE(T) \
   static_assert( ::blaze::IsDefaultConstructible_v<T>, "Non-default constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_DEFAULT_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does provide a default constructor, a compilation error is
// created.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_DEFAULT_CONSTRUCTIBLE_TYPE(T) \
   static_assert( !::blaze::IsDefaultConstructible_v<T>, "Default constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_BE_NOTHROW_DEFAULT_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does not provide a noexcept default constructor, a compilation
// error is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_NOTHROW_DEFAULT_CONSTRUCTIBLE_TYPE(T) \
   static_assert( ::blaze::IsNothrowDefaultConstructible_v<T>, "Non-noexcept default constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_NOTHROW_DEFAULT_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does provide a noexcept default constructor, a compilation
// error is created.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_NOTHROW_DEFAULT_CONSTRUCTIBLE_TYPE(T) \
   static_assert( !::blaze::IsNothrowDefaultConstructible_v<T>, "Noexcept default constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_BE_COPY_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does not provide a copy constructor, a compilation error
// is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_COPY_CONSTRUCTIBLE_TYPE(T) \
   static_assert( ::blaze::IsCopyConstructible_v<T>, "Non-copy constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_COPY_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does provide a copy constructor, a compilation error is
// created.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_COPY_CONSTRUCTIBLE_TYPE(T) \
   static_assert( !::blaze::IsCopyConstructible_v<T>, "Copy constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_BE_NOTHROW_COPY_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does not provide a noexcept copy constructor, a compilation
// error is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_NOTHROW_COPY_CONSTRUCTIBLE_TYPE(T) \
   static_assert( ::blaze::IsNothrowCopyConstructible_v<T>, "Non-noexcept copy constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_NOTHROW_COPY_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does provide a noexcept copy constructor, a compilation
// error is created.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_NOTHROW_COPY_CONSTRUCTIBLE_TYPE(T) \
   static_assert( !::blaze::IsNothrowCopyConstructible_v<T>, "Noexcept copy constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_BE_MOVE_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does not provide a move constructor, a compilation error
// is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_MOVE_CONSTRUCTIBLE_TYPE(T) \
   static_assert( ::blaze::IsMoveConstructible_v<T>, "Non-move constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_MOVE_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does provide a move constructor, a compilation error is
// created.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_MOVE_CONSTRUCTIBLE_TYPE(T) \
   static_assert( !::blaze::IsMoveConstructible_v<T>, "Move constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_BE_NOTHROW_MOVE_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does not provide a noexcept move constructor, a compilation
// error is created.
*/
#define BLAZE_CONSTRAINT_MUST_BE_NOTHROW_MOVE_CONSTRUCTIBLE_TYPE(T) \
   static_assert( ::blaze::IsNothrowMoveConstructible_v<T>, "Non-noexcept move constructible type detected" )
//*************************************************************************************************




//=================================================================================================
//
//  MUST_NOT_BE_NOTHROW_MOVE_CONSTRUCTIBLE_TYPE CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constraint on the data type.
// \ingroup constraints
//
// In case the given data type \a T does provide a noexcept move constructor, a compilation
// error is created.
*/
#define BLAZE_CONSTRAINT_MUST_NOT_BE_NOTHROW_MOVE_CONSTRUCTIBLE_TYPE(T) \
   static_assert( !::blaze::IsNothrowMoveConstructible_v<T>, "Noexcept move constructible type detected" )
//*************************************************************************************************

} // namespace blaze

#endif
