//=================================================================================================
/*!
//  \file blaze/math/proxy/ComplexProxy.h
//  \brief Header file for the ComplexProxy class
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

#ifndef _BLAZE_MATH_PROXY_COMPLEXPROXY_H_
#define _BLAZE_MATH_PROXY_COMPLEXPROXY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Exception.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/system/Inline.h>
#include <blaze/util/constraints/Complex.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Proxy backend for complex types.
// \ingroup math
//
// The ComplexProxy class serves as a backend for the Proxy class. It is used in case the data
// type represented by the proxy is a complex number and augments the Proxy interface by the
// complete interface required of complex numbers.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
class ComplexProxy
{
 public:
   //**Type definitions****************************************************************************
   using value_type = typename CT::value_type;  //!< Value type of the represented complex element.
   using ValueType  = value_type;               //!< Value type of the represented complex element.
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline ValueType real() const;
   inline void      real( ValueType value ) const;
   inline ValueType imag() const;
   inline void      imag( ValueType value ) const;
   //@}
   //**********************************************************************************************

   //**Conversion operators************************************************************************
   /*!\name Conversion operators */
   //@{
   [[deprecated]] BLAZE_ALWAYS_INLINE PT&       operator~()       noexcept;
   [[deprecated]] BLAZE_ALWAYS_INLINE const PT& operator~() const noexcept;

   BLAZE_ALWAYS_INLINE PT&       operator*()       noexcept;
   BLAZE_ALWAYS_INLINE const PT& operator*() const noexcept;
   //@}
   //**********************************************************************************************

 protected:
   //**Special member functions********************************************************************
   /*!\name Special member functions */
   //@{
   ComplexProxy() = default;
   ComplexProxy( const ComplexProxy& ) = default;
   ComplexProxy( ComplexProxy&& ) = default;
   ~ComplexProxy() = default;
   ComplexProxy& operator=( const ComplexProxy& ) = default;
   ComplexProxy& operator=( ComplexProxy&& ) = default;
   //@}
   //**********************************************************************************************

 private:
   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_COMPLEX_TYPE( CT );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the real part of the represented complex number.
//
// \return The current real part of the represented complex number.
//
// This function returns the current value of the real part of the represented complex number.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
inline typename ComplexProxy<PT,CT>::ValueType ComplexProxy<PT,CT>::real() const
{
   return (**this).get().real();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the real part of the represented complex number.
//
// \param value The new value for the real part.
// \return void
//
// This function sets a new value to the real part of the represented complex number.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
inline void ComplexProxy<PT,CT>::real( ValueType value ) const
{
   if( (**this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (**this).get().real( value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the imaginary part of the represented complex number.
//
// \return The current imaginary part of the represented complex number.
//
// This function returns the current value of the imaginary part of the represented complex number.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
inline typename ComplexProxy<PT,CT>::ValueType ComplexProxy<PT,CT>::imag() const
{
   return (**this).get().imag();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the imaginary part of the represented complex number.
//
// \param value The new value for the imaginary part.
// \return void
//
// This function sets a new value to the imaginary part of the represented complex number.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
inline void ComplexProxy<PT,CT>::imag( ValueType value ) const
{
   if( (**this).isRestricted() ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid access to restricted element" );
   }

   (**this).get().imag( value );
}
//*************************************************************************************************




//=================================================================================================
//
//  CONVERSION OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Conversion operator for non-constant proxies.
//
// \return Mutable reference to the actual type of the proxy.
//
// This operator performs the CRTP-based type-safe downcast to the actual type \a PT of the
// proxy. It will return a mutable reference to the actual type \a PT.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
[[deprecated]] BLAZE_ALWAYS_INLINE PT& ComplexProxy<PT,CT>::operator~() noexcept
{
   return static_cast<PT&>( *this );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator for constant proxies.
//
// \return Constant reference to the actual type of the proxy.
//
// This operator performs the CRTP-based type-safe downcast to the actual type \a PT of the
// proxy. It will return a constant reference to the actual type \a PT.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
[[deprecated]] BLAZE_ALWAYS_INLINE const PT& ComplexProxy<PT,CT>::operator~() const noexcept
{
   return static_cast<const PT&>( *this );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator for non-constant proxies.
//
// \return Mutable reference to the actual type of the proxy.
//
// This operator performs the CRTP-based type-safe downcast to the actual type \a PT of the
// proxy. It will return a mutable reference to the actual type \a PT.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
BLAZE_ALWAYS_INLINE PT& ComplexProxy<PT,CT>::operator*() noexcept
{
   return static_cast<PT&>( *this );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion operator for constant proxies.
//
// \return Constant reference to the actual type of the proxy.
//
// This operator performs the CRTP-based type-safe downcast to the actual type \a PT of the
// proxy. It will return a constant reference to the actual type \a PT.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
BLAZE_ALWAYS_INLINE const PT& ComplexProxy<PT,CT>::operator*() const noexcept
{
   return static_cast<const PT&>( *this );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\name ComplexProxy global functions */
//@{
template< typename PT, typename CT >
PT& crtp_cast( ComplexProxy<PT,CT>& proxy );

template< typename PT, typename CT >
const PT& crtp_cast( const ComplexProxy<PT,CT>& proxy );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief CRTP-based conversion operation for non-constant proxies.
//
// \param proxy The proxy to be downcast.
// \return Mutable reference of the actual type of the proxy.
//
// This operator performs the CRTP-based type-safe downcast to the actual type \a PT of the
// proxy. It will return a mutable reference to the actual type \a PT.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
BLAZE_ALWAYS_INLINE PT& crtp_cast( ComplexProxy<PT,CT>& proxy )
{
   return *proxy;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief CRTP-based conversion operation for constant proxies.
//
// \param proxy The proxy to be downcast.
// \return Const reference of the actual type of the proxy.
//
// This operator performs the CRTP-based type-safe downcast to the actual type \a PT of the
// proxy. It will return a constant reference to the actual type \a PT.
*/
template< typename PT    // Type of the proxy
        , typename CT >  // Type of the complex number
BLAZE_ALWAYS_INLINE const PT& crtp_cast( const ComplexProxy<PT,CT>& proxy )
{
   return *proxy;
}
//*************************************************************************************************

} // namespace blaze

#endif
