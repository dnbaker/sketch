//=================================================================================================
/*!
//  \file blaze/util/singleton/Dependency.h
//  \brief Header file for the Dependency class
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

#ifndef _BLAZE_UTIL_SINGLETON_DEPENDENCY_H_
#define _BLAZE_UTIL_SINGLETON_DEPENDENCY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <memory>
#include <blaze/util/constraints/DerivedFrom.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Lifetime dependency on a singleton object.
// \ingroup singleton
//
// The Dependency template class represents a lifetime dependency on a singleton object based
// on the Blaze Singleton functionality. By use of the Dependency template, any class can by
// either public or non-public inheritance or composition define a single or multiple lifetime
// dependencies on one or several singletons, which guarantees that the singleton instance(s)
// will be destroyed after the dependent object. The following example demonstrates both the
// inheritance as well as the composition approach:

   \code
   // Definition of the Viewer class, which is depending on the Logger singleton instance

   // #1: Approach by non-public inheritance
   class Viewer : private Dependency<Logger>
   {
      ...
   };

   // #2: Approach by composition
   class Viewer
   {
    private:
      Dependency<Logger> dependency_;
   };
   \endcode
*/
template< typename T >  // Type of the lifetime dependency
class Dependency
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   inline Dependency();
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::shared_ptr<T> dependency_;  //!< Handle to the lifetime dependency.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for Dependency.
*/
template< typename T >  // Type of the lifetime dependency
inline Dependency<T>::Dependency()
   : dependency_( T::instance() )  // Handle to the lifetime dependency
{
   BLAZE_CONSTRAINT_MUST_BE_DERIVED_FROM( T, typename T::SingletonType );
}
//*************************************************************************************************

} // namespace blaze

#endif
