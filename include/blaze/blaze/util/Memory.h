//=================================================================================================
/*!
//  \file blaze/util/Memory.h
//  \brief Header file for memory allocation and deallocation functionality
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

#ifndef _BLAZE_UTIL_MEMORY_H_
#define _BLAZE_UTIL_MEMORY_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstdlib>
#include <new>
#include <blaze/system/Platform.h>
#include <blaze/util/algorithms/ConstructAt.h>
#include <blaze/util/algorithms/Destroy.h>
#include <blaze/util/algorithms/DestroyAt.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/Exception.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/typetraits/IsBuiltin.h>

#if BLAZE_WIN32_PLATFORM || BLAZE_WIN64_PLATFORM || BLAZE_MINGW32_PLATFORM || BLAZE_MINGW64_PLATFORM
#  include <malloc.h>
#endif


namespace blaze {

//=================================================================================================
//
//  BYTE-BASED ALLOCATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Aligned array allocation.
// \ingroup util
//
// \param size The number of bytes to be allocated.
// \param alignment The required minimum alignment.
// \return Byte pointer to the first element of the aligned array.
// \exception std::bad_alloc Allocation failed.
//
// This function provides the functionality to allocate memory based on the given alignment
// restrictions. For that purpose it uses the according system-specific memory allocation
// functions.
*/
inline byte_t* alignedAllocate( size_t size, size_t alignment )
{
   void* raw( nullptr );

#if BLAZE_WIN32_PLATFORM || BLAZE_WIN64_PLATFORM || BLAZE_MINGW64_PLATFORM
   raw = _aligned_malloc( size, alignment );
   if( raw == nullptr ) {
#elif BLAZE_MINGW32_PLATFORM
   raw = __mingw_aligned_malloc( size, alignment );
   if( raw == nullptr ) {
#else
   alignment = ( alignment < sizeof(void*) ? sizeof(void*) : alignment );
   if( posix_memalign( &raw, alignment, size ) ) {
#endif
      BLAZE_THROW_BAD_ALLOC;
   }

   return reinterpret_cast<byte_t*>( raw );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deallocation of aligned memory.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the
// alignedAllocate() function. For that purpose it uses the according system-specific memory
// deallocation functions.
*/
inline void alignedDeallocate( const void* address ) noexcept
{
#if BLAZE_WIN32_PLATFORM || BLAZE_WIN64_PLATFORM || BLAZE_MINGW64_PLATFORM
   _aligned_free( const_cast<void*>( address ) );
#elif BLAZE_MINGW32_PLATFORM
   __mingw_aligned_free( const_cast<void*>( address ) );
#else
   free( const_cast<void*>( address ) );
#endif
}
//*************************************************************************************************




//=================================================================================================
//
//  TYPE-BASED ALLOCATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Aligned array allocation for built-in data types.
// \ingroup util
//
// \param size The number of elements of the given type to allocate.
// \return Pointer to the first element of the aligned array.
// \exception std::bad_alloc Allocation failed.
//
// The allocate() function provides the functionality to allocate memory based on the alignment
// restrictions of the given built-in data type. For instance, in case SSE vectorization is
// possible, the returned memory is guaranteed to be at least 16-byte aligned. In case AVX is
// active, the memory is even guaranteed to be at least 32-byte aligned.
//
// Examples:

   \code
   // Guaranteed to be 16-byte aligned (32-byte aligned in case AVX is used)
   double* dp = allocate<double>( 10UL );
   \endcode
*/
template< typename T
        , EnableIf_t< IsBuiltin_v<T> >* = nullptr >
T* allocate( size_t size )
{
   constexpr size_t alignment( AlignmentOf_v<T> );

   return reinterpret_cast<T*>( alignedAllocate( size*sizeof(T), alignment ) );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned array allocation for user-specific class types.
// \ingroup util
//
// \param size The number of elements of the given type to allocate.
// \return Pointer to the first element of the aligned array.
// \exception std::bad_alloc Allocation failed.
//
// The allocate() function provides the functionality to allocate memory based on the alignment
// restrictions of the given user-specific class type. For instance, in case the given type has
// the requirement to be 32-byte aligned, the returned pointer is guaranteed to be 32-byte
// aligned. Additionally, all elements of the array are guaranteed to be default constructed.
// Note that the allocate() function provides exception safety similar to the new operator: In
// case any element throws an exception during construction, all elements that have already been
// constructed are destroyed in reverse order and the allocated memory is deallocated again.
*/
template< typename T
        , DisableIf_t< IsBuiltin_v<T> >* = nullptr >
T* allocate( size_t size )
{
   constexpr size_t alignment ( AlignmentOf_v<T> );
   constexpr size_t headersize( ( sizeof(size_t) < alignment ) ? ( alignment ) : ( sizeof( size_t ) ) );

   BLAZE_INTERNAL_ASSERT( headersize >= alignment      , "Invalid header size detected" );
   BLAZE_INTERNAL_ASSERT( headersize % alignment == 0UL, "Invalid header size detected" );

   byte_t* const raw( alignedAllocate( size*sizeof(T)+headersize, alignment ) );

   *reinterpret_cast<size_t*>( raw ) = size;

   T* const address( reinterpret_cast<T*>( raw + headersize ) );
   size_t i( 0UL );

   try {
      for( ; i<size; ++i ) {
         blaze::construct_at( address+i );
      }
   }
   catch( ... ) {
      for( ; i>0UL; --i ) {
         blaze::destroy_at( address+i );
      }
      alignedDeallocate( raw );
      throw;
   }

   return address;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deallocation of memory for built-in data types.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the allocate()
// function.
*/
template< typename T
        , EnableIf_t< IsBuiltin_v<T> >* = nullptr >
void deallocate( T* address ) noexcept
{
   if( address == nullptr )
      return;

   alignedDeallocate( address );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Deallocation of memory for user-specific class types.
// \ingroup util
//
// \param address The address of the first element of the array to be deallocated.
// \return void
//
// This function deallocates the given memory that was previously allocated via the allocate()
// function.
*/
template< typename T
        , DisableIf_t< IsBuiltin_v<T> >* = nullptr >
void deallocate( T* address )
{
   if( address == nullptr )
      return;

   constexpr size_t alignment ( AlignmentOf_v<T> );
   constexpr size_t headersize( ( sizeof(size_t) < alignment ) ? ( alignment ) : ( sizeof( size_t ) ) );

   BLAZE_INTERNAL_ASSERT( headersize >= alignment      , "Invalid header size detected" );
   BLAZE_INTERNAL_ASSERT( headersize % alignment == 0UL, "Invalid header size detected" );

   const byte_t* const raw = reinterpret_cast<byte_t*>( address ) - headersize;
   const size_t size( *reinterpret_cast<const size_t*>( raw ) );

   blaze::destroy_n( address, size );
   alignedDeallocate( raw );
}
//*************************************************************************************************

} // namespace blaze

#endif
