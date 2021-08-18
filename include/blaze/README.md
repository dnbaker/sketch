![blaze300x150.jpg](https://bitbucket.org/blaze-lib/blaze/wiki/images/blaze300x150.jpg)

**Blaze** is an open-source, high-performance C++ math library for dense and sparse arithmetic. With its state-of-the-art *Smart Expression Template* implementation **Blaze** combines the elegance and ease of use of a domain-specific language with HPC-grade performance, making it one of the most intuitive and fastest C++ math libraries available.

The **Blaze** library offers ...

  * ... **high performance** through the integration of BLAS libraries and manually tuned HPC math kernels
  * ... **vectorization** by SSE, SSE2, SSE3, SSSE3, SSE4, AVX, AVX2, AVX-512, FMA, SVML and SLEEF
  * ... **parallel execution** by OpenMP, HPX, C++11 threads and Boost threads
  * ... the **intuitive** and **easy to use** API of a domain specific language
  * ... **unified arithmetic** with dense and sparse vectors and matrices
  * ... **thoroughly tested** matrix and vector arithmetic
  * ... completely **portable**, **high quality** C++ source code

Get an impression of the clear but powerful syntax of **Blaze** in the [Getting Started](https://bitbucket.org/blaze-lib/blaze/wiki/Getting_Started) tutorial and of the impressive performance in the [Benchmarks](https://bitbucket.org/blaze-lib/blaze/wiki/Benchmarks) section.

----

## Download ##

![white20x120.jpg](https://bitbucket.org/blaze-lib/blaze/wiki/images/white20x120.jpg)
[![blaze-3.8.jpg](https://bitbucket.org/blaze-lib/blaze/wiki/images/blaze-3.8.jpg)](https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz)
![white40x120.jpg](https://bitbucket.org/blaze-lib/blaze/wiki/images/white40x120.jpg)
[![blaze-docu-3.8.jpg](https://bitbucket.org/blaze-lib/blaze/wiki/images/blaze-docu-3.8.jpg)](https://bitbucket.org/blaze-lib/blaze/downloads/blaze-docu-3.8.tar.gz)

Older releases of **Blaze** can be found in the [downloads](https://bitbucket.org/blaze-lib/blaze/downloads) section or in our [release archive](https://bitbucket.org/blaze-lib/blaze/wiki/Release Archive).

----

## Blaze Projects ##

[Blaze CUDA](https://github.com/STEllAR-GROUP/blaze_cuda): Add CUDA capabilities to the **Blaze** library (Jules Pénuchot)

[blaze_tensor](https://github.com/STEllAR-GROUP/blaze_tensor): An implementation of 3D tensors for the **Blaze** library (Stellar Group)

[BlazeIterative](https://github.com/tjolsen/BlazeIterative): A collection of iterative solvers (CG, BiCGSTAB, ...) for the **Blaze** library (Tyler Olsen)

[RcppBlaze](https://github.com/ChingChuan-Chen/RcppBlaze): A **Blaze** port for the R language (ChingChuan Chen)

----

## News ##

**15.8.2020**: We are very happy to announce the release of **Blaze** 3.8. Again, we have extended the library with a number of amazing new features:

* Introduction of the [```isinf()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!isinf) and [```isfinite()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!isinf) functions
* Introduction of [groups/tags for vectors and matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Grouping-Tagging)
* Introduction of the [```repeat()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!repeat) function for vectors and matrices
* Introduction of allocators for [```DynamicVector```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Types#!dynamicvector) and [```DynamicMatrix```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Types#!dynamicmatrix)
* Extended support for [custom data types](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20and%20Matrix%20Customization#!custom-data-types)

We hope that you enjoy this new release!

**23.2.2020**: Today we are very proud to release **Blaze** 3.7. This release is packed with a long list of new features and improvements:

* Introduction of [vector generators](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!vector-generators) and [matrix generators](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!matrix-generators)
* Introduction of the [dense matrix exponential](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!matrix-exponential)
* Introduction of the [```solve()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!linear-systems) function for dense linear systems
* Support for 64-bit BLAS and LAPACK libraries
* Enable instance-specific alignment and padding configuration for [```StaticVector```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Types#!staticvector), [```HybridVector```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Types#!hybridvector), [```StaticMatrix```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Types#!staticmatrix), and [```HybridMatrix```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Types#!hybridmatrix)
* ```constexpr```ification of [```HybridVector```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Types#!hybridvector) and [```HybridMatrix```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Types#!hybridmatrix)
* Introduction of [outer sum](https://bitbucket.org/blaze-lib/blaze/wiki/Addition#!outer-sum), [outer difference](https://bitbucket.org/blaze-lib/blaze/wiki/Subtraction#!outer-difference), and [outer quotient](https://bitbucket.org/blaze-lib/blaze/wiki/Vector-Vector%20Division#!outer-quotient) operations
* Introduction of [N-ary ```map()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!map-foreach) operations for dense vectors and matrices (up to ```N<=6```)
* Introduction of the [```select()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!select) function for dense vectors and matrices
* Introduction of the [```rank()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!rank) function for dense matrices
* Introduction of the [```declunilow()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!declunilow) and [```decluniupp()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!decluniupp) functions
* Introduction of the [```declstrlow()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!declstrlow) and [```declstrupp()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!declstrupp) functions
* Introduction of the [```nosimd()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!nosimd) function for vectors and matrices
* Introduction of the [```noalias()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!noalias) function for vectors and matrices
* Introduction of the [```isPositiveDefinite()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!ispositivedefinite) function for dense matrices
* Introduction of the [```eigen()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!eigenvalueseigenvectors) expression
* Introduction of the [```svd()```](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!singular-valuessingular-vectors) expression
* Introduction of a ```std::array``` constructor for all [dense vectors](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!array-construction) and [dense matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix%20Operations#!array-construction)
* Introduction of [```min()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!min-max) and [```max()```](https://bitbucket.org/blaze-lib/blaze/wiki/Vector%20Operations#!min-max) overloads for vector/scalar and matrix/scalar operations
* Optimizations of the dense matrix/dense vector multiplication kernels
* Optimizations of the dense matrix/dense matrix multiplication kernels
* Extended support for C++17 [class template argument deduction (CTAD)](https://en.cppreference.com/w/cpp/language/class_template_argument_deduction)

We hope that these new additions and improvements enable you to get even more out of **Blaze**. Enjoy!

----

## Wiki: Table of Contents ##

* [Configuration and Installation](https://bitbucket.org/blaze-lib/blaze/wiki/Configuration and Installation)
* [Getting Started](https://bitbucket.org/blaze-lib/blaze/wiki/Getting Started)
* [Vectors](https://bitbucket.org/blaze-lib/blaze/wiki/Vectors)
    * [Vector Types](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Types)
        * [Dense Vectors](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Types#!dense-vectors)
        * [Sparse Vectors](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Types#!sparse-vectors)
    * [Vector Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations)
        * [Constructors](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!constructors)
        * [Assignment](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!assignment)
        * [Element Access](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!element-access)
        * [Element Insertion](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!element-insertion)
        * [Element Removal](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!element-removal)
        * [Element Lookup](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!element-lookup)
        * [Non-Modifying Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!non-modifying-operations)
        * [Modifying Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!modifying-operations)
        * [Arithmetic Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!arithmetic-operations)
        * [Reduction Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!reduction-operations)
        * [Norms](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!norms)
        * [Scalar Expansion](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!scalar-expansion)
        * [Vector Expansion](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!vector-expansion)
        * [Vector Repetition](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!vector-repetition)
        * [Statistic Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!statistic-operations)
        * [Declaration Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!declaration-operations)
        * [Vector Generators](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Operations#!vector-generators)
* [Matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Matrices)
    * [Matrix Types](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Types)
        * [Dense Matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Types#!dense-matrices)
        * [Sparse Matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Types#!sparse-matrices)
    * [Matrix Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations)
        * [Constructors](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!constructors)
        * [Assignment](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!assignment)
        * [Element Access](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!element-access)
        * [Element Insertion](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!element-insertion)
        * [Element Removal](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!element-removal)
        * [Element Lookup](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!element-lookup)
        * [Non-Modifying Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!non-modifying-operations)
        * [Modifying Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!modifying-operations)
        * [Arithmetic Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!arithmetic-operations)
        * [Reduction Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!reduction-operations)
        * [Norms](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!norms)
        * [Scalar Expansion](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!scalar-expansion)
        * [Matrix Repetition](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!matrix-repetition)
        * [Statistic Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!statistic-operations)
        * [Declaration Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!declaration-operations)
        * [Matrix Generators](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!matrix-generators)
        * [Matrix Inversion](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!matrix-inversion)
        * [Matrix Exponential](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!matrix-exponential)
        * [Matrix Decomposition](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!matrix-decomposition)
        * [Linear Systems](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!linear-systems)
        * [Eigenvalues/Eigenvectors](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!eigenvalueseigenvectors)
        * [Singular Values/Singular Vectors](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Operations#!singular-valuessingular-vectors)
* [Adaptors](https://bitbucket.org/blaze-lib/blaze/wiki/Adaptors)
    * [Symmetric Matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Symmetric Matrices)
    * [Hermitian Matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Hermitian Matrices)
    * [Triangular Matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Triangular Matrices)
* [Views](https://bitbucket.org/blaze-lib/blaze/wiki/Views)
    * [Subvectors](https://bitbucket.org/blaze-lib/blaze/wiki/Subvectors)
    * [Element Selections](https://bitbucket.org/blaze-lib/blaze/wiki/Element Selections)
    * [Submatrices](https://bitbucket.org/blaze-lib/blaze/wiki/Submatrices)
    * [Rows](https://bitbucket.org/blaze-lib/blaze/wiki/Rows)
    * [Row Selections](https://bitbucket.org/blaze-lib/blaze/wiki/Row Selections)
    * [Columns](https://bitbucket.org/blaze-lib/blaze/wiki/Columns)
    * [Column Selections](https://bitbucket.org/blaze-lib/blaze/wiki/Column Selections)
    * [Bands](https://bitbucket.org/blaze-lib/blaze/wiki/Bands)
* [Arithmetic Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Arithmetic Operations)
    * [Addition](https://bitbucket.org/blaze-lib/blaze/wiki/Addition)
    * [Subtraction](https://bitbucket.org/blaze-lib/blaze/wiki/Subtraction)
    * [Scalar Multiplication](https://bitbucket.org/blaze-lib/blaze/wiki/Scalar Multiplication)
    * [Vector/Vector Multiplication](https://bitbucket.org/blaze-lib/blaze/wiki/Vector-Vector Multiplication)
        * [Componentwise Multiplication](https://bitbucket.org/blaze-lib/blaze/wiki/Vector-Vector Multiplication#!componentwise-multiplication)
        * [Inner Product / Scalar Product / Dot Product](https://bitbucket.org/blaze-lib/blaze/wiki/Vector-Vector Multiplication#!inner-product-scalar-product-dot-product)
        * [Outer Product](https://bitbucket.org/blaze-lib/blaze/wiki/Vector-Vector Multiplication#!outer-product)
        * [Cross Product](https://bitbucket.org/blaze-lib/blaze/wiki/Vector-Vector Multiplication#!cross-product)
        * [Kronecker Product](https://bitbucket.org/blaze-lib/blaze/wiki/Vector-Vector Multiplication#!kronecker-product)
    * [Vector/Vector Division](https://bitbucket.org/blaze-lib/blaze/wiki/Vector-Vector Division)
    * [Matrix/Vector Multiplication](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix-Vector Multiplication)
    * [Matrix/Matrix Multiplication](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix-Matrix Multiplication)
        * [Schur Product](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix-Matrix Multiplication#!componentwise-multiplication-schur-product)
        * [Matrix Product](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix-Matrix Multiplication#!matrix-product)
        * [Kronecker Product](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix-Matrix Multiplication#!kronecker-product)
* [Bitwise Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Arithmetic Operations)
    * [Bitwise Shift](https://bitbucket.org/blaze-lib/blaze/wiki/Bitwise Shift)
    * [Bitwise AND](https://bitbucket.org/blaze-lib/blaze/wiki/Bitwise AND)
    * [Bitwise OR](https://bitbucket.org/blaze-lib/blaze/wiki/Bitwise OR)
    * [Bitwise XOR](https://bitbucket.org/blaze-lib/blaze/wiki/Bitwise XOR)
* [Logical Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Arithmetic Operations)
    * [Logical NOT](https://bitbucket.org/blaze-lib/blaze/wiki/Logical NOT)
    * [Logical AND](https://bitbucket.org/blaze-lib/blaze/wiki/Logical AND)
    * [Logical OR](https://bitbucket.org/blaze-lib/blaze/wiki/Logical OR)
* [Shared-Memory Parallelization](https://bitbucket.org/blaze-lib/blaze/wiki/Shared Memory Parallelization)
    * [HPX Parallelization](https://bitbucket.org/blaze-lib/blaze/wiki/HPX Parallelization)
    * [C++11 Thread Parallelization](https://bitbucket.org/blaze-lib/blaze/wiki/Cpp Thread Parallelization)
    * [Boost Thread Parallelization](https://bitbucket.org/blaze-lib/blaze/wiki/Boost Thread Parallelization)
    * [OpenMP Parallelization](https://bitbucket.org/blaze-lib/blaze/wiki/OpenMP Parallelization)
    * [Serial Execution](https://bitbucket.org/blaze-lib/blaze/wiki/Serial Execution)
* [Serialization](https://bitbucket.org/blaze-lib/blaze/wiki/Serialization)
    * [Vector Serialization](https://bitbucket.org/blaze-lib/blaze/wiki/Vector Serialization)
    * [Matrix Serialization](https://bitbucket.org/blaze-lib/blaze/wiki/Matrix Serialization)
* [Customization](https://bitbucket.org/blaze-lib/blaze/wiki/Customization)
    * [Configuration Files](https://bitbucket.org/blaze-lib/blaze/wiki/Configuration Files)
    * [Vector and Matrix Customization](https://bitbucket.org/blaze-lib/blaze/wiki/Vector and Matrix Customization)
        * [Custom Data Members](https://bitbucket.org/blaze-lib/blaze/wiki/Vector and Matrix Customization#!custom-data-members)
        * [Custom Operations](https://bitbucket.org/blaze-lib/blaze/wiki/Vector and Matrix Customization#!custom-operations)
        * [Custom Data Types](https://bitbucket.org/blaze-lib/blaze/wiki/Vector and Matrix Customization#!custom-data-types)
    * [Grouping/Tagging](https://bitbucket.org/blaze-lib/blaze/wiki/Grouping-Tagging)
    * [Error Reporting Customization](https://bitbucket.org/blaze-lib/blaze/wiki/Error Reporting Customization)
* [BLAS Functions](https://bitbucket.org/blaze-lib/blaze/wiki/BLAS Functions)
* [LAPACK Functions](https://bitbucket.org/blaze-lib/blaze/wiki/LAPACK Functions)
* [Block Vectors and Matrices](https://bitbucket.org/blaze-lib/blaze/wiki/Block Vectors and Matrices)
* [Intra-Statement Optimization](https://bitbucket.org/blaze-lib/blaze/wiki/Intra-Statement Optimization)
* [Frequently Asked Questions (FAQ)](https://bitbucket.org/blaze-lib/blaze/wiki/FAQ)
* [Issue Creation Guidelines](https://bitbucket.org/blaze-lib/blaze/wiki/Issue Creation Guidelines)
* [Blaze References](https://bitbucket.org/blaze-lib/blaze/wiki/Blaze References)
* [Blazemark: The Blaze Benchmark Suite](https://bitbucket.org/blaze-lib/blaze/wiki/Blazemark)
* [Benchmarks/Performance Results](https://bitbucket.org/blaze-lib/blaze/wiki/Benchmarks)
* [Release Archive](https://bitbucket.org/blaze-lib/blaze/wiki/Release Archive)

----

## License ##

The **Blaze** library is licensed under the New (Revised) BSD license. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  * Neither the names of the **Blaze** development group nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----

## Compiler Compatibility ##

**Blaze** supports the C++14 standard and is compatible with a wide range of C++ compilers. In fact, **Blaze** is constantly tested with the GNU compiler collection (version 6.0 through 10.2), the Clang compiler (version 5.0 through 10.0), and Visual C++ 2017 (Win64 only). Other compilers are not explicitly tested, but might work with a high probability.

If you are looking for a C++98 compatible math library you might consider using an older release of **Blaze**. Until the release 2.6 **Blaze** was written in C++-98 and constantly tested with the GNU compiler collection (version 4.5 through 5.0), the Intel C++ compiler (12.1, 13.1, 14.0, 15.0), the Clang compiler (version 3.4 through 3.7), and Visual C++ 2010, 2012, 2013, and 2015 (Win64 only).

----

## Publications ##

* K. Iglberger, G. Hager, J. Treibig, and U. Rüde: **Expression Templates Revisited: A Performance Analysis of Current Methodologies** ([Download](http://epubs.siam.org/sisc/resource/1/sjoce3/v34/i2/pC42_s1)). SIAM Journal on Scientific Computing, 34(2): C42--C69, 2012
* K. Iglberger, G. Hager, J. Treibig, and U. Rüde: **High Performance Smart Expression Template Math Libraries** ([Download](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=06266939)). Proceedings of the 2nd International Workshop on New Algorithms and Programming Models for the Manycore Era (APMM 2012) at HPCS 2012

----

## Contributions ##

[Klaus Iglberger](https://www.linkedin.com/in/klaus-iglberger-2133694/) -- Project initiator and main developer

[Georg Hager](http://www.rrze.uni-erlangen.de/wir-ueber-uns/organigramm/mitarbeiter/index.shtml/georg-hager.shtml) -- Performance analysis and optimization

[Christian Godenschwager](http://www10.informatik.uni-erlangen.de/~godenschwager/) -- Visual Studio 2010/2012/2013/2015 bug fixes and testing

Tobias Scharpff -- Sparse matrix multiplication algorithms

byzhang -- Bug fixes

Emerson Ferreira -- Bug fixes

Fabien Péan -- CMake support

Denis Demidov -- Export CMake package configuration

Jannik Schürg -- AVX-512 support and cache size detection for macOS in CMake

Marcin Copik -- CMake fixes

Hartmut Kaiser -- HPX backend

[Patrick Diehl](http://www.diehlpk.de/) -- Integration of HPX to the Blazemark and maintainer of the **Blaze** Fedora package

Mario Emmenlauer -- Blazemark extensions

Jeff Pollock -- CMake extensions

Darcy Beurle -- Integration of **Blaze** into the Compiler Explorer

Robert Schumacher -- CMake fixes

Jan Rudolph -- CMake fixes

Mikhail Katliar -- LAPACK extensions

Daniel Baker -- Integration of Sleef

Thijs Withaar -- LAPACK extensions
