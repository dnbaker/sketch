//=================================================================================================
/*!
//  \file blaze/math/SparseMatrix.h
//  \brief Header file for all basic SparseMatrix functionality
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

#ifndef _BLAZE_MATH_SPARSEMATRIX_H_
#define _BLAZE_MATH_SPARSEMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/adaptors/DiagonalMatrix.h>
#include <blaze/math/adaptors/HermitianMatrix.h>
#include <blaze/math/adaptors/LowerMatrix.h>
#include <blaze/math/adaptors/SymmetricMatrix.h>
#include <blaze/math/adaptors/UpperMatrix.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/DMatSMatEqualExpr.h>
#include <blaze/math/expressions/DMatSMatKronExpr.h>
#include <blaze/math/expressions/DMatSMatSchurExpr.h>
#include <blaze/math/expressions/DMatTSMatSchurExpr.h>
#include <blaze/math/expressions/DVecSVecOuterExpr.h>
#include <blaze/math/expressions/SMatDeclDiagExpr.h>
#include <blaze/math/expressions/SMatDeclHermExpr.h>
#include <blaze/math/expressions/SMatDeclLowExpr.h>
#include <blaze/math/expressions/SMatDeclStrLowExpr.h>
#include <blaze/math/expressions/SMatDeclStrUppExpr.h>
#include <blaze/math/expressions/SMatDeclSymExpr.h>
#include <blaze/math/expressions/SMatDeclUniLowExpr.h>
#include <blaze/math/expressions/SMatDeclUniUppExpr.h>
#include <blaze/math/expressions/SMatDeclUppExpr.h>
#include <blaze/math/expressions/SMatDMatKronExpr.h>
#include <blaze/math/expressions/SMatDMatSchurExpr.h>
#include <blaze/math/expressions/SMatDVecMultExpr.h>
#include <blaze/math/expressions/SMatEvalExpr.h>
#include <blaze/math/expressions/SMatFixExpr.h>
#include <blaze/math/expressions/SMatMapExpr.h>
#include <blaze/math/expressions/SMatMeanExpr.h>
#include <blaze/math/expressions/SMatNoAliasExpr.h>
#include <blaze/math/expressions/SMatNormExpr.h>
#include <blaze/math/expressions/SMatNoSIMDExpr.h>
#include <blaze/math/expressions/SMatReduceExpr.h>
#include <blaze/math/expressions/SMatRepeatExpr.h>
#include <blaze/math/expressions/SMatScalarDivExpr.h>
#include <blaze/math/expressions/SMatScalarMultExpr.h>
#include <blaze/math/expressions/SMatSerialExpr.h>
#include <blaze/math/expressions/SMatSMatAddExpr.h>
#include <blaze/math/expressions/SMatSMatEqualExpr.h>
#include <blaze/math/expressions/SMatSMatKronExpr.h>
#include <blaze/math/expressions/SMatSMatMultExpr.h>
#include <blaze/math/expressions/SMatSMatSchurExpr.h>
#include <blaze/math/expressions/SMatSMatSubExpr.h>
#include <blaze/math/expressions/SMatStdDevExpr.h>
#include <blaze/math/expressions/SMatSVecMultExpr.h>
#include <blaze/math/expressions/SMatTransExpr.h>
#include <blaze/math/expressions/SMatTSMatAddExpr.h>
#include <blaze/math/expressions/SMatTSMatKronExpr.h>
#include <blaze/math/expressions/SMatTSMatMultExpr.h>
#include <blaze/math/expressions/SMatTSMatSchurExpr.h>
#include <blaze/math/expressions/SMatTSMatSubExpr.h>
#include <blaze/math/expressions/SMatVarExpr.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/expressions/SVecDVecOuterExpr.h>
#include <blaze/math/expressions/SVecSVecOuterExpr.h>
#include <blaze/math/expressions/TDVecSMatMultExpr.h>
#include <blaze/math/expressions/TDVecTSMatMultExpr.h>
#include <blaze/math/expressions/TSMatDMatSchurExpr.h>
#include <blaze/math/expressions/TSMatDVecMultExpr.h>
#include <blaze/math/expressions/TSMatSMatKronExpr.h>
#include <blaze/math/expressions/TSMatSMatMultExpr.h>
#include <blaze/math/expressions/TSMatSMatSchurExpr.h>
#include <blaze/math/expressions/TSMatSMatSubExpr.h>
#include <blaze/math/expressions/TSMatSVecMultExpr.h>
#include <blaze/math/expressions/TSMatTSMatAddExpr.h>
#include <blaze/math/expressions/TSMatTSMatKronExpr.h>
#include <blaze/math/expressions/TSMatTSMatMultExpr.h>
#include <blaze/math/expressions/TSMatTSMatSchurExpr.h>
#include <blaze/math/expressions/TSMatTSMatSubExpr.h>
#include <blaze/math/expressions/TSVecSMatMultExpr.h>
#include <blaze/math/expressions/TSVecTSMatMultExpr.h>
#include <blaze/math/Matrix.h>
#include <blaze/math/serialization/MatrixSerializer.h>
#include <blaze/math/smp/DenseMatrix.h>
#include <blaze/math/smp/SparseMatrix.h>
#include <blaze/math/sparse/SparseMatrix.h>
#include <blaze/math/views/Column.h>
#include <blaze/math/views/Row.h>
#include <blaze/math/views/Submatrix.h>

#endif
