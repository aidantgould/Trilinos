// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file Teko_KrylovSurrogateReducedOperator.hpp
 *
 * Builds the Krylov surrogate's reduced operator \f$\hat{C}\f$ and right-hand
 * side \f$\hat{b}\f$ from recorded preconditioner outputs.
 */

#ifndef __Teko_KrylovSurrogateReducedOperator_hpp__
#define __Teko_KrylovSurrogateReducedOperator_hpp__

#include <vector>

#include "Teuchos_Comm.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"

#include "Tpetra_Operator.hpp"

#include "Teko_ConfigDefs.hpp"
#include "Teko_Utilities.hpp"

namespace Teko {
namespace KrylovSurrogate {

typedef Teuchos::SerialDenseMatrix<int, double> DenseMatrix;
typedef Teuchos::SerialDenseVector<int, double> DenseVector;

/** \brief Truncated SVD \f$M \approx U \Sigma W^T\f$ of one block's small factor. */
struct BlockReduction {
  int rank = 0;
  DenseMatrix U;  //!< k x rank
  std::vector<double> sigma;
  DenseMatrix W;  //!< m x rank
};

/** \brief SVD of a small dense \f$k \times m\f$ matrix on rank 0, truncated at
 *         \f$\sigma_i \ge tol \cdot \sigma_1\f$ (keeping at least one direction),
 *         and broadcast to every rank.
 *
 * Only rank 0's M is used. Running the SVD once and broadcasting it guarantees
 * every rank agrees on the rank, which a redundant SVD cannot.
 */
BlockReduction reduceOnRankZero(const DenseMatrix& M, double tol, const Teuchos::Comm<int>& comm);

/** \brief Block (i, j) of A as a Tpetra operator, or null for a zero block.
 *
 * Throws if the block is nonzero and not backed by Tpetra.
 */
Teuchos::RCP<const Tpetra::Operator<ST, LO, GO, NT> > tpetraBlock(const BlockedLinearOp& A, int i,
                                                                  int j);

/** \brief \f$\hat{C}\f$, \f$\hat{b}\f$ and their diagnostics. */
struct ReducedOperator {
  std::vector<int> blockSizes;
  std::vector<int> ranks;
  DenseMatrix CHat;
  DenseVector bHat;
  double rhsCapture = 0.0;
  std::vector<double> diagonalDeviations;
};

/** \brief Build \f$\hat{C}\f$ and \f$\hat{b}\f$ from preconditioner outputs.
 *
 * For each block j: \f$\tilde{V}_j = A_{jj} Z_j\f$, a tall-skinny QR
 * \f$\tilde{V}_j = Q_j R_j\f$ (padded with zero rows on any rank holding fewer
 * than m rows), the truncated SVD \f$R_j = U_j \Sigma_j W_j^T\f$ on rank 0,
 * \f$\hat{Q}_j = Q_j U_j\f$ and \f$Y_j = Z_j W_j \Sigma_j^{-1} = A_{jj}^{-1} \hat{Q}_j\f$.
 * Then \f$\hat{C}_{ij} = \hat{Q}_i^T A_{ij} Y_j\f$ and \f$\hat{b}_j = \hat{Q}_j^T v_{1,j}\f$.
 * The diagonal blocks are computed, their distance from I recorded, and then
 * set to exactly I. The result is broadcast from rank 0, so it is bitwise
 * identical on every rank. Collective.
 *
 * \param[in] A  the blocked operator, every nonzero block backed by Tpetra
 * \param[in] Z  m recorded preconditioner outputs, blocked like A's domain
 * \param[in] v1 the initial residual direction, one column, blocked like A's range
 * \param[in] truncationTol relative singular value cut
 */
ReducedOperator buildReducedOperator(const BlockedLinearOp& A, const MultiVector& Z,
                                     const MultiVector& v1, double truncationTol);

}  // namespace KrylovSurrogate
}  // end namespace Teko

#endif
