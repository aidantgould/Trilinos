// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_KrylovSurrogateReducedOperator.hpp"

#ifdef TEKO_HAVE_KRYLOV_SURROGATE

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_LAPACK.hpp"

#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"

#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_TsqrAdaptor.hpp"

namespace Teko {
namespace KrylovSurrogate {
namespace Impl {

namespace {

typedef Tpetra::Map<LO, GO, NT> TpetraMap;
typedef Tpetra::MultiVector<ST, LO, GO, NT> TpetraMV;
typedef Thyra::TpetraOperatorVectorExtraction<ST, LO, GO, NT> TOE;
typedef Teuchos::RCP<const Tpetra::Operator<ST, LO, GO, NT> > TpetraOp;

// A k x n dense matrix as a Tpetra multivector, locally replicated: the small
// right-hand factor of a local GEMM, or the result of a reduction.
Teuchos::RCP<TpetraMV> localMV(const DenseMatrix& M,
                               const Teuchos::RCP<const Teuchos::Comm<int> >& comm) {
  auto map = Tpetra::createLocalMapWithNode<LO, GO, NT>(M.numRows(), comm);
  return Teuchos::rcp(
      new TpetraMV(map, Teuchos::ArrayView<const ST>(M.values(), M.stride() * M.numCols()),
                   M.stride(), M.numCols()));
}

// A^T B for distributed A and B: a local GEMM and one reduction, the same
// result on every rank.
DenseMatrix transposeProduct(const TpetraMV& A, const TpetraMV& B) {
  const int k = static_cast<int>(A.getNumVectors());
  const int n = static_cast<int>(B.getNumVectors());
  auto map    = Tpetra::createLocalMapWithNode<LO, GO, NT>(k, A.getMap()->getComm());
  TpetraMV C(map, n);
  C.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, A, B, 0.0);

  DenseMatrix out(k, n);
  if (k > 0 && n > 0) C.get1dCopy(Teuchos::ArrayView<ST>(out.values(), k * n), k);
  return out;
}

Teuchos::RCP<TpetraMV> tpetraOfBlock(const MultiVector& mv, int j) {
  return TOE::getTpetraMultiVector(getBlock(j, toBlockedMultiVector(mv)));
}

}  // namespace

BlockReduction reduceOnRankZero(const DenseMatrix& M, double tol, const Teuchos::Comm<int>& comm,
                                int maxSize) {
  const int k = M.numRows();
  const int m = M.numCols();
  const int s = std::min(k, m);

  BlockReduction red;
  std::vector<double> sigma(s, 0.0);
  DenseMatrix U(k, s), VT(s, m);

  // Rank 0 decides, and every rank throws alike from what it broadcasts: a
  // throw on rank 0 alone would leave the others waiting in the broadcast.
  enum { OK = 0, NOT_FINITE = 1, SVD_FAILED = 2 };
  int verdict[3] = {0, OK, 0};  // size, status, LAPACK info
  if (comm.getRank() == 0) {
    bool finite = true;
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < k; ++i) finite = finite && std::isfinite(M(i, j));

    if (!finite) {
      verdict[1] = NOT_FINITE;
    } else {
      Teuchos::LAPACK<int, double> lapack;
      DenseMatrix A(M);  // GESVD overwrites its input
      int info = 0, lwork = -1;
      double query;
      lapack.GESVD('S', 'S', k, m, A.values(), A.stride(), sigma.data(), U.values(), U.stride(),
                   VT.values(), VT.stride(), &query, lwork, nullptr, &info);
      lwork = static_cast<int>(query);
      std::vector<double> work(lwork);
      lapack.GESVD('S', 'S', k, m, A.values(), A.stride(), sigma.data(), U.values(), U.stride(),
                   VT.values(), VT.stride(), work.data(), lwork, nullptr, &info);
      verdict[2] = info;
      if (info != 0) verdict[1] = SVD_FAILED;
    }

    int size      = 0;
    const int cap = std::min(s, std::max(maxSize, 1));
    while (verdict[1] == OK && size < cap && sigma[size] >= tol * sigma[0] && sigma[size] > 0.0)
      ++size;
    verdict[0] = std::max(size, 1);
  }

  Teuchos::broadcast<int, int>(comm, 0, 3, verdict);
  TEUCHOS_TEST_FOR_EXCEPTION(verdict[1] == NOT_FINITE, std::runtime_error,
                             "Teko::KrylovSurrogate::reduceOnRankZero: a block's samples contain "
                             "a NaN or an infinity, so it has no reduced basis");
  TEUCHOS_TEST_FOR_EXCEPTION(
      verdict[1] == SVD_FAILED, std::runtime_error,
      "Teko::KrylovSurrogate::reduceOnRankZero: GESVD failed, info = " << verdict[2]);
  red.size    = verdict[0];
  const int r = red.size;

  red.sigma.assign(r, 0.0);
  red.U.shape(k, r);
  red.W.shape(m, r);
  if (comm.getRank() == 0) {
    for (int c = 0; c < r; ++c) {
      red.sigma[c] = sigma[c];
      for (int i = 0; i < k; ++i) red.U(i, c) = U(i, c);
      for (int i = 0; i < m; ++i) red.W(i, c) = VT(c, i);
    }
  }
  Teuchos::broadcast<int, double>(comm, 0, r, red.sigma.data());
  Teuchos::broadcast<int, double>(comm, 0, k * r, red.U.values());
  Teuchos::broadcast<int, double>(comm, 0, m * r, red.W.values());

  TEUCHOS_TEST_FOR_EXCEPTION(red.sigma[0] == 0.0, std::runtime_error,
                             "Teko::KrylovSurrogate::reduceOnRankZero: the samples of a block are "
                             "all zero, so it has no reduced basis");
  return red;
}

Teuchos::RCP<const Tpetra::Operator<ST, LO, GO, NT> > tpetraBlock(const BlockedLinearOp& A, int i,
                                                                  int j) {
  LinearOp op = getBlock(i, j, A);
  if (op == Teuchos::null || isZeroOp(op)) return Teuchos::null;

  auto tpetraOp = Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<ST, LO, GO, NT> >(op);
  TEUCHOS_TEST_FOR_EXCEPTION(tpetraOp == Teuchos::null, std::runtime_error,
                             "Teko \"Krylov Surrogate\": block ("
                                 << i << ", " << j
                                 << ") of the operator is not backed by Tpetra. Every nonzero "
                                    "block must be a Thyra::TpetraLinearOp.");
  return tpetraOp->getConstTpetraOperator();
}

void buildReducedOperator(const BlockedLinearOp& A, const MultiVector& Z, const MultiVector& v1,
                          double truncationTol, Report& report) {
  const int nb = blockRowCount(A);
  const int m  = static_cast<int>(Z->domain()->dim());

  report.numBlocks = nb;
  report.blockSizes.resize(nb);
  report.basisSizes.resize(nb);
  report.diagonalDeviations.resize(nb);

  // every block as a Tpetra operator, null for a zero block, fetched once
  std::vector<std::vector<TpetraOp> > blocks(nb, std::vector<TpetraOp>(nb));
  for (int i = 0; i < nb; ++i)
    for (int j = 0; j < nb; ++j) blocks[i][j] = tpetraBlock(A, i, j);

  std::vector<Teuchos::RCP<TpetraMV> > Qhat(nb), Y(nb);
  Teuchos::RCP<const Teuchos::Comm<int> > comm;

  for (int j = 0; j < nb; ++j) {
    const TpetraOp& Ajj = blocks[j][j];
    TEUCHOS_TEST_FOR_EXCEPTION(Ajj == Teuchos::null, std::runtime_error,
                               "Teko \"Krylov Surrogate\": diagonal block "
                                   << j << " is zero, so the surrogate is undefined");
    auto Zj    = tpetraOfBlock(Z, j);
    auto range = Ajj->getRangeMap();
    // every block on the same ranks, as Teko assumes throughout
    TEUCHOS_ASSERT(comm == Teuchos::null || comm->getSize() == range->getComm()->getSize());
    comm                 = range->getComm();
    report.blockSizes[j] = static_cast<long long>(range->getGlobalNumElements());

    // V_tilde_j = A_jj Z_j, computed straight into the top rows of a work
    // multivector padded with zero rows to at least m rows on every rank. TSQR
    // needs that many local rows, and zero rows leave R unchanged. The view onto
    // the block's own map works because Tpetra checks only the local length.
    const size_t nLocal = range->getLocalNumElements();
    const size_t nWork  = std::max(nLocal, static_cast<size_t>(m));
    auto workMap        = Teuchos::rcp(
               new TpetraMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), nWork, 0, comm));
    TpetraMV work(workMap, m);
    Ajj->apply(*Zj, *work.offsetViewNonConst(range, 0));

    TpetraMV Q(workMap, m);
    DenseMatrix R(m, m);
    Tpetra::TsqrAdaptor<TpetraMV> tsqr;
    tsqr.factorExplicit(work, Q, R);

    // at most n_j directions: a block has no more, whatever the roundoff
    const int cap = static_cast<int>(
        std::min<long long>(report.blockSizes[j], std::numeric_limits<int>::max()));
    BlockReduction red   = reduceOnRankZero(R, truncationTol, *comm, cap);
    const int r          = red.size;
    report.basisSizes[j] = r;

    // Q_hat_j = Q_j U_j, keeping only the block's own rows
    TpetraMV QU(workMap, r);
    QU.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, Q, *localMV(red.U, comm), 0.0);
    Qhat[j] = Teuchos::rcp(new TpetraMV(*QU.offsetView(range, 0), Teuchos::Copy));

    // Y_j = Z_j W_j Sigma_j^{-1}, which is A_jj^{-1} Q_hat_j
    DenseMatrix WS(red.W);
    for (int c = 0; c < r; ++c)
      for (int i = 0; i < m; ++i) WS(i, c) /= red.sigma[c];
    Y[j] = Teuchos::rcp(new TpetraMV(Zj->getMap(), r));
    Y[j]->multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *Zj, *localMV(WS, comm), 0.0);
  }

  std::vector<int> offset(nb + 1, 0);
  for (int j = 0; j < nb; ++j) offset[j + 1] = offset[j] + report.basisSizes[j];
  const int R = offset[nb];

  // C_hat_ij = Q_hat_i^T A_ij Y_j. The diagonal blocks are computed by the same
  // loop: in exact arithmetic they are I, so their deviation is the roundoff
  // actually incurred.
  report.CHat.shape(R, R);
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < nb; ++j) {
      if (blocks[i][j] == Teuchos::null) continue;

      TpetraMV AY(blocks[i][j]->getRangeMap(), report.basisSizes[j]);
      blocks[i][j]->apply(*Y[j], AY);
      DenseMatrix Cij = transposeProduct(*Qhat[i], AY);

      if (i == j) {
        double deviation = 0.0;
        for (int c = 0; c < Cij.numCols(); ++c)
          for (int rr = 0; rr < Cij.numRows(); ++rr)
            deviation = std::max(deviation, std::abs(Cij(rr, c) - (rr == c ? 1.0 : 0.0)));
        report.diagonalDeviations[i] = deviation;
        Cij.putScalar(0.0);
        for (int k = 0; k < Cij.numRows(); ++k) Cij(k, k) = 1.0;
      }
      DenseMatrix(Teuchos::View, report.CHat, Cij.numRows(), Cij.numCols(), offset[i], offset[j])
          .assign(Cij);
    }
  }

  // b_hat_j = Q_hat_j^T v1_j, and how much of v1 the trained subspace captures
  report.bHat.size(R);
  double v1NormSq = 0.0;
  for (int j = 0; j < nb; ++j) {
    auto v1j       = tpetraOfBlock(v1, j);
    DenseMatrix bj = transposeProduct(*Qhat[j], *v1j);
    for (int rr = 0; rr < report.basisSizes[j]; ++rr) report.bHat(offset[j] + rr) = bj(rr, 0);

    Teuchos::Array<double> norm(1);
    v1j->norm2(norm());
    v1NormSq += norm[0] * norm[0];
  }
  report.rhsCapture = (v1NormSq > 0.0) ? report.bHat.normFrobenius() / std::sqrt(v1NormSq) : 0.0;

  // Every rank holds the same numbers up to the bits of each reduction. Take
  // rank 0's, so everything downstream is bitwise identical across ranks.
  Teuchos::broadcast<int, double>(*comm, 0, R * R, report.CHat.values());
  Teuchos::broadcast<int, double>(*comm, 0, R, report.bHat.values());
  Teuchos::broadcast<int, double>(*comm, 0, nb, report.diagonalDeviations.data());
  Teuchos::broadcast<int, double>(*comm, 0, 1, &report.rhsCapture);
}

}  // namespace Impl
}  // namespace KrylovSurrogate
}  // end namespace Teko

#endif  // TEKO_HAVE_KRYLOV_SURROGATE
