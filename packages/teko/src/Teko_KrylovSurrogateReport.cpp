// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_KrylovSurrogateReport.hpp"

#include <chrono>
#include <ostream>

#include "Teuchos_Array.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_TwoDArray.hpp"

#include "Tpetra_Map.hpp"

#include "Teko_KrylovSurrogateRecorderOp.hpp"
#include "Teko_KrylovSurrogateReducedOperator.hpp"
#include "Teko_KrylovSurrogateSearch.hpp"

namespace Teko {

Teuchos::RCP<Teuchos::ParameterList> KrylovSurrogateReport::toParameterList() const {
  auto pl = Teuchos::rcp(new Teuchos::ParameterList("Krylov Surrogate Report"));
  pl->set("Number of Blocks", numBlocks);
  pl->set("Number of Samples", numSamples);
  pl->set("Block Sizes", Teuchos::Array<int>(blockSizes));
  pl->set("Ranks", Teuchos::Array<int>(ranks));

  Teuchos::TwoDArray<double> C(CHat.numRows(), CHat.numCols());
  for (int i = 0; i < CHat.numRows(); ++i)
    for (int j = 0; j < CHat.numCols(); ++j) C(i, j) = CHat(i, j);
  pl->set("C_hat", C);
  pl->set("b_hat", Teuchos::Array<double>(bHat.values(), bHat.values() + bHat.length()));

  pl->set("RHS Capture", rhsCapture);
  pl->set("Diagonal Deviations", Teuchos::Array<double>(diagonalDeviations));
  pl->set("Search Wall Time", searchWallTime);
  pl->set("Search Ranks", searchRanks);

  Teuchos::ParameterList& cl = pl->sublist("Candidates");
  for (std::size_t c = 0; c < candidates.size(); ++c) {
    Teuchos::ParameterList& one = cl.sublist("Candidate " + std::to_string(c));
    one.set("Ordering", Teuchos::Array<int>(candidates[c].ordering));
    one.set("Status", candidates[c].status);
    one.set("Iterations", candidates[c].iterations);
    one.set("Final Relative Residual", candidates[c].finalRelativeResidual);
    one.set("Rate", candidates[c].rate);
  }
  return pl;
}

Teuchos::RCP<KrylovSurrogateHandle> KrylovSurrogateHandle::create() {
  return Teuchos::rcp(new KrylovSurrogateHandle);
}

void KrylovSurrogateHandle::attach(const Teuchos::RCP<const KrylovSurrogateRecorderOp>& recorder,
                                   const BlockedLinearOp& A,
                                   const KrylovSurrogateOptions& options) {
  recorder_ = recorder;
  A_        = A;
  options_  = options;
}

KrylovSurrogateReport KrylovSurrogateHandle::buildReport() const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      !isAttached(), std::runtime_error,
      "Teko::KrylovSurrogateHandle::buildReport: no preconditioner has been built from the "
      "\"Krylov Surrogate\" sublist holding this handle. Build it, solve, then ask for the "
      "report.");
  TEUCHOS_TEST_FOR_EXCEPTION(
      recorder_->numSamples() == 0, std::runtime_error,
      "Teko::KrylovSurrogateHandle::buildReport: the preconditioner has not been applied since "
      "it was last built, so there are no samples. Solve first.");

  using namespace KrylovSurrogate;
  ReducedOperator red = buildReducedOperator(A_, recorder_->getSamples(),
                                             recorder_->getFirstInput(), options_.truncationTol);

  const int nb = blockRowCount(A_);
  auto comm    = tpetraBlock(A_, 0, 0)->getRangeMap()->getComm();

  const auto start = std::chrono::steady_clock::now();
  std::vector<std::vector<int> > candidates =
      buildCandidates(nb, options_.orderings, options_.candidateCount, options_.seed);

  KrylovSurrogateReport report;
  report.candidates =
      scoreCandidates(red.CHat, red.bHat, red.ranks, candidates, options_.searchTol, *comm);
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  Teuchos::reduceAll<int, double>(*comm, Teuchos::REDUCE_MAX, 1, &elapsed, &report.searchWallTime);

  report.numBlocks          = nb;
  report.numSamples         = recorder_->numSamples();
  report.blockSizes         = red.blockSizes;
  report.ranks              = red.ranks;
  report.CHat               = red.CHat;
  report.bHat               = red.bHat;
  report.rhsCapture         = red.rhsCapture;
  report.diagonalDeviations = red.diagonalDeviations;
  report.searchRanks        = comm->getSize();
  return report;
}

std::ostream& operator<<(std::ostream& os, const KrylovSurrogateHandle& handle) {
  return os << "Teko::KrylovSurrogateHandle(" << (handle.isAttached() ? "attached" : "not attached")
            << ")";
}

}  // end namespace Teko
