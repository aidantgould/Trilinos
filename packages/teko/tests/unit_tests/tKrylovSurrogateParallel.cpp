// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_DefaultSerialComm.hpp>

#include <cmath>
#include <cstring>
#include <set>

#include "Tpetra_Core.hpp"

#include "Teko_KrylovSurrogateSearch.hpp"

using Teko::KrylovSurrogate::DenseMatrix;
using Teko::KrylovSurrogate::DenseVector;
typedef Teko::KrylovSurrogateReport::Candidate Candidate;

namespace {

void buildSurrogate(DenseMatrix& C, DenseVector& b, std::vector<int>& ranks) {
  ranks = {3, 2, 3, 2};
  int R = 10;
  C.shape(R, R);
  b.size(R);
  std::vector<int> block = {0, 0, 0, 1, 1, 2, 2, 2, 3, 3};
  for (int i = 0; i < R; ++i) {
    b(i) = 1.0 + std::sin(0.3 * i);
    for (int j = 0; j < R; ++j)
      C(i, j) = (block[i] == block[j]) ? (i == j ? 1.0 : 0.0) : 0.4 * std::sin(1.7 * i + 0.9 * j);
  }
}

// Bitwise, so a NaN (an unscored field) compares equal to itself
bool sameBits(double x, double y) { return std::memcmp(&x, &y, sizeof(double)) == 0; }

bool sameCandidates(const std::vector<Candidate>& x, const std::vector<Candidate>& y) {
  if (x.size() != y.size()) return false;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (x[i].ordering != y[i].ordering || x[i].status != y[i].status ||
        x[i].iterations != y[i].iterations ||
        !sameBits(x[i].finalRelativeResidual, y[i].finalRelativeResidual) ||
        !sameBits(x[i].rate, y[i].rate))
      return false;
  }
  return true;
}

}  // namespace

TEUCHOS_UNIT_TEST(tKrylovSurrogateParallel, bitwise_identical_to_serial) {
  auto comm = Tpetra::getDefaultComm();
  DenseMatrix C;
  DenseVector b;
  std::vector<int> ranks;
  buildSurrogate(C, b, ranks);

  // 75 candidates: every ordering of 4 blocks, dealt unevenly over 3 ranks
  auto candidates = Teko::KrylovSurrogate::allConfigurations(4);
  auto parallel   = Teko::KrylovSurrogate::scoreCandidates(C, b, ranks, candidates, 1e-10, *comm);

  Teuchos::SerialComm<int> serialComm;
  auto serial = Teko::KrylovSurrogate::scoreCandidates(C, b, ranks, candidates, 1e-10, serialComm);

  TEST_EQUALITY(parallel.size(), candidates.size());
  TEST_ASSERT(sameCandidates(parallel, serial));

  // every candidate scored exactly once: each ordering appears once in the result
  std::set<std::vector<int> > seen;
  for (const auto& c : parallel) TEST_ASSERT(seen.insert(c.ordering).second);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateParallel, more_ranks_than_candidates) {
  auto comm = Tpetra::getDefaultComm();
  DenseMatrix C;
  DenseVector b;
  std::vector<int> ranks;
  buildSurrogate(C, b, ranks);

  std::vector<std::vector<int> > two = {{0, 1, 2, 3}, {0, 0, 0, 0}};
  auto parallel = Teko::KrylovSurrogate::scoreCandidates(C, b, ranks, two, 1e-10, *comm);
  Teuchos::SerialComm<int> serialComm;
  auto serial = Teko::KrylovSurrogate::scoreCandidates(C, b, ranks, two, 1e-10, serialComm);

  TEST_EQUALITY(parallel.size(), 2u);
  TEST_ASSERT(sameCandidates(parallel, serial));
}
