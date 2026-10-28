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

#include <algorithm>
#include <set>

#include "Teko_KrylovSurrogateSearch.hpp"

using Teko::KrylovSurrogate::DenseMatrix;
using Teko::KrylovSurrogate::DenseVector;
typedef Teko::KrylovSurrogateReport::Candidate Candidate;

namespace {

// C_hat with identity diagonal blocks and the given off-diagonal value pattern
DenseMatrix identityBlocks(const std::vector<int>& ranks) {
  int R = 0;
  for (int r : ranks) R += r;
  DenseMatrix C(R, R);
  for (int i = 0; i < R; ++i) C(i, i) = 1.0;
  return C;
}

DenseVector onesVector(int R) {
  DenseVector b(R);
  for (int i = 0; i < R; ++i) b(i) = 1.0 + 0.1 * i;
  return b;
}

Candidate scoreOne(const DenseMatrix& C, const DenseVector& b, const std::vector<int>& ranks,
                   const std::vector<int>& ordering) {
  Teuchos::SerialComm<int> comm;
  return Teko::KrylovSurrogate::scoreCandidates(C, b, ranks, {ordering}, 1e-10, comm)[0];
}

}  // namespace

TEUCHOS_UNIT_TEST(tKrylovSurrogateSearch, configuration_counts) {
  const double expected[] = {1, 1, 3, 13, 75, 541};
  for (int n = 1; n <= 5; ++n) {
    TEST_EQUALITY(Teko::KrylovSurrogate::numConfigurations(n), expected[n]);

    auto all = Teko::KrylovSurrogate::allConfigurations(n);
    TEST_EQUALITY(static_cast<double>(all.size()), expected[n]);
    std::set<std::vector<int> > unique(all.begin(), all.end());
    TEST_EQUALITY(unique.size(), all.size());
    for (const auto& o : all) TEST_ASSERT(Teko::KrylovSurrogate::isValidOrdering(o, n));
  }

  TEST_ASSERT(!Teko::KrylovSurrogate::isValidOrdering({0, 2}, 2));   // position 1 unused
  TEST_ASSERT(!Teko::KrylovSurrogate::isValidOrdering({0, 1}, 3));   // wrong length
  TEST_ASSERT(!Teko::KrylovSurrogate::isValidOrdering({-1, 0}, 2));  // negative
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateSearch, candidates_sampled_reproducibly) {
  // 7 blocks have 47293 orderings, more than the 50 asked for, so they are sampled
  const int n               = 7;
  std::vector<int> user     = {0, 0, 1, 1, 2, 2, 3};
  auto first                = Teko::KrylovSurrogate::buildCandidates(n, {user}, 50, 3);
  auto second               = Teko::KrylovSurrogate::buildCandidates(n, {user}, 50, 3);
  std::vector<int> identity = {0, 1, 2, 3, 4, 5, 6};
  std::vector<int> merged(n, 0);

  TEST_EQUALITY(first.size(), 50u);
  TEST_ASSERT(first == second);
  TEST_ASSERT(first[0] == user);
  TEST_ASSERT(std::find(first.begin(), first.end(), identity) != first.end());
  TEST_ASSERT(std::find(first.begin(), first.end(), merged) != first.end());
  std::set<std::vector<int> > unique(first.begin(), first.end());
  TEST_EQUALITY(unique.size(), first.size());
  for (const auto& o : first) TEST_ASSERT(Teko::KrylovSurrogate::isValidOrdering(o, n));

  // few enough to enumerate: every one of them
  TEST_EQUALITY(Teko::KrylovSurrogate::buildCandidates(3, {}, 100, 0).size(), 13u);

  TEST_THROW(Teko::KrylovSurrogate::buildCandidates(3, {{0, 2, 2}}, 100, 0), std::invalid_argument);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateSearch, all_merged_is_exact) {
  std::vector<int> ranks = {3, 2, 4};
  DenseMatrix C          = identityBlocks(ranks);
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      if (C(i, j) == 0.0) C(i, j) = 0.05 * std::sin(1.0 + i + 3.0 * j);

  Candidate c = scoreOne(C, onesVector(9), ranks, {0, 0, 0});
  TEST_EQUALITY(c.status, "converged");
  TEST_EQUALITY(c.iterations, 1);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateSearch, block_diagonal_needs_one_iteration) {
  std::vector<int> ranks = {2, 3, 2};
  DenseMatrix C          = identityBlocks(ranks);
  for (const auto& o : Teko::KrylovSurrogate::allConfigurations(3)) {
    Candidate c = scoreOne(C, onesVector(7), ranks, o);
    TEST_EQUALITY(c.status, "converged");
    TEST_EQUALITY(c.iterations, 1);
  }
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateSearch, lower_coupling_and_its_reverse) {
  // C = [I 0; L I]. The identity ordering keeps L, so K = I. Reversed, L is
  // dropped, K = C, and K - I is nilpotent: exactly two iterations.
  std::vector<int> ranks = {2, 2};
  DenseMatrix C          = identityBlocks(ranks);
  C(2, 0)                = 0.7;
  C(2, 1)                = -0.3;
  C(3, 0)                = 0.2;
  C(3, 1)                = 0.9;
  DenseVector b          = onesVector(4);

  Candidate keep = scoreOne(C, b, ranks, {0, 1});
  TEST_EQUALITY(keep.status, "converged");
  TEST_EQUALITY(keep.iterations, 1);

  Candidate drop = scoreOne(C, b, ranks, {1, 0});
  TEST_EQUALITY(drop.status, "converged");
  TEST_EQUALITY(drop.iterations, 2);

  DenseMatrix S = Teko::KrylovSurrogate::gaussSeidelSelection(C, ranks, {1, 0});
  TEST_EQUALITY(S(2, 0), 0.0);
  TEST_EQUALITY(S(0, 0), 1.0);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateSearch, singular_merged_group) {
  // C = [1 1 0; 1 1 1; 0 1 1] is nonsingular, but merging blocks 0 and 1
  // gives the singular group block [1 1; 1 1]
  std::vector<int> ranks = {1, 1, 1};
  DenseMatrix C          = identityBlocks(ranks);
  C(0, 1)                = 1.0;
  C(1, 0)                = 1.0;
  C(1, 2)                = 1.0;
  C(2, 1)                = 1.0;

  Candidate c = scoreOne(C, onesVector(3), ranks, {0, 0, 1});
  TEST_EQUALITY(c.status, "singular group block");

  Candidate gs = scoreOne(C, onesVector(3), ranks, {0, 1, 2});
  TEST_EQUALITY(gs.status, "converged");
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateSearch, sorted_and_complete) {
  std::vector<int> ranks = {2, 2, 2};
  DenseMatrix C          = identityBlocks(ranks);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      if (i / 2 != j / 2) C(i, j) = 0.3 * std::cos(2.0 * i - j);

  Teuchos::SerialComm<int> comm;
  auto candidates = Teko::KrylovSurrogate::allConfigurations(3);
  auto scored =
      Teko::KrylovSurrogate::scoreCandidates(C, onesVector(6), ranks, candidates, 1e-10, comm);
  TEST_EQUALITY(scored.size(), candidates.size());
  for (std::size_t i = 1; i < scored.size(); ++i) {
    if (scored[i].status == scored[i - 1].status)
      TEST_ASSERT(scored[i].iterations >= scored[i - 1].iterations);
  }
  for (const auto& c : scored) {
    if (c.status != "converged") continue;
    TEST_ASSERT(c.finalRelativeResidual <= 1e-9);
    TEST_ASSERT(c.rate >= 0.0 && c.rate < 1.0);
  }
}
