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
#include <Teuchos_LAPACK.hpp>

#include <algorithm>
#include <cmath>

#include "Thyra_MultiVectorStdOps.hpp"

#include "Teko_KrylovSurrogateReducedOperator.hpp"
#include "tKrylovSurrogateProblem.hpp"

using namespace KrylovSurrogateTest;
using Teko::KrylovSurrogate::ReducedOperator;

namespace {

const int kSamples = 10;
const double kTol  = 1e-10;

double zEntry(int j, int r, int c) { return std::sin(0.37 * (r + 1) * (c + 1) + 0.5 * j); }
double vEntry(int j, int r) { return std::cos(0.21 * r + j); }

// The dense reference: V_j = A_jj Z_j, its truncated SVD U_j, Y_j = A_jj^{-1} U_j,
// C_ij = U_i^T A_ij Y_j and b_j = U_j^T v_j
struct Reference {
  std::vector<int> ranks;
  Dense C;
  std::vector<double> b;
};

Reference denseReference(const std::vector<int>& sizes, int m) {
  Teuchos::LAPACK<int, double> lapack;
  const int nb = static_cast<int>(sizes.size());
  Reference ref;
  std::vector<Dense> U(nb), Y(nb);
  for (int j = 0; j < nb; ++j) {
    const int n = sizes[j];
    Dense Ajj   = denseBlock(sizes, j, j), Z(n, m), V(n, m);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < m; ++c) Z(r, c) = zEntry(j, r, c);
    V.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, Ajj, Z, 0.0);

    const int s = std::min(n, m);
    std::vector<double> sigma(s), work(10 * (n + m));
    Dense Us(n, s), VT(s, m);
    int info;
    lapack.GESVD('S', 'S', n, m, V.values(), V.stride(), sigma.data(), Us.values(), Us.stride(),
                 VT.values(), VT.stride(), work.data(), static_cast<int>(work.size()), nullptr,
                 &info);
    int r = 0;
    while (r < s && sigma[r] >= kTol * sigma[0]) ++r;
    ref.ranks.push_back(r);

    U[j].shape(n, r);
    for (int c = 0; c < r; ++c)
      for (int i = 0; i < n; ++i) U[j](i, c) = Us(i, c);

    Y[j] = U[j];
    std::vector<int> piv(n);
    lapack.GESV(n, r, Ajj.values(), Ajj.stride(), piv.data(), Y[j].values(), Y[j].stride(), &info);
  }

  int R = 0;
  std::vector<int> off(nb + 1, 0);
  for (int j = 0; j < nb; ++j) off[j + 1] = R += ref.ranks[j];
  ref.C.shape(R, R);
  ref.b.assign(R, 0.0);
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < nb; ++j) {
      Dense Aij = denseBlock(sizes, i, j), AY(sizes[i], ref.ranks[j]),
            Cij(ref.ranks[i], ref.ranks[j]);
      AY.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, Aij, Y[j], 0.0);
      Cij.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, U[i], AY, 0.0);
      for (int a = 0; a < ref.ranks[i]; ++a)
        for (int c = 0; c < ref.ranks[j]; ++c) ref.C(off[i] + a, off[j] + c) = Cij(a, c);
    }
    for (int a = 0; a < ref.ranks[i]; ++a)
      for (int r = 0; r < sizes[i]; ++r) ref.b[off[i] + a] += U[i](r, a) * vEntry(i, r);
  }
  return ref;
}

ReducedOperator build(const std::vector<int>& sizes, const std::vector<Distribution>& d, int m) {
  Teko::BlockedLinearOp A = buildBlockedOp(sizes, d);
  Teko::MultiVector Z     = Thyra::createMembers(A->domain(), m);
  Teko::MultiVector v1    = Thyra::createMembers(A->range(), 1);
  fillBlocked(Z, [](int j, int r, int c) { return zEntry(j, r, c); });
  fillBlocked(v1, [](int j, int r, int) { return vEntry(j, r); });
  return Teko::KrylovSurrogate::buildReducedOperator(A, Z, v1, kTol);
}

// Each basis vector is determined up to sign, so C_hat matches the reference
// entry by entry in magnitude
void compare(const ReducedOperator& red, const Reference& ref, Teuchos::FancyOStream& out,
             bool& success) {
  TEST_ASSERT(red.ranks == ref.ranks);
  if (red.ranks != ref.ranks) return;
  const int R  = ref.C.numRows();
  double scale = 0.0, worst = 0.0, worstB = 0.0;
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < R; ++j) {
      scale = std::max(scale, std::abs(ref.C(i, j)));
      worst = std::max(worst, std::abs(std::abs(red.CHat(i, j)) - std::abs(ref.C(i, j))));
    }
  for (int i = 0; i < R; ++i)
    worstB = std::max(worstB, std::abs(std::abs(red.bHat(i)) - std::abs(ref.b[i])));
  out << "C_hat max deviation from reference " << worst / scale << ", b_hat " << worstB << "\n";
  TEST_ASSERT(worst <= 1e-8 * scale);
  TEST_ASSERT(worstB <= 1e-10);
  for (double d : red.diagonalDeviations) TEST_ASSERT(d < 1e-8);
}

}  // namespace

TEUCHOS_UNIT_TEST(tKrylovSurrogateReducedOperator, matches_dense_reference) {
  // the last block has fewer rows than samples
  std::vector<int> sizes = {60, 48, 8};
  ReducedOperator red    = build(sizes, {EVEN, EVEN, EVEN}, kSamples);
  Reference ref          = denseReference(sizes, kSamples);
  compare(red, ref, out, success);

  TEST_ASSERT(red.ranks[2] <= sizes[2]);
  TEST_ASSERT(red.rhsCapture > 0.0 && red.rhsCapture <= 1.0 + 1e-12);
  TEST_ASSERT(red.blockSizes == sizes);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateReducedOperator, any_distribution) {
  // a block entirely on one rank: at 4 ranks every other rank holds none of it
  std::vector<int> sizes = {60, 48, 8};
  Reference ref          = denseReference(sizes, kSamples);
  compare(build(sizes, {ALL_ON_ONE_RANK, EVEN, EVEN}, kSamples), ref, out, success);
  compare(build(sizes, {EVEN, EVEN, ALL_ON_ONE_RANK}, kSamples), ref, out, success);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateReducedOperator, resolves_below_gram_floor) {
  // One diagonal block D, with Z = D^{-1} U diag(sigma) for orthonormal sine
  // columns U, so V_tilde = A Z has singular values sigma exactly. With
  // sigma_c = 10^{-1.1 c}, ten lie above the 1e-10 cut; the Gram route could
  // resolve only the seven above 4.2e-8.
  const int n = 40, m = 12;
  auto comm  = Tpetra::getDefaultComm();
  auto map   = Teuchos::rcp(new Map(n, 0, comm));
  auto space = Thyra::tpetraVectorSpace<ST, LO, GO, NT>(map);
  auto D     = Teuchos::rcp(new CrsMatrix(map, 1));
  for (size_t lr = 0; lr < map->getLocalNumElements(); ++lr) {
    const GO r = map->getGlobalElement(lr);
    D->insertGlobalValues(r, Teuchos::tuple<GO>(r), Teuchos::tuple<ST>(1.0 + 0.01 * r));
  }
  D->fillComplete();

  auto op = Thyra::defaultBlockedLinearOp<ST>();
  op->beginBlockFill(1, 1);
  op->setBlock(0, 0, Thyra::tpetraLinearOp<ST, LO, GO, NT>(space, space, D));
  op->endBlockFill();
  Teko::BlockedLinearOp A = op;

  const double pi      = std::acos(-1.0);
  Teko::MultiVector Z  = Thyra::createMembers(A->domain(), m);
  Teko::MultiVector v1 = Thyra::createMembers(A->range(), 1);
  fillBlocked(Z, [&](int, int r, int c) {
    const double u = std::sqrt(2.0 / (n + 1)) * std::sin(pi * (r + 1) * (c + 1) / (n + 1));
    return u * std::pow(10.0, -1.1 * c) / (1.0 + 0.01 * r);
  });
  fillBlocked(v1, [](int, int r, int) { return 1.0 + 0.1 * r; });

  ReducedOperator red = Teko::KrylovSurrogate::buildReducedOperator(A, Z, v1, kTol);
  TEST_EQUALITY(red.ranks[0], 10);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateReducedOperator, rank_zero_svd) {
  auto comm = Tpetra::getDefaultComm();
  Dense M(3, 3);
  M(0, 0)  = 3.0;
  M(1, 1)  = 2.0;
  M(2, 2)  = 1e-12;
  auto red = Teko::KrylovSurrogate::reduceOnRankZero(M, 1e-10, *comm);
  TEST_EQUALITY(red.rank, 2);
  TEST_FLOATING_EQUALITY(red.sigma[0], 3.0, 1e-14);
  TEST_FLOATING_EQUALITY(red.sigma[1], 2.0, 1e-14);
  TEST_EQUALITY(red.U.numRows(), 3);
  TEST_EQUALITY(red.W.numCols(), 2);

  Dense zero(2, 2);
  TEST_THROW(Teko::KrylovSurrogate::reduceOnRankZero(zero, 1e-10, *comm), std::runtime_error);
}
