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

#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"

#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"

#include "Teko_KrylovSurrogateRecorderOp.hpp"

typedef Teko::ST ST;
typedef Teko::LO LO;
typedef Teko::GO GO;
typedef Teko::NT NT;
using Teuchos::RCP;
using Teuchos::rcp;

namespace {

// diag(1, 2, 3, ...) as a modifiable Thyra operator
Teko::ModifiableLinearOp diagonalOp(GO n) {
  auto map = rcp(new Tpetra::Map<LO, GO, NT>(n, 0, Tpetra::getDefaultComm()));
  auto mat = Tpetra::createCrsMatrix<ST, LO, GO, NT>(map, 1);
  for (size_t i = 0; i < map->getLocalNumElements(); ++i) {
    GO gid = map->getGlobalElement(i);
    ST v   = static_cast<ST>(gid + 1);
    mat->insertGlobalValues(gid, Teuchos::tuple<GO>(gid), Teuchos::tuple<ST>(v));
  }
  mat->fillComplete();
  return Thyra::tpetraLinearOp<ST, LO, GO, NT>(Thyra::tpetraVectorSpace<ST, LO, GO, NT>(map),
                                               Thyra::tpetraVectorSpace<ST, LO, GO, NT>(map), mat);
}

double maxDiff(const Teko::MultiVector& a, const Teko::MultiVector& b) {
  Teko::MultiVector d = a->clone_mv();
  Thyra::update(-1.0, *b, d.ptr());
  std::vector<double> norms(d->domain()->dim());
  Thyra::norms_inf<double>(*d, Teuchos::arrayViewFromVector(norms));
  return *std::max_element(norms.begin(), norms.end());
}

Teko::MultiVector filled(const Teko::LinearOp& op, int cols, double seed) {
  Teko::MultiVector x = Thyra::createMembers(op->domain(), cols);
  Thyra::seed_randomize<double>(static_cast<unsigned int>(seed));
  Thyra::randomize(-1.0, 1.0, x.ptr());
  return x;
}

}  // namespace

TEUCHOS_UNIT_TEST(tKrylovSurrogateRecorderOp, records_outputs_in_order) {
  Teko::ModifiableLinearOp M = diagonalOp(12);
  Teko::KrylovSurrogateRecorderOp rec(M, 5);

  Teko::MultiVector x1 = filled(M, 2, 1), x2 = filled(M, 1, 2);
  Teko::MultiVector y1 = Thyra::createMembers(M->range(), 2);
  Teko::MultiVector y2 = Thyra::createMembers(M->range(), 1);
  rec.apply(Thyra::NOTRANS, *x1, y1.ptr(), 1.0, 0.0);
  rec.apply(Thyra::NOTRANS, *x2, y2.ptr(), 1.0, 0.0);

  // the recorder changes nothing
  Teko::MultiVector ref1 = Thyra::createMembers(M->range(), 2);
  M->apply(Thyra::NOTRANS, *x1, ref1.ptr(), 1.0, 0.0);
  TEST_FLOATING_EQUALITY(maxDiff(y1, ref1) + 1.0, 1.0, 1e-15);

  TEST_EQUALITY(rec.numSamples(), 3);
  Teko::MultiVector Z = rec.getSamples();
  TEST_FLOATING_EQUALITY(maxDiff(Z->subView(Teuchos::Range1D(0, 1)), y1) + 1.0, 1.0, 1e-15);
  TEST_FLOATING_EQUALITY(maxDiff(Z->subView(Teuchos::Range1D(2, 2)), y2) + 1.0, 1.0, 1e-15);

  // the first input is the first column of the first apply
  TEST_FLOATING_EQUALITY(maxDiff(rec.getFirstInput(), x1->subView(Teuchos::Range1D(0, 0))) + 1.0,
                         1.0, 1e-15);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateRecorderOp, cap_and_clear) {
  Teko::ModifiableLinearOp M = diagonalOp(8);
  Teko::KrylovSurrogateRecorderOp rec(M, 3);

  Teko::MultiVector x = filled(M, 2, 3);
  Teko::MultiVector y = Thyra::createMembers(M->range(), 2);
  rec.apply(Thyra::NOTRANS, *x, y.ptr(), 1.0, 0.0);
  rec.apply(Thyra::NOTRANS, *x, y.ptr(), 1.0, 0.0);
  TEST_EQUALITY(rec.numSamples(), 3);

  rec.clearSamples();
  TEST_EQUALITY(rec.numSamples(), 0);
  TEST_ASSERT(rec.getSamples() == Teuchos::null);
  TEST_ASSERT(rec.getFirstInput() == Teuchos::null);

  Teko::MultiVector x2 = filled(M, 1, 4);
  rec.apply(Thyra::NOTRANS, *x2, y->subView(Teuchos::Range1D(0, 0)).ptr(), 1.0, 0.0);
  TEST_EQUALITY(rec.numSamples(), 1);
  TEST_FLOATING_EQUALITY(maxDiff(rec.getFirstInput(), x2) + 1.0, 1.0, 1e-15);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateRecorderOp, alpha_and_beta) {
  Teko::ModifiableLinearOp M = diagonalOp(10);
  Teko::KrylovSurrogateRecorderOp rec(M, 4);

  Teko::MultiVector x  = filled(M, 1, 5);
  Teko::MultiVector y  = filled(M, 1, 6);
  Teko::MultiVector y0 = y->clone_mv();
  rec.apply(Thyra::NOTRANS, *x, y.ptr(), 2.0, 3.0);

  Teko::MultiVector Mx = Thyra::createMembers(M->range(), 1);
  M->apply(Thyra::NOTRANS, *x, Mx.ptr(), 1.0, 0.0);

  // y = 2 M x + 3 y0, while the sample is M x itself
  Teko::MultiVector expect = y0->clone_mv();
  Thyra::scale(3.0, expect.ptr());
  Thyra::update(2.0, *Mx, expect.ptr());
  TEST_FLOATING_EQUALITY(maxDiff(y, expect) + 1.0, 1.0, 1e-14);
  TEST_FLOATING_EQUALITY(maxDiff(rec.getSamples(), Mx) + 1.0, 1.0, 1e-15);
}
