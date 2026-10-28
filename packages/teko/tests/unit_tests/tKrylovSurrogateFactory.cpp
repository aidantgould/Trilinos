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

#include <algorithm>
#include <cmath>

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosThyraAdapter.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_MultiVectorStdOps.hpp"

#include "Teko_InverseLibrary.hpp"
#include "Teko_KrylovSurrogateRecorderOp.hpp"
#include "Teko_KrylovSurrogateReport.hpp"
#include "Teko_StratimikosFactory.hpp"
#include "tKrylovSurrogateProblem.hpp"

using namespace KrylovSurrogateTest;
using Teuchos::ParameterList;
using Teuchos::RCP;

namespace {

const std::vector<int> kSizes = {60, 48, 8};

// Block Gauss-Seidel with exact (Amesos2) diagonal solves, and the surrogate
// wrapped around it
ParameterList inverseLibraryList(const RCP<Teko::KrylovSurrogateHandle>& handle) {
  ParameterList ifl;
  ifl.sublist("BGS").set("Type", "Block Gauss-Seidel");
  ifl.sublist("BGS").set("Inverse Type", "Amesos2");
  ParameterList& s = ifl.sublist("Surrogate");
  s.set("Type", "Krylov Surrogate");
  s.set("Inverse Factory", "BGS");
  s.set("Report Handle", handle);
  s.sublist("Search").set("Tolerance", 1e-8);
  return ifl;
}

double maxAbs(const Teko::MultiVector& v) {
  std::vector<double> norms(v->domain()->dim());
  Thyra::norms_inf<double>(*v, Teuchos::arrayViewFromVector(norms));
  return *std::max_element(norms.begin(), norms.end());
}

// Right-preconditioned flexible GMRES; returns the iteration count
int solve(const Teko::LinearOp& A, const Teko::LinearOp& prec, const Teko::MultiVector& b,
          const Teko::MultiVector& x) {
  typedef Thyra::MultiVectorBase<double> MV;
  typedef Thyra::LinearOpBase<double> OP;
  auto problem = Teuchos::rcp(new Belos::LinearProblem<double, MV, OP>(A, x, b));
  problem->setRightPrec(prec);
  problem->setProblem();
  auto params = Teuchos::rcp(new ParameterList);
  params->set("Flexible Gmres", true);
  params->set("Num Blocks", 100);
  params->set("Maximum Iterations", 100);
  params->set("Convergence Tolerance", 1e-10);
  Belos::BlockGmresSolMgr<double, MV, OP> solver(problem, params);
  solver.solve();
  return solver.getNumIters();
}

Teko::MultiVector rhs(const Teko::BlockedLinearOp& A) {
  Teko::MultiVector b = Thyra::createMembers(A->range(), 1);
  fillBlocked(b, [](int j, int r, int) { return 1.0 + std::sin(0.3 * r + j); });
  return b;
}

}  // namespace

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, report_from_a_solve) {
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogateHandle::create();
  auto invLib = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));

  Teko::LinearOp surrogate = Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A);
  Teko::LinearOp bare      = Teko::buildInverse(*invLib->getInverseFactory("BGS"), A);
  TEST_ASSERT(handle->isAttached());

  // the recorder changes nothing about the solve
  Teko::MultiVector b  = rhs(A);
  Teko::MultiVector xs = Thyra::createMembers(A->domain(), 1);
  Teko::MultiVector xb = Thyra::createMembers(A->domain(), 1);
  Thyra::assign(xs.ptr(), 0.0);
  Thyra::assign(xb.ptr(), 0.0);
  const int itS = solve(A, surrogate, b, xs);
  const int itB = solve(A, bare, b, xb);
  TEST_EQUALITY(itS, itB);
  Thyra::update(-1.0, *xb, xs.ptr());
  TEST_ASSERT(maxAbs(xs) == 0.0);

  Teko::KrylovSurrogateReport report = handle->buildReport();
  out << "iterations " << itS << ", samples " << report.numSamples << ", ranks";
  for (int r : report.ranks) out << " " << r;
  out << ", diagonal deviations";
  for (double d : report.diagonalDeviations) out << " " << d;
  out << ", rhs capture " << report.rhsCapture << "\n";

  TEST_EQUALITY(report.numBlocks, 3);
  TEST_ASSERT(report.numSamples >= 1);
  TEST_ASSERT(report.blockSizes == kSizes);
  for (int j = 0; j < 3; ++j) {
    TEST_ASSERT(report.ranks[j] >= 1);
    TEST_ASSERT(report.ranks[j] <= std::min(report.numSamples, kSizes[j]));
    TEST_ASSERT(report.diagonalDeviations[j] < 1e-6);
  }
  TEST_ASSERT(report.rhsCapture > 0.0 && report.rhsCapture <= 1.0 + 1e-12);
  TEST_EQUALITY(report.CHat.numRows(), report.bHat.length());

  // three blocks: all 13 orderings, the all-merged one exact in one iteration
  TEST_EQUALITY(report.candidates.size(), 13u);
  for (const auto& c : report.candidates) {
    TEST_ASSERT(c.status == "converged" || c.status == "max iterations" ||
                c.status == "singular group block");
    if (c.ordering == std::vector<int>({0, 0, 0})) {
      TEST_EQUALITY(c.status, "converged");
      TEST_EQUALITY(c.iterations, 1);
    }
  }
  TEST_EQUALITY(report.searchRanks, Tpetra::getDefaultComm()->getSize());

  auto pl = report.toParameterList();
  TEST_ASSERT(pl->isParameter("C_hat"));
  TEST_ASSERT(pl->sublist("Candidates").isSublist("Candidate 12"));
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, warm_start_and_rebuild) {
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogateHandle::create();
  auto invLib  = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
  auto factory = invLib->getInverseFactory("Surrogate");

  Teko::ModifiableLinearOp prec = Teko::buildInverse(*factory, A);
  Teko::MultiVector b           = rhs(A);
  Teko::MultiVector x           = Thyra::createMembers(A->domain(), 1);
  Thyra::assign(x.ptr(), 0.0);
  solve(A, prec, b, x);
  TEST_ASSERT(handle->getRecorder()->numSamples() > 0);

  // a rebuild clears the samples; the next solve starts warm
  Teko::rebuildInverse(*factory, A, prec);
  TEST_EQUALITY(handle->getRecorder()->numSamples(), 0);

  fillBlocked(x, [](int j, int r, int) { return 0.1 * std::cos(r + 2.0 * j); });
  Teko::MultiVector r0 = b->clone_mv();
  A->apply(Thyra::NOTRANS, *x, r0.ptr(), -1.0, 1.0);
  std::vector<double> norm(1);
  Thyra::norms_2<double>(*r0, Teuchos::arrayViewFromVector(norm));
  Thyra::scale(1.0 / norm[0], r0.ptr());

  solve(A, prec, b, x);
  Teko::MultiVector v1 = handle->getRecorder()->getFirstInput()->clone_mv();
  Thyra::update(-1.0, *r0, v1.ptr());
  out << "first input vs r0 / ||r0||: " << maxAbs(v1) << "\n";
  TEST_ASSERT(maxAbs(v1) < 1e-12);

  TEST_NOTHROW(handle->buildReport());
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, through_stratimikos) {
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogateHandle::create();

  Stratimikos::DefaultLinearSolverBuilder builder;
  Teko::addTekoToStratimikosBuilder(builder);
  auto params = Teuchos::rcp(new ParameterList(*builder.getValidParameters()));
  params->set("Linear Solver Type", "Belos");
  ParameterList& belos = params->sublist("Linear Solver Types").sublist("Belos");
  belos.set("Solver Type", "Block GMRES");
  ParameterList& gmres = belos.sublist("Solver Types").sublist("Block GMRES");
  gmres.set("Flexible Gmres", true);
  gmres.set("Convergence Tolerance", 1e-10);
  gmres.set("Maximum Iterations", 100);
  gmres.set("Num Blocks", 100);
  params->set("Preconditioner Type", "Teko");
  ParameterList& teko = params->sublist("Preconditioner Types").sublist("Teko");
  teko.set("Inverse Type", "Surrogate");
  teko.sublist("Inverse Factory Library") = inverseLibraryList(handle);
  builder.setParameterList(params);

  auto lowsFactory    = builder.createLinearSolveStrategy("");
  auto lows           = Thyra::linearOpWithSolve<double>(*lowsFactory, A);
  Teko::MultiVector b = rhs(A);
  Teko::MultiVector x = Thyra::createMembers(A->domain(), 1);
  Thyra::assign(x.ptr(), 0.0);
  Thyra::solve<double>(*lows, Thyra::NOTRANS, *b, x.ptr());

  TEST_ASSERT(handle->isAttached());
  Teko::KrylovSurrogateReport report = handle->buildReport();
  TEST_ASSERT(report.numSamples > 0);
  TEST_EQUALITY(report.candidates.size(), 13u);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, errors) {
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});

  // no handle
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogateHandle::create());
    ifl.sublist("Surrogate").remove("Report Handle");
    auto invLib = Teko::InverseLibrary::buildFromParameterList(ifl);
    TEST_THROW(invLib->getInverseFactory("Surrogate"), std::runtime_error);
  }
  // a misspelled option
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogateHandle::create());
    ifl.sublist("Surrogate").set("Maximum Sample", 10);
    auto invLib = Teko::InverseLibrary::buildFromParameterList(ifl);
    TEST_THROW(invLib->getInverseFactory("Surrogate"), std::invalid_argument);
  }
  // a report before any build, and before any solve
  {
    auto handle = Teko::KrylovSurrogateHandle::create();
    TEST_THROW(handle->buildReport(), std::runtime_error);
    auto invLib         = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
    Teko::LinearOp prec = Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A);
    TEST_THROW(handle->buildReport(), std::runtime_error);
  }
  // an operator that is not blocked
  {
    auto handle = Teko::KrylovSurrogateHandle::create();
    auto invLib = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
    Teko::LinearOp single = Teko::getBlock(0, 0, A);
    TEST_THROW(Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), single),
               std::runtime_error);
  }
}
