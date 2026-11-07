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
#include <sstream>
#include <string>
#include <type_traits>

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosThyraAdapter.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_TwoDArray.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_MultiVectorStdOps.hpp"

#include "Teko_InverseLibrary.hpp"
#include "Teko_KrylovSurrogatePreconditionerFactory.hpp"
#include "Teko_KrylovSurrogateRecorderOp.hpp"
#include "Teko_KrylovSurrogateReport.hpp"
#include "Teko_StratimikosFactory.hpp"
#include "tKrylovSurrogateProblem.hpp"

using namespace KrylovSurrogateTest;
using Teuchos::ParameterList;
using Teuchos::RCP;
typedef Teko::KrylovSurrogate::Report::Candidate Candidate;

namespace {

const std::vector<int> kSizes = {60, 48, 8};

std::vector<long long> sizesAsReported() {
  return std::vector<long long>(kSizes.begin(), kSizes.end());
}

// Block Gauss-Seidel with exact (Amesos2) diagonal solves, and the surrogate
// wrapped around it
ParameterList inverseLibraryList(const RCP<Teko::KrylovSurrogate::Handle>& handle) {
  ParameterList ifl;
  ifl.sublist("BGS").set("Type", "Block Gauss-Seidel");
  ifl.sublist("BGS").set("Inverse Type", "Amesos2");
  ParameterList& s = ifl.sublist("Surrogate");
  s.set("Type", "Krylov Surrogate");
  s.set("Inverse Factory", "BGS");
  s.set("Report Handle", handle);
  // the solve's tolerance, so predicted iterations compare with real ones
  s.sublist("Search").set("Tolerance", 1e-10);
  return ifl;
}

// The same, without a handle, printing at each rebuild
ParameterList printingLibraryList() {
  ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogate::Handle::create());
  ifl.sublist("Surrogate").remove("Report Handle");
  ifl.sublist("Surrogate").sublist("Report").set("Print", true);
  ifl.sublist("Surrogate").sublist("Report").set("Inverse Type", "Amesos2");
  ifl.sublist("Surrogate").sublist("Report").set("Merged Inverse Type", "BGS");
  return ifl;
}

double maxAbs(const Teko::MultiVector& v) {
  std::vector<double> norms(v->domain()->dim());
  Thyra::norms_inf<double>(*v, Teuchos::arrayViewFromVector(norms));
  return *std::max_element(norms.begin(), norms.end());
}

// Right-preconditioned flexible GMRES; returns the iteration count
int solve(const Teko::LinearOp& A, const Teko::LinearOp& prec, const Teko::MultiVector& b,
          const Teko::MultiVector& x, int numBlocks = 100) {
  typedef Thyra::MultiVectorBase<double> MV;
  typedef Thyra::LinearOpBase<double> OP;
  auto problem = Teuchos::rcp(new Belos::LinearProblem<double, MV, OP>(A, x, b));
  problem->setRightPrec(prec);
  problem->setProblem();
  auto params = Teuchos::rcp(new ParameterList);
  params->set("Flexible Gmres", true);
  params->set("Num Blocks", numBlocks);
  params->set("Maximum Iterations", 100);
  params->set("Convergence Tolerance", 1e-10);
  Belos::BlockGmresSolMgr<double, MV, OP> solver(problem, params);
  TEUCHOS_TEST_FOR_EXCEPTION(solver.solve() != Belos::Converged, std::runtime_error,
                             "the test solve did not converge");
  return solver.getNumIters();
}

Teko::MultiVector rhs(const Teko::BlockedLinearOp& A) {
  Teko::MultiVector b = Thyra::createMembers(A->range(), 1);
  fillBlocked(b, [](int j, int r, int) { return 1.0 + std::sin(0.3 * r + j); });
  return b;
}

Teko::MultiVector zeros(const Teko::BlockedLinearOp& A) {
  Teko::MultiVector x = Thyra::createMembers(A->domain(), 1);
  Thyra::assign(x.ptr(), 0.0);
  return x;
}

// Swaps Teuchos's default output stream for one writing into a string on rank
// 0, and restores it however the scope ends
class CapturedOutput {
 public:
  CapturedOutput() : previous_(Teuchos::VerboseObjectBase::getDefaultOStream()) {
    auto stream = Teuchos::fancyOStream(Teuchos::rcpFromRef(captured_));
    stream->setOutputToRootOnly(0);  // as the default stream does
    Teuchos::VerboseObjectBase::setDefaultOStream(stream);
  }
  ~CapturedOutput() { Teuchos::VerboseObjectBase::setDefaultOStream(previous_); }
  std::string str() const { return captured_.str(); }

 private:
  std::ostringstream captured_;
  RCP<Teuchos::FancyOStream> previous_;
};

// The inverse library entry print() writes, cut out of its output
std::string printedBest(const std::string& printed) {
  const std::size_t begin = printed.find("<ParameterList name=\"Krylov Surrogate Best\">");
  const std::size_t end   = printed.rfind("</ParameterList>");
  if (begin == std::string::npos || end == std::string::npos) return "";
  return printed.substr(begin, end + std::string("</ParameterList>").size() - begin);
}

// Stratimikos: Belos flexible GMRES, with Teko's "Surrogate" from ifl as the
// preconditioner
RCP<Thyra::LinearOpWithSolveFactoryBase<double> > stratimikosSolver(
    const ParameterList& ifl, const std::string& reorderType = "",
    const std::string& stridedBlocking = "1") {
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
  teko.set("Reorder Type", reorderType);
  teko.set("Strided Blocking", stridedBlocking);
  teko.sublist("Inverse Factory Library") = ifl;
  builder.setParameterList(params);
  return builder.createLinearSolveStrategy("");
}

// "Exact Group": unpreconditioned GMRES converged far below any check's
// tolerance, the exact merged-group solve the surrogate scores
void addExactGroup(Teko::InverseLibrary& invLib) {
  ParameterList exact;
  exact.set("Type", "Belos");
  exact.set("Solver Type", "Block GMRES");
  ParameterList& gmres = exact.sublist("Solver Types").sublist("Block GMRES");
  gmres.set("Convergence Tolerance", 1e-13);
  gmres.set("Maximum Iterations", 400);
  gmres.set("Num Blocks", 400);
  invLib.addInverse("Exact Group", exact);
}

int numCalls(const char* timer) {
  auto t = Teuchos::TimeMonitor::lookupCounter(timer);
  return t == Teuchos::null ? 0 : t->numCalls();
}

static_assert(!std::is_copy_constructible<Teko::KrylovSurrogate::Handle>::value,
              "a Handle is shared through RCP only");

}  // namespace

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, report_from_a_solve) {
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogate::Handle::create();
  auto invLib = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));

  Teko::LinearOp surrogate = Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A);
  Teko::LinearOp bare      = Teko::buildInverse(*invLib->getInverseFactory("BGS"), A);
  TEST_ASSERT(handle->isAttached());

  // the recorder changes nothing about the solve
  Teko::MultiVector b  = rhs(A);
  Teko::MultiVector xs = zeros(A);
  Teko::MultiVector xb = zeros(A);
  const int itS        = solve(A, surrogate, b, xs);
  const int itB        = solve(A, bare, b, xb);
  TEST_EQUALITY(itS, itB);
  Thyra::update(-1.0, *xb, xs.ptr());
  TEST_ASSERT(maxAbs(xs) == 0.0);

  const char* timers[]    = {"Teko Krylov Surrogate: Build Surrogate",
                             "Teko Krylov Surrogate: Search"};
  const int callsBefore[] = {numCalls(timers[0]), numCalls(timers[1])};

  Teko::KrylovSurrogate::Report report = handle->buildReport();
  out << "iterations " << itS << ", samples " << report.numSamples << ", basis sizes";
  for (int r : report.basisSizes) out << " " << r;
  out << ", diagonal deviations";
  for (double d : report.diagonalDeviations) out << " " << d;
  out << ", rhs capture " << report.rhsCapture << "\n";

  TEST_EQUALITY(report.numBlocks, 3);
  TEST_ASSERT(report.numSamples >= 1);
  TEST_ASSERT(report.blockSizes == sizesAsReported());
  for (int j = 0; j < 3; ++j) {
    TEST_ASSERT(report.basisSizes[j] >= 1);
    TEST_ASSERT(report.basisSizes[j] <= std::min(report.numSamples, kSizes[j]));
    TEST_ASSERT(report.diagonalDeviations[j] < 1e-6);
  }
  TEST_ASSERT(report.rhsCapture > 0.0 && report.rhsCapture <= 1.0 + 1e-12);
  TEST_EQUALITY(report.CHat.numRows(), report.bHat.length());

  // three blocks: all 13 orderings, the all-merged one exact in one iteration
  TEST_EQUALITY(report.candidates.size(), 13u);
  for (const auto& c : report.candidates) {
    TEST_ASSERT(c.status == Candidate::CONVERGED || c.status == Candidate::NOT_CONVERGED ||
                c.status == Candidate::SINGULAR_GROUP_BLOCK);
    if (c.ordering == std::vector<int>({0, 0, 0})) {
      TEST_EQUALITY(c.status, Candidate::CONVERGED);
      TEST_EQUALITY(c.iterations, 1);
    }
  }
  TEST_EQUALITY(report.searchRanks, Tpetra::getDefaultComm()->getSize());

  // the cost shows up in the application's timer summary, once per report
  for (int t = 0; t < 2; ++t) TEST_EQUALITY(numCalls(timers[t]), callsBefore[t] + 1);

  auto pl = report.toParameterList();
  TEST_ASSERT(pl->isParameter("C_hat"));
  TEST_ASSERT(pl->sublist("Candidates").isSublist("Candidate 12"));
  TEST_EQUALITY(pl->sublist("Candidates").sublist("Candidate 0").get<std::string>("Status"),
                Teko::KrylovSurrogate::statusName(report.candidates[0].status));

  // the parameter list survives a trip through XML, for logging
  std::ostringstream xmlLog;
  Teuchos::writeParameterListToXmlOStream(*pl, xmlLog);
  RCP<ParameterList> logged = Teuchos::getParametersFromXmlString(xmlLog.str());
  TEST_ASSERT(logged->get<Teuchos::Array<long long> >("Block Sizes") ==
              Teuchos::Array<long long>(report.blockSizes));
  const auto C = logged->get<Teuchos::TwoDArray<double> >("C_hat");
  TEST_EQUALITY(static_cast<int>(C.getNumRows()), report.CHat.numRows());
  TEST_FLOATING_EQUALITY(C(0, 0), report.CHat(0, 0), 1e-14);
  TEST_FLOATING_EQUALITY(C(1, 0), report.CHat(1, 0), 1e-14);

  // the printed best is ready to paste: it parses back and builds
  std::ostringstream printed;
  report.print(printed);
  const std::string xml = printedBest(printed.str());
  TEST_ASSERT(!xml.empty());
  RCP<ParameterList> best = Teuchos::getParametersFromXmlString(xml);
  TEST_EQUALITY(best->get<std::string>("Type"), "Hierarchical Block Gauss-Seidel");
  for (int g = 1; best->isSublist("Hierarchical Block " + std::to_string(g)); ++g) {
    ParameterList& group = best->sublist("Hierarchical Block " + std::to_string(g));
    const bool merged = group.get<std::string>("Included Subblocks").find(',') != std::string::npos;
    group.set("Inverse Type", merged ? "BGS" : "Amesos2");
  }
  invLib->addInverse("Printed Best", *best);
  TEST_NOTHROW(Teko::buildInverse(*invLib->getInverseFactory("Printed Best"), A));
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, warm_start_and_rebuild) {
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogate::Handle::create();
  auto invLib  = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
  auto factory = invLib->getInverseFactory("Surrogate");

  Teko::ModifiableLinearOp prec = Teko::buildInverse(*factory, A);
  Teko::MultiVector b           = rhs(A);
  Teko::MultiVector x           = zeros(A);
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

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, restarted_gmres) {
  // Restarts every 4 iterations: the samples span the cycles, and the first
  // input is still the initial residual direction, not reset at a restart
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogate::Handle::create();
  auto invLib         = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
  Teko::LinearOp prec = Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A);

  Teko::MultiVector b = rhs(A);
  const int iters     = solve(A, prec, b, zeros(A), 4);
  out << "iterations " << iters << ", samples " << handle->getRecorder()->numSamples() << "\n";
  TEST_ASSERT(iters > 4);
  TEST_ASSERT(handle->getRecorder()->numSamples() > 4);

  Teko::MultiVector v1 = b->clone_mv();
  std::vector<double> norm(1);
  Thyra::norms_2<double>(*v1, Teuchos::arrayViewFromVector(norm));
  Thyra::scale(1.0 / norm[0], v1.ptr());
  Thyra::update(-1.0, *handle->getRecorder()->getFirstInput(), v1.ptr());
  TEST_ASSERT(maxAbs(v1) < 1e-12);

  Teko::KrylovSurrogate::Report report;
  TEST_NOTHROW(report = handle->buildReport());
  for (double d : report.diagonalDeviations) TEST_ASSERT(d < 1e-6);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, through_stratimikos) {
  // Plain, and reordered: with a reordering, Stratimikos hands the factory a
  // temporary operator, which the handle must not depend on
  for (const std::string reorder : {"", "[0 1 2]"}) {
    Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
    auto handle             = Teko::KrylovSurrogate::Handle::create();
    auto lowsFactory        = stratimikosSolver(inverseLibraryList(handle), reorder);
    auto lows               = Thyra::linearOpWithSolve<double>(*lowsFactory, A);
    Teko::MultiVector b     = rhs(A);
    Teko::MultiVector x     = zeros(A);
    Thyra::solve<double>(*lows, Thyra::NOTRANS, *b, x.ptr());

    out << "\"Reorder Type\" = \"" << reorder << "\"\n";
    TEST_ASSERT(handle->isAttached());
    Teko::KrylovSurrogate::Report report;
    TEST_NOTHROW(report = handle->buildReport());
    TEST_ASSERT(report.numSamples > 0);
    TEST_EQUALITY(report.candidates.size(), 13u);
  }
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, errors) {
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});

  // no handle, and nothing printed: the report could never be seen
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogate::Handle::create());
    ifl.sublist("Surrogate").remove("Report Handle");
    auto invLib = Teko::InverseLibrary::buildFromParameterList(ifl);
    TEST_THROW(Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A), std::runtime_error);
  }
  // the same from code
  {
    auto invLib = Teko::InverseLibrary::buildFromParameterList(
        inverseLibraryList(Teko::KrylovSurrogate::Handle::create()));
    Teko::KrylovSurrogate::KrylovSurrogatePreconditionerFactory factory(
        invLib->getInverseFactory("BGS"), Teuchos::null, Teko::KrylovSurrogate::Options());
    Teko::PreconditionerState state;
    Teko::LinearOp lo = A;
    TEST_THROW(factory.buildPreconditionerOperator(lo, state), std::runtime_error);
  }
  // a misspelled option: the list is validated
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogate::Handle::create());
    ifl.sublist("Surrogate").set("Maximum Sample", 10);
    auto invLib = Teko::InverseLibrary::buildFromParameterList(ifl);
    TEST_THROW(invLib->getInverseFactory("Surrogate"), Teuchos::Exceptions::InvalidParameter);
  }
  // an ordering that does not fit the operator, caught at build time
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogate::Handle::create());
    ifl.sublist("Surrogate")
        .sublist("Search")
        .sublist("Orderings")
        .set("bad", Teuchos::tuple<int>(0, 2, 2));
    auto invLib = Teko::InverseLibrary::buildFromParameterList(ifl);
    TEST_THROW(Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A),
               std::invalid_argument);
  }
  // a report before any build, and before any solve
  {
    auto handle = Teko::KrylovSurrogate::Handle::create();
    TEST_THROW(handle->buildReport(), std::runtime_error);
    auto invLib         = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
    Teko::LinearOp prec = Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A);
    TEST_THROW(handle->buildReport(), std::runtime_error);
  }
  // an operator that is not blocked
  {
    auto handle = Teko::KrylovSurrogate::Handle::create();
    auto invLib = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
    Teko::LinearOp single = Teko::getBlock(0, 0, A);
    TEST_THROW(Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), single),
               std::runtime_error);
  }
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, handle_does_not_keep_the_preconditioner) {
  // The handle holds weak references: once the preconditioner is gone, so are
  // its samples, and the handle says so rather than reporting stale data
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogate::Handle::create();
  auto invLib = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
  {
    Teko::LinearOp prec = Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A);
    solve(A, prec, rhs(A), zeros(A));
    TEST_ASSERT(handle->isAttached());
  }
  TEST_ASSERT(!handle->isAttached());
  TEST_ASSERT(handle->getRecorder() == Teuchos::null);
  TEST_THROW(handle->buildReport(), std::runtime_error);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, hierarchical_gauss_seidel_form) {
  // The form a candidate is reported in must mean what the surrogate scored.
  // With single-block groups: the identity ordering is lower block
  // Gauss-Seidel, and the reversed ordering is upper block Gauss-Seidel.
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto invLib             = Teko::InverseLibrary::buildFromParameterList(
                  inverseLibraryList(Teko::KrylovSurrogate::Handle::create()));
  ParameterList upper;
  upper.set("Type", "Block Gauss-Seidel");
  upper.set("Inverse Type", "Amesos2");
  upper.set("Use Upper Triangle", true);
  invLib->addInverse("BGS Upper", upper);

  // The merged group is scored with an exact solve of its whole block, so the
  // form is checked with one
  addExactGroup(*invLib);

  invLib->addInverse("Identity Form", Teko::KrylovSurrogate::asHierarchicalGaussSeidel(
                                          {0, 1, 2}, "Amesos2", "Exact Group"));
  invLib->addInverse("Reversed Form", Teko::KrylovSurrogate::asHierarchicalGaussSeidel(
                                          {2, 1, 0}, "Amesos2", "Exact Group"));
  const std::vector<int> merge01 = {0, 0, 1};
  invLib->addInverse("Merged Form", Teko::KrylovSurrogate::asHierarchicalGaussSeidel(
                                        merge01, "Amesos2", "Exact Group"));

  Teko::MultiVector x = Thyra::createMembers(A->range(), 1);
  fillBlocked(x, [](int j, int r, int) { return std::sin(1.0 + r + 3.0 * j); });
  auto applied = [&](const std::string& name) {
    Teko::LinearOp op   = Teko::buildInverse(*invLib->getInverseFactory(name), A);
    Teko::MultiVector y = Thyra::createMembers(A->domain(), 1);
    op->apply(Thyra::NOTRANS, *x, y.ptr(), 1.0, 0.0);
    return y;
  };
  auto difference = [&](const std::string& a, const std::string& b) {
    Teko::MultiVector d = applied(a);
    Thyra::update(-1.0, *applied(b), d.ptr());
    return maxAbs(d);
  };

  out << "identity vs lower: " << difference("Identity Form", "BGS")
      << ", reversed vs upper: " << difference("Reversed Form", "BGS Upper") << "\n";
  TEST_ASSERT(difference("Identity Form", "BGS") < 1e-12);
  TEST_ASSERT(difference("Reversed Form", "BGS Upper") < 1e-12);

  // Merged: lower block Gauss-Seidel in group order, so block i sees block j
  // exactly when j's group comes at or before i's, and x_i - sum_j A_ij y_j = 0
  // for every block. That holds only if the group was solved exactly.
  Teko::MultiVector y         = applied("Merged Form");
  Teko::BlockedMultiVector bx = Teko::toBlockedMultiVector(x);
  Teko::BlockedMultiVector by = Teko::toBlockedMultiVector(y);
  double residual             = 0.0;
  for (int i = 0; i < 3; ++i) {
    Teko::MultiVector r = Teko::getBlock(i, bx)->clone_mv();
    for (int j = 0; j < 3; ++j)
      if (merge01[j] <= merge01[i])
        Teko::getBlock(i, j, A)->apply(Thyra::NOTRANS, *Teko::getBlock(j, by), r.ptr(), -1.0, 1.0);
    residual = std::max(residual, maxAbs(r));
  }
  out << "merged form residual: " << residual << " (x is O(1))\n";
  TEST_ASSERT(residual < 1e-9);

  // a merged group is written with the merged inverse, blocks numbered from 1
  ParameterList merged = Teko::KrylovSurrogate::asHierarchicalGaussSeidel({1, 0, 1}, "", "");
  TEST_EQUALITY(merged.sublist("Hierarchical Block 1").get<std::string>("Included Subblocks"), "2");
  TEST_EQUALITY(merged.sublist("Hierarchical Block 2").get<std::string>("Included Subblocks"),
                "1,3");
  TEST_EQUALITY(merged.sublist("Hierarchical Block 2").get<std::string>("Inverse Type"),
                "<Merged Inverse Type>");
  TEST_EQUALITY(merged.get<bool>("Use Upper Triangle"), false);
  TEST_THROW(Teko::KrylovSurrogate::asHierarchicalGaussSeidel({0, 2}, "", ""),
             std::invalid_argument);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, predictions_order_real_solves) {
  // What the surrogate is for: each converged candidate's predicted iteration
  // count, against a real solve with the candidate's printed form. The trained
  // configuration (lower block Gauss-Seidel, the identity ordering) must be
  // predicted exactly, and real counts must not decrease down the report's
  // order. Agreement within an iteration is printed, not asserted: it holds
  // here but is not a property of the method.
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogate::Handle::create();
  auto invLib = Teko::InverseLibrary::buildFromParameterList(inverseLibraryList(handle));
  addExactGroup(*invLib);

  Teko::LinearOp prec = Teko::buildInverse(*invLib->getInverseFactory("Surrogate"), A);
  const int trained   = solve(A, prec, rhs(A), zeros(A));
  const Teko::KrylovSurrogate::Report report = handle->buildReport();

  int previous    = 0;
  bool sawTrained = false;
  for (std::size_t c = 0; c < report.candidates.size(); ++c) {
    const Candidate& cand = report.candidates[c];
    if (cand.status != Candidate::CONVERGED) continue;
    const std::string name = "Candidate " + std::to_string(c);
    invLib->addInverse(name, Teko::KrylovSurrogate::asHierarchicalGaussSeidel(
                                 cand.ordering, "Amesos2", "Exact Group"));
    const int actual =
        solve(A, Teko::buildInverse(*invLib->getInverseFactory(name), A), rhs(A), zeros(A));

    out << "[";
    for (int g : cand.ordering) out << g;
    out << "] predicted " << cand.iterations << ", actual " << actual << "\n";
    TEST_ASSERT(actual >= previous);
    previous = actual;
    if (cand.ordering == std::vector<int>({0, 1, 2})) {
      sawTrained = true;
      TEST_EQUALITY(cand.iterations, trained);
      TEST_EQUALITY(actual, trained);
    }
  }
  TEST_ASSERT(sawTrained);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, print_reports_the_replaced_preconditioner) {
  // One factory, two preconditioners. A new build prints the solves of the
  // preconditioner built before it. A rebuild prints its own preconditioner's,
  // even though the handle now points at the other.
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogate::Handle::create();
  ParameterList ifl       = printingLibraryList();
  ifl.sublist("Surrogate").set("Report Handle", handle);
  auto invLib  = Teko::InverseLibrary::buildFromParameterList(ifl);
  auto factory = invLib->getInverseFactory("Surrogate");

  std::string atNewBuild, atRebuild;
  int firstSamples = 0;
  {
    CapturedOutput captured;
    Teko::ModifiableLinearOp first = Teko::buildInverse(*factory, A);
    solve(A, first, rhs(A), zeros(A));
    firstSamples                    = handle->getRecorder()->numSamples();
    Teko::ModifiableLinearOp second = Teko::buildInverse(*factory, A);
    atNewBuild                      = captured.str();
    TEST_EQUALITY(handle->getRecorder()->numSamples(), 0);

    // the first's printed samples are forgotten: a second solve is reported alone
    solve(A, first, rhs(A), zeros(A));
    TEST_NOTHROW(Teko::rebuildInverse(*factory, A, first));
    atRebuild = captured.str().substr(atNewBuild.size());
  }
  out << atNewBuild << atRebuild;

  TEST_ASSERT(firstSamples > 0);
  const std::string samples = ", " + std::to_string(firstSamples) + " samples";
  if (Tpetra::getDefaultComm()->getRank() == 0) {
    TEST_ASSERT(atNewBuild.find(samples) != std::string::npos);
    TEST_ASSERT(atRebuild.find(samples) != std::string::npos);
  }
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, print_uses_the_recorded_operator) {
  // A rebuild with a different operator prints the report of the solve's own
  // operator, the same as a handle report taken before the rebuild
  Teko::BlockedLinearOp A     = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  Teko::BlockedLinearOp Anext = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN}, 3.0);
  auto handle                 = Teko::KrylovSurrogate::Handle::create();
  ParameterList ifl           = printingLibraryList();
  ifl.sublist("Surrogate").set("Report Handle", handle);
  auto invLib  = Teko::InverseLibrary::buildFromParameterList(ifl);
  auto factory = invLib->getInverseFactory("Surrogate");

  Teko::ModifiableLinearOp prec = Teko::buildInverse(*factory, A);
  solve(A, prec, rhs(A), zeros(A));
  std::ostringstream before;
  handle->buildReport().print(before);

  std::string printed;
  {
    CapturedOutput captured;
    Teko::rebuildInverse(*factory, Anext, prec);
    printed = captured.str();
  }
  out << printed;
  if (Tpetra::getDefaultComm()->getRank() == 0) {
    TEST_EQUALITY(printed, before.str());
  } else {
    TEST_ASSERT(printed.empty());
  }

  // and the handle now reports against the new operator
  TEST_ASSERT(handle->getRecorder()->getForwardOp().get() == Anext.get());
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, print_survives_a_broken_solve) {
  // An application configured only from XML: no handle, the report printed.
  // A solve that broke down leaves NaN samples, so no report can be built. The
  // rebuild an application recovers with must still succeed, and the next
  // solve is reported as usual, ready to paste.
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto invLib             = Teko::InverseLibrary::buildFromParameterList(printingLibraryList());
  auto factory            = invLib->getInverseFactory("Surrogate");

  std::string failed, recovered;
  {
    CapturedOutput captured;
    Teko::ModifiableLinearOp prec = Teko::buildInverse(*factory, A);
    Teko::MultiVector nan         = Thyra::createMembers(prec->domain(), 1);
    Teko::MultiVector y           = Thyra::createMembers(prec->range(), 1);
    Thyra::assign(nan.ptr(), std::nan(""));
    prec->apply(Thyra::NOTRANS, *nan, y.ptr(), 1.0, 0.0);

    TEST_NOTHROW(Teko::rebuildInverse(*factory, A, prec));
    failed = captured.str();

    solve(A, prec, rhs(A), zeros(A));
    TEST_NOTHROW(Teko::rebuildInverse(*factory, A, prec));
    recovered = captured.str().substr(failed.size());
  }
  out << failed << recovered;
  if (Tpetra::getDefaultComm()->getRank() == 0) {
    TEST_ASSERT(failed.find("no report of the solves since the previous build") !=
                std::string::npos);
    TEST_ASSERT(recovered.find("Teko Krylov Surrogate report: 3 blocks") != std::string::npos);
    TEST_ASSERT(recovered.find("Hierarchical Block Gauss-Seidel") != std::string::npos);
    // the best is all-merged here, so the merged inverse is named, and no placeholder
    TEST_ASSERT(recovered.find("value=\"BGS\"") != std::string::npos);
    TEST_ASSERT(recovered.find("Inverse Type&gt;") == std::string::npos);
  } else {
    TEST_ASSERT(failed.empty() && recovered.empty());
  }
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, print_through_stratimikos_strided) {
  // An XML-only application: one monolithic Tpetra matrix, split by "Strided
  // Blocking". Stratimikos builds a new preconditioner at every
  // initialization, and that build prints the previous one's solve.
  const int n      = 24;
  auto crs         = buildInterleavedMatrix(n, 3);
  auto space       = Thyra::tpetraVectorSpace<ST, LO, GO, NT>(crs->getRowMap());
  Teko::LinearOp A = Thyra::tpetraLinearOp<ST, LO, GO, NT>(
      space, space, Teuchos::rcp_implicit_cast<Tpetra::Operator<ST, LO, GO, NT> >(crs));

  auto handle       = Teko::KrylovSurrogate::Handle::create();
  ParameterList ifl = printingLibraryList();
  ifl.sublist("Surrogate").set("Report Handle", handle);
  auto lowsFactory = stratimikosSolver(ifl, "", "1 1 1");

  std::string printed;
  bool silentBeforeReinitialization = false;
  int samples                       = 0;
  {
    CapturedOutput captured;
    auto lows           = Thyra::linearOpWithSolve<double>(*lowsFactory, A);
    Teko::MultiVector b = Thyra::createMembers(A->range(), 1);
    Teko::MultiVector x = Thyra::createMembers(A->domain(), 1);
    Thyra::assign(b.ptr(), 1.0);
    Thyra::assign(x.ptr(), 0.0);
    Thyra::solve<double>(*lows, Thyra::NOTRANS, *b, x.ptr());
    samples                      = handle->getRecorder()->numSamples();
    silentBeforeReinitialization = captured.str().empty();

    Thyra::initializeOp<double>(*lowsFactory, A, lows.ptr());
    printed = captured.str();
  }
  out << printed;

  TEST_ASSERT(samples > 0);
  TEST_ASSERT(silentBeforeReinitialization);
  if (Tpetra::getDefaultComm()->getRank() == 0) {
    TEST_ASSERT(printed.find("Teko Krylov Surrogate report: 3 blocks (sizes 24 24 24), " +
                             std::to_string(samples) + " samples") != std::string::npos);
  } else {
    TEST_ASSERT(printed.empty());
  }
}
