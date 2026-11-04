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

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosThyraAdapter.hpp"

#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
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
  s.sublist("Search").set("Tolerance", 1e-8);
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
  solver.solve();
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

// Swaps Teuchos's default output stream for one writing into a string, and
// restores it however the scope ends
class CapturedOutput {
 public:
  CapturedOutput() : previous_(Teuchos::VerboseObjectBase::getDefaultOStream()) {
    Teuchos::VerboseObjectBase::setDefaultOStream(
        Teuchos::fancyOStream(Teuchos::rcpFromRef(captured_)));
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

int numCalls(const char* timer) {
  auto t = Teuchos::TimeMonitor::lookupCounter(timer);
  return t == Teuchos::null ? 0 : t->numCalls();
}

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
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogate::Handle::create();

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
  Teko::MultiVector x = zeros(A);
  Thyra::solve<double>(*lows, Thyra::NOTRANS, *b, x.ptr());

  TEST_ASSERT(handle->isAttached());
  Teko::KrylovSurrogate::Report report = handle->buildReport();
  TEST_ASSERT(report.numSamples > 0);
  TEST_EQUALITY(report.candidates.size(), 13u);
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, errors) {
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});

  // no handle, and nothing printed: the report could never be seen
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogate::Handle::create());
    ifl.sublist("Surrogate").remove("Report Handle");
    auto invLib = Teko::InverseLibrary::buildFromParameterList(ifl);
    TEST_THROW(invLib->getInverseFactory("Surrogate"), std::runtime_error);
  }
  // a misspelled option
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogate::Handle::create());
    ifl.sublist("Surrogate").set("Maximum Sample", 10);
    auto invLib = Teko::InverseLibrary::buildFromParameterList(ifl);
    TEST_THROW(invLib->getInverseFactory("Surrogate"), Teuchos::Exceptions::InvalidParameter);
  }
  // options out of range
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogate::Handle::create());
    ifl.sublist("Surrogate").sublist("Search").set("Candidate Count", -5);
    auto invLib = Teko::InverseLibrary::buildFromParameterList(ifl);
    TEST_THROW(invLib->getInverseFactory("Surrogate"), Teuchos::Exceptions::InvalidParameter);
  }
  {
    ParameterList ifl = inverseLibraryList(Teko::KrylovSurrogate::Handle::create());
    ifl.sublist("Surrogate").set("Truncation Tolerance", 2.0);
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
  // form is checked with one: unpreconditioned GMRES, converged far below the
  // check's tolerance
  ParameterList exact;
  exact.set("Type", "Belos");
  exact.set("Solver Type", "Block GMRES");
  ParameterList& gmres = exact.sublist("Solver Types").sublist("Block GMRES");
  gmres.set("Convergence Tolerance", 1e-13);
  gmres.set("Maximum Iterations", 400);
  gmres.set("Num Blocks", 400);
  invLib->addInverse("Exact Group", exact);

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

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, print_at_rebuild_without_a_handle) {
  // An application configured only from XML: no handle, the report printed
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto invLib             = Teko::InverseLibrary::buildFromParameterList(printingLibraryList());
  auto factory            = invLib->getInverseFactory("Surrogate");

  std::string printed;
  bool silentBeforeRebuild = false;
  {
    CapturedOutput captured;
    Teko::ModifiableLinearOp prec = Teko::buildInverse(*factory, A);
    solve(A, prec, rhs(A), zeros(A));
    silentBeforeRebuild = captured.str().empty();
    Teko::rebuildInverse(*factory, A, prec);
    printed = captured.str();
  }
  out << printed;

  TEST_ASSERT(silentBeforeRebuild);
  if (Tpetra::getDefaultComm()->getRank() == 0) {
    TEST_ASSERT(printed.find("Teko Krylov Surrogate report: 3 blocks") != std::string::npos);
    TEST_ASSERT(printed.find("Hierarchical Block Gauss-Seidel") != std::string::npos);
    // the best is all-merged here, so the merged inverse is named, and no placeholder
    TEST_ASSERT(printed.find("value=\"BGS\"") != std::string::npos);
    TEST_ASSERT(printed.find("Inverse Type&gt;") == std::string::npos);
  } else {
    TEST_ASSERT(printed.empty());
  }
}

TEUCHOS_UNIT_TEST(tKrylovSurrogateFactory, print_reports_its_own_preconditioner) {
  // One factory, two preconditioners. Rebuilding the first must print the
  // first's samples, even though the handle now points at the second, which has
  // none.
  Teko::BlockedLinearOp A = buildBlockedOp(kSizes, {EVEN, EVEN, EVEN});
  auto handle             = Teko::KrylovSurrogate::Handle::create();
  ParameterList ifl       = printingLibraryList();
  ifl.sublist("Surrogate").set("Report Handle", handle);
  auto invLib  = Teko::InverseLibrary::buildFromParameterList(ifl);
  auto factory = invLib->getInverseFactory("Surrogate");

  std::string printed;
  int firstSamples = 0;
  {
    CapturedOutput captured;
    Teko::ModifiableLinearOp first = Teko::buildInverse(*factory, A);
    solve(A, first, rhs(A), zeros(A));
    firstSamples                    = handle->getRecorder()->numSamples();
    Teko::ModifiableLinearOp second = Teko::buildInverse(*factory, A);
    TEST_EQUALITY(handle->getRecorder()->numSamples(), 0);
    TEST_NOTHROW(Teko::rebuildInverse(*factory, A, first));
    printed = captured.str();
  }
  out << printed;

  TEST_ASSERT(firstSamples > 0);
  if (Tpetra::getDefaultComm()->getRank() == 0) {
    TEST_ASSERT(printed.find(", " + std::to_string(firstSamples) + " samples") !=
                std::string::npos);
  }
}
