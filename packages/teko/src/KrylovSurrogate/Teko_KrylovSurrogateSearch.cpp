// @HEADER
// *****************************************************************************
//      Teko: A package for block and physics based preconditioning
//
// Copyright 2010 NTESS and the Teko contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Teko_KrylovSurrogateSearch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

#ifdef TEKO_HAVE_KRYLOV_SURROGATE
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_SerialDenseSolver.hpp"

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosThyraAdapter.hpp"
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_DetachedMultiVectorView.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#endif

namespace Teko {
namespace KrylovSurrogate {
namespace Impl {

namespace {

double binomial(int n, int k) {
  double c = 1.0;
  for (int i = 1; i <= k; ++i) c = c * (n - k + i) / i;
  return c;
}

// Every unordered set partition of n blocks once, as restricted growth labels
void partitions(int k, int used, std::vector<int>& labels, std::vector<std::vector<int> >& out) {
  if (k == static_cast<int>(labels.size())) {
    out.push_back(labels);
    return;
  }
  for (int v = 0; v <= used; ++v) {
    labels[k] = v;
    partitions(k + 1, std::max(used, v + 1), labels, out);
  }
}

#ifdef TEKO_HAVE_KRYLOV_SURROGATE

typedef Report::Candidate Candidate;

struct Scored {
  int status                   = Candidate::SOLVER_ERROR;
  int iterations               = 0;
  double finalRelativeResidual = std::numeric_limits<double>::quiet_NaN();
  double rate                  = std::numeric_limits<double>::quiet_NaN();
};

typedef Thyra::MultiVectorBase<double> ThyraMV;
typedef Thyra::LinearOpBase<double> ThyraOp;

// Scores candidates one at a time on serial Thyra vectors, reusing one Belos
// solver manager so its timers are registered once. It lives only inside
// scoreCandidates, which owns what the references point at.
class Scorer {
 public:
  Scorer(const DenseMatrix& C, const DenseVector& b, const std::vector<int>& basisSizes, double tol)
      : C_(C), basisSizes_(basisSizes), R_(C.numRows()) {
    space_ = Thyra::defaultSpmdVectorSpace<double>(R_);

    // Belos's error messages go where Teko's do, so a failed candidate leaves a trace
    auto params = Teuchos::rcp(new Teuchos::ParameterList);
    params->set("Num Blocks", R_);
    params->set("Maximum Iterations", R_);
    params->set("Convergence Tolerance", tol);
    params->set("Verbosity", static_cast<int>(Belos::Errors));
    params->set("Output Stream", Teuchos::RCP<std::ostream>(Teko::getOutputStream()));
    params->set("Timer Label", std::string("Teko Krylov Surrogate"));
    solver_ = Teuchos::rcp(new Belos::BlockGmresSolMgr<double, ThyraMV, ThyraOp>);
    solver_->setParameters(params);

    bThyra_ = Thyra::createMembers(space_, 1);
    {
      Thyra::DetachedMultiVectorView<double> view(bThyra_);
      for (int i = 0; i < R_; ++i) view(i, 0) = b(i);
    }
  }

  Scored score(const std::vector<int>& ordering) {
    Scored s;
    // Everything is caught here, on purpose: a candidate that throws on one rank
    // must not leave the other ranks waiting in the gather that follows. The
    // message goes to Teko's output stream instead.
    try {
      // [C]_S1^{-1}, in place. It is singular exactly when a merged group's
      // block is, since singleton groups' blocks are I. Only exact singularity
      // (a zero pivot) is detected here.
      DenseMatrix S = gaussSeidelSelection(C_, basisSizes_, ordering);
      Teuchos::SerialDenseSolver<int, double> inverse;
      inverse.setMatrix(Teuchos::rcpFromRef(S));
      if (inverse.invert() != 0) {
        s.status = Candidate::SINGULAR_GROUP_BLOCK;
        return s;
      }

      // K = C [C]_S1^{-1}, held as a Thyra multivector, which is its own
      // operator. An explicit inverse is less stable than applying an LU
      // factorization, and is kept for simplicity at these sizes.
      DenseMatrix K(R_, R_);
      K.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, C_, S, 0.0);
      auto Kop = Thyra::createMembers(space_, R_);
      {
        Thyra::DetachedMultiVectorView<double> view(Kop);
        for (int j = 0; j < R_; ++j)
          for (int i = 0; i < R_; ++i) view(i, j) = K(i, j);
      }

      auto y = Thyra::createMembers(space_, 1);
      Thyra::assign(y.ptr(), 0.0);
      auto problem = Teuchos::rcp(
          new Belos::LinearProblem<double, ThyraMV, ThyraOp>(Kop, y, bThyra_.getConst()));
      problem->setProblem();
      solver_->setProblem(problem);

      // Belos's own verdict and residual are reliable here: the GMRES is
      // unpreconditioned on a fixed matrix, unlike the flexible solve the
      // surrogate came from. Unconverged covers the iteration limit and a loss
      // of accuracy alike.
      s.status =
          solver_->solve() == Belos::Converged ? Candidate::CONVERGED : Candidate::NOT_CONVERGED;
      s.iterations            = solver_->getNumIters();
      s.finalRelativeResidual = solver_->achievedTol();
      s.rate = s.iterations > 0 ? std::pow(s.finalRelativeResidual, 1.0 / s.iterations) : 0.0;
    } catch (const std::exception& e) {
      *Teko::getOutputStream() << "Teko \"Krylov Surrogate\": scoring a candidate failed: "
                               << e.what() << "\n";
      s = Scored();
    }
    return s;
  }

 private:
  const DenseMatrix& C_;
  const std::vector<int>& basisSizes_;
  int R_;

  Teuchos::RCP<const Thyra::VectorSpaceBase<double> > space_;
  Teuchos::RCP<Belos::BlockGmresSolMgr<double, ThyraMV, ThyraOp> > solver_;
  Teuchos::RCP<ThyraMV> bThyra_;
};

// x before y, with NaN after every number, so sorting stays a strict weak ordering
bool rateBefore(double x, double y) {
  if (std::isnan(x)) return false;
  if (std::isnan(y)) return true;
  return x < y;
}

#endif  // TEKO_HAVE_KRYLOV_SURROGATE

}  // namespace

double numConfigurations(int n) {
  std::vector<double> a(n + 1, 0.0);
  a[0] = 1.0;
  for (int k = 1; k <= n; ++k)
    for (int i = 1; i <= k; ++i) a[k] += binomial(k, i) * a[k - i];
  return a[n];
}

std::vector<std::vector<int> > allConfigurations(int n) {
  std::vector<std::vector<int> > out;
  if (n == 0) return out;

  std::vector<std::vector<int> > groupings;
  std::vector<int> labels(n, 0);
  partitions(1, 1, labels, groupings);

  // every relabelling of each grouping's groups gives each ordering once
  for (const auto& g : groupings) {
    std::vector<int> perm(*std::max_element(g.begin(), g.end()) + 1);
    std::iota(perm.begin(), perm.end(), 0);
    do {
      std::vector<int> ordering(n);
      for (int k = 0; k < n; ++k) ordering[k] = perm[g[k]];
      out.push_back(ordering);
    } while (std::next_permutation(perm.begin(), perm.end()));
  }
  return out;
}

std::vector<int> randomConfiguration(int n, std::mt19937_64& rng) {
  // The first group is any nonempty subset and the rest an ordering of what is
  // left, so numConfigurations(n) = sum_k C(n, k) numConfigurations(n - k).
  // Drawing the first group's size with those weights, then its members
  // uniformly, makes every ordering equally likely.
  std::vector<int> ordering(n, 0), left(n);
  std::iota(left.begin(), left.end(), 0);
  for (int position = 0; !left.empty(); ++position) {
    const int remaining = static_cast<int>(left.size());
    std::vector<double> weights(remaining);
    for (int k = 1; k <= remaining; ++k)
      weights[k - 1] = binomial(remaining, k) * numConfigurations(remaining - k);
    const int size = 1 + std::discrete_distribution<int>(weights.begin(), weights.end())(rng);

    std::shuffle(left.begin(), left.end(), rng);
    for (int k = 0; k < size; ++k) ordering[left[k]] = position;
    left.erase(left.begin(), left.begin() + size);
  }
  return ordering;
}

bool isValidOrdering(const std::vector<int>& ordering, int n) {
  if (static_cast<int>(ordering.size()) != n || n == 0) return false;
  // every position in range first, so used[] below is never sized from a bad entry
  for (int p : ordering)
    if (p < 0 || p >= n) return false;
  const int groups = *std::max_element(ordering.begin(), ordering.end()) + 1;
  std::vector<bool> used(groups, false);
  for (int p : ordering) used[p] = true;
  return std::all_of(used.begin(), used.end(), [](bool u) { return u; });
}

std::vector<std::vector<int> > buildCandidates(int n,
                                               const std::vector<std::vector<int> >& userOrderings,
                                               int count, int seed) {
  std::vector<std::vector<int> > out;
  std::set<std::vector<int> > seen;
  auto add = [&](const std::vector<int>& o) {
    if (seen.insert(o).second) out.push_back(o);
  };

  for (const auto& o : userOrderings) {
    TEUCHOS_TEST_FOR_EXCEPTION(!isValidOrdering(o, n), std::invalid_argument,
                               "Teko \"Krylov Surrogate\": an ordering in \"Orderings\" is not a "
                               "valid ordering of "
                                   << n
                                   << " blocks. Give one position per block, using every "
                                      "position from 0 up to the number of groups minus one.");
    add(o);
  }
  std::vector<int> identity(n);
  std::iota(identity.begin(), identity.end(), 0);
  add(identity);
  add(std::vector<int>(n, 0));

  if (numConfigurations(n) <= count) {
    for (const auto& o : allConfigurations(n)) add(o);
  } else {
    std::mt19937_64 rng(seed);
    for (long attempts = 100L * count; static_cast<int>(out.size()) < count && attempts > 0;
         --attempts)
      add(randomConfiguration(n, rng));
  }
  return out;
}

DenseMatrix gaussSeidelSelection(const DenseMatrix& C, const std::vector<int>& basisSizes,
                                 const std::vector<int>& ordering) {
  const int nb = static_cast<int>(basisSizes.size());
  std::vector<int> offset(nb + 1, 0);
  for (int j = 0; j < nb; ++j) offset[j + 1] = offset[j] + basisSizes[j];

  DenseMatrix S(C);
  for (int i = 0; i < nb; ++i)
    for (int j = 0; j < nb; ++j)
      if (ordering[i] < ordering[j])
        for (int c = offset[j]; c < offset[j + 1]; ++c)
          for (int r = offset[i]; r < offset[i + 1]; ++r) S(r, c) = 0.0;
  return S;
}

#ifdef TEKO_HAVE_KRYLOV_SURROGATE

std::vector<Report::Candidate> scoreCandidates(const DenseMatrix& C, const DenseVector& b,
                                               const std::vector<int>& basisSizes,
                                               const std::vector<std::vector<int> >& candidates,
                                               double tol, const Teuchos::Comm<int>& comm) {
  TEUCHOS_TEST_FOR_EXCEPTION(b.normFrobenius() == 0.0, std::runtime_error,
                             "Teko \"Krylov Surrogate\": the initial residual has no component in "
                             "the trained subspace, so there is nothing to score");

  // Deal round-robin, then gather with one sum: each candidate's slot is
  // written by exactly one rank and zero on every other, so the sum is exact.
  const int nc    = static_cast<int>(candidates.size());
  const int rank  = comm.getRank();
  const int nproc = comm.getSize();
  std::vector<double> mine(4 * nc, 0.0), all(4 * nc, 0.0);
  Scorer scorer(C, b, basisSizes, tol);
  for (int i = rank; i < nc; i += nproc) {
    Scored s        = scorer.score(candidates[i]);
    mine[4 * i + 0] = s.status;
    mine[4 * i + 1] = s.iterations;
    mine[4 * i + 2] = s.finalRelativeResidual;
    mine[4 * i + 3] = s.rate;
  }
  if (nc > 0)
    Teuchos::reduceAll<int, double>(comm, Teuchos::REDUCE_SUM, 4 * nc, mine.data(), all.data());

  std::vector<Report::Candidate> out(nc);
  for (int i = 0; i < nc; ++i) {
    out[i].ordering              = candidates[i];
    out[i].status                = static_cast<Candidate::Status>(static_cast<int>(all[4 * i + 0]));
    out[i].iterations            = static_cast<int>(all[4 * i + 1]);
    out[i].finalRelativeResidual = all[4 * i + 2];
    out[i].rate                  = all[4 * i + 3];
  }

  std::stable_sort(out.begin(), out.end(), [](const Candidate& x, const Candidate& y) {
    if (x.status != y.status) return x.status < y.status;
    if (x.iterations != y.iterations) return x.iterations < y.iterations;
    return rateBefore(x.rate, y.rate);
  });
  return out;
}

#endif  // TEKO_HAVE_KRYLOV_SURROGATE

}  // namespace Impl
}  // namespace KrylovSurrogate
}  // end namespace Teko
