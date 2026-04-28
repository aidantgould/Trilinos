// @HEADER
// *****************************************************************************
//                 Belos: Block Linear Solvers Package
//
// Copyright 2004-2016 NTESS and the Belos contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER
//
// Test for BelosGmresRitzPairs.hpp: computeRitzPairs() utility.
//
// Uses a 6x6 matrix with known eigenvalues {1+0.5i, 1-0.5i, 2, 2, 3, 3.0001}.
// The double eigenvalue at 2 is 2*I in its 2x2 block, so GMRES converges in
// 5 effective steps.  A fixed-seed random orthogonal similarity transform
// Q (seed 42) is applied so the CrsMatrix is dense and non-trivially structured
// while preserving eigenvalues.  H_m is Q-invariant, so all Ritz-value
// assertions hold regardless of Q.
//
// Key assertions:
//   m=2: harmonic Ritz values differ from standard Ritz values.
//   m=4: exactly 1 Ritz value within 0.01 of 3 (near pair unresolved).
//   m=5: exactly 2 Ritz values within 0.01 of 3 (near pair split).
//   Both m=4/m=5 checks hold for Standard and Harmonic.
//   Physical vectors: null by default, non-null when requested.
//   Physical vector residual at m=4: max ||A*x - lambda*x|| / ||x|| < 0.1
//     (complex pairs use coupled equations: A*xr - lr*xr + li*xi = 0, etc.)
//   Complex scalar: same Ritz values as double on this real matrix.
//

#include "BelosConfigDefs.hpp"
#include "BelosGmresRitzPairs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosBlockFGmresIter.hpp"
#include "BelosGmresIteration.hpp"
#include "BelosStatusTest.hpp"
#include "BelosTypes.hpp"

#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>

using Teuchos::RCP;
using Teuchos::rcp;

// -----------------------------------------------------------------------
// Captures a deep copy of H at every curDim seen from BlockFGmresIter.
// -----------------------------------------------------------------------
template <class SC, class MV, class OP>
class StateCapture : public Belos::StatusTest<SC, MV, OP> {
 public:
  using SDM = Teuchos::SerialDenseMatrix<int, SC>;

  struct Snapshot {
    int curDim = 0;
    RCP<const SDM> H;
    RCP<const MV>  V;  // Krylov basis; columns 0..curDim-1 are valid
  };

  std::map<int, Snapshot> snapshots;
  bool sawFGmresIter = false;

  Belos::StatusType checkStatus(Belos::Iteration<SC, MV, OP>* it) override {
    auto* fg = dynamic_cast<Belos::BlockFGmresIter<SC, MV, OP>*>(it);
    if (fg) {
      sawFGmresIter = true;
      auto s = fg->getState();
      if (s.curDim > 0 && !s.H.is_null()) {
        Snapshot snap;
        snap.curDim = s.curDim;
        snap.H = rcp(new SDM(*s.H));  // deep copy: H is modified in-place each step
        snap.V = s.V;                 // RCP share: no restarts so columns are stable
        snapshots[s.curDim] = snap;
      }
    }
    return Belos::Undefined;
  }
  Belos::StatusType getStatus() const override { return Belos::Undefined; }
  void reset() override { snapshots.clear(); sawFGmresIter = false; }
  void print(std::ostream& os, int indent = 0) const override {
    os << std::string(indent, ' ') << "StateCapture\n";
  }
};

// -----------------------------------------------------------------------
// Build a random 6x6 orthogonal matrix Q via QR of a seeded random matrix.
// -----------------------------------------------------------------------
Teuchos::SerialDenseMatrix<int, double> buildQ6()
{
  Teuchos::SerialDenseMatrix<int, double> R(6, 6);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  for (int j = 0; j < 6; ++j)
    for (int i = 0; i < 6; ++i)
      R(i, j) = dist(rng);

  Teuchos::LAPACK<int, double> lapack;
  std::vector<double> tau(6), work(1);
  int lwork = -1, info = 0;

  lapack.GEQRF(6, 6, R.values(), R.stride(), tau.data(), work.data(), lwork, &info);
  lwork = static_cast<int>(work[0]);
  work.resize(lwork);
  lapack.GEQRF(6, 6, R.values(), R.stride(), tau.data(), work.data(), lwork, &info);

  lwork = -1;
  work.resize(1);
  lapack.ORGQR(6, 6, 6, R.values(), R.stride(), tau.data(), work.data(), lwork, &info);
  lwork = static_cast<int>(work[0]);
  work.resize(lwork);
  lapack.ORGQR(6, 6, 6, R.values(), R.stride(), tau.data(), work.data(), lwork, &info);

  return R;  // now contains Q
}

// -----------------------------------------------------------------------
// Count Ritz values within tol of center (in absolute value).
// -----------------------------------------------------------------------
template <class MT>
int countNear(const std::vector<std::complex<MT>>& vals,
              double center, double tol)
{
  int n = 0;
  for (const auto& v : vals)
    if (std::abs(v - std::complex<MT>(static_cast<MT>(center), MT(0))) < static_cast<MT>(tol))
      ++n;
  return n;
}

// -----------------------------------------------------------------------
// Check eigenvalue equation H_m * v = lambda * v (or coupled equations for
// complex conjugate pairs in real SC) for all Ritz values.
// Returns max relative residual.
// -----------------------------------------------------------------------
template <class SC, class MV>
typename Teuchos::ScalarTraits<SC>::magnitudeType
eigenvalueResidual(const Teuchos::SerialDenseMatrix<int, SC>& H,
                   const Belos::RitzPairs<SC, MV>& ritz)
{
  using MT  = typename Teuchos::ScalarTraits<SC>::magnitudeType;
  using STS = Teuchos::ScalarTraits<SC>;
  using STM = Teuchos::ScalarTraits<MT>;
  using SDM = Teuchos::SerialDenseMatrix<int, SC>;

  const int m = ritz.values.size();
  SDM Hm(Teuchos::Copy, H, m, m);
  MT maxerr = STM::zero();

  if constexpr (!STS::isComplex) {
    // Real SC: GEEV stores conjugate pairs as consecutive real columns.
    // Paired columns satisfy coupled equations; real eigenvalues use H_m*v = λv.
    for (int i = 0; i < m; ) {
      MT lambda_r = ritz.values[i].real();
      MT lambda_i = ritz.values[i].imag();
      bool is_pair = (std::abs(lambda_i) > MT(1e-10)) && (i + 1 < m);

      SDM y0(m, 1), Hy0(m, 1);
      for (int j = 0; j < m; ++j) y0(j, 0) = ritz.projectedVectors(j, i);
      Hy0.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, STS::one(), Hm, y0, STS::zero());

      if (!is_pair) {
        MT numer = STM::zero(), denom = STM::zero();
        for (int j = 0; j < m; ++j) {
          SC diff = Hy0(j, 0) - SC(lambda_r) * y0(j, 0);
          numer += STS::magnitude(diff) * STS::magnitude(diff);
          denom += STS::magnitude(y0(j, 0)) * STS::magnitude(y0(j, 0));
        }
        if (denom > STM::zero())
          maxerr = std::max(maxerr, STM::squareroot(numer / denom));
        i += 1;
      } else {
        // Coupled residuals for complex pair (columns i = real part, i+1 = imag part):
        //   r0 = H_m*y0 - lambda_r*y0 + lambda_i*y1
        //   r1 = H_m*y1 - lambda_r*y1 - lambda_i*y0
        SDM y1(m, 1), Hy1(m, 1);
        for (int j = 0; j < m; ++j) y1(j, 0) = ritz.projectedVectors(j, i + 1);
        Hy1.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, STS::one(), Hm, y1, STS::zero());

        MT numer = STM::zero(), denom = STM::zero();
        for (int j = 0; j < m; ++j) {
          SC r0 = Hy0(j, 0) - SC(lambda_r) * y0(j, 0) + SC(lambda_i) * y1(j, 0);
          SC r1 = Hy1(j, 0) - SC(lambda_r) * y1(j, 0) - SC(lambda_i) * y0(j, 0);
          numer += STS::magnitude(r0) * STS::magnitude(r0)
                 + STS::magnitude(r1) * STS::magnitude(r1);
          denom += STS::magnitude(y0(j, 0)) * STS::magnitude(y0(j, 0))
                 + STS::magnitude(y1(j, 0)) * STS::magnitude(y1(j, 0));
        }
        if (denom > STM::zero())
          maxerr = std::max(maxerr, STM::squareroot(numer / denom));
        i += 2;
      }
    }
  } else {
    // Complex SC: every column of projectedVectors is a complete complex eigenvector.
    for (int i = 0; i < m; ++i) {
      SC lambda(ritz.values[i].real(), ritz.values[i].imag());

      SDM y(m, 1), Hy(m, 1);
      for (int j = 0; j < m; ++j) y(j, 0) = ritz.projectedVectors(j, i);
      Hy.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, STS::one(), Hm, y, STS::zero());

      MT numer = STM::zero(), denom = STM::zero();
      for (int j = 0; j < m; ++j) {
        SC diff = Hy(j, 0) - lambda * y(j, 0);
        numer += STS::magnitude(diff) * STS::magnitude(diff);
        denom += STS::magnitude(y(j, 0)) * STS::magnitude(y(j, 0));
      }
      if (denom > STM::zero())
        maxerr = std::max(maxerr, STM::squareroot(numer / denom));
    }
  }
  return maxerr;
}

// -----------------------------------------------------------------------
// Main test for double scalar type.
// -----------------------------------------------------------------------
bool runRitzTest(bool verbose)
{
  using SC  = double;
  using LO  = typename Tpetra::MultiVector<SC>::local_ordinal_type;
  using GO  = typename Tpetra::MultiVector<SC>::global_ordinal_type;
  using NT  = typename Tpetra::MultiVector<SC>::node_type;
  using MV  = Tpetra::MultiVector<SC, LO, GO, NT>;
  using OP  = Tpetra::Operator<SC, LO, GO, NT>;
  using SDM = Teuchos::SerialDenseMatrix<int, SC>;

  auto comm = Tpetra::getDefaultComm();
  const int me = comm->getRank();

  // --- Build the structured 6x6 matrix -----------------------------------
  // Eigenvalues: 1+0.5i, 1-0.5i, 2, 2, 3, 3.0001
  SDM A_struct(6, 6);  // zero-initialized
  A_struct(0, 0) =  1.0;  A_struct(0, 1) = -0.5;
  A_struct(1, 0) =  0.5;  A_struct(1, 1) =  1.0;
  A_struct(2, 2) =  2.0;
  A_struct(3, 3) =  2.0;
  A_struct(4, 4) =  3.0;
  A_struct(5, 5) =  3.0001;

  // --- Apply similarity transform: A_tilde = Q*A*Q^T, b_tilde = Q*ones ---
  SDM Q = buildQ6();

  SDM tmp(6, 6), A_tilde(6, 6);
  tmp.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, Q, A_struct, 0.0);
  A_tilde.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, 1.0, tmp, Q, 0.0);

  SDM b_dense(6, 1);
  for (int i = 0; i < 6; ++i) b_dense(i, 0) = 1.0;
  SDM b_tilde(6, 1);
  b_tilde.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, Q, b_dense, 0.0);

  // --- Build Tpetra objects -----------------------------------------------
  const GO n = 6;
  auto map  = rcp(new Tpetra::Map<LO, GO, NT>(n, 0, comm));

  auto A = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NT>(map, n));
  for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
    GO g = map->getGlobalElement(i);
    Teuchos::Array<GO> cols(n);
    Teuchos::Array<SC> vals(n);
    for (GO j = 0; j < n; ++j) { cols[j] = j; vals[j] = A_tilde(g, j); }
    A->insertGlobalValues(g, cols, vals);
  }
  A->fillComplete();

  auto b = rcp(new MV(map, 1));
  auto x = rcp(new MV(map, 1));
  x->putScalar(0.0);
  {
    auto bv = b->getDataNonConst(0);
    for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i)
      bv[i] = b_tilde(map->getGlobalElement(i), 0);
  }

  // Identity right preconditioner to keep isFlexible_=true (BlockFGmresIter)
  auto I = rcp(new Tpetra::CrsMatrix<SC, LO, GO, NT>(map, 1));
  for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
    GO g = map->getGlobalElement(i);
    I->insertGlobalValues(g, Teuchos::tuple(g), Teuchos::tuple(1.0));
  }
  I->fillComplete();

  auto problem = rcp(new Belos::LinearProblem<SC, MV, OP>(A, x, b));
  problem->setRightPrec(I);
  problem->setProblem();

  auto params = rcp(new Teuchos::ParameterList);
  params->set("Num Blocks",            6);
  params->set("Maximum Restarts",      0);
  params->set("Maximum Iterations",    6);
  params->set("Convergence Tolerance", 1e-12);
  params->set("Flexible Gmres",        true);
  params->set("Keep Hessenberg",       true);
  params->set("Verbosity",             Belos::Errors);

  auto capture = rcp(new StateCapture<SC, MV, OP>());
  Belos::BlockGmresSolMgr<SC, MV, OP> solver(problem, params);
  solver.setDebugStatusTest(capture);
  solver.solve();

  if (!capture->sawFGmresIter) {
    if (me == 0)
      std::cerr << "  FAIL: BlockFGmresIter never used.\n";
    return false;
  }

  // --- Check snapshots at m=2, 4, 5 were captured ------------------------
  for (int m : {2, 4, 5}) {
    if (capture->snapshots.count(m) == 0) {
      if (me == 0)
        std::cerr << "  FAIL: no snapshot at m=" << m << ".\n";
      return false;
    }
  }

  // --- Helper: build a minimal state from a snapshot ---------------------
  auto makeState = [](const typename StateCapture<SC,MV,OP>::Snapshot& snap) {
    Belos::GmresIterationState<SC, MV> st;
    st.curDim = snap.curDim;
    st.H      = snap.H;
    st.V      = snap.V;
    return st;
  };

  bool ok = true;

  // --- At every captured m: basic correctness ----------------------------
  for (auto& [m, snap] : capture->snapshots) {
    for (auto type : {Belos::RitzType::Standard, Belos::RitzType::Harmonic}) {
      auto ritz = Belos::computeRitzPairs(makeState(snap), type);

      if (static_cast<int>(ritz.values.size()) != m) {
        if (me == 0)
          std::cerr << "  FAIL: m=" << m << " wrong value count "
                    << ritz.values.size() << ".\n";
        ok = false;
      }
      for (const auto& v : ritz.values) {
        if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) {
          if (me == 0)
            std::cerr << "  FAIL: m=" << m << " non-finite Ritz value.\n";
          ok = false;
        }
      }
      // Eigenvalue equation H_m*v = λ*v holds for Standard Ritz (eigenvectors
      // of H_m) but not for Harmonic (eigenvectors of the modified target).
      if (type == Belos::RitzType::Standard) {
        auto relerr = eigenvalueResidual(*snap.H, ritz);
        if (relerr > 1e-10) {
          if (me == 0)
            std::cerr << "  FAIL: m=" << m << " eigenvalue equation residual "
                      << relerr << " > 1e-10.\n";
          ok = false;
        }
      }
    }
  }

  // --- m=2: harmonic and standard must differ ----------------------------
  {
    auto& snap = capture->snapshots.at(2);
    auto std_ritz  = Belos::computeRitzPairs(makeState(snap), Belos::RitzType::Standard);
    auto harm_ritz = Belos::computeRitzPairs(makeState(snap), Belos::RitzType::Harmonic);

    // Sort by (real, imag) so comparison is order-independent.
    // Magnitude-based sort is unstable for 1±0.5i (equal magnitudes).
    auto byRealImag = [](const auto& a, const auto& b) {
      if (std::real(a) != std::real(b)) return std::real(a) < std::real(b);
      return std::imag(a) < std::imag(b);
    };
    std::sort(std_ritz.values.begin(), std_ritz.values.end(), byRealImag);
    std::sort(harm_ritz.values.begin(), harm_ritz.values.end(), byRealImag);

    double maxdiff = 0.0;
    for (int i = 0; i < 2; ++i)
      maxdiff = std::max(maxdiff,
                         std::abs(std_ritz.values[i] - harm_ritz.values[i]));
    if (maxdiff <= 1e-12) {
      if (me == 0)
        std::cerr << "  FAIL: m=2 standard and harmonic Ritz values agree "
                     "(max diff = " << maxdiff << "); harmonic modification "
                     "appears inactive.\n";
      ok = false;
    } else if (verbose && me == 0) {
      std::cout << "  PASS m=2: standard/harmonic differ (max diff = "
                << maxdiff << ").\n";
    }
  }

  // --- m=4 and m=5: near-pair resolution ---------------------------------
  // The near pair {3, 3.0001} is unresolved at m=4 (1 value near 3) and
  // resolved at m=5 (2 values near 3 — both 3 and 3.0001 fall within 0.01).
  // Assertions verified empirically for b=ones, seed=42.
  for (auto type : {Belos::RitzType::Standard, Belos::RitzType::Harmonic}) {
    const char* tname = (type == Belos::RitzType::Standard) ? "Standard" : "Harmonic";

    auto ritz4 = Belos::computeRitzPairs(makeState(capture->snapshots.at(4)), type);
    auto ritz5 = Belos::computeRitzPairs(makeState(capture->snapshots.at(5)), type);

    int n4 = countNear(ritz4.values, 3.0, 0.01);
    int n5 = countNear(ritz5.values, 3.0, 0.01);

    if (n4 != 1) {
      if (me == 0)
        std::cerr << "  FAIL (" << tname << "): m=4 expected 1 Ritz value "
                     "near 3, found " << n4 << ".\n";
      ok = false;
    } else if (verbose && me == 0) {
      std::cout << "  PASS (" << tname << ") m=4: " << n4
                << " Ritz value near 3 (pair unresolved).\n";
    }

    if (n5 != 2) {
      if (me == 0)
        std::cerr << "  FAIL (" << tname << "): m=5 expected 2 Ritz values "
                     "near 3, found " << n5 << ".\n";
      ok = false;
    } else if (verbose && me == 0) {
      std::cout << "  PASS (" << tname << ") m=5: " << n5
                << " Ritz values near 3 (pair split).\n";
    }
  }

  // --- Physical vectors: null when not requested, non-null when requested --
  {
    auto& snap5 = capture->snapshots.at(5);
    auto ritz_no_phys = Belos::computeRitzPairs(makeState(snap5),
                                                Belos::RitzType::Standard, false);
    if (!ritz_no_phys.physicalVectors.is_null()) {
      if (me == 0)
        std::cerr << "  FAIL: physicalVectors non-null with computePhysicalVectors=false.\n";
      ok = false;
    }
  }

  // --- Physical vector residual check at m=4 --------------------------------
  // All m=4 Ritz vectors should satisfy the physical-space eigenvector equation.
  // For real SC, GEEV produces:
  //   real eigenvalue λ at col i:       A*x  - λ*x  = 0  (single equation)
  //   complex pair λ=λr+iλi at cols i,i+1 (xr,xi):
  //     A*xr - λr*xr + λi*xi = 0   (real part)
  //     A*xi - λr*xi - λi*xr = 0   (imaginary part)
  // Both cases are handled in one loop — no pairs are skipped.
  {
    using MVT = Belos::MultiVecTraits<SC, MV>;
    using MT  = Teuchos::ScalarTraits<SC>::magnitudeType;

    auto& snap4 = capture->snapshots.at(4);
    auto ritz4  = Belos::computeRitzPairs(makeState(snap4),
                                          Belos::RitzType::Standard, true);

    if (ritz4.physicalVectors.is_null()) {
      if (me == 0)
        std::cerr << "  FAIL: physicalVectors null at m=4 even though V was set.\n";
      ok = false;
    } else {
      const int m = 4;
      auto AphysVecs = MVT::Clone(*ritz4.physicalVectors, m);
      A->apply(*ritz4.physicalVectors, *AphysVecs);

      MT maxres = 0.0;
      for (int i = 0; i < m; ) {
        MT lambda_r = ritz4.values[i].real();
        MT lambda_i = ritz4.values[i].imag();
        bool is_pair = (std::abs(lambda_i) > MT(1e-10)) && (i + 1 < m);

        std::vector<int> col0 = {i};
        auto x0  = MVT::CloneView(*ritz4.physicalVectors, col0);
        auto Ax0 = MVT::CloneView(*AphysVecs, col0);

        std::vector<MT> x0_norm(1);
        MVT::MvNorm(*x0, x0_norm);
        MT denom = x0_norm[0];

        auto r0  = MVT::Clone(*x0, 1);
        auto tmp = MVT::Clone(*x0, 1);
        // tmp = A*x0 - λr*x0
        MVT::MvAddMv(SC(1), *Ax0, SC(-lambda_r), *x0, *tmp);

        if (!is_pair) {
          // Real eigenvalue: r0 = A*x0 - λ*x0
          MVT::MvAddMv(SC(1), *tmp, SC(0), *x0, *r0);
          i += 1;
        } else {
          std::vector<int> col1 = {i + 1};
          auto x1  = MVT::CloneView(*ritz4.physicalVectors, col1);
          auto Ax1 = MVT::CloneView(*AphysVecs, col1);

          // r0 = (A*x0 - λr*x0) + λi*x1
          MVT::MvAddMv(SC(1), *tmp, SC(lambda_i), *x1, *r0);

          // r1 = (A*x1 - λr*x1) - λi*x0
          auto r1   = MVT::Clone(*x1, 1);
          auto tmp1 = MVT::Clone(*x1, 1);
          MVT::MvAddMv(SC(1), *Ax1, SC(-lambda_r), *x1, *tmp1);
          MVT::MvAddMv(SC(1), *tmp1, SC(-lambda_i), *x0, *r1);

          std::vector<MT> x1_norm(1), r1_norm(1);
          MVT::MvNorm(*x1, x1_norm);
          MVT::MvNorm(*r1, r1_norm);
          denom = std::sqrt(x0_norm[0]*x0_norm[0] + x1_norm[0]*x1_norm[0]);
          if (denom > MT(0))
            maxres = std::max(maxres, r1_norm[0] / denom);
          i += 2;
        }

        std::vector<MT> r0_norm(1);
        MVT::MvNorm(*r0, r0_norm);
        if (denom > MT(0))
          maxres = std::max(maxres, r0_norm[0] / denom);
      }

      const MT tol = MT(0.1);
      if (maxres > tol) {
        if (me == 0)
          std::cerr << "  FAIL: physical vector residual at m=4: max = "
                    << maxres << " > " << tol << ".\n";
        ok = false;
      } else if (verbose && me == 0) {
        std::cout << "  PASS: physical vector residual at m=4: max ‖A*x-λx‖/‖x‖ = "
                  << maxres << " (all " << m << " eigenpairs).\n";
      }
    }
  }

  if (verbose && ok && me == 0)
    std::cout << "  PASS: all double assertions.\n";

  return ok;
}

// -----------------------------------------------------------------------
// Complex consistency: same Ritz values as double on this real matrix.
// Builds a complex copy of a captured H and verifies the two code paths agree.
// Only compiled when Teuchos complex scalar support is available.
// -----------------------------------------------------------------------
#ifdef HAVE_TEUCHOS_COMPLEX
bool checkComplexConsistency(
    const Teuchos::SerialDenseMatrix<int, double>& H_real,
    int curDim, bool verbose)
{
  using CSC = std::complex<double>;
  using LO  = typename Tpetra::MultiVector<double>::local_ordinal_type;
  using GO  = typename Tpetra::MultiVector<double>::global_ordinal_type;
  using NT  = typename Tpetra::MultiVector<double>::node_type;
  using CMV = Tpetra::MultiVector<CSC, LO, GO, NT>;
  using RMV = Tpetra::MultiVector<double, LO, GO, NT>;

  auto comm = Tpetra::getDefaultComm();
  const int me = comm->getRank();

  // Build complex copy of H_real
  auto H_complex = rcp(new Teuchos::SerialDenseMatrix<int, CSC>(
      H_real.numRows(), H_real.numCols()));
  for (int j = 0; j < H_real.numCols(); ++j)
    for (int i = 0; i < H_real.numRows(); ++i)
      (*H_complex)(i, j) = CSC(H_real(i, j), 0.0);

  Belos::GmresIterationState<CSC, CMV> cstate;
  cstate.curDim = curDim;
  cstate.H      = H_complex;

  Belos::GmresIterationState<double, RMV> rstate;
  rstate.curDim = curDim;
  rstate.H      = rcp(new Teuchos::SerialDenseMatrix<int, double>(H_real));

  auto critz = Belos::computeRitzPairs(cstate, Belos::RitzType::Standard);
  auto rritz = Belos::computeRitzPairs(rstate, Belos::RitzType::Standard);

  // Sort by (real, imag) lexicographically so conjugate pairs land in the same
  // relative order in both lists (magnitude-only sort is unstable for pairs
  // like 1+0.5i / 1-0.5i which have identical magnitude).
  auto byRealImag = [](const auto& a, const auto& b) {
    if (std::real(a) != std::real(b)) return std::real(a) < std::real(b);
    return std::imag(a) < std::imag(b);
  };
  std::sort(critz.values.begin(), critz.values.end(), byRealImag);
  std::sort(rritz.values.begin(), rritz.values.end(), byRealImag);

  double maxdiff = 0.0;
  for (int i = 0; i < curDim; ++i) {
    double cr = static_cast<double>(std::real(critz.values[i]));
    double ci = static_cast<double>(std::imag(critz.values[i]));
    double rr = static_cast<double>(std::real(rritz.values[i]));
    double ri = static_cast<double>(std::imag(rritz.values[i]));
    maxdiff = std::max(maxdiff, std::abs(std::complex<double>(cr - rr, ci - ri)));
  }

  if (maxdiff > 1e-10) {
    if (me == 0)
      std::cerr << "  FAIL complex consistency: max |complex - real| = "
                << maxdiff << " > 1e-10.\n";
    return false;
  }
  if (verbose && me == 0)
    std::cout << "  PASS complex consistency: max diff = " << maxdiff << ".\n";
  return true;
}
#endif  // HAVE_TEUCHOS_COMPLEX

// -----------------------------------------------------------------------
int main(int argc, char* argv[])
{
  Tpetra::ScopeGuard tpetraScope(&argc, &argv);

  bool verbose = false;
  bool success = false;

  try {
    auto comm = Tpetra::getDefaultComm();
    const int me = comm->getRank();

    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]) == "--verbose") verbose = true;

    if (verbose && me == 0)
      std::cout << "\nRitz pairs: double scalar test\n";
    bool ok = runRitzTest(verbose);

    // Re-run to get a snapshot for the complex consistency check.
    // Simplest: re-capture from a fresh solve and extract m=5 H.
    if (ok) {
      using SC = double;
      using LO = typename Tpetra::MultiVector<SC>::local_ordinal_type;
      using GO = typename Tpetra::MultiVector<SC>::global_ordinal_type;
      using NT = typename Tpetra::MultiVector<SC>::node_type;
      using MV = Tpetra::MultiVector<SC, LO, GO, NT>;
      using OP = Tpetra::Operator<SC, LO, GO, NT>;

      // Rebuild A_tilde (identical to runRitzTest — same seed, same Q)
      Teuchos::SerialDenseMatrix<int,SC> A_struct(6,6);
      A_struct(0,0)= 1.0; A_struct(0,1)=-0.5;
      A_struct(1,0)= 0.5; A_struct(1,1)= 1.0;
      A_struct(2,2)= 2.0; A_struct(3,3)= 2.0;
      A_struct(4,4)= 3.0; A_struct(5,5)= 3.0001;

      auto Q = buildQ6();
      Teuchos::SerialDenseMatrix<int,SC> tmp(6,6), A_tilde(6,6), b_dense(6,1), b_tilde(6,1);
      tmp.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, Q, A_struct, 0.0);
      A_tilde.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, 1.0, tmp, Q, 0.0);
      for (int i = 0; i < 6; ++i) b_dense(i,0) = 1.0;
      b_tilde.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, Q, b_dense, 0.0);

      auto map = rcp(new Tpetra::Map<LO,GO,NT>(GO(6), 0, comm));
      auto A   = rcp(new Tpetra::CrsMatrix<SC,LO,GO,NT>(map, GO(6)));
      for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
        GO g = map->getGlobalElement(i);
        Teuchos::Array<GO> cols(6); Teuchos::Array<SC> vals(6);
        for (GO j = 0; j < 6; ++j) { cols[j]=j; vals[j]=A_tilde(g,j); }
        A->insertGlobalValues(g, cols, vals);
      }
      A->fillComplete();
      auto b = rcp(new MV(map,1)), x = rcp(new MV(map,1));
      x->putScalar(0.0);
      { auto bv = b->getDataNonConst(0);
        for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i)
          bv[i] = b_tilde(map->getGlobalElement(i), 0); }
      auto I = rcp(new Tpetra::CrsMatrix<SC,LO,GO,NT>(map,1));
      for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
        GO g = map->getGlobalElement(i);
        I->insertGlobalValues(g, Teuchos::tuple(g), Teuchos::tuple(SC(1)));
      }
      I->fillComplete();
      auto problem = rcp(new Belos::LinearProblem<SC,MV,OP>(A,x,b));
      problem->setRightPrec(I); problem->setProblem();
      auto params = rcp(new Teuchos::ParameterList);
      params->set("Num Blocks",6); params->set("Maximum Restarts",0);
      params->set("Maximum Iterations",6);
      params->set("Convergence Tolerance",1e-12);
      params->set("Flexible Gmres",true); params->set("Keep Hessenberg",true);
      params->set("Verbosity", Belos::Errors);
      auto cap2 = rcp(new StateCapture<SC,MV,OP>());
      Belos::BlockGmresSolMgr<SC,MV,OP> solver(problem, params);
      solver.setDebugStatusTest(cap2);
      solver.solve();

#ifdef HAVE_TEUCHOS_COMPLEX
      if (cap2->snapshots.count(5)) {
        if (verbose && me == 0)
          std::cout << "\nRitz pairs: complex scalar consistency check (m=5)\n";
        ok &= checkComplexConsistency(*cap2->snapshots.at(5).H, 5, verbose);
      }
#else
      if (verbose && me == 0)
        std::cout << "\nRitz pairs: complex scalar consistency check: SKIPPED"
                     " (HAVE_TEUCHOS_COMPLEX not enabled)\n";
#endif
    }

    success = ok;
    if (me == 0) {
      if (success) std::cout << "\nEnd Result: TEST PASSED\n";
      else         std::cout << "\nEnd Result: TEST FAILED\n";
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
