// @HEADER
// *****************************************************************************
//                 Belos: Block Linear Solvers Package
//
// Copyright 2004-2016 NTESS and the Belos contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef BELOS_GMRES_RITZ_PAIRS_HPP
#define BELOS_GMRES_RITZ_PAIRS_HPP

/** \file BelosGmresRitzPairs.hpp
 *  \brief Free function for extracting Ritz and harmonic Ritz pairs from a
 *         GmresIterationState.
 *
 *  Requires that the solver was run with "Keep Hessenberg" = true so that
 *  state.H contains the raw (pre-QR) upper Hessenberg matrix.
 *
 *  Usage:
 *  \code
 *    // Standard Ritz pairs (eigenvalues of H_m)
 *    auto ritz = Belos::computeRitzPairs(state);
 *
 *    // Harmonic Ritz pairs
 *    auto harm = Belos::computeRitzPairs(state, Belos::RitzType::Harmonic);
 *
 *    // Also compute physical Ritz vectors (V * projected eigenvectors)
 *    auto full = Belos::computeRitzPairs(state, Belos::RitzType::Standard,
 *                                        true);
 *  \endcode
 *
 *  For real ScalarType, complex eigenvalues appear as conjugate pairs.
 *  LAPACK stores their eigenvectors across consecutive real columns of
 *  projectedVectors: column i holds the real part and column i+1 holds the
 *  imaginary part when imag(values[i]) != 0.
 */

#include "BelosConfigDefs.hpp"
// SerialDenseMatrix must be fully defined before BelosGmresIteration.hpp is
// processed, because GmresIterationState members use it via Teuchos::RCP and
// incomplete-type instantiation causes those members to silently drop.
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "BelosGmresIteration.hpp"
#include "BelosMultiVecTraits.hpp"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"

#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace Belos {

enum class RitzType { Standard, Harmonic };

/** Result of computeRitzPairs(). */
template <class SC, class MV>
struct RitzPairs {
  using MT = typename Teuchos::ScalarTraits<SC>::magnitudeType;

  /** Ritz values, always stored as complex regardless of SC. */
  std::vector<std::complex<MT>> values;

  /**
   * Ritz vectors in the projected (Krylov) space.  Column j corresponds to
   * values[j].  For real SC, consecutive columns encode conjugate-pair
   * eigenvectors (real part then imaginary part) when imag(values[j]) != 0.
   */
  Teuchos::SerialDenseMatrix<int, SC> projectedVectors;

  /**
   * Physical Ritz vectors: V * projectedVectors.  Null unless
   * computeRitzPairs() was called with computePhysicalVectors = true and
   * state.V is non-null.  Computing these requires O(n*m) work and memory.
   */
  Teuchos::RCP<MV> physicalVectors;
};

/**
 * Compute Ritz or harmonic Ritz pairs from a GMRES iteration state.
 *
 * \param state   Snapshot from GmresIteration::getState().  state.H must be
 *                the raw Hessenberg (requires "Keep Hessenberg" = true).
 * \param type    RitzType::Standard or RitzType::Harmonic.
 * \param computePhysicalVectors  If true and state.V is non-null, compute
 *                physical Ritz vectors (V * eigenvectors).  Default false.
 *
 * Harmonic Ritz math follows getHarmonicVecs1() in BelosGCRODRSolMgr.hpp:
 *   target = H_m + h_{m+1,m}^2 * H_m^{-H} * e_m * e_m^H
 * where H_m is the curDim x curDim leading submatrix of state.H.
 *
 * \note Only correct for block size 1.  For block size > 1 the Hessenberg
 *       layout differs and h_{m+1,m} would need to be addressed differently.
 */
template <class SC, class MV>
RitzPairs<SC, MV> computeRitzPairs(
    const GmresIterationState<SC, MV>& state,
    RitzType type                  = RitzType::Standard,
    bool    computePhysicalVectors = false)
{
  using STS = Teuchos::ScalarTraits<SC>;
  using MT  = typename STS::magnitudeType;
  using SDM = Teuchos::SerialDenseMatrix<int, SC>;
  using SDV = Teuchos::SerialDenseVector<int, SC>;
  using MVT = Belos::MultiVecTraits<SC, MV>;

  TEUCHOS_TEST_FOR_EXCEPTION(
      state.H.is_null(), std::invalid_argument,
      "Belos::computeRitzPairs: state.H is null. "
      "Run the solver with \"Keep Hessenberg\" = true.");

  const int m = state.curDim;
  TEUCHOS_TEST_FOR_EXCEPTION(
      m <= 0, std::invalid_argument,
      "Belos::computeRitzPairs: state.curDim = " << m << " must be positive.");
  TEUCHOS_TEST_FOR_EXCEPTION(
      state.H->numRows() < m + 1 || state.H->numCols() < m,
      std::invalid_argument,
      "Belos::computeRitzPairs: state.H (" << state.H->numRows() << " x "
      << state.H->numCols() << ") is too small for curDim = " << m << ".");

  // blockSize = numRows - numCols of the (numBlocks+1)*b x numBlocks*b Hessenberg
  const int blockSize = state.H->numRows() - state.H->numCols();
  TEUCHOS_TEST_FOR_EXCEPTION(
      blockSize != 1, std::invalid_argument,
      "Belos::computeRitzPairs: block size " << blockSize << " > 1 is not "
      "supported. The Hessenberg is block-structured and h_{m+1,m} is a "
      "block, not a scalar. Only single right-hand-side solves (block size 1) "
      "are currently handled.");

  Teuchos::LAPACK<int, SC> lapack;
  int info = 0;

  // H_m: leading m x m submatrix of the (m+1) x m Hessenberg
  SDM Hm(Teuchos::Copy, *state.H, m, m);

  // Build the matrix whose eigenvalues are the desired Ritz values
  SDM target(Teuchos::Copy, Hm);

  if (type == RitzType::Harmonic) {
    // h_{m+1,m}: subdiagonal entry below H_m (row m, col m-1 in 0-based)
    SC h_sub = (*state.H)(m, m - 1);
    // Use |h_{m+1,m}|^2 (real, non-negative) so the rank-1 update has the
    // right structure for both real and complex SC.  For real SC this equals
    // h_sub*h_sub; for complex SC the complex square h_sub^2 would be wrong.
    MT mag   = STS::magnitude(h_sub);
    SC d     = SC(mag * mag);

    // Solve H_m^H x = e_m  =>  x = H_m^{-H} e_m
    SDM HHt(Hm, Teuchos::CONJ_TRANS);
    SDV e_m(m);
    e_m[m - 1] = STS::one();
    std::vector<int> ipiv(m);
    lapack.GESV(m, 1, HHt.values(), HHt.stride(), ipiv.data(),
                e_m.values(), e_m.stride(), &info);
    TEUCHOS_TEST_FOR_EXCEPTION(
        info != 0, std::runtime_error,
        "Belos::computeRitzPairs: LAPACK GESV failed (info = " << info << ").");

    // target = H_m + h_{m+1,m}^2 * x * e_m^H
    for (int i = 0; i < m; ++i)
      target(i, m - 1) += d * e_m[i];
  }

  // Eigenvector matrix (m x m, column j = right eigenvector j)
  SDM vr(m, m, false);

  RitzPairs<SC, MV> result;
  result.values.resize(m);

  // GEEV signature differs between real and complex SC:
  //   real:    GEEV(..., WR*, WI*, VL, ldvl, VR, ..., RWORK*, info*)
  //   complex: GEEV(..., W*,       VL, ldvl, VR, ..., RWORK*, info*)
  if constexpr (!STS::isComplex) {
    std::vector<MT> wr(m), wi(m);
    std::vector<SC> work(1);
    std::vector<MT> rwork(2 * m);
    int lwork = -1;

    lapack.GEEV('N', 'V', m, target.values(), target.stride(),
                wr.data(), wi.data(), nullptr, 1,
                vr.values(), vr.stride(), work.data(), lwork, rwork.data(), &info);
    lwork = std::abs(static_cast<int>(STS::real(work[0])));
    work.resize(lwork);
    lapack.GEEV('N', 'V', m, target.values(), target.stride(),
                wr.data(), wi.data(), nullptr, 1,
                vr.values(), vr.stride(), work.data(), lwork, rwork.data(), &info);
    TEUCHOS_TEST_FOR_EXCEPTION(
        info != 0, std::runtime_error,
        "Belos::computeRitzPairs: LAPACK GEEV failed (info = " << info << ").");

    for (int i = 0; i < m; ++i)
      result.values[i] = std::complex<MT>(wr[i], wi[i]);
  } else {
    std::vector<SC> w(m);
    std::vector<SC> work(1);
    std::vector<MT> rwork(2 * m);
    int lwork = -1;

    lapack.GEEV('N', 'V', m, target.values(), target.stride(),
                w.data(), nullptr, 1,
                vr.values(), vr.stride(), work.data(), lwork, rwork.data(), &info);
    lwork = std::abs(static_cast<int>(STS::real(work[0])));
    work.resize(lwork);
    lapack.GEEV('N', 'V', m, target.values(), target.stride(),
                w.data(), nullptr, 1,
                vr.values(), vr.stride(), work.data(), lwork, rwork.data(), &info);
    TEUCHOS_TEST_FOR_EXCEPTION(
        info != 0, std::runtime_error,
        "Belos::computeRitzPairs: LAPACK GEEV failed (info = " << info << ").");

    for (int i = 0; i < m; ++i)
      result.values[i] = std::complex<MT>(STS::real(w[i]), STS::imag(w[i]));
  }

  result.projectedVectors = vr;

  if (computePhysicalVectors && !state.V.is_null()) {
    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    Teuchos::RCP<const MV> Vm = MVT::CloneView(*state.V, idx);

    result.physicalVectors = MVT::Clone(*Vm, m);
    MVT::MvTimesMatAddMv(STS::one(), *Vm, vr,
                         STS::zero(), *result.physicalVectors);
  }

  return result;
}

}  // namespace Belos

#endif  // BELOS_GMRES_RITZ_PAIRS_HPP
