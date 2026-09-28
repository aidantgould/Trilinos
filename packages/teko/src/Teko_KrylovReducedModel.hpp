// Teko_KrylovReducedModel.hpp — Krylov-informed reduced model (the surrogate
// math), split out of Teko_KrylovSurrogate.hpp.
//
// Builds the cheap reduced operator C_hat and reduced RHS b_hat from a flexible
// GMRES Krylov state (V, Z) and the blocked operator A, via per-block thin SVDs
// computed through the Gram matrix G_j = V_j^T V_j (so the distributed,
// problem-sized left singular vectors Q_j are never formed — see the caveat on
// computeCHat). This header is pure construction math: it has NO dependency on
// the JSON handshake, preconditioner rebuild, or Belos hook orchestration in
// Teko_KrylovSurrogate.hpp (the dependency runs only the other way).
//
//   reduceKrylovBlock(V_j, curDim, svd_tol) -> {rank, sigma, W}   (per block)
//   computeCHat(state, A_blocked, b, block_sizes) -> CHatData{ranks, C_hat, b_hat}
#pragma once

// ── Standard library ──────────────────────────────────────────────────────
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ── Teuchos ───────────────────────────────────────────────────────────────
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

// ── Belos (Krylov state / linear problem types) ────────────────────────────
#include "BelosGmresIteration.hpp"
#include "BelosLinearProblem.hpp"

// ── Thyra ─────────────────────────────────────────────────────────────────
#include "Thyra_BlockedLinearOpBase.hpp"
#include "Thyra_LinearOpBase.hpp"
#include "Thyra_MultiVectorBase.hpp"
#include "Thyra_ProductMultiVectorBase.hpp"
#include "Thyra_TpetraLinearOp.hpp"
#include "Thyra_TpetraMultiVector.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"

// ── Tpetra ────────────────────────────────────────────────────────────────
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"

namespace Teko {
namespace KrylovSurrogate {

// ── Type aliases matching teko_ext.cpp ────────────────────────────────────
// SC is fixed to double (the surrogate's dense math and the Belos adaptive
// hook are double-only by design). LO/GO/Node track the Trilinos build's
// Tpetra defaults rather than being hardcoded, so the dynamic_casts to
// TpetraLinearOp/CrsMatrix below match whatever node/ordinals the build uses
// (Serial, OpenMP, CUDA, a different GO, ...) instead of silently returning
// null on a mismatched build. (GPU nodes additionally need host-accessible
// data for the getData() loops here.)
using SC   = double;
using LO   = Tpetra::Map<>::local_ordinal_type;
using GO   = Tpetra::Map<>::global_ordinal_type;
using Node = Tpetra::Map<>::node_type;

using SDM      = Teuchos::SerialDenseMatrix<int, SC>;
using TpetraMV = Tpetra::MultiVector<SC, LO, GO, Node>;
using MV       = Thyra::MultiVectorBase<SC>;
using OP       = Thyra::LinearOpBase<SC>;
using Problem  = Belos::LinearProblem<SC, MV, OP>;
using State    = Belos::GmresIterationState<SC, MV>;

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: distributed helpers
// ═══════════════════════════════════════════════════════════════════════════

// Return block j of a Thyra ProductMultiVector as its underlying (possibly
// distributed) Tpetra::MultiVector. No data is copied or gathered.
inline Teuchos::RCP<const TpetraMV> getBlockTpetraMV(
    Teuchos::RCP<const MV> mv,
    int j)
{
    auto pmv  = Teuchos::rcp_dynamic_cast<const Thyra::ProductMultiVectorBase<SC>>(mv);
    TEUCHOS_TEST_FOR_EXCEPTION(pmv.is_null(), std::runtime_error,
        "Teko::KrylovSurrogate::getBlockTpetraMV: mv is not a ProductMultiVectorBase");

    auto blk  = pmv->getMultiVectorBlock(j);
    auto tmv  = Teuchos::rcp_dynamic_cast<const Thyra::TpetraMultiVector<SC, LO, GO, Node>>(blk);
    TEUCHOS_TEST_FOR_EXCEPTION(tmv.is_null(), std::runtime_error,
        "Teko::KrylovSurrogate::getBlockTpetraMV: block " + std::to_string(j) +
        " is not a TpetraMultiVector");

    return tmv->getConstTpetraMultiVector();
}

// Block (i,j) of A_blocked as a Tpetra CrsMatrix, or null when the block is
// absent or is not a Tpetra CRS operator.
inline Teuchos::RCP<const Tpetra::CrsMatrix<SC, LO, GO, Node>> getBlockAsCrs(
    Teuchos::RCP<const Thyra::BlockedLinearOpBase<SC>> A_blocked, int i, int j)
{
    auto blk = A_blocked->getBlock(i, j);
    if (blk.is_null()) return Teuchos::null;
    auto lo = Teuchos::rcp_dynamic_cast<
        const Thyra::TpetraLinearOp<SC, LO, GO, Node>>(blk);
    if (lo.is_null()) return Teuchos::null;
    return Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<SC, LO, GO, Node>>(
        lo->getConstTpetraOperator());
}

// C := A^T B as a replicated SerialDenseMatrix (numVecs(A) × numVecs(B)).
// A and B may be distributed arbitrarily (they must share a row distribution).
// Internally Tpetra computes the local GEMM and sum-reduces the small result
// across ranks (Tpetra::MultiVector::multiply with a locally-replicated
// target), so every rank gets the bit-identical matrix — no broadcasts needed
// downstream. This is the only cross-rank reduction in computeCHat(); nothing
// problem-sized is ever gathered.
inline SDM mvTransMv(const TpetraMV& A, const TpetraMV& B)
{
    const int ka = static_cast<int>(A.getNumVectors());
    const int kb = static_cast<int>(B.getNumVectors());

    auto repMap = Teuchos::rcp(new Tpetra::Map<LO, GO, Node>(
        static_cast<Tpetra::global_size_t>(ka), 0,
        A.getMap()->getComm(), Tpetra::LocallyReplicated));
    TpetraMV C(repMap, kb);
    C.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, A, B, 0.0);

    SDM out(ka, kb);
    for (int c = 0; c < kb; ++c) {
        auto view = C.getVector(c)->getData();
        for (int r = 0; r < ka; ++r)
            out(r, c) = view[r];
    }
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Section 2: CHat data structure and computation
// ═══════════════════════════════════════════════════════════════════════════

// Reciprocal of a singular value, guarding the Sigma^{-1} projections below
// against a (numerically) zero sigma — a direction at or below the Gram-matrix
// resolution floor contributes nothing rather than an O(1/sigma) blow-up.
inline SC invSigmaOrZero(SC s) { return (std::abs(s) > 1e-14) ? 1.0 / s : 0.0; }

struct CHatData {
    int                nb;          // number of blocks
    int                krylov_dim;  // curDim used
    std::vector<int>   block_sizes;
    std::vector<int>   ranks;       // r_j per block

    // blocks[i][j] is (ranks[i] × ranks[j]) — reduced operator
    std::vector<std::vector<SDM>>  blocks;
    // b_hat[j] is (ranks[j] × 1) = Q[j]^T * b_j — reduced right-hand side
    std::vector<SDM>               b_hat;
};

// Reduced basis of one sampled block — the core "reduced model" construction.
// From the thin SVD V_tilde_j = Q_j Sigma_j W_j^T (paper eq 3.9), obtained via
// the Gram matrix G_j = V_tilde_j^T V_tilde_j (eigendecomposition) WITHOUT ever
// forming the distributed Q_j. Holds the truncated rank r_j, the leading r_j
// singular values (descending), and the right singular vectors W_j
// (curDim × r_j). b_hat_j and the reduced operator blocks C_hat_{ij} are
// projections onto this basis.
struct BlockReduction {
    int             rank = 0;   // r_j
    std::vector<SC> sigma;      // length r_j, descending singular values
    SDM             W;          // curDim × r_j, right singular vectors
};

// Smallest singular-value ratio the Gram-matrix route can resolve, and
// therefore the truncation threshold used here.
//
// The paper's rule is sigma_i >= 1e-10 * sigma_max (eq 3.10), and pyautoteko's
// reduce_Z uses rank_tol * eps_mach = 2.2e-10. Neither is reachable through
// G_j = V_tilde_j^T V_tilde_j: forming the Gram matrix squares the condition
// number, so a direction with sigma/sigma_max = t shows up in G_j at
// lambda/lambda_max = t^2, and anything with t below sqrt(8 eps_mach) is buried
// in the roundoff of lambda_max. Any requested tolerance at or below this floor
// therefore produces identical results, which is why the floor is named here
// instead of being passed in as a tolerance that looks like it does something.
//
// The consequence is deliberate and one-sided: this retains FEWER directions
// than the paper does, and each retained direction carries less amplified
// roundoff (a direction kept at ratio t costs about eps_mach/t relative error in
// its column of C_hat). Reaching 1e-10 would mean forming Q_j explicitly with a
// distributed tall-skinny QR instead of going through G_j.
constexpr double kGramSigmaFloor = 4.2146848510894035e-08;  // sqrt(8 * eps_mach)

// Build the reduced basis of block Vj (using its first curDim columns), where
// Vj is V_tilde_j = A_jj Z_j. svd_tol truncates singular values relative to the
// largest, and is clamped from below by kGramSigmaFloor. Collective —
// mvTransMv does a local GEMM + Allreduce, so the small dense eigenproblem and
// its truncation are identical on every rank (no broadcasts).
inline BlockReduction reduceKrylovBlock(const TpetraMV& Vj, int curDim, double svd_tol)
{
    Teuchos::LAPACK<int, SC> lapack;

    // G_j (curDim × curDim, replicated identically on every rank).
    SDM Gj = mvTransMv(Vj, Vj);

    // SYEV: eigenvalues ascending; eigenvectors overwrite Gj's columns.
    std::vector<SC> eig(curDim);
    int lwork = -1, info = 0;
    SC work_query;
    lapack.SYEV('V', 'U', curDim, Gj.values(), Gj.stride(),
                eig.data(), &work_query, lwork, &info);
    lwork = static_cast<int>(work_query);
    std::vector<SC> work(lwork);
    lapack.SYEV('V', 'U', curDim, Gj.values(), Gj.stride(),
                eig.data(), work.data(), lwork, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error,
        "Teko::KrylovSurrogate::reduceKrylovBlock: SYEV failed (info=" +
        std::to_string(info) + ")");

    // Truncate against the leading eigenvalue. A cut at sigma_k >= t*sigma_max
    // on singular values is lambda_k >= t^2*lambda_max on G_j's eigenvalues,
    // since lambda_k = sigma_k^2. t is clamped up to kGramSigmaFloor because
    // below that the eigenvalue test is comparing against the roundoff of
    // lambda_max, and keeping such a direction would inject an O(1/sigma)
    // garbage column into C_hat.
    const SC t          = std::max(svd_tol, kGramSigmaFloor);
    const SC lambda_max = eig[curDim - 1];
    const SC threshold  = (lambda_max > 0.0 ? t * t * lambda_max : 0.0);
    int rj = 0;
    while (rj < curDim && eig[curDim - 1 - rj] >= threshold) ++rj;
    if (rj == 0) rj = 1;  // keep at least one direction

    // Descending order: r-th largest eigenpair is SYEV column curDim-1-r.
    BlockReduction red;
    red.rank = rj;
    red.sigma.resize(rj);
    red.W.shape(curDim, rj);
    for (int r = 0; r < rj; ++r) {
        const int src = curDim - 1 - r;
        red.sigma[r] = std::sqrt(std::max(eig[src], 0.0));
        for (int kd = 0; kd < curDim; ++kd)
            red.W(kd, r) = Gj(kd, src);
    }
    return red;
}

// Compute C_hat from the final FGMRES state.
//   state      — from BlockFGmresIter::getState(); only Z and curDim are used
//   A_blocked  — the blocked system operator
//   b          — the right-hand side of the linear system (for b_hat = Q^T b)
//   block_sizes — sizes of each block
//   svd_tol    — truncation threshold relative to leading singular value,
//                clamped up to kGramSigmaFloor
//
// The basis is sampled from a forward multiply, V_tilde_j := A_jj Z_j (paper
// eq 3.3), NOT from the FGMRES basis state.V. See the comment at the forward
// multiply below for why.
//
//   V_tilde_j = Q_j Sigma_j W_j^T  (thin SVD, eq 3.9),
//   C_hat_{ij} = Q_{i,ri}^T A_{ij} Z_j W_{j,rj} Sigma_{j,rj}^{-1}   (eq 3.18),
//   b_hat_j    = Q_{j,rj}^T b_j,
//
// Multi-rank (Gram-matrix) formulation: Q_j (problem-sized, distributed) is
// eliminated via Q_j = V_tilde_j W_j Sigma_j^{-1}, leaving only small replicated
// quantities:
//
//   G_j   := V_tilde_j^T V_tilde_j = W_j Sigma_j^2 W_j^T  (curDim × curDim, SPD)
//     → SYEV eigendecomposition gives W_j and sigma_{j,k} = sqrt(lambda_{j,k})
//   b_hat_j    = Sigma_{j,rj}^{-1} W_{j,rj}^T (V_tilde_j^T b_j)
//   C_hat_{ij} = Sigma_{i,ri}^{-1} W_{i,ri}^T (V_tilde_i^T A_{ij} Z_j) W_{j,rj} Sigma_{j,rj}^{-1}
//
// Every V_tilde^T(...) product is a mvTransMv() reduction (local GEMM +
// Allreduce of a curDim-sized matrix); the sparse applies stay fully
// distributed. All ranks therefore compute identical C_hat/b_hat with no gathers
// or broadcasts. This is a collective: every rank must call it.
//
// Numerical caveat: the Gram matrix squares the condition number, so the
// achievable truncation floor is kGramSigmaFloor rather than the paper's 1e-10
// (see that constant). SYEV's eigenvector sign convention may also flip
// individual rows/columns of C_hat/b_hat relative to a direct GESVD; magnitudes
// and norms are unaffected.
inline CHatData computeCHat(
    const State&                                       state,
    Teuchos::RCP<const Thyra::BlockedLinearOpBase<SC>> A_blocked,
    Teuchos::RCP<const MV>                             b,
    const std::vector<int>&                            block_sizes,
    double                                             svd_tol = kGramSigmaFloor)
{
    const int nb       = static_cast<int>(block_sizes.size());
    const int curDim   = state.curDim;

    CHatData chat;
    chat.nb          = nb;
    chat.krylov_dim  = curDim;
    chat.block_sizes = block_sizes;
    chat.ranks.resize(nb, 0);
    chat.b_hat.resize(nb);
    chat.blocks.assign(nb, std::vector<SDM>(nb));

    const Teuchos::Range1D krylovCols(0, curDim - 1);

    // Per-block distributed views of the preconditioned vectors (first curDim
    // columns; no copies). state.V is deliberately not used: see below.
    std::vector<Teuchos::RCP<const TpetraMV>> Zt(nb);
    for (int j = 0; j < nb; ++j)
        Zt[j] = getBlockTpetraMV(state.Z, j)->subView(krylovCols);

    // V_tilde_j := A_jj Z_j, one sparse apply per block (paper eq 3.3).
    //
    // The sampled right-hand side is built by multiplying the stored Z forward
    // rather than by reading the FGMRES basis V. That is what makes
    //
    //     A_jj^-1 V_tilde_j = Z_j                      (paper eq 3.4)
    //
    // hold exactly instead of only to the inner solver's tolerance, and it also
    // removes the off-diagonal [A]_{S0 \ D0} Z contribution that V carries under
    // Gauss-Seidel. Two things downstream depend on it: the per-block residual
    // that would otherwise have to be absorbed by the drop tolerance is
    // identically zero, and C_hat_ii is genuinely the identity (paper eq 3.19)
    // rather than an identity asserted by fiat.
    std::vector<Teuchos::RCP<const TpetraMV>> Vt(nb);
    for (int j = 0; j < nb; ++j) {
        auto A_jj = getBlockAsCrs(A_blocked, j, j);
        TEUCHOS_TEST_FOR_EXCEPTION(A_jj.is_null(), std::runtime_error,
            "Teko::KrylovSurrogate::computeCHat: diagonal block " +
            std::to_string(j) + " is absent or is not a Tpetra::CrsMatrix, so "
            "V_tilde = A_jj Z_j cannot be formed and the surrogate is undefined.");

        auto Vt_j = Teuchos::rcp(new TpetraMV(A_jj->getRangeMap(), curDim));
        A_jj->apply(*Zt[j], *Vt_j, Teuchos::NO_TRANS, 1.0, 0.0);
        Vt[j] = Vt_j;
    }

    // ── Per-block: eigendecomposition of the Gram matrix G_j = V_j^T V_j ──
    std::vector<SDM> WTrunc(nb);            // WTrunc[j] = curDim × r_j
    std::vector<std::vector<SC>> Sigma(nb); // Sigma[j]  = leading r_j sing. values

    for (int j = 0; j < nb; ++j) {
        // Reduced basis of block j (Gram-matrix thin SVD + truncation).
        BlockReduction red = reduceKrylovBlock(*Vt[j], curDim, svd_tol);
        const int rj  = red.rank;
        chat.ranks[j] = rj;
        WTrunc[j]     = red.W;          // curDim × rj
        Sigma[j]      = red.sigma;      // length rj

        // b_hat[j] = Sigma_j^{-1} W_j^T (V_j^T b_j)   (rj × 1)
        SDM Vtb = mvTransMv(*Vt[j], *getBlockTpetraMV(b, j));   // curDim × 1
        chat.b_hat[j].shape(rj, 1);
        chat.b_hat[j].multiply(Teuchos::TRANS, Teuchos::NO_TRANS,
                               1.0, WTrunc[j], Vtb, 0.0);
        for (int r = 0; r < rj; ++r)
            chat.b_hat[j](r, 0) *= invSigmaOrZero(Sigma[j][r]);
    }

    // ── Per (i,j) block of C_hat ──────────────────────────────────────────
    // C_hat[i][j] = Sigma_i^{-1} W_i^T (V_tilde_i^T A_ij Z_j) W_j Sigma_j^{-1}.
    // A_ij Z_j is the existing distributed sparse apply; V_tilde_i^T(...) is a
    // mvTransMv reduction.

    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < nb; ++j) {
            const int ri = chat.ranks[i];
            const int rj = chat.ranks[j];
            chat.blocks[i][j].shape(ri, rj);

            if (i == j) {
                // Exactly the identity (paper eq 3.19): substituting
                // A_ii Z_i = V_tilde_i gives Q_i^T V_tilde_i W_i Sigma_i^{-1} =
                // Q_i^T Q_i Sigma_i Sigma_i^{-1} = I. This holds because
                // V_tilde is the forward multiply; it was only approximate when
                // the basis came from the raw FGMRES V.
                for (int d = 0; d < std::min(ri, rj); ++d)
                    chat.blocks[i][j](d, d) = 1.0;
                continue;
            }

            auto A_ij = A_blocked->getBlock(i, j);
            if (A_ij.is_null()) {
                // Zero operator block → C_hat[i][j] = 0 (already zero-initialized)
                continue;
            }

            // Extract A_ij as Tpetra CrsMatrix for direct apply.
            auto A_ij_tpetra_lo =
                Teuchos::rcp_dynamic_cast<const Thyra::TpetraLinearOp<SC, LO, GO, Node>>(A_ij);
            if (A_ij_tpetra_lo.is_null()) {
                // Fall back: try applying via Thyra (slower but general)
                // For now, just zero this block and continue.
                continue;
            }
            auto A_crs = Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<SC, LO, GO, Node>>(
                A_ij_tpetra_lo->getConstTpetraOperator());
            if (A_crs.is_null()) continue;

            // AijZj = A_ij * Z_j — fully distributed sparse apply, directly on
            // block j's Tpetra view (no extract/repack).
            auto AijZj_tmv = Teuchos::rcp(new TpetraMV(A_crs->getRangeMap(), curDim));
            A_crs->apply(*Zt[j], *AijZj_tmv, Teuchos::NO_TRANS, 1.0, 0.0);

            // M_ij = V_i^T (A_ij Z_j)   (curDim × curDim, replicated)
            SDM Mij = mvTransMv(*Vt[i], *AijZj_tmv);

            // T = M_ij * W_j  (curDim × rj), columns scaled by 1/sigma_{j,c}
            SDM T(curDim, rj);
            T.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
                       1.0, Mij, WTrunc[j], 0.0);
            for (int c = 0; c < rj; ++c) {
                const SC inv_s = invSigmaOrZero(Sigma[j][c]);
                for (int r = 0; r < curDim; ++r)
                    T(r, c) *= inv_s;
            }

            // C_hat[i][j] = W_i^T * T  (ri × rj), rows scaled by 1/sigma_{i,r}
            chat.blocks[i][j].multiply(Teuchos::TRANS, Teuchos::NO_TRANS,
                                       1.0, WTrunc[i], T, 0.0);
            for (int r = 0; r < ri; ++r) {
                const SC inv_s = invSigmaOrZero(Sigma[i][r]);
                for (int c = 0; c < rj; ++c)
                    chat.blocks[i][j](r, c) *= inv_s;
            }
        }
    }

    return chat;
}

}  // namespace KrylovSurrogate
}  // namespace Teko
