// Teko_KrylovSurrogate.hpp — the Belos adaptive hook that drives the
// Krylov-informed block-preconditioner reconfiguration loop.
//
// This header is the orchestration only. The pieces it coordinates live in
// three sibling headers it includes:
//   * Teko_KrylovReducedModel.hpp — the surrogate math (C_hat / b_hat from the
//     Krylov state via per-block Gram-matrix thin SVDs) and the shared type
//     aliases.
//   * Teko_KrylovReconfigIO.hpp   — the JSON file formats (request / reconfig /
//     conv) and the watcher handshake.
//   * Teko_KrylovReconfigPrec.hpp — assembling the reconfigured ("as-if-
//     original") flat blocked system and its preconditioner from an ordering.
//
// Usage (transparent to callers):
//   adaptiveLoop() is registered as the BelosAdaptiveHook and fires after
//   every flexible GMRES solve on a blocked operator (converged or stalled).
//   The request/response/convergence files live in TEKO_RECONFIG_REQUESTS_DIR
//   if set, otherwise defaultRequestsDir() below (teko-reconfig-requests under
//   the working directory the application was launched from).
//
// adaptiveLoop():
//   1. Computes C_hat and writes s<N>_request.json to requests_dir, where
//      N = nextRequestNumber(requests_dir) (0, 1, 2, ... — never reused).
//   2. Polls for s<N>_reconfig.json in the same directory (left in place after
//      being read — its presence is how wait_for_request.py recognizes an
//      already-answered request).
//   3. Solves and times each requested ordering as a test on a fresh flat
//      blocked operator (as if the application had been called with that
//      grouping originally). The row flagged use_ordering, if any, is not
//      tested: it is the final solve. Without one, the test that performed
//      best is re-solved as the final.
//   4. Overwrites the LHS with the final solve's result and writes
//      s<N>_conv.json, one row per solve that ran.
//
// Depends on: Belos, Thyra, Tpetra, Teko, Stratimikos, Teuchos.
// No external JSON library — JSON is written/parsed manually.
#pragma once

// ── Standard library ──────────────────────────────────────────────────────
#include <chrono>
#include <cstdlib>    // getenv
#include <iostream>
#include <numeric>
#include <string>
#include <system_error>   // error_code, for the non-throwing fs::current_path
#include <utility>
#include <vector>

// ── Teuchos ───────────────────────────────────────────────────────────────
#include "Teuchos_Comm.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

// ── Belos ─────────────────────────────────────────────────────────────────
#include "BelosAdaptiveHook.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosThyraAdapter.hpp"
#include "BelosTypes.hpp"

// ── Thyra ─────────────────────────────────────────────────────────────────
#include "Thyra_BlockedLinearOpBase.hpp"
#include "Thyra_LinearOpBase.hpp"
#include "Thyra_MultiVectorBase.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"

// ── Teko ──────────────────────────────────────────────────────────────────
#include "Teko_BlockLowerTriInverseOp.hpp"
#include "Teko_FactorTimeRegistry.hpp"
#include "Teko_InverseFactory.hpp"
#include "Teko_KrylovReconfigIO.hpp"     // JSON file formats + record structs
#include "Teko_KrylovReconfigPrec.hpp"   // reconfigured system + preconditioner
#include "Teko_KrylovReducedModel.hpp"   // type aliases + computeCHat (surrogate math)
#include "Teko_Utilities.hpp"

namespace Teko {
namespace KrylovSurrogate {

// (comm, rank) from block 0 of a product multivector — used to gate the JSON
// file I/O in adaptiveLoop() to rank 0. (getBlockTpetraMV is defined in
// Teko_KrylovReducedModel.hpp.)
inline std::pair<Teuchos::RCP<const Teuchos::Comm<int>>, int>
getCommAndRank(Teuchos::RCP<const MV> mv)
{
    auto comm = getBlockTpetraMV(mv, 0)->getMap()->getComm();
    return {comm, comm->getRank()};
}

// Default location for s<N>_request.json / s<N>_reconfig.json / s<N>_conv.json
// when TEKO_RECONFIG_REQUESTS_DIR is unset. All file types live together in
// this single directory. Hardcoded for now, relative to the Trilinos checkout
// this package lives in.
// Where the request/reconfig/convergence JSON goes when
// TEKO_RECONFIG_REQUESTS_DIR is unset: "teko-reconfig-requests" under the
// process's current working directory. That is the directory the application was
// LAUNCHED from (or whatever it last chdir'd to), not the build tree, not the
// install prefix, and not the executable's own directory. Rank 0 creates it.
//
// Resolved on each call rather than once at load, so it follows a chdir, and
// returned absolute so the paths this hook prints are unambiguous in a log.
// Falls back to the bare relative name if the cwd cannot be read (it can fail
// if the directory has been deleted under the process), which the filesystem
// then resolves the same way.
//
// This used to be a hardcoded absolute path from the container the hook was
// first developed in, which existed on no other machine: with the flag set and
// the variable unset, an unrelated application would try to create
// /home/node/..., fail, and then block for the full waitForOrdering timeout
// with no watcher able to find the directory.
inline std::string defaultRequestsDir()
{
    std::error_code ec;
    const auto cwd = fs::current_path(ec);
    return ec ? std::string("teko-reconfig-requests")
              : (cwd / "teko-reconfig-requests").string();
}

// adaptiveLoop()'s Phase 4 below runs a second flexible FGMRES solve, which
// (if it converges) would re-invoke this same hook recursively on the
// already-reconfigured system — at best redundant work, at worst an unbounded
// reconfigure/solve recursion. g_inAdaptiveLoop / ScopedGuard
// below prevent this: the outermost call sets the flag before Phase 4, so a
// recursive invocation from solver2.solve() sees it set and returns
// immediately at the top of adaptiveLoop(). thread_local since Belos solves
// (and therefore this hook) could in principle run on different threads.
namespace detail {
thread_local bool g_inAdaptiveLoop = false;
}

// RAII guard around g_inAdaptiveLoop, reset on scope exit even if
// solver2.solve() throws (e.g. BlockGmresSolMgrOrthoFailure).
class ScopedAdaptiveLoopGuard {
public:
    ScopedAdaptiveLoopGuard()  { detail::g_inAdaptiveLoop = true; }
    ~ScopedAdaptiveLoopGuard() { detail::g_inAdaptiveLoop = false; }
    ScopedAdaptiveLoopGuard(const ScopedAdaptiveLoopGuard&) = delete;
    ScopedAdaptiveLoopGuard& operator=(const ScopedAdaptiveLoopGuard&) = delete;
};

// Build the as-if-original flat system for `ordering`, solve it, and return
// timing/convergence. Factor and iterate wall times are max-reduced across
// ranks (critical path) so every rank gets identical numbers — which makes
// any time-based selection deterministic across ranks. If writeToLHS is true
// and the solve converged, the solution is unpacked into problem->getLHS().
// The hook-recursion guard must already be held by the caller (this runs a
// flexible solve, which would otherwise re-enter the hook).
//
// Preconditioner construction is guarded: if buildReconfiguredPrec throws on
// ANY rank (e.g. a singular merged block), the failure flag is max-reduced and
// every rank returns a non-converged result without entering the collective
// solve — so one bad candidate in a sweep can't deadlock the run (other ranks
// would otherwise hang in solver.solve()).
inline SolveRecord solveOrdering(
    const std::vector<int>&                            ordering,
    Teuchos::RCP<const Thyra::BlockedLinearOpBase<SC>> A_blocked,
    int                                                nb,
    const std::string&                                 method,
    Teuchos::RCP<Teko::InverseFactory>                 invFact,
    Teuchos::RCP<Belos::AdaptiveHook::Problem>         problem,
    Teuchos::RCP<const Teuchos::ParameterList>         orig_params,
    Teuchos::RCP<const Teuchos::Comm<int>>             comm,
    double                                             b_norm,
    bool                                               writeToLHS)
{
    SolveRecord r;
    r.ordering         = ordering;
    r.initial_residual = b_norm;

    ReconfiguredSystem recon;
    int local_err = 0;
    try {
        recon = buildReconfiguredPrec(ordering, A_blocked, nb, method, invFact);
    } catch (const std::exception&) {
        local_err = 1;
    }
    int glob_err = local_err;
    if (comm->getSize() > 1)
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_err, &glob_err);
    if (glob_err) {
        // Build failed somewhere; do not enter the collective solve. r is left
        // as not-converged with zero timings.
        return r;
    }

    auto newProblem = Teuchos::rcp(new Problem());
    newProblem->setOperator(Teuchos::rcp_dynamic_cast<const OP>(recon.flatOp));
    newProblem->setRightPrec(Teuchos::rcp_dynamic_cast<const OP>(recon.precOp));

    auto x_new = Thyra::createMembers(recon.flatOp->domain(), 1);
    Thyra::assign(x_new.ptr(), SC(0));
    auto b_new = Thyra::createMembers(recon.flatOp->range(), 1);
    copyOriginalToGrouped(problem->getRHS(), b_new, recon.groups);
    newProblem->setLHS(x_new);
    newProblem->setRHS(b_new);
    newProblem->setProblem();

    auto solverParams = Teuchos::rcp(new Teuchos::ParameterList(*orig_params));
    Belos::BlockGmresSolMgr<SC, MV, OP> solver(newProblem, solverParams);

    const auto t0 = std::chrono::steady_clock::now();
    Belos::ReturnType ret = solver.solve();
    double iterate_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    double factor_sec = recon.factor_wall_time_sec;
    if (comm->getSize() > 1) {
        double loc_f = factor_sec, loc_i = iterate_sec;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &loc_f, &factor_sec);
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &loc_i, &iterate_sec);
    }

    // Verify Belos's verdict against the explicitly recomputed residual before
    // trusting it. Belos reports convergence from its implicit estimate only
    // (see trueRelativeResidual), and a candidate that converges falsely stops
    // early, so it would otherwise report converged with a low iteration count,
    // a small residual and a short iterate time: it would win every tiebreaker
    // in both selection comparators. Recording the true residual here is also
    // what makes final_residual in conv.json meaningful.
    const double tol = solverParams->get(
        "Convergence Tolerance",
        static_cast<double>(Belos::DefaultSolverParameters::convTol));
    const double true_res = trueRelativeResidual(
        Teuchos::rcp_dynamic_cast<const Thyra::LinearOpBase<SC>>(recon.flatOp),
        b_new, x_new);

    r.iters                 = solver.getNumIters();
    r.converged             = (ret == Belos::Converged) &&
                              (true_res <= kResidualSlack * tol);
    r.final_residual        = true_res;
    r.factor_wall_time_sec  = factor_sec;
    r.iterate_wall_time_sec = iterate_sec;
    r.total_wall_time_sec   = factor_sec + iterate_sec;

    if (writeToLHS && r.converged)
        copyGroupedToOriginal(x_new, problem->getLHS(), recon.groups);
    return r;
}

// This is the function registered as Belos::AdaptiveHook::HookFn.
// It is called by BelosBlockGmresSolMgr::solve() after a flexible GMRES
// solve converges.
//
// On entry: problem->getLHS() contains the first solve's result.
// On exit:  problem->getLHS() is overwritten with the selected ordering's
//           result (if one was attempted and converged).
// Returns a HookResult: when the selected re-solve converged, it carries that
// solve's iteration count / achieved tolerance so the solver manager can
// report the (possibly rescued) outcome. Fires on both converged and stalled
// first solves — a stall (curDim > 0) takes the full reconfigure path too, so
// a short, deliberately-low-max-iter first solve can be rescued.
inline Belos::AdaptiveHook::HookResult adaptiveLoop(
    const Belos::AdaptiveHook::State&                         state,
    Teuchos::RCP<Belos::AdaptiveHook::Problem>                problem,
    Teuchos::RCP<const Thyra::BlockedLinearOpBase<SC>>        A_blocked,
    Teuchos::RCP<const Teuchos::ParameterList>                orig_params,
    const Belos::AdaptiveHook::SolveMetrics&                  solve1_metrics)
{
    // Bail out immediately if this call is the recursive re-entry from
    // Phase 4's solver2.solve() below (see ScopedAdaptiveLoopGuard).
    if (detail::g_inAdaptiveLoop) return {};

    // Opt-in gate. The hook is registered globally at libteko load, so it would
    // otherwise fire on *every* flexible blocked GMRES solve in any process
    // that links Teko — hijacking unrelated solves and blocking them on the
    // reconfig handshake. Do nothing unless TEKO_ADAPTIVE_RECONFIG is set to a
    // truthy value. (A per-solve ParameterList flag is not used because Belos
    // validates the solver parameter list and rejects unknown keys.)
    // The predicate itself lives in Teko_KrylovReconfigPrec.hpp, so the front
    // end can ask the same question without including this header.
    //
    // Announced on every solve that reaches here, open or closed, and on every
    // rank: this is the line that distinguishes "the hook is switched off" from
    // "the hook was never invoked at all", which look identical from the
    // outside and are completely different problems. It reports the raw
    // variable too, so a value that is set but falsy (0, false) is visible as
    // such rather than looking unset. Note the recursion guard above returns
    // first, so Phase 4's re-solve does not print a second time.
    {
        const char* raw = std::getenv("TEKO_ADAPTIVE_RECONFIG");
        const bool enabled = adaptiveEnabled();
        announceBoth(std::string("[TekoAdaptive] gate: TEKO_ADAPTIVE_RECONFIG=")
                     + (raw ? (*raw ? raw : "(empty)") : "(unset)")
                     + " -> hook " + (enabled ? "ACTIVE" : "inert") + "\n");
        if (!enabled) return {};
    }

    // requests_dir comes from TEKO_RECONFIG_REQUESTS_DIR if set, otherwise
    // defaultRequestsDir(). Announced either way, on rank 0 below, because a
    // watcher has to be pointed at this exact directory and guessing it from a
    // launch directory is how an afternoon disappears.
    const char* reconfig_dir_env = std::getenv("TEKO_RECONFIG_REQUESTS_DIR");
    const std::string requests_dir = reconfig_dir_env ? std::string(reconfig_dir_env)
                                                      : defaultRequestsDir();

    if (A_blocked.is_null()) {
        std::cerr << "[TekoAdaptive] operator is not blocked; skipping.\n";
        return {};
    }
    if (state.curDim == 0) {
        std::cerr << "[TekoAdaptive] curDim==0; nothing to compute.\n";
        return {};
    }

    // All file I/O (request/reconfig/convergence JSON) is rank-0-only; the
    // numerics (computeCHat, buildReconfiguredPrec, second solve) are
    // collective and run identically on every rank.
    auto [comm, rank] = getCommAndRank(state.V);

    if (rank == 0) {
        // Name the directory once per process, and say whether it was chosen or
        // defaulted: the default depends on where the application was launched
        // from, and a watcher pointed anywhere else leaves this solve blocking
        // in waitForOrdering until it times out.
        static bool dir_announced = false;
        if (!dir_announced) {
            dir_announced = true;
            std::cout << "[TekoAdaptive] requests dir: " << requests_dir
                      << (reconfig_dir_env ? "  (TEKO_RECONFIG_REQUESTS_DIR)"
                                           : "  (default: $PWD/teko-reconfig-requests)")
                      << "\n";
        }
        fs::create_directories(requests_dir);
    }
    // All request/reconfig/convergence files live together in requests_dir.
    const std::string convergence_dir = requests_dir;

    // Determine block structure from A_blocked.
    const int nb = A_blocked->productRange()->numBlocks();
    std::vector<int> block_sizes(nb);
    for (int j = 0; j < nb; ++j)
        block_sizes[j] =
            static_cast<int>(A_blocked->productRange()->getBlock(j)->dim());

    if (rank == 0)
        std::cout << "[TekoAdaptive] first solve done. curDim=" << state.curDim
                  << "  nb=" << nb << "\n";

    // ||b||: shared initial residual for both solves. Both start from x0 = 0,
    // and reordering (Phase 3) only regroups vector blocks, which preserves
    // the 2-norm.
    double b_norm = 0.0;
    {
        Teuchos::Array<SC> norms(1);
        Thyra::norms_2(*problem->getRHS(), norms());
        b_norm = norms[0];
    }

    // Setup (factorization) time of the first solve: everything the
    // application spent inside Teko::buildInverse / Teko::rebuildInverse since
    // the previous hook invocation — i.e. building the preconditioner that
    // the solve which just converged used. Drain the registry now (Phase 3's
    // buildReconfiguredPrec will bump it again; it is timed separately), and
    // re-drain on every exit path below so solve 2's factorizations never
    // leak into the *next* solve's setup figure. Factorization is rank-local
    // work, so report the max across ranks (the wall-clock critical path).
    double solve1_factor_sec = Teko::FactorTimeRegistry::read();
    Teko::FactorTimeRegistry::reset();
    if (comm->getSize() > 1) {
        double local = solve1_factor_sec;
        Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local, &solve1_factor_sec);
    }
    struct RegistryResetGuard {
        ~RegistryResetGuard() { Teko::FactorTimeRegistry::reset(); }
    } registry_reset_guard;

    // Solve 1's residual is recomputed explicitly rather than taken from
    // solve1_metrics.achieved_tol, which is Belos's implicit estimate. On entry
    // problem->getLHS() holds solve 1's result, so this measures what the
    // application actually got. Without it conv.json would compare solve 1's
    // estimate against solve 2's verified residual, which is not like for like.
    const double solve1_true_res = trueRelativeResidual(
        Teuchos::rcp_dynamic_cast<const Thyra::LinearOpBase<SC>>(A_blocked),
        problem->getRHS(), problem->getLHS());

    // The initial row of conv.json. Its ordering is the application's own
    // blocking, every block its own group, and its converged flag gets the same
    // true-residual check every reconfigured solve gets in solveOrdering.
    const double solve1_tol =
        orig_params->isParameter("Convergence Tolerance")
            ? orig_params->get<double>("Convergence Tolerance")
            : static_cast<double>(Belos::DefaultSolverParameters::convTol);
    SolveRecord s1;
    s1.ordering.resize(nb);
    std::iota(s1.ordering.begin(), s1.ordering.end(), 0);
    s1.iters                 = solve1_metrics.num_iters;
    s1.type                  = "initial";
    s1.converged             = solve1_metrics.converged &&
                               (solve1_true_res <= kResidualSlack * solve1_tol);
    s1.initial_residual      = b_norm;
    s1.final_residual        = solve1_true_res;
    s1.factor_wall_time_sec  = solve1_factor_sec;
    s1.iterate_wall_time_sec = solve1_metrics.wall_time_sec;
    s1.total_wall_time_sec   = solve1_factor_sec + solve1_metrics.wall_time_sec;
    std::vector<SolveRecord> records{s1};

    // A stalled first solve (hit max iterations / lost accuracy) is NOT a dead
    // end — it is exactly the case reconfiguration should rescue, and the
    // surrogate builds fine from its Krylov data (curDim == max iters). So it
    // falls through to the same surrogate / watcher / sweep / re-solve path as
    // a converged solve; s1 still records the stall (iterations == max iters,
    // time-to-stall), and if the selected re-solve converges this hook reports
    // it back so the manager returns success instead of the original stall.
    if (rank == 0 && !solve1_metrics.converged)
        std::cerr << "[TekoAdaptive] first solve stalled ("
                  << solve1_metrics.num_iters
                  << " iters); attempting reconfiguration rescue.\n";

    // ── Phase 1: compute C_hat, b_hat and write s<N>_request.json ──────────
    // computeCHat is collective (mvTransMv reductions + distributed applies);
    // its result is replicated identically on every rank, so only rank 0
    // writes the request file.
    CHatData chat = computeCHat(state, A_blocked, problem->getRHS(), block_sizes);
    int request_id = -1;
    if (rank == 0) request_id = writeRequestJson(chat, requests_dir);

    // ── Phase 2: wait for s<N>_reconfig.json ───────────────────────────────
    // The response arrives from an external process via the filesystem, so
    // rank 0 polls. Every rank needs the orderings and which row (if any) is
    // flagged use_ordering; the rest of each row (surrogate_iters, the flags)
    // only goes into conv.json, which rank 0 writes, so it stays on rank 0.
    ReconfigResponse resp;
    if (rank == 0) resp = waitForReconfig(requests_dir, request_id);

    int num_solves = static_cast<int>(resp.solves.size());
    Teuchos::broadcast(*comm, 0, 1, &num_solves);
    if (rank != 0) resp.solves.assign(num_solves, RequestedSolve());
    for (int t = 0; t < num_solves; ++t) {
        auto& ord = resp.solves[t].ordering;
        int len = static_cast<int>(ord.size());
        Teuchos::broadcast(*comm, 0, 1, &len);
        ord.resize(len);
        if (len > 0) Teuchos::broadcast(*comm, 0, len, ord.data());
    }

    // The first row flagged use_ordering is the final solve; -1 means none was,
    // and the final is chosen from how the tests actually performed.
    int use_index = -1;
    if (rank == 0) {
        for (int t = 0; t < num_solves; ++t)
            if (hasFlag(resp.solves[t], kUseOrderingFlag)) { use_index = t; break; }
    }
    Teuchos::broadcast(*comm, 0, 1, &use_index);

    // The initial solve's ordering may also have been scored by the surrogate;
    // if the watcher sent it, the initial row carries that prediction too.
    if (rank == 0) {
        for (const auto& s : resp.solves)
            if (s.ordering == records[0].ordering) {
                copySurrogateFields(s, records[0]);
                break;
            }
    }

    if (num_solves == 0) {
        if (rank == 0) {
            std::cerr << "[TekoAdaptive] "
                      << (resp.received ? "reconfig requested no solves"
                                        : "no reconfig received")
                      << "; returning first solve result.\n";
            writeConvergenceJson(convergence_dir, request_id, records);
        }
        return {};
    }

    // ── Phase 3: determine method, then test / select / final solve ────────
    // Forced to block Gauss-Seidel (block lower triangular) regardless of what
    // the first solve used. Previously this dynamic-cast probed
    // problem->getRightPrec() for BlockLowerTriInverseOp and fell back to
    // "jacobi", which made the reconfigured preconditioner's shape depend on
    // the calling application's choice rather than on this experiment's.
    const std::string method = "gs";

    // Each candidate solve runs a flexible GMRES that would otherwise re-enter
    // this hook; hold the guard across the whole sweep + final solve.
    ScopedAdaptiveLoopGuard recursion_guard;

    // One per-block solver factory (GMRES + RILUK), reused for every candidate
    // build below (factory construction is untimed setup; sharing it avoids
    // rebuilding the Stratimikos solve strategy once per ordering).
    auto invFact = makeBlockSolverInverseFactory();

    // Every requested row except the use_ordering one is a test.
    std::vector<int> test_rows;
    for (int t = 0; t < num_solves; ++t)
        if (t != use_index) test_rows.push_back(t);

    // Warm-up before timed tests. The first build+solve inside this hook pays
    // one-time costs (kernel first-touch, Stratimikos/Ifpack2 setup paths)
    // that the factorization warm-up does not cover; left unaddressed they
    // would be charged entirely to the first test and bias the best-time pick.
    // When there are tests to time, run one throwaway solve on the identity
    // (all-separate) ordering first so every recorded test is measured warm.
    // The result is discarded (writeToLHS = false) and not recorded.
    if (!test_rows.empty()) {
        std::vector<int> identity(nb);
        std::iota(identity.begin(), identity.end(), 0);
        solveOrdering(identity, A_blocked, nb, method, invFact, problem,
                      orig_params, comm, b_norm, /*writeToLHS=*/false);
    }

    // Tests: build, solve, and time each on the freshly assembled flat
    // (as-if-original) system, so all are timed comparably to the first solve.
    std::vector<SolveRecord> tests;
    tests.reserve(test_rows.size());
    for (int t : test_rows) {
        SolveRecord rec = solveOrdering(resp.solves[t].ordering, A_blocked, nb, method,
                                        invFact, problem, orig_params, comm, b_norm,
                                        /*writeToLHS=*/false);
        rec.type = "test";
        copySurrogateFields(resp.solves[t], rec);
        tests.push_back(std::move(rec));
    }

    // Selection. use_ordering, when given, is respected. Otherwise the best
    // test wins: converged first, then least total wall time, and among tests
    // that all failed the least bad by convergence. The comparators operate
    // only on cross-rank-reduced, deterministic quantities, so every rank
    // selects the same row.
    auto convBetter = [](const SolveRecord& a, const SolveRecord& b) {
        if (a.converged != b.converged) return a.converged;          // converged first
        if (a.iters != b.iters) return a.iters < b.iters;
        return a.final_residual < b.final_residual;
    };
    auto timeBetter = [&](const SolveRecord& a, const SolveRecord& b) {
        if (a.converged != b.converged) return a.converged;
        if (a.converged && b.converged)
            return a.total_wall_time_sec < b.total_wall_time_sec;
        return convBetter(a, b);  // neither converged: least-bad by convergence
    };

    int selected_row = use_index;
    std::string selection_mode = kUseOrderingFlag;
    if (use_index < 0) {
        int best = 0;
        for (int i = 1; i < static_cast<int>(tests.size()); ++i)
            if (timeBetter(tests[i], tests[best])) best = i;
        selected_row   = test_rows[best];
        selection_mode = "best_time";
    }

    // ── Phase 4: the final solve ───────────────────────────────────────────
    // Solve the selected ordering (a best-time pick is re-solved, since no
    // test keeps its solution) and write its result into the LHS the
    // application reads back.
    SolveRecord finalRes = solveOrdering(resp.solves[selected_row].ordering, A_blocked,
                                         nb, method, invFact, problem, orig_params,
                                         comm, b_norm, /*writeToLHS=*/true);
    finalRes.type = "final from selection mode " + selection_mode;
    copySurrogateFields(resp.solves[selected_row], finalRes);

    if (rank == 0) {
        records.insert(records.end(), tests.begin(), tests.end());
        records.push_back(finalRes);
        writeConvergenceJson(convergence_dir, request_id, records);

        if (!finalRes.converged)
            std::cerr << "[TekoAdaptive] final solve did not converge ("
                      << finalRes.iters << " iters). Using first result.\n";
        else
            std::cout << "[TekoAdaptive] final solve (" << selection_mode
                      << ") converged (" << finalRes.iters << " iters).\n";
    }

    // Report the final solve back to the manager. Only when it converged (and
    // so wrote its result into the LHS) does this override the manager's
    // result — turning a rescued stall into a reported success. If it did not
    // converge, converged stays false and the manager keeps its own
    // (first-solve) result.
    Belos::AdaptiveHook::HookResult result;
    result.converged    = finalRes.converged;
    result.num_iters    = finalRes.iters;
    result.achieved_tol = finalRes.final_residual;
    return result;
}

} // namespace KrylovSurrogate
} // namespace Teko
