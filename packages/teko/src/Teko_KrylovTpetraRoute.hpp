// Teko_KrylovTpetraRoute.hpp — the adaptive hook for Tpetra-typed Belos solves.
//
// An application that calls Belos::BlockGmresSolMgr with Tpetra::MultiVector /
// Tpetra::Operator never reaches adaptiveLoop: that hook needs a blocked Thyra
// operator, and the Thyra call site only exists in the Thyra instantiation.
// Belos instead hands such a solve to the type-erased slot
// (Belos::AdaptiveHook::invokeErased), and this is what Teko registers there.
//
// It finds the block structure Teko itself built for the preconditioner, a
// TpetraHelpers::TpetraOperatorWrapper whose Thyra operator is blocked, from
//   a. the problem's operator (a BlockedTpetraOperator / StridedTpetraOperator
//      handed to Belos), else
//   b. the right preconditioner as an InverseFactoryOperator, via
//      getForwardOp() (the usual case: Belos holds the flat CrsMatrix and Teko
//      blocked a wrapper of it), else
//   c. the same for the left preconditioner.
// It then copies the Krylov data and the RHS/LHS into blocked Thyra
// multivectors with that wrapper's mapping strategy, runs the unchanged
// adaptiveLoop on a Thyra problem, and copies a converged re-solve back into
// the application's Tpetra LHS. See
// teko-reconfig/claudes_world/plans/tpetra_hook_route_plan.md.
#pragma once

#include <string>

#include "Teuchos_Range1D.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Tpetra_Operator.hpp"

#include "BelosAdaptiveHook.hpp"

#include "Teko_KrylovSurrogate.hpp"
#include "Teko_TpetraInverseFactoryOperator.hpp"
#include "Teko_TpetraOperatorWrapper.hpp"

namespace Teko {
namespace KrylovSurrogate {

using TpetraOp      = Tpetra::Operator<SC, LO, GO, Node>;
using TpetraProblem = Belos::LinearProblem<SC, TpetraMV, TpetraOp>;
using TpetraState   = Belos::GmresIterationState<SC, TpetraMV>;

// The wrapper whose blocked Thyra operator and mapping strategy stand in for
// the application's flat Tpetra system.
struct BlockSource {
    Teuchos::RCP<const TpetraHelpers::TpetraOperatorWrapper> wrapper;
    Teuchos::RCP<const Thyra::BlockedLinearOpBase<SC>>       blocked;
    std::string                                              from;
};

// Accept `op` if it is a Teko wrapper over a blocked Thyra operator.
inline bool tryBlockSource(const std::string& name, Teuchos::RCP<const TpetraOp> op,
                           BlockSource& out)
{
    auto wrapper = Teuchos::rcp_dynamic_cast<const TpetraHelpers::TpetraOperatorWrapper>(op);
    if (wrapper.is_null()) return false;
    auto blocked = Teuchos::rcp_dynamic_cast<const Thyra::BlockedLinearOpBase<SC>>(
        wrapper->getThyraOp());
    if (blocked.is_null() || wrapper->getMapStrategy().is_null()) return false;
    out = BlockSource{wrapper, blocked, name};
    return true;
}

inline BlockSource findBlockSource(const TpetraProblem& problem)
{
    BlockSource src;
    if (tryBlockSource("operator", problem.getOperator(), src)) return src;
    const std::pair<const char*, Teuchos::RCP<const TpetraOp>> precs[] = {
        {"rightPrec", problem.getRightPrec()}, {"leftPrec", problem.getLeftPrec()}};
    for (const auto& [name, prec] : precs) {
        if (prec.is_null()) continue;
        auto ifo = Teuchos::rcp_dynamic_cast<const TpetraHelpers::InverseFactoryOperator>(prec);
        if (ifo.is_null()) continue;
        if (tryBlockSource(std::string(name) + ".getForwardOp()", ifo->getForwardOp(), src))
            return src;
    }
    return src;
}

// The first ncols columns of X as a new blocked Thyra multivector on `space`.
inline Teuchos::RCP<MV> toBlocked(const TpetraMV& X, int ncols,
                                  const TpetraHelpers::MappingStrategy& map,
                                  const Teuchos::RCP<const Thyra::VectorSpaceBase<SC>>& space)
{
    auto cols = X.subView(Teuchos::Range1D(0, ncols - 1));
    auto Y = Thyra::createMembers(space, ncols);
    map.copyTpetraIntoThyra(*cols, Y.ptr());
    return Y;
}

// Registered with Belos::AdaptiveHook::registerErasedHook.
inline Belos::AdaptiveHook::HookResult tpetraAdaptiveLoop(
    const Belos::AdaptiveHook::ErasedArgs& args)
{
    // The hook's own re-solve is Thyra-typed, so this only guards against an
    // application solve nested inside a running adaptiveLoop.
    if (detail::g_inAdaptiveLoop) return {};

    // Only the Tpetra types Teko is instantiated on can be converted.
    if (*args.mv != typeid(TpetraMV) || *args.op != typeid(TpetraOp)) return {};

    const auto& state   = *static_cast<const TpetraState*>(args.state);
    auto&       problem = *static_cast<Teuchos::RCP<TpetraProblem>*>(args.problem);
    if (state.curDim == 0 || state.Z.is_null() || state.V.is_null()) return {};

    // No blocked Teko operator among operator / rightPrec / leftPrec: the hook
    // cannot run on this solve.
    const BlockSource src = findBlockSource(*problem);
    if (src.blocked.is_null()) return {};

    const auto& map  = *src.wrapper->getMapStrategy();
    const int   nrhs = static_cast<int>(problem->getRHS()->getNumVectors());

    State thyraState;
    thyraState.curDim = state.curDim;
    thyraState.V = toBlocked(*state.V, state.curDim, map, src.blocked->range());
    thyraState.Z = toBlocked(*state.Z, state.curDim, map, src.blocked->domain());
    thyraState.H = state.H;
    thyraState.R = state.R;
    thyraState.z = state.z;

    auto x = toBlocked(*problem->getLHS(), nrhs, map, src.blocked->domain());
    auto b = toBlocked(*problem->getRHS(), nrhs, map, src.blocked->range());
    auto thyraProblem = Teuchos::rcp(new Problem(
        Teuchos::rcp_implicit_cast<const OP>(src.blocked), x, b));

    const Belos::AdaptiveHook::HookResult r =
        adaptiveLoop(thyraState, thyraProblem, src.blocked, args.params, args.metrics);

    // A converged re-solve wrote x; otherwise the application's LHS is left as
    // solve 1 produced it.
    if (r.converged) map.copyThyraIntoTpetra(x, *problem->getLHS());
    return r;
}

}  // namespace KrylovSurrogate
}  // namespace Teko
