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
// the application's Tpetra LHS.
//
// Diagnostic branch: K1-K4 report each step (see
// teko-reconfig/claudes_world/plans/tpetra_hook_route_plan.md).
#pragma once

#include <sstream>
#include <string>

#include "Teuchos_Range1D.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Tpetra_Operator.hpp"

#include "BelosAdaptiveDiag.hpp"
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
    using Belos::AdaptiveDiag::dynType;
    std::ostringstream m;
    m << "candidate " << name << ": type=" << dynType(op);
    auto wrapper = Teuchos::rcp_dynamic_cast<const TpetraHelpers::TpetraOperatorWrapper>(op);
    if (wrapper.is_null()) {
        m << " -> rejected, not a Teko TpetraOperatorWrapper";
        Belos::AdaptiveDiag::print("K2", m.str());
        return false;
    }
    auto blocked = Teuchos::rcp_dynamic_cast<const Thyra::BlockedLinearOpBase<SC>>(
        wrapper->getThyraOp());
    m << " thyraOp=" << dynType(wrapper->getThyraOp());
    if (blocked.is_null() || wrapper->getMapStrategy().is_null()) {
        m << " -> rejected, " << (blocked.is_null() ? "Thyra operator not blocked"
                                                   : "no mapping strategy");
        Belos::AdaptiveDiag::print("K2", m.str());
        return false;
    }
    m << " -> accepted";
    Belos::AdaptiveDiag::print("K2", m.str());
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
        if (ifo.is_null()) {
            Belos::AdaptiveDiag::print("K2", std::string("candidate ") + name + ": type="
                + Belos::AdaptiveDiag::dynType(prec)
                + " -> rejected, not a Teko InverseFactoryOperator");
            continue;
        }
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
    using Belos::AdaptiveDiag::demangle;
    using Belos::AdaptiveDiag::print;

    // The hook's own re-solve is Thyra-typed, so this only guards against an
    // application solve nested inside a running adaptiveLoop.
    if (detail::g_inAdaptiveLoop) {
        print("K1", "tpetraAdaptiveLoop entered inside a running adaptiveLoop; returning");
        return {};
    }

    const bool typesMatch = (*args.mv == typeid(TpetraMV)) && (*args.op == typeid(TpetraOp));
    {
        std::ostringstream m;
        m << "tpetraAdaptiveLoop entered: types " << (typesMatch ? "match" : "MISMATCH")
          << " got MV=" << demangle(args.mv->name()) << " OP=" << demangle(args.op->name());
        if (!typesMatch)
            m << " expected MV=" << demangle(typeid(TpetraMV).name())
              << " OP=" << demangle(typeid(TpetraOp).name());
        print("K1", m.str());
    }
    if (!typesMatch) return {};

    const auto& state   = *static_cast<const TpetraState*>(args.state);
    auto&       problem = *static_cast<Teuchos::RCP<TpetraProblem>*>(args.problem);
    if (state.curDim == 0 || state.Z.is_null() || state.V.is_null()) {
        print("K1", "no Krylov data (curDim=" + std::to_string(state.curDim) + "); returning");
        return {};
    }

    const BlockSource src = findBlockSource(*problem);
    if (src.blocked.is_null()) {
        print("K3", "no blocked Teko operator found among operator / rightPrec / leftPrec;"
                    " hook cannot run on this solve");
        return {};
    }
    {
        std::ostringstream m;
        const int nb = src.blocked->productRange()->numBlocks();
        m << "block source=" << src.from << " blocks=" << nb << " sizes=[";
        for (int i = 0; i < nb; ++i)
            m << (i ? "," : "") << src.blocked->productRange()->getBlock(i)->dim();
        m << "] diagBlockTypes=[";
        for (int i = 0; i < nb; ++i)
            m << (i ? "," : "") << Belos::AdaptiveDiag::dynType(src.blocked->getBlock(i, i));
        m << "] curDim=" << state.curDim;
        print("K3", m.str());
    }

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

    if (r.converged) {
        map.copyThyraIntoTpetra(x, *problem->getLHS());
        print("K4", "re-solve converged (iters=" + std::to_string(r.num_iters)
                    + "); copied into the application's Tpetra LHS");
    } else {
        print("K4", "no converged re-solve; application's LHS left as solve 1 produced it");
    }
    return r;
}

}  // namespace KrylovSurrogate
}  // namespace Teko
