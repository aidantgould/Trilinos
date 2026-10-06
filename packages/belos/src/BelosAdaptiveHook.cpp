// BelosAdaptiveHook.cpp — out-of-line storage for the adaptive-hook registry
// declared in BelosAdaptiveHook.hpp.  See that header for why this lives in
// exactly one translation unit (compiled into libbelos.so) instead of being
// a header-only inline function.

#include "BelosAdaptiveHook.hpp"
#include "BelosAdaptiveDiag.hpp"

namespace Belos {
namespace AdaptiveHook {

namespace {
HookFn& globalHook() {
    static HookFn hook;
    return hook;
}

// Diagnostic: announce this copy of libbelos and its slot when it loads. One
// L1 line per copy, so two lines with different slots mean two copies.
struct AnnounceLoad {
    AnnounceLoad() {
        std::ostringstream m;
        m << "libbelos with adaptive hook registry loaded, slot=" << &globalHook();
        AdaptiveDiag::print("L1", m.str());
    }
};
static AnnounceLoad g_announceLoad;
} // namespace

const void* slotAddress() { return &globalHook(); }
bool        hookIsSet()   { return static_cast<bool>(globalHook()); }

void registerHook(HookFn fn) {
    globalHook() = std::move(fn);
    std::ostringstream m;
    m << "registerHook: slot=" << &globalHook()
      << " now " << (globalHook() ? "SET" : "EMPTY");
    AdaptiveDiag::print("H1", m.str());
}

HookResult invoke(
    const State&                                              state,
    Teuchos::RCP<Problem>                                     problem,
    Teuchos::RCP<const Thyra::BlockedLinearOpBase<SC>>        blocked_op,
    Teuchos::RCP<const Teuchos::ParameterList>                params,
    const SolveMetrics&                                       metrics)
{
    auto& h = globalHook();
    {
        std::ostringstream k;
        k << "invoke: slot=" << &h << " hook " << (h ? "SET, calling it" : "EMPTY, returning default")
          << " converged=" << metrics.converged;
        std::ostringstream m;
        m << k.str() << " (iters=" << metrics.num_iters << " curDim=" << state.curDim << ")";
        AdaptiveDiag::print("H2", m.str(), k.str());
    }
    if (h) return h(state, problem, blocked_op, params, metrics);
    return HookResult{};
}

} // namespace AdaptiveHook
} // namespace Belos
