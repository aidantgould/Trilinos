// Teko_KrylovSurrogateInit.cpp
//
// Registers Teko's adaptive hooks with Belos::AdaptiveHook:
//   adaptiveLoop        for Thyra-typed solves (registerHook), and
//   tpetraAdaptiveLoop  for Tpetra-typed solves (registerErasedHook).
//
// Two entry points reach ensureAdaptiveHookRegistered(): the file-scope static
// object below, whose constructor runs when this object file is loaded, and
// the free Teko::buildInverse / rebuildInverse functions. The second exists for
// static links. A shared libteko.so loads every object file, so the
// constructor always runs; a static libteko.a contributes only object files
// the application references, and nothing referenced this one, so the
// constructor was silently dropped and no hook was ever registered. The call
// from buildInverse is such a reference.

#include "Teko_KrylovSurrogateRegistration.hpp"
#include "Teko_KrylovTpetraRoute.hpp"
#include "BelosAdaptiveDiag.hpp"

#include <mutex>
#include <sstream>

namespace Teko {
namespace KrylovSurrogate {

void ensureAdaptiveHookRegistered(const char* caller)
{
    static std::once_flag once;
    std::call_once(once, [caller] {
        // Diagnostic: R1 names the path that registered first. If it is
        // missing, neither the static constructor nor buildInverse ran here.
        std::ostringstream m;
        m << "Teko registration ran via " << caller
          << ": slot=" << Belos::AdaptiveHook::slotAddress()
          << " before=" << (Belos::AdaptiveHook::hookIsSet() ? "SET" : "EMPTY")
          << " erasedSlot=" << Belos::AdaptiveHook::erasedSlotAddress()
          << " before=" << (Belos::AdaptiveHook::erasedHookIsSet() ? "SET" : "EMPTY");
        Belos::AdaptiveDiag::print("R1", m.str());
        Belos::AdaptiveHook::registerHook(&adaptiveLoop);
        Belos::AdaptiveHook::registerErasedHook(&tpetraAdaptiveLoop);
    });
}

}  // namespace KrylovSurrogate
}  // namespace Teko

namespace {

struct TekoAdaptiveRegistration {
    TekoAdaptiveRegistration() {
        Teko::KrylovSurrogate::ensureAdaptiveHookRegistered("static initializer");
    }
};

// One static instance: its constructor fires when this object file is loaded,
// which under a static link is only if something references the file (see the
// header comment).
static TekoAdaptiveRegistration g_tekoAdaptiveRegistration;

}  // anonymous namespace
