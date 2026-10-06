// Teko_KrylovSurrogateInit.cpp
//
// Registers Teko::KrylovSurrogate::adaptiveLoop as the global adaptive hook
// with Belos::AdaptiveHook.  The registration happens via a file-scope
// static object whose constructor runs before main().
//
// Because Teko is a shared library (libteko.so) all translation units are
// linked, so this constructor always executes when teko_ext.so loads Teko.

#include "Teko_KrylovSurrogate.hpp"
#include "BelosAdaptiveDiag.hpp"

#include <sstream>

namespace {

struct TekoAdaptiveRegistration {
    TekoAdaptiveRegistration() {
        // Diagnostic: if this line is missing, this object file was never
        // linked in (a static link drops it, since nothing references it).
        std::ostringstream m;
        m << "Teko registration ctor ran: slot=" << Belos::AdaptiveHook::slotAddress()
          << " before=" << (Belos::AdaptiveHook::hookIsSet() ? "SET" : "EMPTY");
        Belos::AdaptiveDiag::print("R1", m.str());
        Belos::AdaptiveHook::registerHook(&Teko::KrylovSurrogate::adaptiveLoop);
    }
};

// One static instance — constructor fires at dynamic-library load time.
static TekoAdaptiveRegistration g_tekoAdaptiveRegistration;

} // anonymous namespace
