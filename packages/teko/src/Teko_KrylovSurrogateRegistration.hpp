// Teko_KrylovSurrogateRegistration.hpp
//
// Registers Teko's adaptive hooks with Belos::AdaptiveHook: the Thyra one
// (adaptiveLoop) and the type-erased one for Tpetra-typed solves
// (tpetraAdaptiveLoop). Idempotent.
//
// Defined in Teko_KrylovSurrogateInit.cpp and called from the free
// Teko::buildInverse / rebuildInverse functions as well as from that file's
// static constructor. The call from buildInverse is the one that matters under
// a static link: the linker only pulls in object files something references,
// and nothing referenced the registration object until this function, so its
// constructor was dropped and the hook never registered. Kept free of heavy
// includes so Teko_InverseFactory.cpp can include it cheaply.
#pragma once

namespace Teko {
namespace KrylovSurrogate {

void ensureAdaptiveHookRegistered(const char* caller);

}  // namespace KrylovSurrogate
}  // namespace Teko
