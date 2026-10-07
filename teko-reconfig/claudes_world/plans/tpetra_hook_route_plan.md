# Plan: register under a static link, and reach the hook from a Tpetra-typed solve

**Status: built 2026-10-06.** Decisions: commit on `aidantgould/teko-reconfig-diag` (port the fixes to `teko-reconfig-request` later, without the prints), test the full loop with the watcher, resolver tries all three candidates.

The diag branch (`hook_diag_branch_plan.md`) located two blockers in the
external app:

1. **No `R1`.** `Teko_KrylovSurrogateInit.o` never runs its registration
   constructor. The app links Teko statically and nothing references that
   object, so the linker drops it.
2. **`B1 ... thyraTypes=no`, MV `Tpetra::MultiVector`, OP `Tpetra::Operator`.**
   The app calls Belos with Tpetra types, flexible requested. The hook's call
   sites exist only in the `double`/Thyra instantiation, so this solve has none.

## Fix 1: registration that a static link cannot drop

`Teko_KrylovSurrogateInit.cpp` gains an exported function, declared in the
light header `Teko_KrylovSurrogateRegistration.hpp`,

```cpp
namespace Teko { namespace KrylovSurrogate {
void ensureAdaptiveHookRegistered(const char* caller);   // idempotent, std::call_once
}}
```

which registers both hooks (the Thyra one as today, and the Tpetra one from fix
2). The static constructor calls it, as before. New: the free
`Teko::buildInverse` / `rebuildInverse` overloads in `Teko_InverseFactory.cpp`
call it too, next to the branch tracer. The app is known to reach
`buildInverse` (the tracer printed), so that reference pulls
`Teko_KrylovSurrogateInit.o` into a static link and registration runs before
the first solve, whichever path gets there first.

## Fix 2: a Tpetra route into the hook

Belos cannot name Teko types, and Belos core does not name Tpetra types either,
so the second slot is type-erased:

```cpp
// BelosAdaptiveHook.hpp
struct ErasedArgs {
    const std::type_info& mv;      // typeid(MV) of the solve
    const std::type_info& op;      // typeid(OP)
    const void* state;             // const GmresIterationState<double, MV>*
    void*       problem;           // Teuchos::RCP<LinearProblem<double, MV, OP>>*
    Teuchos::RCP<const Teuchos::ParameterList> params;
    SolveMetrics metrics;
};
using ErasedHookFn = std::function<HookResult(const ErasedArgs&)>;
void       registerErasedHook(ErasedHookFn fn);
HookResult invokeErased(const ErasedArgs& args);
```

`BlockGmresSolMgr::solve` gets an `else` branch on its existing type check:
for any `double` instantiation that is not Thyra, at both call sites (converged
and stalled), it calls `invokeErased` under the same conditions as the Thyra
site minus the blocked-operator cast (`isFlexible_`, the right path, an
`BlockFGmresIter`). The block structure is Teko's to find.

Teko registers `Teko::KrylovSurrogate::tpetraAdaptiveLoop(const ErasedArgs&)`,
in a new header `Teko_KrylovTpetraRoute.hpp`:

1. Return unless `mv`/`op` are exactly `Tpetra::MultiVector<double,int,long
   long,Node>` / `Tpetra::Operator<...>` (the hook's aliases).
2. Find a `TpetraHelpers::TpetraOperatorWrapper` whose `getThyraOp()` is a
   `Thyra::BlockedLinearOpBase`, trying in order:
   a. the problem's operator itself (a `BlockedTpetraOperator` or
      `StridedTpetraOperator` handed to Belos);
   b. the right preconditioner as a `TpetraHelpers::InverseFactoryOperator`,
      then its `getForwardOp()` (the usual case: Belos holds the flat
      `CrsMatrix`, Teko blocked a wrapper of it to build the preconditioner);
   c. the same for the left preconditioner.
3. With that wrapper's blocked Thyra operator and mapping strategy, copy the
   first `curDim` columns of `state.V` and `state.Z`, plus `b` and `x`, into
   blocked Thyra multivectors (`copyTpetraIntoThyra`), build a Thyra
   `LinearProblem` and `GmresIterationState`, and call the existing
   `adaptiveLoop` unchanged.
4. If the re-solve converged, copy its `x` back into the app's Tpetra LHS
   (`copyThyraIntoTpetra`) and return its `HookResult`, so `BlockGmresSolMgr`
   reports the rescued outcome exactly as on the Thyra path.

No change to `adaptiveLoop`, the watcher, or the JSON protocol.

## Prints, for when these two fixes are not enough

New IDs, same format and rate limit as the diag branch. `R1` now comes from
`ensureAdaptiveHookRegistered` and names the path that registered first
(`static initializer` or `Teko::buildInverse`) and both slots; `L1` names both
slots too.

| ID | Where | Says |
|---|---|---|
| `H3` | `registerErasedHook` | erased slot filled |
| `H4` | `invokeErased` | erased slot SET/EMPTY, MV/OP names |
| `B4` | Tpetra converged-path site | `isFlexible`, `isConverged`, fgmresIter, `-> calling invokeErased` or why not |
| `B5` | Tpetra stalled-path site | same for the stalled path |
| `K1` | `tpetraAdaptiveLoop` entry | type match or mismatch (expected vs got) |
| `K2` | each resolver candidate | which candidate, its dynamic type, blocked or not, accepted or rejected and why |
| `K3` | after resolving | block count, block sizes, each block's dynamic type (the surrogate needs `TpetraLinearOp` over `CrsMatrix`) |
| `K4` | after the hook returns | converged re-solve copied back into the app's LHS, or not |

Changed: `B1`, `P1`, `S1` add the right and left preconditioner's dynamic type
(demangled `typeid`), so the Teko-preconditioned solve labels itself among
unrelated ones. `B1` for a non-Thyra `double` solve now says
`erased hook path` instead of "compiled out". The per-key cap becomes
`TEKO_DIAG_MAX` (default 3).

## Tests

Extend `tests/hook_diag/`. Built as below; `test_hook_diag.py` is 11 tests, all
passing, and `demo.py`, `test_warmup_nested.py` (5) and
`test_reconfig_protocol.py` (2) still pass.

- `hook_diag tpetra-flex interleaved`: Belos `BlockGmresSolMgr` with Tpetra
  types and flexible, the operator the flat `CrsMatrix`, the right
  preconditioner a Teko `InverseFactoryOperator` built over a
  `BlockedTpetraOperator`. With the gate off: `B4 -> calling invokeErased`,
  `H4 SET`, `K2` accepting candidate b, `K3` two blocks, gate line inert.
- `tpetra-pseudo`: same with Pseudo Block GMRES, prints `P1` only.
- Full loop: the `tpetra-flex` case with `TEKO_ADAPTIVE_RECONFIG=1`, a temp
  requests dir, and `wait_for_request.py` started by the test. Expects
  `s0_conv.json` with a `solve2` entry and the app's returned `x` having a
  small true residual.
- Static-link guard: `nm` on `Teko_InverseFactory.cpp.o` shows
  `ensureAdaptiveHookRegistered` undefined.
- Real static link (added during the build, cheaper than expected): pack this
  build's Teko object files into a `libteko.a`, relink `hook_diag` against it
  with everything else shared, and check that `ldd` loads no `libteko`, that the
  link map shows `Teko_KrylovSurrogateInit.cpp.o` pulled in by
  `Teko_InverseFactory.cpp.o`'s reference to `ensureAdaptiveHookRegistered`,
  and that the run prints `R1` and reaches the hook.
- Rerun the existing suites: `test_hook_diag.py`, `demo.py`,
  `test_warmup_nested.py`, `test_reconfig_protocol.py`.

## Non-goals

- Tpetra types other than the hook's exact aliases (other `LO`/`GO`/`Node`).
- Block structure from anything other than Teko's Tpetra wrappers; an app
  whose blocks Teko never saw stays unsupported, and `K2` will say so.
- Pseudo Block GMRES (no call site, as before).
- A full static Trilinos build (the packed-archive test covers Teko, the one
  library whose linking was the problem).
