# Plan: port the static-link and Tpetra-route fixes from the diag branch to the feature branch

**Status: built 2026-10-19.** One clean commit on `teko-reconfig-request`, pushed. Diag branch left as is. Built as planned except as noted under "Changes from the draft".

`aidantgould/teko-reconfig-diag` is `aidantgould/teko-reconfig-request` (at
`ebc3f035246`) plus two commits:

- `aba2830a3bf`: `[TekoDiag]` prints on every check. Diagnostic only.
- `33a184324d9`: the two fixes (`tpetra_hook_route_plan.md`), with their own
  `[TekoDiag]` prints woven through.

The fixes go to the feature branch; no print does. A cherry-pick would conflict
on every print's context, so the port is a hand-written commit on
`teko-reconfig-request` carrying the fix logic only.

## What is carried

| File | Carried | Left behind |
| --- | --- | --- |
| `belos/src/BelosAdaptiveHook.hpp` | `ErasedArgs`, `ErasedHookFn`, `registerErasedHook`, `invokeErased`, `<typeinfo>` | `slotAddress`, `hookIsSet`, `erasedSlotAddress`, `erasedHookIsSet` |
| `belos/src/BelosAdaptiveHook.cpp` | `globalErasedHook`, `registerErasedHook`, `invokeErased` | `AnnounceLoad`, every print |
| `belos/src/BelosBlockGmresSolMgr.hpp` | the converged-path and stalled-path `else if constexpr` erased call sites | B1 to B5 prints |
| `teko/src/Teko_InverseFactory.cpp` | the four `ensureAdaptiveHookRegistered` calls and the include | the `kBranch` rename |
| `teko/src/Teko_KrylovSurrogateInit.cpp` | `ensureAdaptiveHookRegistered` (call_once, both registrations), the static object calling it, the header comment | R1 print |
| `teko/src/Teko_KrylovSurrogateRegistration.hpp` | whole file (new) | |
| `teko/src/Teko_KrylovTpetraRoute.hpp` | whole file minus prints: type check, `findBlockSource` (operator, then rightPrec, then leftPrec), conversion, copy-back | K1 to K4 prints, `BelosAdaptiveDiag.hpp` include, the "Diagnostic branch" comment line |
| `teko-reconfig/claudes_world/plans/tpetra_hook_route_plan.md` | as the record of the fix, with a note that the port dropped the prints | |

Not carried: `BelosAdaptiveDiag.hpp`, its CMakeLists line, the PseudoBlock,
Thyra-factory, `Thyra_BelosLinearOpWithSolve_def.hpp` and `Tpetra_Core.cpp`
prints, `hook_diag_branch_plan.md`, `tests/hook_diag/`.

The silent rejections in `tryBlockSource` / `findBlockSource` stay silent: the
feature branch reports only through `[TekoAdaptive]` lines, and the hook
already prints its gate line once reached.

## ACTIVATION.md

- "Activates only when ALL hold": the type list gains the Tpetra-typed route
  (`Tpetra::MultiVector` / `Tpetra::Operator`, the operator or a Teko
  `InverseFactoryOperator` preconditioner wrapping a blocked Teko operator),
  and the link bullet says registration also runs from `buildInverse`, so a
  static `libteko.a` registers too.
- "Where the implementation lives" gains the two new headers.
- Line references fixed (`Teko_KrylovSurrogate.hpp` gate, `Teko_InverseFactory.cpp`
  warm-up).

## Test

`teko-reconfig/claudes_world/tests/tpetra_route/`: a print-free cut of
`hook_diag.cpp` (the Tpetra driver plus the Thyra blocked case), its
`CMakeLists.txt`, `.gitignore` for `build/`, and `test_tpetra_route.py` in the
style of the other tests (pytest or run directly). It observes through the
`[TekoAdaptive] gate:` line, `TRUERES`, `SOLVE` and the conv.json files:

1. `Teko_InverseFactory.cpp.o` references `ensureAdaptiveHookRegistered` (`nm`).
2. A static `libteko.a` link pulls in the registration object (link map), and
   the run reaches the hook (gate line says inert).
3. Thyra-typed flexible solve reaches the hook (gate line).
4. Tpetra-typed flexible solve reaches the hook (gate line).
5. Tpetra-typed, gate on, watcher answering, first solve stalled by a low
   `Maximum Iterations`: the run reports converged with `TRUERES` under
   tolerance, which only a re-solve copied into the application's Tpetra
   vector can produce.

Plus the existing `test_reconfig_protocol.py`, `test_surrogate_search.py` and
`warmup_nested/` on the rebuilt tree.

## Build

`trilinos-build` in place, 12 jobs. Switching branches changes
`BelosBlockGmresSolMgr.hpp`, so much of Belos, Stratimikos and Teko recompiles.
`teko_ext` rebuilt against it.

## Non-goals

- No change to the diag branch.
- No new diagnostics on the feature branch.
- No rebase or other change to the diag branch afterwards.

## Changes from the draft

- **A bug the new test found, fixed in its own commit before the port.** With
  the gate on, a Thyra-typed solve through Stratimikos aborted in
  `computeCHat`: `getBlockTpetraMV: block 0 is not a TpetraMultiVector`. A
  Stratimikos application solving one right-hand side hands Belos
  `Thyra::createMember` vectors, whose blocks are `Thyra::TpetraVector`, and
  that class does not derive from `Thyra::TpetraMultiVector`. The diag branch
  never ran this path with the gate on, and `teko_ext` and the Tpetra route
  both build multivectors, so nothing had hit it. `constTpetraOfBlock` /
  `tpetraOfBlock` in `Teko_KrylovReducedModel.hpp` now accept either, and
  `getBlockTpetraMV`, `copyOriginalToGrouped` and `copyGroupedToOriginal` use
  them.
- Test 5 runs on both routes, and says which x the application holds by
  matching the driver's `TRUERES` against conv.json's residuals (relative
  1e-6), with solve 1 stalled at `STALL_ITERS = 2`. Solve 1 lands near 7e-6,
  the use_ordering final near 5e-10.
- `ensureAdaptiveHookRegistered` keeps its `caller` argument, unused on this
  branch, so the declaration matches the diag branch's.

## Results

- `tpetra_route/test_tpetra_route.py`: 4 of 4.
- `warmup_nested/`: 5 of 5. `test_surrogate_search.py` and
  `test_reconfig_protocol.py`: 25 of 25.
- `teko_ext` rebuilt (12 jobs, not `build.sh`'s `nproc`), since it
  instantiates the changed `BlockGmresSolMgr`.
