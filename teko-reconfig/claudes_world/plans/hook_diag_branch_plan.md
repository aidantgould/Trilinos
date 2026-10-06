# Plan: a diagnostic branch that says why the hook did not fire

**Status: built 2026-10-06.** Decisions: build in place in `trilinos-build` (12 jobs), test with a small Stratimikos app plus `demo.py`, commit and push. Built as planned except as noted under "Changes from the draft".

An external app built against this fork prints no `[TekoAdaptive]` line at
all, so `adaptiveLoop` is never entered. Every check between the app and the
gate is silent, so the output cannot tell the suspects apart: hook never
registered (static link drops `Teko_KrylovSurrogateInit.o`, or two copies of
Belos's slot), the solve never reaching `BlockGmresSolMgr`, flexible off (or
silently switched off by `checkStatusTest` when there is no right
preconditioner), the operator Belos holds not being a
`Thyra::BlockedLinearOpBase`, or the call site compiled out (non-Thyra types,
no `HAVE_BELOS_THYRA`, or stock Belos headers).

Branch `aidantgould/teko-reconfig-diag` off `aidantgould/teko-reconfig-request`
at `ebc3f035246`. Diagnostic only: no control flow changes.

## Print format

Every line is

```
[TekoDiag <ID>] pid=<pid> <message>
```

written to stdout and stderr, each flushed, the same as the branch tracer.
IDs are unique per call site so one `grep TekoDiag` reconstructs the path.
Lines are rate-limited per (ID, key): each distinct key prints at most 3 times
per process, then one `... further <ID> lines like this suppressed: <key>`
line. The key is the message, minus iteration counts where it carries them
(`B2`, `H2`), so inner solves cannot swamp a run or use up the budget of the
outer solve whose checks come out differently.

Helper, header-only, no Thyra dependency so every solver manager can include
it: `packages/belos/src/BelosAdaptiveDiag.hpp`,
`Belos::AdaptiveDiag::print(const char* id, const std::string& msg)`.

## Call sites

| ID | Where | Says |
|---|---|---|
| `L1` | `BelosAdaptiveHook.cpp`, static object | this libbelos copy loaded, its slot address |
| `R1` | `Teko_KrylovSurrogateInit.cpp`, registration ctor | registration ran, slot address before and after |
| `H1` | `registerHook` | slot address filled |
| `H2` | `invoke` | slot address, hook SET or EMPTY |
| `B1` | `BlockGmresSolMgr::solve` entry, after `checkStatusTest` | label, SC/MV/OP type names, `HAVE_BELOS_THYRA`, Thyra types matched (else "hook compiled out of this instantiation"), `Flexible Gmres` requested vs effective, left/right prec null, number of RHS, operator description, operator blocked |
| `B2` | converged call site | `isFlexible_`, `isConverged`, fgmres cast null, operator blocked, invoking or which check failed |
| `B3` | stalled call site | same as `B2`, for the non-converged path |
| `P1` | `PseudoBlockGmresSolMgr::solve` entry | entered, types, operator description when Thyra; this solver has no hook |
| `S1` | Stratimikos `BelosLinearOpWithSolveFactory`, after operator and prec are set | solver type, operator description, blocked or not, prec side chosen |
| `S2` | Stratimikos `BelosLinearOpWithSolve`, before `iterativeSolver_->solve()` | solver manager description (includes Flexible for Block GMRES) |
| `T1` | `adaptiveLoop` recursion-guard return | re-entry from the hook's own re-solve, expected |

How to read the output:

- No `L1`: the fork's libbelos is not loaded. Two `L1` with different slots: two copies.
- `L1` but no `R1`: Teko's registration object was dropped.
- `R1` slot differs from `H2` slot: registration filled a different copy.
- `H2 ... EMPTY`: invoke ran with no hook.
- Only `P1` and `S2` naming Pseudo Block GMRES: wrong outer solver.
- `B1` and `B2`/`B3` naming the failed check: flexible off or operator not blocked.
- `S2` shows Block GMRES but no `B1`: the app's `BlockGmresSolMgr` came from other headers.

## Tests

Location `teko-reconfig/claudes_world/tests/hook_diag/`, in the style of
`warmup_nested` (small C++ binary built against `trilinos-build`, pytest
driver running it per case in a fresh process):

`hook_diag <pseudo|flex> <flat|blocked> [inner-gmres]`, Stratimikos with a
Teko block Gauss-Seidel preconditioner, `TEKO_ADAPTIVE_RECONFIG` unset so a
reached hook prints its gate as inert and needs no watcher.
`test_hook_diag.py`, 6 tests, all passing:

1. Every line appears identically on stdout and stderr.
2. One `L1`; `R1` ran with the slot empty before; `H1` filled it.
3. flex + blocked reaches the hook: `B2 -> calling invoke`, `H2 ... SET`,
   one slot address across `L1`, `R1`, `H1`, `H2`, and the gate line.
4. flex + flat names the cause: `S1`, `B1`, `B2` all say `opBlocked=no`, no
   `H2`. (Teko's `Strided Blocking` splits the operator inside the
   preconditioner, so Belos holds a flat `TpetraLinearOp`.)
5. flex + blocked + inner-gmres: inner non-flexible Block GMRES block solves
   print their own `B1`/`B2`, and the outer solve's `B2 -> calling invoke`
   still comes through.
6. pseudo, either layout: `S1` names Pseudo Block GMRES, `P1`, no `B1`, no `H2`.

Also run by hand: `demo.py` with the hook live (outer `B2 -> calling invoke`,
gate ACTIVE five times, `T1` on each re-solve), `test_warmup_nested.py` (5
passed), `test_reconfig_protocol.py` (2 passed).

## Changes from the draft

- `B9` folded into `B1`, which reports `HAVE_BELOS_THYRA` and whether the
  instantiation's types keep the call site.
- Rate limit per (ID, key) instead of per ID. The first version capped each ID
  at 5, and in `demo.py` the inner Block GMRES solves inside the hook's own
  block inverses used up `B2`'s budget before the outer solve reached its call
  site, hiding exactly the line that matters.
- `S1` first read `Solver Type` as a string and threw, since validation stores
  it as an enum. It now reads it through the validator inside a try, so no
  diagnostic can throw.
- `BelosAdaptiveDiag.hpp` is C++17 (detection idiom, no `requires`), since an
  application compiles it with its own standard.

## Non-goals

- No fix for any of the causes; this branch only names them.
- No env-var gate on the prints, so the diagnostic cannot itself fail to
  propagate.
- No rank in the prefix; pid separates processes and avoids pulling MPI into
  Belos headers.
