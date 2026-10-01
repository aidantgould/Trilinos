# Plan: gate the per-block solver on the adaptive flag

**Built 2026-09-30.** Implemented as planned, all three decisions as
recommended. Verified by the harness below: with the flag off the gated build
lands exactly on the old Ifpack2 timing (ratio 0.99x, 1.00x, 1.00x at n = 200,
400, 800), the pre-gate behavior is 2.2x to 2.8x, and the hook fires on exactly
the flag=on cells and no others (0 gate/hook disagreements). Two additions
beyond the draft: an `always_gmres` arm, so the regression being fixed is
measured rather than remembered, and a separate timeout budget for flagged
cells, since those run the whole adaptive loop and the watcher's
541-ordering search, which is slow for reasons that are not a hang.

`teko_ext.cpp:225` builds the main solve's block inverses with the hook's shared
factory (inner Belos GMRES to 1e-4 preconditioned by RILUK ILU(0)) whatever
`TEKO_ADAPTIVE_RECONFIG` says. Measured 2026-09-30 in
`pyautoteko/claudes_world/adaptive_flag_ab_experiment`: that costs 2.2x to 2.7x
on the solve, and a third arm with the inner GMRES capped at ONE iteration was
exactly as slow, so the expense is routing every block inverse through a Belos
solve manager at all, not the inner iteration count. Tuning `kBlockSolveTol` or
`kBlockSolveMaxIter` therefore cannot recover it. Only not using a solver can.

Goal: an unflagged run costs what it used to, a flagged run is unchanged.

## 1. One shared predicate (Trilinos)

The truthiness test currently lives inline in `adaptiveLoop`'s gate. Lift it, so
the front end and the hook cannot drift on what "the hook is on" means:

```cpp
// Teko_KrylovReconfigPrec.hpp, next to makeBlockSolverInverseFactory
inline bool adaptiveEnabled()          // TEKO_ADAPTIVE_RECONFIG, "" / 0 / false / FALSE are off
```

`Teko_KrylovSurrogate.hpp`'s Phase 0 gate becomes `if (!adaptiveEnabled()) return {};`
with its existing comment kept. `Teko_KrylovReconfigPrec.hpp` is the home because
it is the header `teko_ext.cpp` already includes for the factory, so the front end
does not have to pull in the hook itself.

## 2. Pick the factory on it (pyfront)

```cpp
// teko_ext.cpp, replacing the unconditional call at line 225
static Teuchos::RCP<Teko::InverseFactory> makeBlockInverseFactory();
```

- **Hook live**: `Teko::KrylovSurrogate::makeBlockSolverInverseFactory()`, exactly
  as now, so solve 1 and solve 2 still invert their blocks identically, which is
  the property that factory was shared for.
- **Hook off**: the previous Stratimikos/Ifpack2 factory, `Linear Solver Type`
  Belos plus `Preconditioner Type` Ifpack2 through `InverseLibrary`, which gives
  Ifpack2's default ILUT as a single apply per block. Decided 2026-09-30 over
  applying RILUK ILU(0) directly: restoring exactly the old factory is known-good
  and keeps unflagged timings comparable with everything before this branch,
  where the RILUK route would need a path that honors preconditioner settings
  (`buildFromStratimikos` discards them) and might not drop in cleanly.

Plus one diagnostic line, rank 0, once per process, naming which of the two was
chosen and why, so "which path am I on?" never costs an afternoon again:

    [Teko] block inverse: Ifpack2 ILUT, single apply (TEKO_ADAPTIVE_RECONFIG off)
    [Teko] block inverse: Belos GMRES 1e-4 + RILUK ILU(0) (adaptive hook live)

Everything else stays, the explicit true-residual check included: it is
orthogonal to the factory and it is what caught the false convergence in demo 2.

## 3. Tests: extend the existing harness

`pyautoteko/claudes_world/adaptive_flag_ab_experiment` already builds arms and
times them. Two changes make it verify the gate rather than just measure it:

- `solve_once.py` gains `--flag {off,on}`. With `on` it leaves
  `TEKO_ADAPTIVE_RECONFIG=1` set and uses the `pyteko` driver, which spawns the
  watcher so the handshake is answered. It already reports `hook_fired` from new
  `s<N>_request.json` files.
- `run_ab.py` arms become `ifpack2` (patched to the old factory) and `gated` (the
  committed source), run as three cells: `ifpack2`+off, `gated`+off, `gated`+on.

Two assertions, both falsifiable:

1. **The gate agrees with the hook.** `hook_fired` must equal `flag == on` for
   every cell. A `gated`+off run that fires means the two predicates disagree,
   which is the bug this refactor exists to prevent.
2. **The speed is back.** `gated`+off must land on the `ifpack2` timing, and
   `gated`+on must stay at the ~2.5x it is now.

## Non-goals

- No change to what a flagged run does. Solve 2 keeps its hardcoded `method =
  "gs"` and its RILUK inner GMRES; that is the experiment's choice, not a bug to
  fix here.
- No attempt to make the unflagged factorization match the flagged one (ILUT
  against ILU(0)). The unflagged path is restored to exactly what it was, so its
  numbers are comparable with the history rather than with the flagged path.
- No new knob on `pyTeko()`; the env var is the only control.
- No change to the hook's own behavior, the watcher, or the surrogate search.
