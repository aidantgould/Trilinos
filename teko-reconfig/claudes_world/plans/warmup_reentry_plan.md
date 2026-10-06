# Plan: make the factorization warm-up safe under nested Teko preconditioners

**Status: built 2026-10-06**, as planned (outermost-only timing and the full
env-matrix test, both decided as recommended). Verified:

- The repro hangs on the unfixed `Teko_InverseFactory.cpp` (HEAD's copy,
  rebuilt, both gates unset: killed by a 20 s timeout right after the
  `Teko::buildInverse(factory,A)` tracer line, the application's symptom), and
  completes after the fix.
- `test_warmup_nested.py`: 5 of 5 pass. `test_surrogate_search.py`: 19 of 19.
- pyfront `demo.py`: all PASS lines, the singular-block case raises as
  designed, and `s<N>_conv.json` still records nonzero factor times for both
  solves. `teko_ext` needed no rebuild, since only libteko's `.cpp` changed.

One addition beyond the draft: `ACTIVATION.md` gained a paragraph on the
warm-up's gate and the outermost-only rule, next to the other things the flag
controls. `warmup_nested/.gitignore` keeps its `build/` out of git.

## The bug

`maybeWarmupFactor` (`packages/teko/src/Teko_InverseFactory.cpp`) runs inside
the free function `Teko::buildInverse(factory, A)` and guards its one untimed
factorization with a process-wide `std::call_once`. When `factory` is a block
preconditioner (Gauss-Seidel, SIMPLE, LSC, and so on), building it calls
`Teko::buildInverse` again on each diagonal block
(`Teko_BlockInvDiagonalStrategy.cpp:91`). That nested call re-enters the same
`call_once` on the same thread while the outer one is still running, and it
waits forever. A 3-line recursive `call_once` reproduces the hang under the
system libstdc++.

pyTeko never hits this because it calls `buildInverse` on leaf solvers only.
An application that reaches Teko through Stratimikos with an XML block
preconditioner hits it on its first solve. Since e591175623c (2026-06-16) the
warm-up is on by default, so `TEKO_ADAPTIVE_RECONFIG` being unset does not
help: the branch is not inert to such an application.

## Changes

All in `Teko_InverseFactory.cpp`, anonymous namespace, plus docs.

1. **Re-entry guard.** A `thread_local int t_buildDepth` and an RAII
   `BuildDepthGuard` that increments it for its lifetime. Every free
   `buildInverse` / `rebuildInverse` overload opens with

   ```cpp
   const bool outermost = (t_buildDepth == 0);
   BuildDepthGuard depth;
   ```

   and `buildInverse(factory, A)` calls `maybeWarmupFactor` only when
   `outermost`. The warm-up's own `factory.buildInverse(A)` then runs at depth
   1, so the nested calls it makes see a non-zero depth and skip the warm-up.

2. **Outermost-only timing.** `FactorStopwatch` takes the `outermost` flag and
   records nothing when it is false. Today a nested build is timed twice, once
   by the inner stopwatch and once inside the outer one's span, and the nested
   calls made *during* the warm-up would add warm-up time to the registry,
   which is the cost the warm-up exists to keep out.

3. **Default gate.** `TEKO_FACTOR_WARMUP` set: its own truthiness, as now
   (empty, `0`, `false`, `FALSE` mean off). Unset: follows
   `TEKO_ADAPTIVE_RECONFIG`, parsed with the same rule as `adaptiveEnabled()`.
   The parse is repeated locally (a two-line `envTruthy` helper) rather than
   including `Teko_KrylovReconfigPrec.hpp` into this translation unit.

4. **Docs.** `ACTIVATION.md`'s knob table: default becomes "follows
   `TEKO_ADAPTIVE_RECONFIG`", with the line number refreshed. pyfront's
   `python_front_end_interface.py` comment ("on by default in libteko") is
   corrected. Its behavior is unchanged, since it sets both variables.

## Tests

`teko-reconfig/claudes_world/tests/warmup_nested/`:

- `warmup_nested.cpp`: a 2x2 blocked Tpetra operator with tridiagonal
  diagonal blocks. The top-level factory is a counting `InverseFactory` that
  wraps a Teko block Gauss-Seidel factory over Ifpack2 sub-inverses (from
  `InverseLibrary`), so a nested `Teko::buildInverse` really happens. It calls
  `Teko::buildInverse(counting, A)` once and prints how many times the counting
  factory was asked to build: 2 with the warm-up, 1 without.
- `CMakeLists.txt`: links `Teko::teko` from `trilinos-build/cmake_packages`,
  the way pyfront's `CMakeLists.txt` does.
- `test_warmup_nested.py`: runs the binary in a fresh process per environment
  combination (the gates are read once per process), each under a timeout so a
  regression shows up as a failure rather than a hung test:

  | `TEKO_ADAPTIVE_RECONFIG` | `TEKO_FACTOR_WARMUP` | expected builds |
  | --- | --- | --- |
  | unset | unset | 1 |
  | `1` | unset | 2 |
  | `1` | `0` | 1 |
  | unset | `1` | 2 |
  | `0` | unset | 1 |

  Same style as `test_surrogate_search.py`: plain functions, pytest or a
  `__main__` runner.

Regression on the existing path: rebuild `libteko` and `teko_ext`, then run
pyfront's `demo.py`, which should behave as before.

Build within the compute share: `-j12` at most during 10:00 to 22:00.

## Non-goals

- No change to which factorization is warmed up (still the first outermost
  `buildInverse(factory, A)` in the process) or to the
  `buildInverse(factory, A, precOp)` overload not warming up.
- No fix for the static-link hook registration in
  `Teko_KrylovSurrogateInit.cpp`.
- No commit, unless asked.
