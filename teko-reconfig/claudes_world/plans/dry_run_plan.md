# Plan: TEKO_RECONFIG_DRY_RUN, run the reconfig but always report solve 2 as failed

**Status: built 2026-10-19** (name `TEKO_RECONFIG_DRY_RUN`, commit and push). Built as planned except as noted under "Changes from the draft". Lands on `aidantgould/teko-reconfig-request`
after `diag_port_plan.md`.

## Purpose

Collect data on what the reconfiguration would have done without letting it
change the application's result. Everything runs as now (surrogate, watcher,
tests, selection, final solve) and conv.json records it, but the application
gets solve 1's solution, status, iteration count and tolerance unchanged.

## Interface

Env var **`TEKO_RECONFIG_DRY_RUN`**, same truthy rule as
`TEKO_ADAPTIVE_RECONFIG` (unset, empty, `0`, `false`, `FALSE` are off). Read by
a predicate `dryRunEnabled()` beside `adaptiveEnabled()` in
`Teko_KrylovReconfigPrec.hpp`. Only meaningful with the gate on.

## Changes in `adaptiveLoop` (`Teko_KrylovSurrogate.hpp`)

1. Phase 4's final solve passes `writeToLHS = !dryRun`. Without this the
   Thyra route would overwrite the application's LHS on a converged final
   while Belos reports solve 1's status.
2. The returned `HookResult` is `{}` when the flag is on, so
   `BlockGmresSolMgr` keeps its own result, and the Tpetra route skips its
   copy-back (it copies only on `r.converged`).
3. The rank-0 message names what happened: final converged (or not), and
   "not applied (TEKO_RECONFIG_DRY_RUN)".

conv.json's final row keeps its real `converged`, `iters` and residual: that
is the would-be behaviour being collected.

## conv.json

A new top-level field, `"applied"`, after `request_id`: true exactly when the
application received the final solve's solution (final converged and the flag
off), false otherwise, including the no-solves early return. Written on every
run, so a file says by itself whether its final row reached the application.
`writeConvergenceJson` gains the argument.

Readers: `postprocess.py` ignores unknown fields; `test_reconfig_protocol.py`
asserts the top-level keys exactly and is updated to the new list.

## ACTIVATION.md

The env-var table gains the flag; a short section under "What C++ does with
the rows" says what it keeps and what it skips, and documents `applied`.

## Tests

- `test_reconfig_protocol.py`: `applied` is present, and true for every
  request in the shipped demo (all finals converge there).
- `tpetra_route/test_tpetra_route.py`, flag on, first solve stalled:
  - Tpetra route: `SOLVE` reports unconverged, `TRUERES` equals the conv.json
    initial row's `final_residual` (to 1e-10 relative), the final row is
    converged, `applied` is false.
  - Thyra route (Stratimikos blocked case): the same.
- Flag off on the same stalled case: rescued, `applied` true (the port's test 5).

## Non-goals

- No observe mode that skips the final solve (the costlier but schema-free
  choice was taken: in `best_time` mode the final duplicates a test).
- No watcher-side switch; the watcher is unchanged.
- No per-step control.

## Changes from the draft

- The truthy rule is now one function, `envFlagOn(name)`, which
  `adaptiveEnabled()` and `dryRunEnabled()` both call.
- The hook announces a dry run on both streams right after the gate line, and
  the rank-0 final message says "not applied (TEKO_RECONFIG_DRY_RUN). Using
  first result."
- `test_reconfig_protocol.py` checks `applied == final["converged"]` rather
  than `applied` being true everywhere, so it does not depend on every demo
  final converging.
- Added at the user's request mid-build: a sample run command at the top of
  `wait_for_request.py`'s docstring. It starts the watcher with
  `TEKO_WATCHER_EMIT_TEST_ORDERINGS=1` and runs the application with the gate
  and `TEKO_RECONFIG_DRY_RUN` on. Run as written against
  `route_driver tpetra 2`, it gave `applied: false`, the two test orderings
  and a converged would-be final in conv.json, and solve 1's stall to the
  application. The stale `RUN_SECONDS` sentence in that docstring now
  describes the real stop conditions.

## Results

- `tpetra_route/test_tpetra_route.py`: 6 of 6. The two dry-run tests cover both
  routes, stalled and converged.
- `warmup_nested/`: 5 of 5. `test_surrogate_search.py` and
  `test_reconfig_protocol.py`: 25 of 25.
