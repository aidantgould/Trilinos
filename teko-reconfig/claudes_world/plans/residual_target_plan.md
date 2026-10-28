# Plan: relative residuals in conv.json, and the surrogate told its target

**Status: built 2026-10-25, not committed.** Decisions: `initial_residual` stays absolute
and `initial_relative_residual` is added beside it, every row also records
`target_tol`, verdicts use the converted target, and `"User Provided"` sends
the raw tolerance.

## Why

Two defects found against a real application (a right-hand side with
||b|| around 1e-4):

1. `conv.json` mixes units. `initial_residual` is the absolute ||b||
   (`Teko_KrylovSurrogate.hpp`, `solveOrdering` and the solve 1 row), while
   `final_residual` is ||b - A x|| / ||b||. The demos hid it because their
   ||b|| is 2.4 to 16.6. A solve that went from 1 to 1e-8 reads as 1e-4 to 1e-8.
2. The surrogate's stopping tolerance is a constant, `SURROGATE_TOL = 1e-8` in
   `surrogate_search.py`, relative to ||b_hat||. The test solves it predicts
   inherit the application's Belos list (`orig_params`): its
   `"Convergence Tolerance"` and its `"Implicit Residual Scaling"`. Nothing in
   the request carries either, so whenever the application's target is not a
   1e-8 reduction from ||b||, the surrogate predicts a different solve from the
   one it is scored against.

The same conversion also exposes a third, smaller defect: the true-residual
check compares the relative residual against `kResidualSlack * tol` with the
raw tolerance, which is wrong under `"None"` scaling (absolute) and, for solve
1, under a warm start with initial-residual scaling.

## The one new helper (`Teko_KrylovReconfigPrec.hpp`, beside `trueRelativeResidual`)

```cpp
// The relative residual ||b - A x|| / ||b|| at which a Belos GMRES solve run
// with params stops, given ||b|| and ||r0|| = ||b - A x0||. Belos scales its
// implicit test by "Implicit Residual Scaling":
//   Norm of Initial Residual,
//   Norm of Preconditioned Initial Residual   tol * ||r0|| / ||b||
//   Norm of RHS                               tol
//   None                                      tol / ||b||
//   User Provided                             tol (the scale is unknown here)
// The preconditioned variant equals ||r0|| only without a left preconditioner,
// which holds for every solve the hook runs. ||b|| = 0 returns tol, matching
// trueRelativeResidual's absolute fallback.
inline double relativeTarget(const Teuchos::ParameterList& params,
                             double b_norm, double r0_norm);
```

Defaults when a key is absent: `Belos::DefaultSolverParameters::convTol` and
`BlockGmresSolMgr`'s own `"Norm of Preconditioned Initial Residual"`.

## Changes

**Hook (`Teko_KrylovSurrogate.hpp`)**

- `SolveRecord` (`Teko_KrylovReconfigIO.hpp`) gains
  `initial_relative_residual` (||r0|| / ||b||) and `target_tol`.
  `initial_residual` keeps its meaning, the absolute ||r0||.
- Solve 1: ||r0|| from `problem->getInitResVec()`, which Belos computed from the
  application's x0 at `setProblem`. Falls back to ||b|| (x0 = 0) when null.
  The solve 1 row records `initial_residual = ||r0||` (today it records ||b||,
  which differs only under a warm start), `initial_relative_residual =
  ||r0|| / ||b||` and
  `target_tol = relativeTarget(*orig_params, b_norm, r0_norm)`.
- Tpetra route (`Teko_KrylovTpetraRoute.hpp`): the Thyra `Problem` it builds is
  never `setProblem`'d, so it has no initial residual. The route converts the
  Tpetra problem's `getInitResVec()` and sets it with `setInitResVec`, so
  `adaptiveLoop` reads ||r0|| the same way on both routes.
- `solveOrdering`: x0 = 0, so `initial_residual` stays ||b||,
  `initial_relative_residual = 1` (0 when ||b|| = 0) and
  `target_tol = relativeTarget(*solverParams, b_norm, b_norm)`.
- Verdicts (solve 1 and `solveOrdering`): `true_res <= kResidualSlack *
  target_tol`, in place of the raw tolerance. Identical under the default
  scaling from x0 = 0, and fixes the false "not converged" under `"None"`.
- Request: `writeRequestJson(chat, requests_dir, target_tol)` writes
  `"target_tol"`, the tests' relative target. Computed once in `adaptiveLoop`.

**Watcher (`surrogate_search.py`, `wait_for_request.py`)**

- `surrogate_iters(C_hat, b_hat, jitter=..., tol=SURROGATE_TOL)`.
- `search()` reads `request.get("target_tol", SURROGATE_TOL)`, so recorded
  requests without the field behave as before.
- The watcher's request printout names the target.

**Docs (`ACTIVATION.md`)**: the request schema gains `target_tol`, the conv.json
sample and field notes add `initial_relative_residual` and `target_tol` and say
`initial_residual` is absolute, and the surrogate section says where its tolerance comes from.

## Tests

- `test_surrogate_search.py`: a request's `target_tol` reaches the surrogate
  (a looser target scores fewer or equal iterations on the same request, and
  equal to calling `surrogate_iters` with that tol), and a request without it
  scores exactly as `SURROGATE_TOL`.
- `test_reconfig_protocol.py` (pyfront demo, x0 = 0, default scaling): every
  row has `initial_relative_residual == 1`, `initial_residual` equal to
  ||b|| as before, and `target_tol` equal to the demo's tolerance. The request has
  `target_tol` equal to the demo's tolerance.
- `tpetra_route` (the real-application route): `route_driver` gains three
  environment knobs, `ROUTE_B_SCALE` (b is that constant), `ROUTE_X0` (x0 is
  that constant), `ROUTE_RES_SCALING` (the Belos scaling string). New cases on
  both routes:
  - b = 1e-4, default scaling: `initial_relative_residual` 1 on every row,
    `target_tol` equal to the tolerance, every verdict converged.
  - b = 1e-4, `"None"` scaling: request and test rows carry
    `target_tol = tol / ||b||`, and the tests' verdicts are converged.
  - Warm start: solve 1's `initial_relative_residual` is ||b - A x0|| / ||b||, computed
    independently in the test from the driver's x0.

## Non-goals

- `teko_ext.cpp`'s own boundary check (trilinos-teko-pyfront) keeps the raw
  tolerance. The pyfront always uses x0 = 0 and default scaling, where the two
  agree.
- The candidate tests still start from x0 = 0, not from the application's x0.
- b_hat stays Q^T b. Seeding it from r0 is a separate change.
- No change to how the surrogate measures its own relative residual (against
  ||b_hat||, not ||b||).
- No change to the paper.

## As built

As planned, with these differences:

- `writeRequestJson` takes `target_tol` as a third argument and writes it after
  `equation_ends`. `SolveRecord.target_tol` in `solveOrdering` comes from
  `orig_params` at the top, so a candidate whose build fails still records it.
- `relativeTarget` needed `BelosTypes.hpp` in `Teko_KrylovReconfigPrec.hpp`.
- `route_driver` gained `INITRES` output beside `TRUERES`, and its Belos list
  is built by one `setBelosParams` shared by both routes.
- The absolute-scaling case runs at b = 1e-6, not 1e-4. The driver's system
  converges in one or two iterations and overshoots any target, so at 1e-4
  every residual sat under the old raw-tolerance bound and the case could not
  tell the old verdict from the new one. At 1e-6 the one-iteration solves stop
  at about 1.2e-5, above that bound of 1e-5.

Results: `test_surrogate_search.py` 27 of 27 (two new),
`test_reconfig_protocol.py` 2 of 2 (field list and residual checks extended),
`tpetra_route` 9 of 9 (three new, both routes each), `warmup_nested` 5 of 5.
Built with `build_trilinos.sh` (belos, teko), the pyfront `build.sh`, and both
test drivers, at 12 jobs.
