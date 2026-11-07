# Plan: act on proto1 code review 3

**Written 2026-11-04.** Working tree of `aidantgould/teko-auto-proto1`: `b5c3877ce9c`
plus the uncommitted review 2 fixes. Source: `proto1_code_review_3.md`, whose labels
(E1, S1, ...) are used below. Baseline: 9 of 9 `KrylovSurrogate` ctest entries pass.

## 1. The prediction test and the tolerance (section 2, E1)

- New `Factory.predictions_order_real_solves`. It scores at `"Tolerance" = 1e-10`, the
  solve's tolerance. For each candidate, it builds the printed Hierarchical Gauss-Seidel
  form, with Amesos2 for single-block groups and "Exact Group" GMRES at $10^{-13}$ for
  merged groups, and runs the real solve. It asserts two things:
  - the trained ordering `[0 1 2]` is predicted exactly;
  - the actual iteration counts are non-decreasing in the report's order.
  The $\pm 1$ agreement is printed but not asserted.
- `inverseLibraryList` scores at `1e-10`, so every test models correct use.
- `"Tolerance"` documentation (header XML and the valid-parameter string): set it to
  the outer solve's convergence tolerance, or the predicted counts are not comparable
  with the user's own.

## 2. Idiom (section 5)

- **Drop `rankOf`.** Output goes to `Teko::getOutputStream()`, which is root-only by
  default, like Teko's 18 other writes. The tests' `CapturedOutput` sets
  `setOutputToRootOnly(0)`, to mirror the default.
- **`Impl::buildReport(const RecorderOp&, const Options&)`** is declared in
  `Teko_KrylovSurrogateSearch.hpp`, under the feature guard, beside `scoreCandidates`.
  - Both `Handle::buildReport` (after its user-facing checks) and `printAndClear` call
    it.
  - The private handle in `printAndClear` goes, and `Handle::attach` keeps the factory
    as its one caller.
  - The `#else` branch in `Handle::buildReport` stays: it is the function body when
    the feature is off.

## 3. Tests (sections 3.2, 3.3)

- **Cut:** `Search.block_diagonal_needs_one_iteration`.
- **Strengthen:** `Search.sorted_and_complete` asserts that statuses are
  non-decreasing.
- **Merge:** `Parallel.more_ranks_than_candidates` becomes a second candidate list in
  `bitwise_identical_to_serial`.
- **Rename:** `ReducedOperator.resolves_below_gram_floor` becomes
  `keeps_small_singular_values`. Its comment states the property,
  $\sigma_k \ge 10^{-10}\sigma_1$, without the Gram route.
- **`Factory.errors`:**
  - Validator cases go from 5 to 1: the misspelled option, which shows validation is
    wired in.
  - The `TEST_NOTHROW` for a truncation tolerance of 0 is dropped.
  - The `static_assert` moves to file scope.
- **Merge:** `Factory.print_at_rebuild_without_a_handle` moves into
  `print_survives_a_broken_solve`'s second half, keeping its XML assertions.
- **Recorder tests:** `TEST_FLOATING_EQUALITY(x + 1.0, 1.0, tol)` becomes
  `TEST_COMPARE(x, <=, tol)`.
- **`solve()` helper:** it asserts Belos converged.
- The serial search tests keep sharing an executable with the parallel ones. The
  split is optional, and it costs 2 s.

## 4. The print rule (E2, S1)

Owner decision (2026-11-04): **document now, and fix Stratimikos separately.**

- Document "one factory per system" beside `"Print"`.
- Section 6 of this plan is the note for a separate Teko PR.

## 5. Deliberately not done

- **S2 (compile only with the feature).** Review 1 decided to keep the tailored error,
  and review 3 does not recommend changing that.
- **Splitting the search executable.**

## 6. Note for a later Teko PR: an in-place rebuild in Stratimikos's strided route

- **Today.** The multi-block branch of `StratimikosFactory::initializePrec_Thyra`
  builds a new `BlockedTpetraOperator` and `InverseFactoryOperator` at every
  initialization.
- **Change.** When `prec_Op` already exists, recover its `InverseFactoryOperator`
  from the `Thyra::TpetraLinearOp` and rebuild the blocked wrapper (`RebuildOps`, or
  a new one on the new operator). Then call `rebuildInverseOperator`, as the
  single-block branch already does through `Teko::rebuildInverse`.
- **What it buys.**
  - Every strided user gets a rebuild that can reuse the inner inverses' symbolic
    setup.
  - Krylov surrogate's `lastRecorder_` and the new-build print rule can go, and E2
    with them.
- **Why separately.** It changes behavior for every strided Teko user, so it needs a
  Teko maintainer's agreement.

## 7. What was built (differences from the plan above)

- `Impl::buildReport` is defined in `Search.cpp`, next to `scoreCandidates`. The two
  report timers moved there with it. `Report.cpp` no longer includes `CommHelpers`,
  `TimeMonitor`, `Tpetra_Map` or `ReducedOperator.hpp`.
- `Handle::buildReport` keeps its three user-facing checks: never attached, the
  preconditioner gone, and no samples. `Impl::buildReport` asserts its own
  precondition (an operator and samples) as a `logic_error`.
- The "Exact Group" inverse is now a shared test helper, `addExactGroup`, used by
  both `hierarchical_gauss_seidel_form` and the prediction test.
- `print_survives_a_broken_solve` now opens as the XML-only (no handle) test, and its
  other ranks assert that nothing was printed.
- Test counts:
  - Factory: still 12, with `print_at_rebuild_without_a_handle` out and
    `predictions_order_real_solves` in.
  - Search executable: from 9 to 7. One search test was cut, and one parallel test
    merged.
  - `errors`: 3 validator cases fewer.

## 8. Verification (2026-11-04)

- `trilinos-build-auto`: no warnings. `ctest -R Teko`: 24 of 24 pass, 7.4 s in all.
- `predictions_order_real_solves` reproduces review 3's table exactly, at 1 and 4
  ranks:
  - the trained `[0 1 2]` is predicted at 8 and solved in 8;
  - three candidates are off by one (6 predicted, 7 actual);
  - the actual counts are non-decreasing in report order.
- Feature off: the five sources and `Teko_PreconditionerFactory.cpp` compile with no
  warnings.
- clang-format 14 has been run on every touched file.
- Not run: a feature-off configure, and a device build.
