# Plan: act on proto1 code review 2

**Written 2026-11-04.** Branch `aidantgould/teko-auto-proto1` at `b5c3877ce9c`. Source:
`proto1_code_review_2.md`, whose finding labels (H1, B2, ...) are used below.
Baseline: all 9 `KrylovSurrogate` tests pass in `trilinos-build-auto`.

Each finding was checked against the code before it went into this plan.

## 1. The recorder holds its operator (H1, H4)

- `RecorderOp` gains `setForwardOp(const BlockedLinearOp& A)` and
  `BlockedLinearOp getForwardOp() const`. They hold a strong reference to the operator
  the samples are recorded against. The factory sets it at a build. At a rebuild it
  replaces it, after any print of the old samples.
- `Handle` holds a weak reference to the recorder only, so `attach(recorder, options)`
  no longer takes `A`. `isAttached()` means the recorder still exists. `buildReport()`
  uses `recorder->getForwardOp()`. The weak-handle intent is kept: a copy of a
  parameter list still does not keep the samples alive.
- The rebuild print uses the recorder's old operator, not the new `lo`.
- Docs (`Handle`, `"Print"`): a report uses the operator's values as they are when
  the report is built. If the application refills the same operator in place, the
  report sees the refilled values.
- Tests:
  - `through_stratimikos` gains a reordered variant (`"Reorder Type" = "[0 1 2]"`)
    that builds a handle report after the solve.
  - `print_uses_the_recorded_operator`: build a handle report and print it, then
    rebuild with `"Print"` on against a different operator object (scaled off-diagonal
    blocks). The text printed at the rebuild must equal the handle report's text.

## 2. A print never aborts a rebuild (H3)

- The factory gets one private helper,
  `printAndClear(const Teuchos::RCP<RecorderOp>&) const`. It builds the report through
  a private `Handle` inside `try`/`catch (const std::exception&)`. When `buildReport`
  throws, it prints `e.what()` in place of the report.
- A comment beside the catch says why catching is safe on every rank: every throw
  inside `buildReport` is collective.
- The helper then clears the samples whether or not the report was built.
- Test: `print_survives_a_broken_solve`. Apply the preconditioner to a NaN vector,
  then rebuild with `"Print"` on. `rebuildInverse` must not throw, and rank 0 prints the
  error.

## 3. Print in the Stratimikos strided route (H2)

Owner decision (2026-11-04), the recommended rule: **a build prints the solves of the preconditioner it replaces.**

- At a rebuild, that is the preconditioner's own samples, as today.
- At a first build, it is the factory's `lastRecorder_`, a weak reference to the
  recorder this factory built or rebuilt most recently, if it still exists and has
  samples.
- In Stratimikos's strided route, every `initializePrec` builds a new preconditioner
  while the old one is still alive, so the old one is printed.
- A printed recorder's samples are cleared, so nothing is printed twice.
- Header and parameter documentation are rewritten to match.
- Test: `print_through_stratimikos_strided`. Use a monolithic `Tpetra::CrsMatrix`
  assembled from the same entries, with `"Strided Blocking" = "1 1 1"`, and run
  `initializePrec` twice with a solve between. The second must print, on rank 0, a
  report with the first solve's sample count.
- `print_reports_its_own_preconditioner` changes meaning. Building the second
  preconditioner now prints the first's samples. The test is reworded to check that
  rule.

## 4. Smaller fixes

- **B1:** in `record()`, reallocate the store when its range is not compatible with
  the preconditioner's.
- **B2:** owner decision: 0 stays allowed. The documentation says that 0 keeps
  roundoff directions (for example a non-flexible restart's dependent column), which
  $\Sigma_j^{-1}$ turns into noise.
- **B3:** `Report::print` counts `SOLVER_ERROR` candidates and says that only the
  messages of candidates scored on rank 0 reach Teko's stream.
- **B4:** `handle_` may be null in both construction paths. The build requires a
  handle or `"Print"`, and the dummy handle goes away.
- **B5:**
  - `Handle`'s copy constructor and copy assignment are deleted.
  - The `"Tolerance"` validator range becomes 0 to 0.5. At 1, the stopping test is
    met at iteration 0, the rate is 0, and that candidate sorts as best.
- `"Seed"` documentation says a seed reproduces candidates only within one standard
  library.

## 5. Idioms taken (review section 4)

1. $\hat{C}$ and $\hat{b}$ assembled through `DenseMatrix` views.
2. `Tpetra::createLocalMapWithNode` replaces the two hand-built locally replicated
   maps.
3. `validParameters()` builds the list once, as a function-local static.
4. The recorder's const views stay, because they select the const overload of an
   input, which is the right one. Only the misleading comment is fixed.

## 6. Tests added

- the H1, H4, H3 and H2 tests above;
- `report.toParameterList()` written with `writeParameterListToXmlString` and read
  back (review test gap 5);
- out-of-range values for the new validator bounds.

## 7. Deliberately not done

- **`Report` as `Teuchos::Describable` (idiom 4).** It is a new API, and the user
  guide work was deferred until the API settles.
- **One feature guard in `buildReport` (idiom 6).** The `#else` stays. Removing it
  only moves it into a new `Impl` function.
- **The hand-written XML writer (idiom 7).** The existing comment and the round-trip
  test answer it.
- **A test for B1.** Changing an operator's maps under an in-place rebuild needs an
  inner inverse that supports it, and the test suite has none. The fix is one line.
- **Wiki (review section 7).** The stale "no explicit clamp" line in
  `gram-matrix-svd-floor.md` gets a one-line fix in the wiki repo, left uncommitted
  there.

## 8. What was built (differences from the plan above)

- **B1:** the store check is in `clearSamples()`, not in `record()`. Every rebuild
  clears after the inner inverse is rebuilt, so the new range is known by then, and
  `record()` never has to drop samples it already holds.
- **H3:** `printAndClear` takes the caller's strong `RCP`. A `rcpFromRef` would leave
  the private handle's weak copy dead on arrival.
- **H3:** the rank that prints comes from the first nonzero block (`rankOf`), not from
  block (0, 0), which may be zero.
- **B4:** the "a handle or Print" check moved from `initializeFromParameterList` to the
  build. A misconfigured XML list now fails at `buildInverse`, not at
  `getInverseFactory`, and the `errors` test changed accordingly.
- **H1:** the reorder test is a second pass of the `through_stratimikos` loop, not a
  separate test. Both use a new `stratimikosSolver` helper, which the strided test
  shares.
- **H2:** `print_reports_its_own_preconditioner` is renamed
  `print_reports_the_replaced_preconditioner`. It checks both halves of the rule: the
  new build prints the first preconditioner's samples, and a later rebuild of the
  first prints only its second solve.
- **`tKrylovSurrogateProblem.hpp`:**
  - `buildBlockedOp` gains an `offDiagonalScale`, used by the H4 test.
  - A new `buildInterleavedMatrix` builds the monolithic matrix for the strided test:
    3 fields of 24 rows, interleaved, distributed in whole nodes.
- **Idiom 1:** $\hat{b}$ keeps its one-line loop. Only $\hat{C}$ goes through views.

## 9. Verification (2026-11-04)

- Teko and all its tests built with no warnings in `trilinos-build-auto`.
- `ctest -R Teko`: 24 of 24 pass. That includes the factory tests at 1 and 4 ranks,
  now 12 tests in the executable.
- The new tests print sensible output:
  - The NaN rebuild prints the collective `reduceOnRankZero` error.
  - The next solve is then reported normally: 8 samples, a diagonal deviation of
    $2.7\times10^{-14}$.
  - The strided route prints at the second `initializePrec`.
  - The printed report at a rebuild onto a scaled operator equals, byte for byte, the
    handle report taken before it.
- Feature off: the five sources and `Teko_PreconditionerFactory.cpp` compile with
  `-Wall -Wextra -Wreturn-type` and no warnings.
- clang-format 14 has been run on every touched file.
- Not run: a check that the reorder test fails on the old code. The review reproduced
  that failure. Also not run: an actual feature-off Trilinos configure, and a device
  build.
- Wiki: the stale clamp line in `gram-matrix-svd-floor.md` is fixed, and left
  uncommitted.
