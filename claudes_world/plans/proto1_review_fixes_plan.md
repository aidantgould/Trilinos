# Plan: act on the proto1 code review

**Written 2026-11-04.** Branch `aidantgould/teko-auto-proto1` at `d491285f47e`. Source:
`proto1_code_review.md` (section numbers below are its own). Baseline: all 9
`KrylovSurrogate` tests pass in `trilinos-build-auto`.

Every finding was checked against the code and against what Teko itself does. A
finding is acted on when it is a real defect, or a reviewer comment that is cheap to
answer in code. It is skipped when Teko's own convention points the other way, or
when the fix costs more than it buys. Skips are listed with the reason, so the PR
description can answer them.

## 1. Real bugs (each with a test)

1. **Rank-0-only throw in `reduceOnRankZero`** (2.3, 3.8). Rank 0 sets a status
   (0 ok, 1 non-finite input, 2 `GESVD` failure) and broadcasts it together with
   `size`, as one `int[2]`. Every rank then throws the same message. Checking for
   non-finite input first makes the path testable: NaN samples (a solve that broke
   down) are realistic input, and a forced `GESVD` failure is not.
   Test: a NaN matrix throws on every rank (the np4 run would hang otherwise).
2. **Printed report through the shared handle** (3.4). At a rebuild with `"Print"`
   on, the factory builds a private `Handle`, attaches the local recorder and operator,
   and prints that one's report, so it never reads another preconditioner's
   samples. No new free function. Test: one factory, two preconditioners, solve with
   the first, build the second, rebuild the first. Today that throws ("no samples").
3. **`isValidOrdering({-3, -2}, 2)` throws `std::length_error`** (3.9). Range-check
   every entry first. Test added to the existing ordering test.
4. **NaN rate breaks the sort's strict weak ordering** (3.9). Compare rates with NaN
   last.
5. **Retained rank above the block dimension** (wiki `gram-matrix-svd-floor`, not in
   the review). `reduceOnRankZero` gains `maxSize`. `buildReducedOperator` passes
   $n_j$. This matters once `Truncation Tolerance = 0` is allowed (item 10). Test:
   `rank_zero_svd` with a cap.

## 2. Build integration (2.1, 3.4, 3.12)

6. **The feature macro comes from TriBITS.** Use the precedent Teko already has for
   `TEKO_HAVE_EPETRA`: `COMBINED_OPTION(TEKO_HAVE_KRYLOV_SURROGATE DEP_OPTIONS_NAMES
   TpetraCore_ENABLE_TSQR Teko_ENABLE_Belos ...)` in `packages/teko/CMakeLists.txt`,
   plus `#cmakedefine TEKO_HAVE_KRYLOV_SURROGATE` in `Teko_Config.h.in`. Remove the
   `#define` block and the `TpetraCore_config.h` include from `Report.hpp`.
7. **Declare the TSQR dependency.** Add `TpetraTSQR` to `LIB_OPTIONAL_DEP_PACKAGES`,
   so enabling Teko with optional packages turns TSQR on.
8. **Test guards** mirror the library macro: `IF(TEKO_HAVE_KRYLOV_SURROGATE)`. The
   factory test also needs `Teko_ENABLE_Amesos2` (3.13), because its exact block
   solves are the point of several assertions. A fallback to inexact `Ifpack2` would
   change what they test.
9. **Feature-off branch.** Add `return Teuchos::null;` after the throw. Name both CMake
   options in the message. Wrap the declarations in `ReducedOperator.hpp` and
   `Search.hpp` in the guard (2.2). Verify by compiling the five sources with
   `-fsyntax-only -Wall -Wreturn-type` against a copy of `Teko_Config.h` with the macro
   removed. That is cheap, and a feature-off build of Trilinos is not.

## 3. Parameters and API

10. **Validators** (3.4): `Teuchos::EnhancedNumberValidator` for `"Maximum Samples"`
    (at least 1), `"Candidate Count"` (at least 1), `"Truncation Tolerance"` (0 to 1)
    and `"Tolerance"` (0 to 1). They replace the hand-written `maxSamples < 1` checks in
    the factory. `RecorderOp`'s own check stays, because it is a constructor
    precondition. Test: `"Candidate Count" = -5` throws.
11. **`getValidParameters()`** (3.3), returning the same list. Stratimikos users
    expect it.
12. **Rename** `KrylovSurrogate::PreconditionerFactory` to
    `KrylovSurrogate::KrylovSurrogatePreconditionerFactory` (3.3). It matches the
    `NS::LSCPreconditionerFactory` precedent and the file name.
13. **Public status enum** (3.1, 3.9): `Report::Candidate::Status { CONVERGED,
    NOT_CONVERGED, SINGULAR_GROUP_BLOCK, SOLVER_ERROR }` and `statusName()`. "max
    iterations" becomes "not converged", because Belos's `Unconverged` also covers loss
    of accuracy. Strings stay in `toParameterList` and `print`.
14. **64-bit block sizes** (3.1): `std::vector<long long> blockSizes`, and
    `Array<long long>` in `toParameterList`.
15. **Internal API surface** (2.2), owner decision: everything declared in
    `ReducedOperator.hpp` and `Search.hpp` moves into `Teko::KrylovSurrogate::Impl`
    (`DenseMatrix`, `DenseVector`, `BlockReduction`, `reduceOnRankZero`, `tpetraBlock`,
    `buildReducedOperator`, `numConfigurations`, `allConfigurations`,
    `randomConfiguration`, `isValidOrdering`, `buildCandidates`,
    `gaussSeidelSelection`, `scoreCandidates`). Headers stay installed, and each file
    comment says it is not user API. The user API is `Handle`, `Report`, `Options`,
    `asHierarchicalGaussSeidel`, the factory, and `RecorderOp` (an operator like
    Teko's public `DiagnosticLinearOp`, which `Handle::getRecorder` returns).
16. **`Handle::attach` private**, with the factory as a `friend` (2.2). `getRecorder`
    stays, documented as a diagnostic: two tests need it, and it does no harm.
16b. **Weak references in the handle** (2.4), owner decision: `Handle` holds
    `create_weak()` copies of the recorder and the operator, so a parameter list copy
    no longer keeps the samples alive. `buildReport()` throws a clear error when either
    has gone, and `isAttached()` reports false once it has. Test: drop the
    preconditioner, and `buildReport()` throws.

## 4. Memory, robustness, meaning

17. **Grow the sample store** (3.6): start at `min(maxSamples, 16)` columns and double
    up to the cap, copying on growth. Today 100 full-length vectors are allocated on
    the first apply. Document the peak (store plus about 3x during the build, 3.8) in
    `"Maximum Samples"`. Test: the recorder keeps every column across two growths.
18. **Search errors leave a trace** (3.9). The catch-all stays, because a throw on one
    rank would leave the others in the gather, and a comment now says so. The caught
    message, and Belos's own `Errors` output (today sent to a black hole), go to
    `Teko::getOutputStream()`, which is rank 0's by default. `bad_alloc` is not
    treated differently, for the same reason.
19. **Merged-group meaning** (3.2, 3.13). Document in `asHierarchicalGaussSeidel`, in
    `"Merged Inverse Type"`, and in the factory header that a score assumes an exact
    solve of the merged group. The test's merged form uses an exact merged inverse
    (Belos GMRES to $10^{-13}$ on the blocked group), and it checks the form against
    a hand-computed lower block Gauss-Seidel, not just that it runs.
20. **Documentation in the header** (2.3, 3.3, 3.4): the factory's build and rebuild
    are collective when `"Print"` is on, the last solve of a run is never printed, the
    XML caveat of an `RCP` in the list (writing it works, reading it back throws), and
    the handle's weak references (2.4).

## 5. Tests added or tightened

- the four bug tests above, the rank cap, recorder growth, and the validators;
- invalid ordering rejected at build time;
- restarted GMRES (`"Num Blocks"` below the iteration count): the first input is
  still $r_0 / \lVert r_0\rVert$, and the samples span the cycles;
- the printed best XML parses back with `getParametersFromXmlString` and builds;
- the timer check compares `numCalls()` before and after (3.13);
- the default-stream swap in the print test is restored by an RAII guard (3.13);
- `tKrylovSurrogateProblem.hpp` nits (an unused parameter, a missing
  `<algorithm>`).

## 6. Owner decisions (2026-11-04)

- Internal API (2.2): an `Impl` namespace, headers still installed (item 15).
- Handle references (2.4): weak (item 16b).
- User guide entry and example (2.6): deferred until the API settles after this pass.

## 7. Deliberately not done

- **`CATEGORIES` and timeouts on tests** (3.12). No other Teko test uses them.
- **`typedef` to `using`, `override`** (3.3, 3.7). Teko uses `typedef` and plain
  `virtual` throughout.
- **Member timers instead of function-local statics** (3.2). Function-local statics are
  the common Trilinos form, and Belos itself uses it.
- **Packing the four per-block broadcasts** (3.8). The review says only if asked.
  Status and size are packed anyway (item 1).
- **Condition estimate for merged group blocks** (3.9). It needs a threshold that
  nothing in the wiki settles yet (`selection-invertibility`). Recorded as an open
  question instead.
- **Lazily built `asHierarchicalGaussSeidel` per candidate**, **zero-padded candidate
  names**, **enumeration cap**, **`getSamples` returning const**. Low value. The
  last needs const variants of Teko's block helpers.
- **Bitwise solve comparison on device** (2.5). Host is all that is built here. The PR
  description answers it (and the TSQR-on-device question) instead.
- **Splitting the PR** (2.7). That is a process decision for when the PR is opened.

## 8. What was built (differences from the plan above)

- Item 1: the broadcast is `int[3]` (size, status, LAPACK info), so the `GESVD`
  message on every rank still carries `info`.
- Item 3: `isValidOrdering` also rejects an entry of `n` or more.
- Item 9: only the scoring in `Search.cpp` is guarded. The combinatorics
  (`numConfigurations` and the rest) need neither TSQR nor Belos, and
  `asHierarchicalGaussSeidel` (always compiled) now calls `Impl::isValidOrdering`
  and throws on an invalid ordering (3.2, L38).
- Item 10: the factory's `maxSamples < 1` check is gone. The validator replaces it.
- Item 13: the status enum is `Report::Candidate::Status`, and `statusName()` is a
  free function in `Teko::KrylovSurrogate`.
- Item 17: `kInitialCapacity = 16`. The test crosses growths at 16 and 32, up to
  a cap of 40.
- Item 19: the merged-form test checks the lower block Gauss-Seidel residual
  $x_i - \sum_{j \preceq i} A_{ij} y_j$ on every block, with Belos GMRES to $10^{-13}$
  as the merged inverse.
- `Report::print` formats into a local stream, so the caller's flags are untouched.
  It sizes the ordering column to fit, prints rates as `%.3e`, and says when no
  candidate converged.
- `buildReport` takes strong copies of the weak references for its own duration.
- The handle-lifetime test is `handle_does_not_keep_the_preconditioner`, and the
  two-preconditioner test is `print_reports_its_own_preconditioner`.

## 9. Verification (2026-11-04)

- `trilinos-build-auto` reconfigured: `TEKO_HAVE_KRYLOV_SURROGATE = ON`. Teko and all
  its tests built with no warnings.
- `ctest -R KrylovSurrogate`: 9 of 9 pass (1 to 4 ranks). The new tests print sensible
  numbers: merged-form residual $7.6\times10^{-14}$, the restarted solve (restart 4)
  kept all 8 samples, and diagonal deviations are about $10^{-14}$.
- The whole Teko suite: 24 of 24 pass.
- Feature off: the five `KrylovSurrogate/*.cpp` files and
  `Teko_PreconditionerFactory.cpp` compile with `-Wall -Wextra -Wreturn-type` and no
  warnings, against a copy of `Teko_Config.h` without the macro. Preprocessing
  confirms that the off branch was the one compiled.
- clang-format 14 with Teko's `.clang-format`: clean on every touched file.
- Not run: an actual feature-off Trilinos configure, and any device build.
