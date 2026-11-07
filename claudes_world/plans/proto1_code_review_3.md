# Code review 3: Krylov surrogate in Teko (tests, correctness, idiom, simplicity)

**Written 2026-11-04.** Branch `aidantgould/teko-auto-proto1` at `b5c3877ce9c`, plus the
uncommitted work of `proto1_review_2_fixes_plan.md` (the recorder holds its operator, the
print cannot abort a build, the last-recorder print rule, the smaller fixes). This review
is of that working tree.

**Questions asked.** Are there too many tests, and which matter most? Is the code free of
errors? Is it idiomatic? Is it as simple as it reasonably can be?

**How it was reviewed.** Every line of all six test files (this was not done in
review 2) and the library diff since review 2, on top of review 2's full read. Each
suspicion was checked against the Teko, Thyra or Teuchos code it depends on.

**What was run.**
- `ctest -R KrylovSurrogate`: 9 of 9 pass. The factory executable: 12 of 12. No source
  is newer than the binaries, so this is the working tree.
- One scratch test, compiled outside the repo
  (`/tmp/claude-1003/-home-atgould-teko-Trilinos/0538c11a-a98e-4b27-bf8b-bebb61b56c1b/scratchpad/predict/tPredict.cpp`,
  session scratch, not kept). It compares each candidate's predicted iteration count with
  a real solve using that candidate's printed Hierarchical Gauss-Seidel form. The results
  are in section 2.

Nothing in the repo was changed.

## 1. Verdict

- **Too many tests?** Not by count or cost. There are 30 tests in about 1,170 code lines
  against about 1,190 library lines. All 9 ctest entries run in 7 seconds. For dense
  numerics with MPI invariance claims, about 1:1 is normal. The problems are elsewhere:
  - **Where the tests sit.** Five tests (about 180 lines) cover `"Print"`, a
    diagnostic, and none checks that the surrogate's predictions are right.
  - **About six tests or cases can be cut or merged** without losing coverage
    (section 3).
- **Error free?** I found no bug in any route the tests exercise, and the review 2
  findings are fixed and tested. Two things remain:
  - The `"Tolerance"` default silently offsets predicted iteration counts (E1, measured).
  - The last-recorder print rule can attribute a report to the wrong preconditioner
    (E2, by reading).
- **Idiomatic?** Mostly. Three local departures: `rankOf` (Teko never gates output by
  rank), the `x + 1.0` floating-equality trick in the recorder tests, and using a
  `Handle` as an internal report builder.
- **As simple as it reasonably can be?** The numerical core is: compact, with nothing
  worth removing. `"Print"` is not. It now has two triggers and mutable state shared
  across preconditioners, and it has the most tests. The simplest fix is outside the
  feature, in Teko's Stratimikos adapter (S1).

## 2. The most important missing test (run)

On the test problem ($n = 60, 48, 8$, lower block Gauss-Seidel with Amesos2 blocks, 8
iterations and 8 samples), with the search tolerance set to the solve's $10^{-10}$:

| ordering | predicted | actual |   | ordering | predicted | actual |
|---|---|---|---|---|---|---|
| [000] | 1 | 1 | | [101] | 6 | 7 |
| [110] | 5 | 5 | | [102] | 6 | 7 |
| [001] | 5 | 5 | | [210] | 6 | 7 |
| [100] | 6 | 6 | | [201] | 8 | 8 |
| [011] | 6 | 6 | | [120] | 8 | 8 |
| [010] | 6 | 6 | | **[012]** (trained) | **8** | **8** |
| [021] | 6 | 6 | | | | |

The trained configuration is predicted exactly. Every other candidate is within one
iteration. Sorted by prediction, the actual counts never decrease, so the ranking is
exactly right. The 4-rank run gives the same result.

This is the one claim a Trilinos reviewer will ask about ("does it predict anything?"),
and no current test checks it. `hierarchical_gauss_seidel_form` shows that the printed
form is the configuration that was scored. It does not show that the score is right.
**Add `predictions_order_real_solves`.** Assert that the trained configuration is
predicted exactly, and that actual iterations are non-decreasing in the report's order.
Do not assert the $\pm 1$: that holds on this problem but is not a property of the
method. The test costs 13 small solves, about 0.4 s.

## 3. The tests: what matters, what to cut

### 3.1 If only eight could stay

1. `ReducedOperator.matches_dense_reference`, with `any_distribution` as its second
   loop. It checks the numerical core against an independent dense SVD reference,
   including TSQR's zero-row padding and a block with fewer rows than samples.
2. The new `predictions_order_real_solves` (section 2). It checks that the method works.
3. `Factory.report_from_a_solve`. The recorder is transparent (the solve with it is
   bitwise identical to the solve without it), the report is complete, and it survives
   an XML round trip.
4. `Factory.hierarchical_gauss_seidel_form`. What is printed for the user to paste
   means what was scored, including the merged group.
5. `Search.lower_coupling_and_its_reverse`. It pins the orientation of the Gauss-Seidel
   selection with exact iteration counts (1 and 2). Getting this backwards would invert
   every recommendation while every other test still passed.
6. `Parallel.bitwise_identical_to_serial`. Rank invariance, the design's load-bearing
   promise.
7. `ReducedOperator.non_finite_throws_on_every_rank`. The hang guard. At 4 ranks it
   fails by never finishing, which is the point.
8. `Factory.through_stratimikos` (both passes) and `print_through_stratimikos_strided`.
   The two routes real applications use, and the two review 2 found broken.

The next tier is worth keeping but would not be missed first: the recorder's
`records_outputs_in_order` and `cap_and_clear`, `warm_start_and_rebuild` and
`restarted_gmres`, `handle_does_not_keep_the_preconditioner`, and
`print_survives_a_broken_solve`. The two first-input tests pin a Belos behaviour the
design relies on (the first apply is $r_0 / \lVert r_0\rVert$, and a restart does not
reset it), so they guard something outside this code.

### 3.2 Cut or merge

| test | action | why |
|---|---|---|
| `Search.block_diagonal_needs_one_iteration` | cut | With $\hat{C} = I$, every selection is $I$ and $K = I$. `all_merged_is_exact` already covers $K = I$, and `lower_coupling_and_its_reverse` covers the selection. |
| `Search.sorted_and_complete` | strengthen or cut | It checks only that iterations rise within a status, never that statuses are in order. Add `TEST_ASSERT(scored[i].status >= scored[i - 1].status)`, or drop it, since the new prediction test checks the order that matters. |
| `Parallel.more_ranks_than_candidates` | merge | Make it a second candidate list inside `bitwise_identical_to_serial`. Same assertion, same helper. |
| `ReducedOperator.resolves_below_gram_floor` | rename | It names a design (the Gram route) that this code never had, so a Trilinos reviewer cannot place it. What it tests is real: a singular value between $10^{-10}$ and $4\times10^{-8}$ is kept. Call it `keeps_small_singular_values`, and say $\sigma_k \ge 10^{-10}\sigma_1$ in the comment. |
| `Factory.errors`, validator cases | trim 5 to 1 | Four of the five cases test that `Teuchos::EnhancedNumberValidator` rejects out-of-range values, which is Teuchos's job. One case (the misspelled option) shows that validation is wired in. The `TEST_NOTHROW` for tolerance 0 tests nothing. |
| `Factory.errors`, `static_assert` | move | A compile-time check inside a runtime test. Put it at file scope, or drop it: `= delete` is its own documentation. |
| `Factory.print_at_rebuild_without_a_handle` | merge | `print_survives_a_broken_solve`'s second half already prints a normal report at a rebuild with no handle. Move this test's XML assertions (`value="BGS"`, no placeholder) there. |

Net: about 150 lines and 4 tests fewer, plus one test added.

### 3.3 Test idiom and robustness

- **`TEST_FLOATING_EQUALITY(maxDiff(...) + 1.0, 1.0, 1e-15)`** (recorder tests, 6 places)
  turns a relative check into an absolute one. The direct form is
  `TEST_COMPARE(maxDiff(...), <=, 1e-15)`. The diagonal operator makes these differences
  exactly 0 anyway, as `store_grows_and_keeps_every_column` already asserts with
  `TEST_EQUALITY(worst, 0.0)`.
- **`solve()` ignores Belos's verdict**
  ([tKrylovSurrogateFactory.cpp:96](../../packages/teko/tests/unit_tests/tKrylovSurrogateFactory.cpp#L96)).
  A solve that stops converging would then fail some later assertion with a misleading
  message, or pass. Assert `solver.solve() == Belos::Converged` in the helper.
- **The serial search tests run at four rank counts.**
  `tKrylovSurrogateSearch.cpp` uses `SerialComm` throughout, but it shares an executable
  with the parallel tests, so CMake runs it at 1, 2, 3 and 4 ranks. That costs about
  2 seconds, and the CMake comment justifies only the parallel half. Splitting the
  executable is optional.

## 4. Correctness

### E1. The search tolerance silently offsets predicted iteration counts (major for usability, measured)

`inverseLibraryList` scores at `"Tolerance" = 1e-8`, the default, while `solve()` runs to
$10^{-10}$. With that mismatch, the same probe gives the trained configuration as 6
predicted against 8 actual, and the counts are off by up to 2. The ranking happened to
stay monotone. With the tolerances matched, the trained configuration is exact. The
factory cannot see the outer solver's tolerance, so every user who compares the printed
iterations with their own solve will see a disagreement they cannot explain.

**Fix.** Document on `"Tolerance"`, in the header XML and in the valid-parameter string,
that it should equal the outer solve's convergence tolerance for the predicted counts to
be comparable. Set the test fixture's value to $10^{-10}$, so the tests model correct
use.

### E2. The last-recorder rule can print the wrong preconditioner (minor, by reading)

The new-build print uses `lastRecorder_`, the recorder this **factory** built most
recently. The rule is documented, and it is right for the case it was written for (one
system, Stratimikos's strided route). It goes wrong when one factory serves two live
preconditioners:

- **It clears a live preconditioner's samples.** Build preconditioner 1 for system A,
  solve, then build preconditioner 2 for system B. That build prints preconditioner 1's
  samples, as documented, and also forgets them while preconditioner 1 is still in use.
- **It misattributes in Stratimikos.** Teko's `StratimikosFactory` keeps one
  `invFactory_` and replaces it whenever any new preconditioner object starts over
  ([Teko_StratimikosFactory.cpp:140-155](../../packages/teko/src/Teko_StratimikosFactory.cpp#L140-L155)).
  With two systems solved through one Stratimikos builder in the strided route, system
  A's re-initialization goes through system B's surrogate factory, so it prints system
  B's samples and never system A's.
- **Edge case.** If the two preconditioners live on different communicators, the print
  is collective over the old one's communicator, called from a build on the new one's.

The report does not say which preconditioner it describes, so none of this is visible
in the output. See S1 for the fix that removes the rule. Short of that, document "one
factory per system" beside `"Print"`.

### Checked and fine

- **B1's check** compares Thyra's `mapCode()`, which encodes local sizes. That is exactly
  what decides whether the old store's local layout can be reused, so the check is
  sufficient. Same-size repartitioning keeps the store, and local data is all that
  `record()` and `Tpetra::CrsMatrix::apply` use.
- **H4's test** compares the text printed at a rebuild onto a scaled operator with a
  handle report taken before it. That text would differ under the old code (the rates
  change), so the test does pin the fix.
- **`printAndClear` ordering.** The new-build print runs after validation and before the
  inner build, and the rebuild print runs after `rebuildInverse`, so a throwing inner
  build never loses an unprinted report, and a print never sees half-rebuilt state.
- **`validParameters()` as a function-local static.** Initialization is thread-safe
  since C++11, and the list is never modified.

## 5. Idiom

1. **Drop `rankOf`**
   ([Factory.cpp, anonymous namespace](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogatePreconditionerFactory.cpp)).
   Teko writes to `Teko::getOutputStream()` in 18 other places and never checks the rank.
   Root-only output is the stream's job (`setOutputToRootOnly(0)` on Teuchos's default).
   A user who installs an all-ranks stream has asked for all ranks. The tests'
   `CapturedOutput` then sets `setOutputToRootOnly(0)` to mirror the default, and their
   `printed.empty()` checks on other ranks still hold. This removes 12 lines, a feature
   guard, and a loop over every block at each print.
2. **Build reports without a private `Handle`.** `printAndClear` creates a `Handle` only
   to call `buildReport()`. A free `Impl::buildReport(const RecorderOp&, const Options&)`,
   called by both `Handle::buildReport` and `printAndClear`, says what happens. The
   handle then goes back to being only the user's way in, and `attach` has one caller.
3. **The recorder tests' assertions** (section 3.3).

The review 2 items the fixes plan declined (`Describable`, one feature guard, the XML
writer) stay declined. The plan's reasons hold.

## 6. Simplicity

**The numerical core is as simple as it reasonably can be.** The TSQR with zero-row
padding, the rank-0 SVD, the explicit $K$ and the round-robin gather each exist for a
stated reason (rank invariance, collective errors, resolving small singular values), and
none could be dropped without losing one. The combinatorics (about 70 lines) are
justified by uniform sampling. The 35-line XML writer is justified by readability.

### S1. `"Print"` is where the complexity went (recommended, an owner decision)

`"Print"` now has two triggers (a rebuild, and a new build replacing `lastRecorder_`), a
`mutable` weak member, state that crosses preconditioners (E2), and five tests. All of
the new-build half exists because Teko's Stratimikos adapter rebuilds nothing in the
strided route.

Teko already has what an in-place rebuild there needs:
`TpetraHelpers::InverseFactoryOperator::rebuildInverseOperator` and
`BlockedTpetraOperator::RebuildOps`. The single-block branch of the same function already
rebuilds in place. A roughly 10-line change to the multi-block branch would make it do
the same: when `prec_Op` exists, recover its `InverseFactoryOperator`, rebuild the
blocked wrapper, and call `rebuildInverseOperator`. After that:

- `lastRecorder_`, the replaced-preconditioner rule and its documentation go away.
- `print_reports_the_replaced_preconditioner` goes back to the simpler rebuild-only test.
- E2 cannot happen.
- Every strided Teko user gains an in-place rebuild, which can reuse the inner inverses'
  symbolic setup.

The cost is that it changes Teko's Stratimikos adapter for every strided user, so it
belongs in its own small PR with a Teko maintainer's agreement. Until that lands, the
current rule is a correct workaround. It also covers any other caller that builds anew
rather than rebuilding, which is its one advantage over the Stratimikos fix.

### S2. Feature guards (noted, not recommended)

`TEKO_HAVE_KRYLOV_SURROGATE` appears at 15 sites in 7 files. Compiling
`KrylovSurrogate/*.cpp` only with the feature, and registering the type only with it,
would remove all but about two. The cost is the tailored "configure with
`Trilinos_ENABLE_TpetraTSQR=ON` ..." error, replaced by Teko's generic unknown-type
error. That was decided in review 1, and the tailored message is worth more than the
guards cost. It is listed here only because it is the largest simplification left.

## 7. Suggested order

1. Add `predictions_order_real_solves`, and set the fixture's search tolerance to
   $10^{-10}$ (section 2, E1). This is the strongest evidence for the PR.
2. Document `"Tolerance"` (E1).
3. Drop `rankOf`, add the free `Impl::buildReport` (idioms 1 and 2).
4. The test cuts, merges and fixes (sections 3.2 and 3.3).
5. Decide S1. If yes, a separate Teko PR for the Stratimikos strided rebuild, then remove
   `lastRecorder_`. If no, document "one factory per system" (E2).
