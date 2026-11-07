# Code review 2: Krylov surrogate in Teko (proto1, after the review fixes)

**Written 2026-11-04.** Branch `aidantgould/teko-auto-proto1` at `b5c3877ce9c`. The code
under review is `097f36f0b90` (the fixes from `proto1_code_review.md`, as recorded in
`proto1_review_fixes_plan.md`). The branch's own diff is against `0cf62aaf091` (develop,
2026-10-27), not against local `master`, which is from April and makes upstream Teko
work (the ML removal, cmake-format) look like part of this branch.

**How it was reviewed.** Every line of the ten library files, the build and
registration changes, and the tests by name and where a finding touched them. Each
suspicion was checked against the Teko, Stratimikos, Belos, Tpetra or Teuchos code it
depends on. Severity follows the first review: **blocker**, **major**, **minor**, **nit**.

**What was run.** `ctest -R KrylovSurrogate` in `trilinos-build-auto`: 9 of 9 pass, and
no source is newer than the test binaries, so that is HEAD. Finding H1 was reproduced by
compiling a scratch copy of `tKrylovSurrogateFactory.cpp` with one added test, outside
the repo (`/tmp/claude-1003/-home-atgould-teko-Trilinos/0538c11a-a98e-4b27-bf8b-bebb61b56c1b/scratchpad/weakA/tWeakA.cpp`, session scratch, not kept). Nothing in the repo was
changed. Findings marked "by reading" were not run.

## 1. Summary

The numerical core is still sound. The first review's collective-error work holds up:
every throw inside `buildReport` that I traced is taken identically on every rank. The
dense reference tests are strong, and clang-format and the feature-off compile were
already checked.

The new defects are where the feature meets **how Teko and Stratimikos actually call
it**. The tests drive the factory directly, or through Stratimikos with neither reordering
nor strided blocking. In those routes the operator object outlives the build and is
reused at each rebuild. The two routes a production application is most likely to use
break those assumptions:

- **H1 (blocker, run).** Since the weak-handle change, the handle cannot report in the
  Stratimikos `"Reorder Type"` route, or when the surrogate is a nested inverse.
- **H2 (major, by reading).** `"Report" > "Print"` never prints in the Stratimikos
  `"Strided Blocking"` route, which is the route an XML-only application uses, and
  `"Print"` exists for XML-only applications.
- **H3 (major).** A printed report can throw out of a rebuild and abort the simulation.
- **H4 (major).** A printed report pairs the previous solve's samples with the new
  operator.

H1 and H4 share one fix: the recorder holds the operator it recorded against.

## 2. High priority

### H1. The weak handle loses the operator when the operator is a temporary (blocker, verified by running)

[Teko_KrylovSurrogateReport.cpp:188](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateReport.cpp#L188)
stores `A.create_weak()`, so the handle is attached only while someone else holds the
blocked operator. Nothing in Teko's preconditioner path does:
`PreconditionerInverseFactory::buildInverse` keeps only the preconditioner, and
`PreconditionerFactory::initializePrec` never stores `A`. So the handle works only when
the application itself holds the exact `BlockedLinearOp` object the factory was given.
That object is a temporary in two ordinary routes:

- **Stratimikos with `"Reorder Type"`**:
  [Teko_StratimikosFactory.cpp:191](../../packages/teko/src/Teko_StratimikosFactory.cpp#L191)
  builds `blockedFwdOp = Teko::buildReorderedLinearOp(...)` locally and drops it.
- **Nested inside another Teko inverse**, for example as a merged group's inverse in
  Hierarchical Block Gauss-Seidel:
  [Teko_HierarchicalGaussSeidelPreconditionerFactory.cpp:250](../../packages/teko/src/Teko_HierarchicalGaussSeidelPreconditionerFactory.cpp#L250)
  passes `A_bb = extractSubblockMatrix(...)`, a fresh object. (By reading, same mechanism.)

**Reproduced.** `through_stratimikos` plus `teko.set("Reorder Type", "[0 1 2]")`, with the
`lows` and its preconditioner still in scope after the solve:

```
isAttached after a reordered Stratimikos solve: 0
Throw test that evaluated to true: !isAttached()
... the preconditioner (or its operator) this handle was attached to no longer exists ...
```

The same test without the reorder passes. The error message also misleads: it tells the
user the preconditioner was destroyed when it was not.

**Fix.** Give `RecorderOp` a strong reference to the blocked operator it records against,
set at build and replaced at rebuild, and have the handle hold a weak reference to the
recorder only. The weak-handle intent (2.4 in the first review: a parameter list copy
must not keep the samples alive) is kept, because the recorder still dies with the
preconditioner. The cost is keeping a blocked wrapper alive whose blocks the inner
inverse and the application already hold. `isAttached()` becomes "the recorder exists".
Add the reorder test above to the suite.

### H2. `"Print"` never fires in the Stratimikos strided-blocking route (major, by reading)

`"Print"` reports at a **rebuild**
([Teko_KrylovSurrogatePreconditionerFactory.cpp:127](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogatePreconditionerFactory.cpp#L127)).
In the Stratimikos multi-block branch
([Teko_StratimikosFactory.cpp:208-258](../../packages/teko/src/Teko_StratimikosFactory.cpp#L208-L258))
there is no rebuild. Every `initializePrec` builds a new `BlockedTpetraOperator` and a new
`InverseFactoryOperator`, and calls `buildInverseOperator`, which calls
`Teko::buildInverse` with a fresh state. The surrogate factory only ever takes its
first-build path, and each preconditioner, with its samples, is discarded unprinted.

That route (a monolithic Tpetra matrix, `"Strided Blocking"` in XML) is the normal one
for the XML-only applications `"Print"` was added for. The factory documentation's
promise ("a rebuild first builds and prints the report") is true only of Teko's own
`rebuildInverse`.

**Fix (recommended).** Print from the factory at **any** build, not only a rebuild. The
factory instance survives Stratimikos re-initializations (`invFactory_` is rebuilt only
when `startingOver`), and while the new preconditioner is being built, the old
`prec_Op` is still alive. So the factory keeps a weak reference `lastRecorder_` to the
last recorder it built or rebuilt. At each build it prints that recorder's samples if it
is alive and nonempty, then clears them. This is collective in the same way the current
rebuild print is. It also changes the documented meaning, from "the solves since the
previous build of this preconditioner" to "of this factory", which is what an XML user
can observe anyway. Test it through Stratimikos with a monolithic `Tpetra::CrsMatrix`
and `"Strided Blocking" = "1 1 1"`. The suite has no strided-route test at all today.

The alternative, printing from `~RecorderOp`, is worse: collective work in a destructor
hangs if ranks destroy in different orders, and a destructor cannot report a failure.

### H3. A printed report can throw out of a rebuild (major)

[Teko_KrylovSurrogatePreconditionerFactory.cpp:127-135](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogatePreconditionerFactory.cpp#L127-L135)
calls `buildReport()` inside the rebuild, after `rebuildInverse` and before
`clearSamples()` and `handle_->attach`. `buildReport` throws by design on NaN or infinite
samples (a solve that broke down), an all-zero block, a $\hat{b}$ with no component in
the trained subspace, or a GESVD failure. Any of these turns a diagnostic print into a
failed preconditioner rebuild. The worst case is the realistic one: a solve breaks down,
the application tries to recover by rebuilding (a smaller time step, a damped Newton
step), and the print kills the recovery. If the application catches the exception, the
recorder still holds stale samples and the handle points at the old state.

**Fix.** Wrap the print in `try { ... } catch (const std::exception& e)` and write
`e.what()` to Teko's stream in place of the report. This is safe on every rank only
because every throw inside `buildReport` is collective, which the first review's fixes
made true and should now be stated in a comment beside the catch. Then `clearSamples()`
and `attach` run unconditionally. Test: a recorder fed NaN samples, with `"Print"` on, and
`rebuildInverse` must not throw.

### H4. The printed report mixes the old samples with the new operator (major)

At a rebuild, `lo` is the **new** operator, and the printed report is built from
`own->attach(recorder, blo, ...)`
([line 129](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogatePreconditionerFactory.cpp#L129)),
so $\tilde{V}_j = A_{jj} Z_j$ and $\hat{C}_{ij} = \hat{Q}_i^T A_{ij} Y_j$ use the new
blocks with the previous solve's $Z$, and $\hat{b}$ uses the previous solve's $v_1$.
The result is well defined (the identity $A_{jj}^{-1}\tilde{V}_j = Z_j$ holds for any
$A_{jj}$), but it is a surrogate of the new operator on the old trial space, not the
report of "the solves since the previous build" that the header promises. In a Newton
loop the printed iteration counts are then compared with a solve on a different matrix.
The handle has the same issue in a weaker form: a report requested after the
application refills the matrix in place uses the refilled values. Every rebuild test
reuses the same `A` with the same values, so none can see it.

**Fix.** With H1, the recorder holds the operator it recorded against. Print from that
operator, not from `lo`. When the application refills the same object in place, the
old values are gone and nothing can recover them, so document in the header and in
`Handle::buildReport` that the report uses the operator's values at the time it is
built. Test: rebuild with a different operator object (scaled off-diagonal blocks), and
check that the printed $\hat{C}$ matches a handle report taken before the rebuild.

## 3. Other bugs

### B1. The sample store outlives a change of the operator's spaces (minor)

`clearSamples()` keeps `store_`
([Teko_KrylovSurrogateRecorderOp.cpp:80](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateRecorderOp.cpp#L80)),
and `record()` reuses it while it has room
([line 61](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateRecorderOp.cpp#L61)).
If a rebuild changes the operator's maps (repartitioning, adaptivity), new samples are
assigned into a multivector on the old maps. Thyra checks space compatibility only in
debug builds, and Tpetra's copy checks local lengths, so in release the assignment
either throws far from the cause or, at equal local lengths, silently mislabels rows.
**Fix:** in `record()`, reallocate when
`!store_->range()->isCompatible(*prec_->range())`. One line.

### B2. `Truncation Tolerance = 0` is accepted and keeps noise directions (minor)

The validator range is $[0, 1]$
([Teko_KrylovSurrogatePreconditionerFactory.cpp:52](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogatePreconditionerFactory.cpp#L52)).
With non-flexible restarted GMRES and a linear $M_0^{-1}$ (exact inner solves), every
restart records $M_0^{-1}(V y) = Z y$, a column exactly dependent on earlier ones. Its
singular value is roundoff, about $\varepsilon_{\text{mach}}\sigma_1$. At tolerance 0 it
is kept, $\Sigma_j^{-1}$ amplifies it by about $10^{16}$, and $Y_j$ is noise. The
$r_j \le n_j$ cap does not help, because this dependence has nothing to do with $n_j$.
The diagonal deviation would show it, but only to someone reading it. **Fix:** a
validator minimum of about $100\,\varepsilon_{\text{mach}}$, or document that 0 means
"keep roundoff".

### B3. The search's error trace exists only for candidates scored on rank 0 (minor)

Both the caught message
([Teko_KrylovSurrogateSearch.cpp:140](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateSearch.cpp#L140))
and Belos's `Errors` output
([line 83](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateSearch.cpp#L83))
go to `Teko::getOutputStream()`, which is Teuchos's default stream with
`setOutputToRootOnly(0)`. On $P$ ranks, the failures of the $(P-1)/P$ of candidates
dealt elsewhere leave no trace, contrary to the fixes plan's item 18. **Fix:** have
`Report::print` count `SOLVER_ERROR` candidates and say where messages appear. Gathering
the strings costs more than it buys.

### B4. The two construction paths disagree on whether a handle is required (minor)

`initializeFromParameterList` accepts no handle when `"Print"` is on, and creates a
dummy one
([line 160](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogatePreconditionerFactory.cpp#L160)).
The programmatic constructor, given a null handle and `printReport = true`, throws at
build
([line 96](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogatePreconditionerFactory.cpp#L96)).
**Fix:** let `handle_` be null in both paths, skip `attach` when it is, and require
"a handle or Print" once, in the build. The dummy handle then goes away. If H2 is taken,
the factory's own `lastRecorder_` makes it unnecessary in any case.

### B5. Nits

- `Handle` is copyable despite "no stack instances"
  ([Report.hpp:192](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateReport.hpp#L192)):
  `Handle h = *handle;` compiles. Delete the copy constructor and assignment.
- `"Tolerance" = 1` is accepted. Belos then converges at iteration 0, `rate` is set to
  0, and the candidate sorts as best. Use an upper bound below 1.

## 4. More idiomatic

None of these is a defect. They are the comments a Trilinos reviewer is likely to make,
best value first.

1. **Assemble $\hat{C}$ with dense views.** The index loops at
   [ReducedOperator.cpp:230-251](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateReducedOperator.cpp#L230-L251)
   become `DenseMatrix blk(Teuchos::View, report.CHat, ri, cj, offset[i], offset[j]);
   blk.assign(Cij);`, with the deviation taken from `Cij` before the identity is
   written. The same works for $\hat{b}$. It is shorter and states the block structure.
2. **`Tpetra::createLocalMapWithNode<LO, GO, NT>(k, comm)`** in place of the two
   hand-built `LocallyReplicated` maps (`localMV` at
   [line 42](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateReducedOperator.cpp#L42),
   `transposeProduct` at
   [line 55](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateReducedOperator.cpp#L55)).
3. **Cache the valid parameter list.** `validParameters()`
   ([Factory.cpp:40](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogatePreconditionerFactory.cpp#L40))
   rebuilds the list and four validators on every call. Belos and Ifpack2 build it once,
   in a function-local `static Teuchos::RCP<const Teuchos::ParameterList>`.
4. **`Report` as `Teuchos::Describable`.** `RecorderOp` already implements `describe`.
   Giving `Report` a `describe(FancyOStream&, EVerbosityLevel)` (LOW: header and best;
   MEDIUM: top 10, today's `print`; HIGH: every candidate; EXTREME: $\hat{C}$ too) is
   the Trilinos way to size output, and lets `"Print"` take a verbosity rather than a
   bool.
5. **Drop the const-view dance in the recorder.** Teko's `MultiVector` is an
   `RCP<MultiVectorBase>` to a non-const object, so
   `x->subView(Range1D(0, 0))->clone_mv()` works directly. The `cx` and `cz` copies at
   [RecorderOp.cpp:32](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateRecorderOp.cpp#L32)
   and in `record()` only select the const overload, and the comment "x arrives
   const-only" is not accurate.
6. **One feature guard in `buildReport`.** The `#else` branch with an "unreachable" throw
   could go if the guarded body moved into an `Impl::` function declared only with the
   feature. That matches how `scoreCandidates` is already guarded.
7. **The hand-written XML writer**
   ([Report.cpp:52](../../packages/teko/src/KrylovSurrogate/Teko_KrylovSurrogateReport.cpp#L52))
   is defensible, because `Teuchos::writeParameterListToXmlOStream` adds `id`,
   `docString`, `isDefault` and `isUsed` attributes nobody pastes by hand. Expect the
   question anyway. The answer is the comment that is already there, plus the round-trip
   test the fixes plan added.

Still declined, for the reasons in the fixes plan section 7: `typedef` to `using`,
`override`, member timers, the class-name stutter.

## 5. Test gaps

In priority order. The first four would each have caught one of section 2's findings.

1. Stratimikos with `"Reorder Type"`, handle report after the solve (H1). The scratch
   test is ready to paste.
2. Stratimikos with a monolithic `Tpetra::CrsMatrix` and `"Strided Blocking"`, with both a
   handle and `"Print"`, across two `initializePrec` calls (H2).
3. `"Print"` on with a degenerate report (NaN samples), and `rebuildInverse` must not
   throw (H3).
4. A rebuild with a different operator, with the printed $\hat{C}$ compared to a report
   taken before the rebuild (H4).
5. `report.toParameterList()` written with `writeParameterListToXmlString` and read back.
   The converters exist (`Array<long long>` and `TwoDArray<double>` are both
   registered, checked), so this is cheap insurance for the "for logging" promise.

## 6. Checked and fine

These looked suspicious and are not. They are listed so a later pass does not redo them.

- **Inverse rebuilt in place.** `recorder->getModifiableOp()` returns a copy of the
  `RCP`, and `Teko::rebuildInverse` updates the pointee; it does not reassign. Both
  `PreconditionerInverseFactory` and `SolveInverseFactory` rebuild in place, and
  `DiagnosticPreconditionerFactory` relies on the same pattern.
- **Reusing one `BlockGmresSolMgr`.** `setProblem` resets the status tests and the
  iteration (`isSTSet_ = false; needsIterRebuild_ = true`), so scoring candidates in
  sequence is safe.
- **NaN through the gather.** A failed candidate's NaN fields are summed with zeros from
  every other rank, and NaN survives the sum, so the sorted result is the same on every
  rank.
- **Strided route and H1.** `InverseFactoryOperator` keeps `fwdOp_`, which keeps the
  Thyra blocked operator alive, so H1 does not occur there (H2 does).
- **`TpetraCore_ENABLE_TSQR` in `COMBINED_OPTION`.** It is in TpetraCore's exported
  package variables, so it is defined at Teko's configure, including against an
  installed Tpetra.
- **Reproducibility of sampled candidates.** `std::discrete_distribution` and
  `std::shuffle` are not specified bit-for-bit across standard libraries, so a seed
  reproduces only within one toolchain. Every rank of one run shares a toolchain, and no
  test hardcodes a sampled sequence, so nothing depends on it. Worth one line in the
  `"Seed"` documentation.

## 7. The wiki disagrees with the code

`wiki/projects/teko-surrogate/gram-matrix-svd-floor.md` says there is "no explicit clamp
$r_j \leftarrow \min(r_j, n_j)$". `097f36f0b90` added it (`reduceOnRankZero`'s `maxSize`,
passed $n_j$ from `buildReducedOperator`). The page is stale. That is an `/ingest` job.

## 8. Suggested order

1. H1 and H4 together: the recorder holds its operator, the handle holds a weak
   reference to the recorder, the print uses the recorder's operator. Add tests 1 and 4.
2. H3: catch around the print, then clear and attach unconditionally. Add test 3.
3. H2: print at any build from the factory's last recorder, then rewrite the header's
   description of `"Print"`. Add test 2. This is the largest change and touches the
   documented meaning, so it needs an owner decision before it is built.
4. B1, B2, B4 (a few lines each), then the idioms in section 4 that are cheap (1 to 3, 5).
