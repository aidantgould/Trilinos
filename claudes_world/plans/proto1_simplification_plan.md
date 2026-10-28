# Plan: simplify proto1

**Status: built 2026-10-28** (see As built at the end). Applied on `aidantgould/teko-auto-proto1`. These are only
the changes recommended in the proto1 review. Each is meant to make the code shorter or
clearer at no meaningful performance cost.

## Changes

1. **Belos decides convergence when scoring candidates** (`Teko_KrylovSurrogateSearch.cpp`).
   - Status is `converged` when `solve()` returns `Belos::Converged`, and
     `max iterations` otherwise. `finalRelativeResidual` is `achievedTol()`.
   - Removes `kResidualSlack`, the copy of `y` out of Thyra, and the residual matvec.
   - Why this is safe here: the GMRES is unpreconditioned on a fixed dense $K$, so
     Belos's implicit residual is reliable. The bias the hack guarded against needs a
     flexible solve with a changing preconditioner (wiki `belos-residual-verification`).
2. **No duplicate struct.** `ReducedOperator` goes away, and
   `buildReducedOperator(A, Z, v1, tol, report)` fills the report's fields directly.
3. **`ranks` becomes `basisSizes`** everywhere it means $r_j$: the report field,
   function parameters, `BlockReduction::rank` (as `size`), and the pyfront dict key
   `basis_sizes`. MPI rank names are unchanged.
4. **One availability guard.** A `TEKO_HAVE_KRYLOV_SURROGATE` macro, defined in one
   header when both `HAVE_TPETRA_TSQR` and `Teko_ENABLE_Belos` hold. It wraps the
   reduced operator and the search bodies, and the factory keeps the single
   user-facing error.
5. **`Teuchos::SerialDenseSolver::invert()`** replaces the hand-written
   `GETRF`/`GETRI`. A nonzero return from `factor()` still means
   `singular group block`.
6. **Delete `operator<<`** for the handle. Teuchos prints any `RCP` itself.
7. **Standard parameter validation.** `getValidParameters()` lists every option with
   its default and documentation. `initializeFromParameterList` calls
   `validateParameters` on a copy, then reads the values. `"Orderings"` disables
   recursive validation, since its entry names are arbitrary. `checkNames` goes away.
8. **Smaller items.**
   - Fetch every Tpetra block into a grid once in `buildReducedOperator`.
   - Fold `tKrylovSurrogateSearch.cpp` into the parallel test executable, which is
     registered at 1 to 4 ranks: one fewer CMake target, the same tests.

## Tests

- No new tests: every existing one must still pass.
- `tKrylovSurrogateFactory`'s misspelled-option test keeps its intent, but the
  exception type may change with `validateParameters`.
- Run the full Teko suite and the pyfront `test_teko_auto.py` at 1 and 4 ranks.

## Logistics (decided 2026-10-28)

- Commit the "Lessons from proto1" section on `teko-auto`, then add the same section to
  proto1's plan, after its "As built" section.
- Rename the pyfront dict key to `basis_sizes` on its `teko-auto` branch, in the same
  pass.
- Commit on proto1 and pyfront `teko-auto`. Don't push.

## Not doing

- Keeping, not removing, `toParameterList()`, the general `alpha`/`beta` path in the
  recorder, the uniform random draw of orderings, and the copy of $K$ into a Thyra
  multivector.
- No behaviour changes beyond item 1's reported residual.

## As built

All eight items applied as planned.

**Results.**

- The library went from 1446 to 1391 lines. Item 7's documented valid-parameter list
  added about 25 of those, so the other items removed about 80.
- Feature guards went from 7 sites to 4 (factory, report, reduced operator, search),
  all on the one macro defined in `Teko_KrylovSurrogateReport.hpp`.
- Teko's suite: 24 tests, all passing. That is 15 stock tests and 9 surrogate ones,
  one fewer than before because the search and parallel tests now share
  `Teko_KrylovSurrogateSearch`, which runs all 9 of its tests at 1 to 4 ranks.
- End-to-end numbers are unchanged: 8 samples, basis sizes 8 8 8, diagonal
  deviations up to 3e-14, a warm-start first input within 3e-17 of r0/||r0||.
- The pyfront test passes 16/16 at 1 and 4 ranks, with the dict key `basis_sizes`,
  and the demo runs.

**Beyond the plan.**

- The surrogate tests' CMake registration is wrapped in
  `IF(TpetraCore_ENABLE_TSQR)`, since they cannot link without the feature.
- `buildReport()` has an `#else` branch returning an empty report. It is
  unreachable, because without the feature the factory throws before any handle is
  attached.
- A misspelled option now throws `Teuchos::Exceptions::InvalidParameter`, from
  `validateParametersAndSetDefaults`, instead of `std::invalid_argument`. The test
  checks the new type.
