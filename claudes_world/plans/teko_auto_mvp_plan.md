# Plan: Krylov surrogate report in Teko (MVP, production)

**Status:** proposed 2026-10-27, revised 2026-10-28, nothing built.
**Branch:** `aidantgould/teko-auto`, cut from upstream `develop` at `0cf62aaf091`.
This plan lives in the branch's `claudes_world/`, which is stripped before any upstream
PR. Nothing from the research tree (`teko-reconfig/` on the hack branch) is carried over.

## 1. Goal

A user who solves a blocked system with flexible GMRES and a Teko block preconditioner
turns on one option and, after the solve, gets back a **surrogate report**:

- the reduced operator $\hat{\mathcal{C}}$ and reduced right-hand side $\hat{b}$, built
  from that solve's own preconditioner applies;
- a list of alternative block configurations (grouping and ordering of the physics
  blocks), each with how it converged **on the surrogate**: iterations, whether it
  converged, its final residual, and its average per-iteration residual reduction.

That is all. The MVP does not act on the report: no re-solve, no reconfigured
preconditioner, no automatic pick. It is the measurement half of the research hack,
rebuilt to go upstream into Trilinos and kept as simple as possible.

What makes it production code rather than the hack:

- **Self-contained in Teko.** No Belos or Tpetra changes, no Python, no files, no
  environment variables. Opt-in is a Teko parameter list entry, and retrieval works
  however the preconditioner is built, Stratimikos included.
- **Free when off.** The feature is a preconditioner type, so a user who doesn't name
  it runs none of it.
- **One numerical path for the bases.** Tpetra's tall-skinny QR replaces the hack's Gram
  matrix, removing its $4.21\times10^{-8}$ resolution floor so the paper's $10^{-10}$
  truncation rule is reachable.
- **No new Krylov solver and no adapter code.** Candidates are scored with Belos GMRES
  through Belos's existing Thyra adapter.
- **Parallel search.** Candidates are dealt across MPI ranks, and the report is bitwise
  identical at any rank count. (The hack scored every candidate in one Python process.)
- **Warm starts handled** with no extra input.
- **Tested** in Teko's normal TriBITS suite, serial and MPI.

## 2. Where everything lives

All new code is in `packages/teko/`, whose CMake globs `src/*.cpp` and `src/*.hpp`, so no
library CMake edits are needed. Tests are registered in `packages/teko/tests/CMakeLists.txt`.

| Piece | File | Role |
| --- | --- | --- |
| The option | `src/Teko_KrylovSurrogatePreconditionerFactory.{hpp,cpp}` | Teko type `"Krylov Surrogate"`, wrapping any other Teko inverse as `"Diagnostic Inverse"` does |
| Sample capture | `src/Teko_KrylovSurrogateRecorderOp.{hpp,cpp}` | Thyra wrapper that applies the real preconditioner and keeps its outputs and its first input |
| Surrogate construction | `src/Teko_KrylovSurrogateReducedOperator.{hpp,cpp}` | $\hat{\mathcal{C}}$ and $\hat{b}$ in plain Tpetra: padded TSQR and a small SVD on rank 0 |
| Surrogate scoring | `src/Teko_KrylovSurrogateSearch.{hpp,cpp}` | Candidate configurations, explicit $K = \hat{\mathcal{C}}[\hat{\mathcal{C}}]_{S_1}^{-1}$, Belos GMRES on $K$, round-robin dealing over MPI ranks |
| Report and retrieval | `src/Teko_KrylovSurrogateReport.{hpp,cpp}` | Report struct, `toParameterList()`, and the handle passed in the parameter list |
| Tests | `tests/unit_tests/tKrylovSurrogate*.cpp` | Dense math, recorder, rank invariance, end to end (direct and Stratimikos) |
| Parity checks (not shipped) | `claudes_world/teko_auto_parity/` | Comparisons against the hack and pyautoteko |

## 3. Detailed plan

**Notation.** $N$ physics blocks; block $j$ has $n_j$ rows. $m$ is the number of recorded
samples (at most `Maximum Samples`). $Z_j$ is block $j$ of the recorded preconditioner
outputs ($n_j \times m$). $r_j$ is the number of directions block $j$ retains, and
$R = \sum_j r_j$ is the size of $\hat{\mathcal{C}}$. $P$ is the number of MPI ranks. All
matrix quantities are block matrices: inverses are explicit and products are never
reordered.

### 3.1 Key decision: record at the preconditioner, don't hook Belos

The hack's `computeCHat` never uses the FGMRES basis $V$. It needs only $Z$, the blocks of
$\mathcal{A}$, and the right-hand side, because defining $\tilde{V}_j := A_{jj}Z_j$ (wiki
`vtilde-from-z`) makes $A_{jj}^{-1}\tilde{V}_j = Z_j$ exact for **any** $Z$. So nothing
has to come out of Belos. A thin operator wrapped around the preconditioner sees every
$z = \mathcal{M}_0^{-1}v$ as it is produced, and that is the whole harvest.

- No Belos hook, no `BlockGmresSolMgr` edits, no type-erased slot, no static
  registration problems: the hardest-to-upstream part of the hack disappears.
- The Thyra and Tpetra paths behave the same, since Teko's `TpetraOperatorWrapper`
  applies the same Thyra preconditioner underneath.
- Samples from every restart cycle accumulate (sampling-agnostic lemma, wiki
  `restarted-gmres`), not only the last cycle's.
- Every recorded apply is a valid sample. Applies outside the Krylov solve only make
  the subspace less informative, never wrong.

The solve itself must be flexible GMRES, because inexact inner solves make
$\mathcal{M}_0$ non-stationary. It must also be right-preconditioned (Belos's default),
which the right-hand side capture in 3.3 relies on.

### 3.2 User interface

**The option**, in a Teko inverse library parameter list. The wrapped inverse is the
preconditioner actually applied, unchanged:

```xml
<ParameterList name="Surrogate GS">
  <Parameter name="Type"                 type="string" value="Krylov Surrogate"/>
  <Parameter name="Inverse Factory"      type="string" value="GS"/>    <!-- any Teko block inverse -->
  <Parameter name="Maximum Samples"      type="int"    value="100"/>   <!-- columns of Z kept -->
  <Parameter name="Truncation Tolerance" type="double" value="1e-10"/> <!-- paper eq 3.10 -->
  <ParameterList name="Search">
    <Parameter name="Tolerance"       type="double" value="1e-8"/>  <!-- relative to ||b_hat|| -->
    <Parameter name="Candidate Count" type="int"    value="100"/>
    <Parameter name="Seed"            type="int"    value="0"/>
    <ParameterList name="Orderings">                                <!-- optional, always scored -->
      <Parameter name="merge 0 and 1" type="Array(int)" value="{0,0,1}"/>
    </ParameterList>
  </ParameterList>
</ParameterList>
```

**Retrieval, through a handle in the parameter list.** Most Sandia applications build
Teko through Stratimikos (often via Panzer) and never hold the preconditioner object, so
retrieval can't depend on holding it. The user creates a handle, puts it in the
sublist, and asks it for the report after the solve. Teuchos parameter lists carry
`RCP`s of any type, and copying a list copies the `RCP`, so the handle reaches the
factory by whatever route the list takes:

```cpp
auto handle = Teko::KrylovSurrogateHandle::create();
tekoInverseLibraryParams.sublist("Surrogate GS").set("Report Handle", handle);
// build however the application does (Teko::buildInverse, a Teko preconditioner handed
// to Belos, or Stratimikos with Teko as the preconditioner), then solve with Belos
// flexible GMRES, right preconditioned
Teko::KrylovSurrogateReport report = handle->buildReport();
auto pl = report.toParameterList();   // for logging, XML output, or a Python front end
```

- On every build and rebuild, the factory attaches its recorder, the blocked
  $\mathcal{A}$ and the options to the handle, so the handle always refers to the
  preconditioner most recently built from that sublist.
- `buildReport()` takes no vectors (3.3 explains the right-hand side). It is collective:
  every rank calls it and every rank gets the same report.
- Errors, each saying what to do: no `"Report Handle"` in the sublist, a non-blocked
  operator, or a block not backed by Tpetra (all at build time), and `buildReport()`
  before anything was built or recorded.
- To confirm while building: that Stratimikos and Teko's `StratimikosFactory` pass a
  non-XML entry through unchanged. The end-to-end tests cover this path (3.8).
- The cost of this design: an application configured purely from XML must add the two
  lines above. A lookup by inverse name would avoid that, but it is global state, so it
  waits until someone asks for it.

**The report and the handle** (`Teko_KrylovSurrogateReport.hpp`):

```cpp
struct KrylovSurrogateReport {
  int numBlocks, numSamples;
  std::vector<int> blockSizes, ranks;            // ranks[j] = r_j
  Teuchos::SerialDenseMatrix<int, double> CHat;  // R x R, block j at offset r_0 + ... + r_{j-1}
  Teuchos::SerialDenseVector<int, double> bHat;  // length R, projection of the initial residual direction
  double rhsCapture;                             // ||b_hat|| / ||r0 direction||, at most 1
  std::vector<double> diagonalDeviations;        // per block: max |C_hat_jj - I| as computed (3.4)
  struct Candidate {
    std::vector<int> ordering;                   // ordering[k] = position of block k's group
    std::string status;                          // "converged", "max iterations", "singular group block", "solver error"
    int iterations;                              // k
    double finalRelativeResidual;                // ||r_k|| / ||r_0||, recomputed (3.5)
    double rate;                                 // finalRelativeResidual^(1/k)
  };
  std::vector<Candidate> candidates;             // converged first, then fewest iterations, then best rate
  double searchWallTime;                         // cross-rank maximum
  int searchRanks;
  Teuchos::RCP<Teuchos::ParameterList> toParameterList() const;
};

class KrylovSurrogateHandle {                    // what the user puts in the parameter list
 public:
  static Teuchos::RCP<KrylovSurrogateHandle> create();
  KrylovSurrogateReport buildReport() const;     // collective
  // attach() is called by the factory; not for users
};
```

`rhsCapture` costs nothing and is the one honest warning the MVP can give. Under a
Gauss-Seidel baseline the right-hand side need not lie in the trained subspace (wiki
`gauss-seidel-rhs-and-basis`), and a small value says $\hat{b}$ misses much of it.

### 3.3 Sample recorder (`Teko_KrylovSurrogateRecorderOp`)

- Subclass `Teko::ImplicitLinearOp`, following `DiagnosticLinearOp`, and stay pure
  Thyra. It holds the wrapped preconditioner, one preallocated product multivector on
  the preconditioner's range space with `Maximum Samples` columns, and a fill count.
  Tpetra appears only in 3.4.
- `implicitApply(X, Y, alpha, beta)` applies the wrapped operator. With `alpha == 1` and
  `beta == 0` (what Belos does), it copies `Y`'s columns into the store directly.
  Otherwise it applies into a temporary first, so the store holds $\mathcal{M}_0^{-1}X$
  and not the scaled update. A multi-column apply appends every column.
- When the store is full, recording stops: the first `Maximum Samples` columns are kept.
  That is the simplest correct policy. Streaming compression and eviction (wiki
  `eviction-policy`) are later work.
- The factory calls `clearSamples()` on every rebuild, so a report describes the
  current operator. Repeated solves without a rebuild accumulate samples, which is
  documented and harmless.
- Memory is `Maximum Samples` + 1 full-length vectors, allocated on the first apply.
  The parameter documentation says so, since it is the main cost of turning this on.

**Right-hand side, warm starts included.** The recorder also keeps the first input
column it sees after a clear. In right-preconditioned GMRES the first preconditioner
apply is on $v_1 = r_0/\|r_0\|$, where $r_0 = b - \mathcal{A}x_0$. So $v_1$ is the
initial residual's direction: $b$'s for a cold start, $r_0$'s for a warm one. The
surrogate treats every solve as warm. It solves $\hat{\mathcal{C}}\hat{x} = \hat{b}$ from
$\hat{x}_0 = 0$, with $\hat{b}$ projected from $v_1$, which is exactly the correction
problem the real solve faced. Scale doesn't matter, since every count and rate is
relative to $\|\hat{b}\|$. The cost is one extra vector and no API. The handle documents
what this relies on: right preconditioning, and the report describing the first solve
after a rebuild.

### 3.4 Reduced operator (`Teko_KrylovSurrogateReducedOperator`)

The math is the hack's `computeCHat` (`Teko_KrylovReducedModel.hpp` on the
`teko-reconfig-request` branch). What changes is how each block's SVD is computed, and
that the whole construction is plain Tpetra.

**Tpetra only.** Teko requires Tpetra, so at report time each block $A_{ij}$ is extracted
once as a `Tpetra::Operator`, and each recorded $Z_j$ and $v_{1,j}$ ($v_{1,j}$ is block $j$
of the recorded first input) once as a `Tpetra::MultiVector`, through
`Thyra::TpetraOperatorVectorExtraction`. Everything after that is `apply`,
`MultiVector::multiply` (with `TRANS`, into a locally replicated multivector, for the
small products) and TSQR, with no Thyra in between. A block that isn't backed by Tpetra
is an error at build time (3.2).

**The algorithm.** For each block $j$:

1. Compute $\tilde{V}_j = A_{jj}Z_j$ (a sparse apply on $m$ columns) straight into the top
   rows of a TSQR work multivector with $\max(n_{\text{local}}, m)$ rows per rank, whose
   remaining rows are zero (padding, below).
2. TSQR on the work multivector: $Q_j$, $R_j$.
3. `reduceOnRankZero(R_j, tol)`: SVD $R_j = U_j\Sigma_jW_j^T$, truncate, broadcast (below).
4. $\hat{Q}_j = Q_jU_{j,r}$, a local GEMM with no communication, keeping only the top
   $n_{\text{local}}$ rows. Free $Q_j$.
5. $Y_j = Z_jW_{j,r}\Sigma_{j,r}^{-1}$, a local GEMM and column scaling. Mathematically
   $Y_j = A_{jj}^{-1}\hat{Q}_j$.

Then, across blocks:

- $\hat{b}_j = \hat{Q}_j^T v_{1,j}$, one reduction.
- $\hat{C}_{ij} = \hat{Q}_i^T(A_{ij}Y_j)$, a sparse apply on $r_j$ columns and one
  reduction. A null $A_{ij}$ is a zero block. Every block is applied as a
  `Tpetra::Operator`, so the hack's silent zeroing of a block that isn't a CRS matrix
  goes away.
- $\hat{C}_{jj}$ is **computed** by the same loop, not set, and $\max|\hat{C}_{jj} - I|$
  goes into `diagonalDeviations`. Theory says it is exactly $I$, so the deviation
  measures the roundoff actually incurred: the free acceptance test that
  `gram-matrix-svd-floor` notes the hack throws away. The assembled $\hat{\mathcal{C}}$
  then gets exactly $I$ on its diagonal blocks, as in the paper.
- Rank 0 assembles one $R \times R$ `SerialDenseMatrix` and broadcasts it with $\hat{b}$
  (at most a few MB). MPI doesn't promise that an `Allreduce` returns identical bits on
  every rank, and the bitwise-identical report depends on identical inputs.

**Why TSQR instead of the Gram matrix.** The hack takes the SVD of $\tilde{V}_j$ from
$G_j = \tilde{V}_j^T\tilde{V}_j$, which squares the condition number. A direction at
ratio $t = \sigma/\sigma_{\max}$ appears in $G_j$ at $t^2$, so nothing below
$\sqrt{8\varepsilon_{\text{mach}}} = 4.21\times10^{-8}$ can be resolved, and the paper's
$10^{-10}$ rule is unreachable (wiki `gram-matrix-svd-floor`, which names this fix).
TSQR factors $\tilde{V}_j$ itself and resolves singular values down to about
$m\,\varepsilon_{\text{mach}}$. It also lets $\hat{b}$ be computed without a
$\Sigma^{-1}$, so $\hat{b}$ carries no amplified roundoff.

**The Trilinos call.** `Tpetra::TsqrAdaptor<Tpetra::MultiVector<ST, LO, GO, NT>>`
(`packages/tpetra/core/src/Tpetra_TsqrAdaptor.hpp`), with Teko's `ST`, `LO`, `GO`, `NT`
typedefs.

- `factorExplicit(A, Q, R)` uses one communication round (a reduction tree), returns an
  explicit distributed $Q$ ($n_j \times m$), and replicates $R$ ($m \times m$) on every
  rank (`Tsqr.hpp`, `factorExplicitRaw`). It overwrites $A$, which is fine because the
  work multivector isn't needed afterwards. Use one adaptor per block, since an adaptor
  is tied to the distribution of the first multivector it sees.
- TSQR is compiled in when Tpetra is configured with it (`HAVE_TPETRA_TSQR`), which is
  the default whenever the TpetraTSQR subpackage is enabled. The feature requires it.
  Without it, `"Krylov Surrogate"` is still registered, but building it throws a message
  saying Tpetra must be configured with TSQR.
- `revealRank` from the same adaptor isn't used. It acts only when $R$ is rank-deficient,
  doesn't return $\Sigma$ and $W$ separately, and runs the SVD on every rank. Belos's
  `TsqrOrthoManager` wraps the same adaptor for Krylov orthogonalization and adds
  nothing here.

**Zero-row padding, always.** TSQR throws if any rank holds fewer local rows of the
block than there are columns (`factorExplicitRaw`: "input matrix A has N local rows, and
M columns"), including a rank holding none. That happens for a block spread unevenly (a
field living on a subregion) and for a block smaller than the sample count (a 0-D
circuit field, a handful of multipliers). So every block is padded, with no branch:

- The work multivector lives on a contiguous map with $\max(n_{\text{local}}, m)$ local
  rows per rank. Step 1 applies $A_{jj}$ into a view of its top $n_{\text{local}}$ rows
  (`offsetViewNonConst` onto the block's own map, to confirm while building), so there
  is no copy in. On a rank with at least $m$ rows there are no padded rows at all.
- Zero rows leave $R$ unchanged, since
  $\begin{bmatrix}\tilde{V}_j \\ 0\end{bmatrix} = \begin{bmatrix}Q \\ 0\end{bmatrix}R$. The
  kept directions of $Q$ lie in the padded matrix's range, so their padded rows are zero
  to roundoff, and step 4 drops them.
- A block with fewer rows than samples needs nothing more: $R$ is still $m \times m$, its
  SVD shows rank at most $n_j$, and truncation discards the rest.
- Cost: no communication beyond TSQR's own, and at most $m^2$ extra doubles (about
  80 KB at $m = 100$) on a short rank.

**The small SVD runs on rank 0 and is broadcast.** The TSQR author's own FIXME in
`revealRankRaw`: "We _should_ compute the SVD of R ... on Proc 0 only. This would ensure
that all processors get the same SVD and rank (esp. in a heterogeneous computing
environment). For now, we just do this computation redundantly, and hope that all the
returned rank values are the same." For us a disagreement is worse than a small
inaccuracy: ranks that disagree on $r_j$ size their next collective differently, and the
job hangs or silently mixes data. So in `reduceOnRankZero`, rank 0 runs `GESVD` on $R_j$,
truncates at $\sigma_k \ge \text{tol}\cdot\sigma_1$ keeping at least one direction, and
broadcasts $r_j$, $U_{j,r}$, $\Sigma_{j,r}$, $W_{j,r}$ (at most $m \times m$ each, about
80 KB per block at $m = 100$).

**Error that remains.** $\hat{Q}_j$ is orthonormal to machine precision, so $\hat{b}$ and
the left side of each $\hat{C}_{ij}$ are clean. What remains is in $Y_j$: a direction
kept at ratio $t$ carries about $\varepsilon_{\text{mach}}/t$ relative error in its
column, so at the $10^{-10}$ default the weakest kept column can be off by about
$10^{-6}$. `diagonalDeviations` reports how large it actually was.

**Cost and memory.** Building $\hat{\mathcal{C}}$ costs about one full-operator apply per
retained direction ($R$ columns through the block columns of $\mathcal{A}$). That is
roughly the solve's own matrix-apply count, with none of its preconditioner work, the
same as the hack paid. At peak, block $j$ holds about $3m$ vectors of its length ($Z_j$,
the work multivector, $Q_j$). Kept until the end: $\hat{Q}_j$ and $Y_j$ for every block,
$2R$ vectors in all. Both go in the parameter documentation next to `Maximum Samples`.

### 3.5 Surrogate scoring (`Teko_KrylovSurrogateSearch`)

**Candidates.** An ordering assigns each block $k$ the position of its group, so
candidates are ordered set partitions, as in pyautoteko's `random_search.py`: 1, 3, 13,
75, 541 and 4683 of them for 1 to 6 blocks.

- `numConfigurations(n)`, `allConfigurations(n)` and `randomConfiguration(n, rng)` port
  `num_configs`, `all_configs` and `random_ordering` (the uniform draw, group order
  included).
- The set is the user's `Orderings`, the identity and all-merged extremes, and then
  either every configuration (when there are at most `Candidate Count`) or a uniform
  sample up to `Candidate Count`, without duplicates. Sampling uses one
  `std::mt19937_64` seeded with `Seed`. The set is cheap, so every rank builds the same
  one rather than distributing it.

**Scoring one candidate: form the preconditioned matrix explicitly.**

1. Copy $\hat{\mathcal{C}}$ and zero the blocks Gauss-Seidel drops: keep block $(i, j)$
   when block $i$'s group comes at or after block $j$'s group, zero it otherwise. That
   gives $[\hat{\mathcal{C}}]_{S_1}$: lower block Gauss-Seidel in group order with each
   group kept whole, matching pyautoteko and Teko's `"Block Gauss-Seidel"` default. No
   permutation is needed. A reordering is an orthogonal similarity, under which GMRES
   with a consistently permuted $\hat{b}$ produces the same residual norms.
2. Invert it with `GETRF` then `GETRI`. Because $[\hat{\mathcal{C}}]_{S_1}$ is block
   triangular over the groups and a singleton group's block is $I$, it is singular
   exactly when some merged group's block is. A zero pivot gives status
   `"singular group block"`, not an exception (wiki `selection-invertibility`: it can
   genuinely happen).
3. Multiply, with one `GEMM`: $K = \hat{\mathcal{C}}[\hat{\mathcal{C}}]_{S_1}^{-1}$.
4. Run unpreconditioned GMRES on $Ky = \hat{b}$ from $y_0 = 0$. Its residual equals that of
   $\hat{\mathcal{C}}\hat{x} = \hat{b}$ at $\hat{x} = [\hat{\mathcal{C}}]_{S_1}^{-1}y$, so the
   iteration count is the right-preconditioned one.

**The GMRES is Belos, over Thyra's serial vectors.** No adapter code:

- Vectors and the operator live on `Thyra::defaultSpmdVectorSpace<double>(R)` with a
  serial communicator: plain host memory with BLAS underneath, so a GPU build launches
  no device kernels. A Thyra multivector is a linear operator, so $K$ is stored as an
  $R \times R$ multivector on that space (filled from the `SerialDenseMatrix`) and handed
  to Belos as the operator directly. $\hat{b}$ and $y$ are vectors on the same space.
- Belos runs on them through its existing Thyra adapter
  (`packages/stratimikos/adapters/belos/src/BelosThyraAdapter.hpp`; Teko already requires
  Stratimikos).
- To confirm while building: that Belos accepts the multivector's domain space as
  matching its range space. Thyra compares serial spaces by a map code
  (`Thyra_SpmdVectorSpaceDefaultBase_def.hpp:95`), so two of the same size should match.
  If they don't, a 20-line Thyra operator wrapping $K$ fixes it.
- `Belos::BlockGmresSolMgr`, not flexible, no preconditioner. `Num Blocks` and
  `Maximum Iterations` are both $R$, which exact arithmetic never needs to exceed.
  `Convergence Tolerance` is `Tolerance`, scaled by the initial residual, which from
  $y_0 = 0$ is $\|\hat{b}\|$.
- Afterwards, recompute the final relative residual as $\|\hat{b} - Ky\|/\|\hat{b}\|$ with
  one matvec. It decides the status, so the verdict never rests on Belos's implicit
  estimate (wiki `belos-residual-verification`), and it gives the rate
  $(\|r_k\|/\|r_0\|)^{1/k}$ from the iteration count alone. No residual history is kept.
- A Belos exception on one candidate becomes that candidate's `"solver error"`, not a
  failed report.
- Each rank builds **one** solver manager, with output to a null stream, and reuses it
  through `setProblem` for every candidate. Belos registers timers when the manager is
  built, so the application's timer summary gains one `"Teko Krylov Surrogate"` line,
  not one per candidate.

**Cost.** About $4R^3$ flops per candidate (`GETRF`, `GETRI`, `GEMM`) plus one dense
matvec per GMRES iteration. Measured 2026-10-28 on one core against the Netlib reference
BLAS and LAPACK that `trilinos-build` links (scratch benchmark `kbench.cpp`), which ran
at 6.7 GFLOP/s for forming $K$ and 4.4 GFLOP/s for matvecs:

| $R$ | Form $K$ | One matvec | Per candidate, explicit $K$ | Per candidate, forward substitution (modelled) |
| --- | --- | --- | --- | --- |
| 150 | 2.1 ms | 11 µs | about 3 ms ($k = 30$) | about 1 ms |
| 300 | 15.8 ms | 41 µs | about 20 ms ($k = 50$) | about 5 ms |
| 500 | 74 ms | 112 µs | about 82 ms ($k = 50$) | about 11 ms |
| 1000 | 603 ms | 443 µs | about 660 ms ($k = 100$) | about 75 ms |

The last column models the performance-note route below from the same kernel rates: 1.5
matvecs per iteration, merged-group factors at $\sum_g n_g^3$, and an assumed 20 µs per
iteration of Belos overhead shared by both routes. Forming $K$ is about 3 to 9 times
slower per candidate, roughly $R/k$, and the gap grows with $R$. For a 100-candidate
search that is about 2 s against 0.5 s on one rank at $R = 300$ (0.12 s against 0.03 s
on 16 ranks), negligible next to the solve and the construction of $\hat{\mathcal{C}}$.
At $R = 1000$ it is about 66 s against 8 s on one rank (4 s against 0.5 s on 16 ranks),
which is noticeable, and is reached only at 10 blocks with 100 samples each. An
optimized BLAS speeds up the matrix-matrix kernels far more than matvecs, so it should
narrow the gap to roughly 1.5 to 3 times (estimated, not measured: no optimized BLAS is
installed here).

**No jitter, for now.** The hack scores candidates with pyautoteko's jittered FGMRES, a
fresh random perturbation on every group solve, because without it the surrogate tells
orderings apart poorly. pyautoteko measured this (`fgmres.py`, the `solve_reduced`
docstring). Unperturbed, orderings needing 54 to 100 real iterations all converged within
1 or 2 surrogate iterations of each other. Pooled over 20 seeds, the correlation with real
iteration counts was 0.858 unperturbed, 0.872 with a fixed perturbation, and 0.955 with
fresh jitter. A fixed explicit $K$ can carry only a fixed perturbation, which buys almost
nothing, so the MVP scores unperturbed and its rankings should be expected to be weaker
than the hack's. 3.9 measures how much weaker on real solves.

**Performance note: the better way, at the cost of more code.** Don't form $K$. Apply
$[\hat{\mathcal{C}}]_{S_1}^{-1}$ as a Belos right preconditioner by block forward
substitution over the groups, factoring each merged group's block once (singleton
groups are free, their block being $I$) and caching factors across candidates that share
a group. Per candidate that costs $\sum_g n_g^3$ ($n_g$ the group's reduced size) plus
about $R^2$ per iteration, instead of $4R^3$. That operator is also where fresh jitter
goes back in, with flexible GMRES, so this one upgrade both removes the $R^3$ cost and
restores the 0.955 correlation.

**Parallel search.** The search is the only part of the report whose cost grows with the
number of candidates, and without parallelism every rank would repeat the same work for
every candidate. Scoring is deterministic and depends only on $\hat{\mathcal{C}}$,
$\hat{b}$, the ordering and the options, so any rank can score any candidate and get the
same answer:

- **Communicator:** the one under the blocked operator's range space, so a solve on a
  subcommunicator searches on that subcommunicator. A serial or non-SPMD space is one
  rank.
- **Dealing:** candidate $i$ goes to rank $i \bmod P$. Every rank computes the same
  assignment from the same list, with no communication, then scores its candidates one
  after another.
- **Gather:** one `Teuchos::reduceAll` (sum) over a flat buffer of four doubles per
  candidate (status code, iterations, final residual, rate). Each slot is written by
  exactly one rank and is zero everywhere else, so the sum is exact and the report is
  bitwise identical on every rank and at every rank count.
- **More ranks than candidates:** the extra ranks score nothing and just join the
  reduction.

**Why no threads within a rank.** Concurrent Belos solves on host threads aren't safe in
a standard Trilinos build. Every Belos solve and orthogonalization starts and stops
`Teuchos::TimeMonitor` timers, and each start or stop also pushes into one process-wide
`Teuchos::StackedTimer`, which exists by default and has no lock (`Teuchos_TimeMonitor.cpp`,
the `TimeMonitor` constructor and destructor). Turning timers off is a configure-time
option (`Belos_ENABLE_TEUCHOS_TIME_MONITOR=OFF`) production code can't count on, and
swapping out the global timer around the search would be a hack. Dealing across ranks
keeps almost all the benefit, since most applications already run one rank per core.
Threads can come later if Belos's timers gain a thread-safe mode, with no change to any
result.

### 3.6 Factory (`Teko_KrylovSurrogatePreconditionerFactory`)

- Subclass `Teko::PreconditionerFactory`, modelled on `DiagnosticPreconditionerFactory`.
  `initializeFromParameterList` reads the options, takes the `"Report Handle"`, and
  builds the wrapped inverse factory from the inverse library.
  `buildPreconditionerOperator` builds the wrapped inverse, checks that the operator is
  blocked with Tpetra-backed blocks, returns the recorder holding it, and attaches the
  recorder, the blocked $\mathcal{A}$ and the options to the handle. A rebuild rebuilds
  the wrapped inverse, clears the samples and re-attaches.
- Register it as `"Krylov Surrogate"` in `PreconditionerFactory.cpp`'s builder, next to
  `"Diagnostic Inverse"`. (`"Adaptive"` is already taken upstream, by an unrelated
  adaptive sub-block solver.)
- Validate the parameter list: an unknown name is an error, so a misspelled option
  doesn't silently do nothing.
- A Doxygen block with the XML example and the handle snippet, in Teko's style.

### 3.7 Tests

In `packages/teko/tests/unit_tests/`, registered like `DiagnosticLinearOp_test`, run at 1
and 4 ranks unless noted.

1. **`tKrylovSurrogateSearch.cpp`**, dense only:
   - configuration counts are 1, 3, 13, 75, 541, `allConfigurations` has no duplicates,
     and seeded sampling is reproducible and includes the extremes;
   - Belos accepts $K$, stored as a Thyra multivector on a serial space, as its operator;
   - the all-merged ordering gives $K = I$ and converges in 1 iteration, and on a
     block-diagonal $\hat{\mathcal{C}}$ every ordering converges in 1;
   - on a strictly block-lower-coupled $2\times2$ case, the identity ordering keeps the
     coupling, so $K = I$ and it converges in 1. The reversed ordering drops it, so
     $K = \hat{\mathcal{C}}$ with $K - I$ nilpotent, and it takes exactly 2;
   - a merged group with a singular block reports `"singular group block"`;
   - many candidates add one surrogate timer to the timer summary, not one per candidate.
2. **`tKrylovSurrogateParallel.cpp`**, at 1, 2, 3 and 4 ranks (3, so the candidate count
   doesn't divide evenly):
   - the report is **bitwise** identical to the 1-rank report at every rank count;
   - the round-robin assignment covers every candidate exactly once;
   - more ranks than candidates still gives the full report.
3. **`tKrylovSurrogateRecorderOp.cpp`:** the recorder's output equals the wrapped
   operator's, columns land in order, the cap holds, `alpha` and `beta` are handled, the
   first input after a clear is kept, and `clearSamples` works.
4. **`tKrylovSurrogateFactory.cpp`**, end to end on a small 3-block Tpetra system built
   with the test utilities:
   - the inverse library builds `"Krylov Surrogate"` over `"Block Gauss-Seidel"`, and a
     Belos flexible GMRES solve converges with the same iteration count and solution as
     the bare `"Block Gauss-Seidel"` (the recorder changes nothing);
   - the handle returns the report both when the preconditioner is built with
     `Teko::buildInverse` and when Stratimikos builds it with Teko as the preconditioner;
   - a warm-started solve (nonzero $x_0$) gives $\hat{b}_j = \hat{Q}_j^T(r_0/\|r_0\|)_j$,
     checked against $r_0$ computed in the test;
   - `diagonalDeviations` are at roundoff, ranks are at most the sample count, and with
     exact inner solves $\hat{\mathcal{C}}$ matches a dense reference
     $\mathcal{Q}^T\mathcal{C}\mathcal{Q}$ assembled in the test (comparing singular
     values, which are insensitive to SVD sign flips);
   - for a block whose $Z$ has singular values spread from 1 down to $10^{-12}$, the
     retained rank and singular values match a dense `GESVD` reference: every direction
     above $10^{-10}$ is kept, none lost to the old $4.21\times10^{-8}$ Gram floor;
   - at 4 ranks, the same problem distributed three ways (evenly, with one block's rows
     all on rank 2, and with one block smaller than the sample count) gives the same
     ranks and singular values of $\hat{\mathcal{C}}$ to about $10^{-12}$, the last two
     exercising padded rows;
   - the 1-rank and 4-rank reports agree on ranks, singular values of $\hat{\mathcal{C}}$,
     and candidate iteration counts;
   - a missing handle, a non-blocked operator and a block not backed by Tpetra throw at
     build time, and `buildReport()` with no samples throws.

### 3.8 Parity checks and measurements (not shipped)

In `claudes_world/teko_auto_parity/`:

- **Against the hack.** Run one problem through `teko-reconfig-request` and `teko-auto`
  with the same samples (cap the hack's restart to match). Where the hack's Gram floor
  doesn't bite, ranks and singular values of $\hat{\mathcal{C}}$ should agree to about
  $10^{-8}$. Where it does, `teko-auto` should keep more directions, never fewer.
- **Against pyautoteko.** On the same $Z$, ranks should now nearly agree, since
  pyautoteko's direct SVD truncates at $2.2\times10^{-10}$, close to the new default. Fed
  `teko-auto`'s $\hat{\mathcal{C}}$ and $\hat{b}$, pyautoteko's FGMRES at jitter 0 does the
  same math, so iteration counts per ordering should agree within about one iteration. A
  larger gap is a bug.
- **Jitter loss.** The hack's `conv.json` files record real iteration counts for the
  orderings it tested. Score the same orderings unperturbed with `teko-auto` and jittered
  with the hack, and compare each against the real counts. That puts a number on what
  shipping without jitter costs, for scheduling the upgrade in 3.5.
- **Truncation default.** Retained rank drives search quality (wiki
  `drop-tolerance-fidelity-tradeoff`), while the extra weak directions $10^{-10}$ admits
  carry about $10^{-6}$ error each. Score the same problem at $10^{-10}$, $10^{-8}$ and
  $10^{-6}$, and keep whichever ranks the real solves best.
- **Scaling.** Search wall time against rank count at a few hundred candidates.

### 3.9 Order of work, one commit each

0. A worktree for `aidantgould/teko-auto` beside this one, and a build directory with
   Teko, Belos, Thyra, Tpetra (with TSQR), Ifpack2, Amesos2, Stratimikos, MPI and tests
   enabled.
1. `Teko_KrylovSurrogateSearch`: candidates, the explicit $K$ on Thyra serial vectors,
   Belos scoring, dealing and the gather, with the dense and rank-invariance tests.
2. `Teko_KrylovSurrogateRecorderOp`, first-input capture included, and its tests.
3. `Teko_KrylovSurrogateReducedOperator`: `reduceOnRankZero` first, tested alone on dense
   matrices, then padded TSQR.
4. The factory and its registration, `Teko_KrylovSurrogateReport` with the handle, and
   the end-to-end tests, Stratimikos and warm start included.
5. The parity checks and measurements in 3.8 (untracked).
6. clang-format with Teko's configuration, a full Teko test run, and a drafted PR
   description. Not opened until asked.

## 4. Deliberately left out

Everything that acts on the report, or that only the research loop needed:

- the re-solve, reconfigured system assembly, and the "rescue" of a stalled solve
  (`Teko_KrylovReconfigPrec.hpp`);
- the Belos hook, the Tpetra type-erased route, and the static registration;
- JSON files, the watcher, Python and pyautoteko at run time, environment variables, dry
  run, and the branch tracer;
- factor-time measurement and warm-up (`Teko_FactorTimeRegistry.hpp`,
  `Teko_InverseFactory.cpp` edits);
- the automatic pick (the hack's 20%-of-range, fewest-merges rule) and any cost model;
- converting Belos's residual scaling into a relative target (the uncommitted
  `residual_target_plan.md` work), since the user sets `Tolerance` directly;
- jitter and the preconditioned surrogate solve (the performance note in 3.5);
- per-candidate residual histories (the rate comes from the final residual);
- retrieval without code changes (a lookup by inverse name for XML-only applications);
- blocks not backed by Tpetra;
- streaming compression, eviction, Schur-block prediction, inner-tolerance selection;
- host threads within a rank, blocked on Belos's timers (3.5).

## 5. Decisions

**Still open:**

1. **Lower block Gauss-Seidel only**, as in the hack and pyautoteko. Block Jacobi or
   upper Gauss-Seidel would be a later `Search` option.
2. **Default sample and candidate counts** (100 each). With the search parallel, a few
   hundred candidates may be affordable. Settle it from the scaling measurement (3.8).
3. **Truncation default** ($10^{-10}$), to be confirmed by the measurement in 3.8.

**Decided** (2026-10-27 and 2026-10-28):

- Record $Z$ at the preconditioner, with no Belos hook (3.1).
- Retrieve through a handle passed in the parameter list, which works through
  Stratimikos (3.2).
- Treat every solve as warm: the right-hand side is the first preconditioner input, the
  initial residual's direction, and the surrogate starts from zero (3.3).
- Build $\hat{\mathcal{C}}$ in plain Tpetra, with blocks not backed by Tpetra rejected
  (3.4).
- Per-block bases from `Tpetra::TsqrAdaptor::factorExplicit`, every block always padded
  with zero rows so there is one path, and no Gram route. The feature requires Tpetra
  built with TSQR. The SVD of each $R_j$ runs on rank 0 and is broadcast, and so are the
  final $\hat{\mathcal{C}}$ and $\hat{b}$ (3.4).
- Score each candidate by forming $K = \hat{\mathcal{C}}[\hat{\mathcal{C}}]_{S_1}^{-1}$
  explicitly and running unpreconditioned Belos GMRES on it, through Belos's Thyra
  adapter on Thyra's serial vectors, with no adapter code of our own (3.5).
- Report iterations, final residual and the rate computed from them, with no residual
  history (3.5).
- No jitter for now, accepting the measured loss (0.955 to 0.858 correlation in
  pyautoteko). The jitter-loss measurement still runs, so the gap on real solves is on
  record when the upgrade is scheduled (3.5, 3.8).
- Deal candidates round-robin across MPI ranks, with no cost balancing and no threads
  (3.5).
- Five source pairs: the factory, the recorder, the reduced operator, the search (which
  includes the parallel dealing), and the report (which includes the handle) (section 2).

## 6. Execution (proto1)

How the first implementation is built and tested. Added 2026-10-28.

**Git.**

1. On `aidantgould/teko-reconfig-request`: commit the residual-target work (the 11
   modified files and `residual_target_plan.md`) and the overprediction probe's code,
   then push to `origin`.
2. Check out `aidantgould/teko-auto` in the main tree. Add `claudes_world/` at the repo
   root, holding a short `README.md` (what lives where) and this plan in
   `claudes_world/plans/`. Commit.
3. Branch `aidantgould/teko-auto-proto1` off it and implement there, one commit per step
   in 3.9. Neither `teko-auto` nor `proto1` is pushed until asked.
4. `claudes_world/` is committed on these research branches and is stripped before
   any upstream PR.

**Trilinos build.** A new build directory, `/home/atgould/teko/trilinos-build-auto`,
configured like `trilinos-build` plus what this feature and its tests need: TpetraTSQR
on (`Trilinos_ENABLE_TpetraTSQR=ON`, `TpetraCore_ENABLE_TSQR=ON`), `Teko_ENABLE_TESTS=ON`
(which needs Galeri), and Amesos2 for exact inner solves in the end-to-end test.
Everything else in Trilinos keeps tests off. `trilinos-build` is left alone.

**Front end.** A pyfront branch, `teko-auto`, made off `main` and checked out in the main
`/home/atgould/teko/trilinos-teko-pyfront` checkout, replacing the hack front end there.
On it:

- the configure and build scripts and `CMakeLists.txt` point at `trilinos-build-auto`;
- `teko_ext.cpp` drops every hack dependency (`Teko_FactorTimeRegistry.hpp`,
  `Teko_KrylovReconfigPrec.hpp`, the adaptive gate) and builds the preconditioner
  through a Teko inverse library: `"Block Gauss-Seidel"` or `"Block Jacobi"` over the
  existing Ifpack2 block inverse, wrapped in `"Krylov Surrogate"` when a surrogate report
  is requested;
- `solve(..., surrogate=False)` returns `x`, or `(x, report)` with the report as a dict
  (`C_hat` and `b_hat` as numpy arrays, the candidates as a list);
- `python_front_end_interface.py` drops the watcher and its environment variables, and
  `pyTeko(..., surrogate=False)` passes the flag through;
- a test script checks the report end to end at 1 and 4 ranks.

**Scope of this run** (decided 2026-10-28): the build, the Trilinos unit tests, and the
front-end end-to-end tests. None of 3.8's parity checks or measurements; those wait for
a later run.

## 7. As built (proto1, 2026-10-28)

Built on `aidantgould/teko-auto-proto1` and the pyfront branch `teko-auto`, as planned
except where noted. Nothing pushed.

**Results.**

- Teko's full suite, 25 tests including the 10 new ones, passes in
  `trilinos-build-auto`. The 10 new ones: search, recorder, parallel at 1 to 4 ranks,
  and reduced operator and factory at 1 and 4 ranks.
- The reduced operator matches the dense reference to 3e-15 in every distribution
  (even, a block entirely on one rank, a block smaller than the sample count), and
  keeps the 10 directions above 1e-10 in the conditioning test where the Gram route
  would keep 7.
- End to end at 4 ranks: 8 FGMRES iterations gave 8 samples, ranks 8 8 8, diagonal
  deviations up to 3e-14, rhs capture 0.998. The solution is bitwise unchanged by the
  recorder. A warm start's recorded first input matches r0/||r0|| to 3e-17.
- The handle reaches the factory through Stratimikos unchanged.
- Front end: `test_teko_auto.py` passes all 16 checks at 1 and 4 ranks, and the demo
  runs, including the new surrogate demo.

**Differences from the plan.**

- Two commits, library then tests, instead of one per step in 3.9.
- `Teko_Config.h.in` gained `#cmakedefine Teko_ENABLE_Belos`: Belos is an optional
  Teko dependency and the search needs it. Without Belos or TSQR, building the type
  throws.
- `KrylovSurrogateHandle::getRecorder()` was added, mainly so the warm-start test can
  check the recorded first input.
- A candidate counts as converged when the recomputed residual is at most 10 times
  `Tolerance` (`kResidualSlack` in `Teko_KrylovSurrogateSearch.cpp`).
- `buildReport()` throws if $\hat{b} = 0$, since there is then nothing to score.
- Tests: the reduced-operator checks (dense reference, distributions, conditioning,
  rank-0 SVD) got their own file, `tKrylovSurrogateReducedOperator.cpp`, sharing a
  formula-defined problem in `tKrylovSurrogateProblem.hpp`. The "one timer per rank"
  test was not written.
- Neither fallback was needed: Belos accepted $K$ as a Thyra multivector operator
  on the serial space, and `offsetViewNonConst` worked for computing into the
  padded TSQR buffer.
- Front end: the hack's `kResidualSlack = 1e3` check on the real solve was kept, now
  local to `teko_ext.cpp`.
