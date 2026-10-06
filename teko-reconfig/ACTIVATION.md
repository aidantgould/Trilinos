# How the Krylov-surrogate hack activates (and what breaks it)

The whole mechanism is gated on **`TEKO_ADAPTIVE_RECONFIG`** — set it to a
truthy value or nothing below happens (the hook returns immediately, so merely
linking Teko never hijacks a solve). With it set, after a flexible, blocked
GMRES solve `Belos::BlockGmresSolMgr::solve()` calls a hook that `libteko`
registers at load time (`Teko_KrylovSurrogateInit.cpp` →
`Belos::AdaptiveHook::registerHook`). The hook
(`Teko::KrylovSurrogate::adaptiveLoop`) builds the surrogate, gets a response
from the watcher (a list of solves, each an ordering with its surrogate
prediction and flags), builds-and-times each requested ordering as a test,
and finishes with a final solve: the row flagged `use_ordering`, or failing
that the test that converged fastest. Every solve is recorded in
`s<N>_conv.json`.

**This fires on both converged and stalled first solves.** A solve that hit its
max-iteration limit (`curDim > 0`) is exactly the case reconfiguration should
rescue, so it takes the same path: the surrogate builds fine from its Krylov
data, and the re-solve runs. If that re-solve converges, the hook reports it
back and `BlockGmresSolMgr::solve()` returns **converged** with the re-solve's
iteration count / residual — so a deliberately short, low-max-iter first solve
that stalls is rescued and the caller sees success rather than the original
stall. `s<N>_conv.json` records both the initial row (the stall: `iters` ==
max iters) and the final row (the rescue). Caveat: the re-solve inherits the first
solve's parameters, including `Maximum Iterations` — so if max-iters is set
very low for cheap data-gathering, the re-solve is capped at that same low
value and may not converge; give it a larger budget if you need the rescue to
land. (A `curDim == 0` breakdown still returns early — nothing to build from.)

## Where the implementation lives

The hook is split across four sibling headers in `packages/teko/src/` (one
concept each); chase references by file, not just line:

| Header | Holds |
| --- | --- |
| [`Teko_KrylovReducedModel.hpp`](../packages/teko/src/Teko_KrylovReducedModel.hpp) | the surrogate math (`computeCHat`) + the shared `SC/LO/GO/Node` type aliases |
| [`Teko_KrylovReconfigIO.hpp`](../packages/teko/src/Teko_KrylovReconfigIO.hpp) | the four JSON file formats + the watcher handshake |
| [`Teko_KrylovReconfigPrec.hpp`](../packages/teko/src/Teko_KrylovReconfigPrec.hpp) | reconfigured ("as-if-original") system + preconditioner assembly |
| [`Teko_KrylovSurrogate.hpp`](../packages/teko/src/Teko_KrylovSurrogate.hpp) | the Belos hook itself (`adaptiveLoop`) |

## Where the knobs are defined

| Env var | Read in | Default |
| --- | --- | --- |
| `TEKO_ADAPTIVE_RECONFIG` | [`Teko_KrylovSurrogate.hpp:228`](../packages/teko/src/Teko_KrylovSurrogate.hpp#L228) | unset → inert |
| `TEKO_RECONFIG_REQUESTS_DIR` | [`Teko_KrylovSurrogate.hpp`](../packages/teko/src/Teko_KrylovSurrogate.hpp) | `$PWD/teko-reconfig-requests`, via `defaultRequestsDir()` |
| `TEKO_FACTOR_WARMUP` | [`Teko_InverseFactory.cpp:198`](../packages/teko/src/Teko_InverseFactory.cpp#L198) | follows `TEKO_ADAPTIVE_RECONFIG` |
| `TEKO_WATCHER_IDLE_TIMEOUT` | [`wait_for_request.py`](wait_for_request.py) | 1000 s |
| `TEKO_WATCHER_EMIT_TEST_ORDERINGS` | [`wait_for_request.py`](wait_for_request.py) | off |
| `TEKO_WATCHER_SEND_USE_ORDERING` | [`wait_for_request.py`](wait_for_request.py) | on |

## What else the flag now controls

Beyond the hook itself, `TEKO_ADAPTIVE_RECONFIG` decides how
`trilinos-teko-pyfront`'s `teko_ext.cpp` inverts the diagonal blocks of its OWN
(first) solve:

| flag | block inverse |
| --- | --- |
| on | Belos GMRES to `kBlockSolveTol` + Ifpack2 RILUK, the same factory the re-solve uses, so solve 1 and solve 2 are comparable |
| off | one Ifpack2 apply (ILUT), which is what it was before the hook existed |

Both read the one predicate, `Teko::KrylovSurrogate::adaptiveEnabled()` in
[`Teko_KrylovReconfigPrec.hpp`](../packages/teko/src/Teko_KrylovReconfigPrec.hpp),
so the front end's choice cannot drift from the gate the hook fires on. The
extension prints which it chose once per process on rank 0:

```
[Teko] block inverse: Ifpack2 (ILUT), single apply (TEKO_ADAPTIVE_RECONFIG off)
[Teko] block inverse: Belos GMRES + Ifpack2 RILUK (adaptive hook live)
```

The reason for the split is cost: wrapping every block inverse in a Belos solve
manager is 2.2x to 2.8x on the whole solve, and an arm with the inner GMRES
capped at a single iteration is exactly as slow, so it is the wrapper and not
the inner iteration count. A run with no re-solve to be comparable with has
nothing to buy for that. Measured in
`pyautoteko/claudes_world/adaptive_flag_ab_experiment`.

The flag also sets the default for the factorization warm-up, the one untimed
factorization the first `Teko::buildInverse(factory, A)` in a process does so
that cold-start cost stays out of solve 1's factor time.
`TEKO_FACTOR_WARMUP` overrides it either way. With neither set, a stock Teko
application does no extra work.

The warm-up and the factor-time stopwatch act only on the outermost
`buildInverse`/`rebuildInverse` call on a thread. A block preconditioner
(Gauss-Seidel, SIMPLE, LSC) builds its diagonal-block inverses through those
same functions, and before this was enforced the nested call re-entered the
warm-up's `call_once` and hung on the first solve, flag or no flag (fixed
2026-10-06, tested by `claudes_world/tests/warmup_nested/`). Nested builds
were also timed twice.

## Activates only when ALL hold
- **`TEKO_ADAPTIVE_RECONFIG` is set** (truthy) — the gate above; inert otherwise.
- The app **links/loads `libteko`** (its static initializer registers the
  hook; otherwise `invoke()` is a silent no-op).
- Solve uses **`Belos::BlockGmresSolMgr`** with **`"Flexible Gmres": true`**.
- Template types are exactly **`SC=double`, `MV=Thyra::MultiVectorBase<double>`,
  `OP=Thyra::LinearOpBase<double>`** (the `if constexpr` guard in the sol mgr).
- The operator casts to **`Thyra::BlockedLinearOpBase<double>`** (physically
  blocked), and its blocks are **`Thyra::TpetraLinearOp`** over
  **`Tpetra::CrsMatrix<double,int,long long,OpenMP-node>`** (the aliases in
  `Teko_KrylovReducedModel.hpp`).
- The first solve produced Krylov data (`curDim > 0`) — true whether it
  converged or stalled at max-iters. A `curDim == 0` breakdown returns early.
- A **watcher** answers `s<N>_reconfig.json` in the requests dir
  (`TEKO_RECONFIG_REQUESTS_DIR`, else `defaultRequestsDir()`, which is
  `teko-reconfig-requests` under the directory the application was launched
  from. The hook prints the resolved path once, and says whether it was set or
  defaulted, so a watcher can be pointed at the same place.)
  Needed for both the converged and the stalled (rescue) paths, since both
  build the surrogate and reconfigure.

## File I/O formats

All files live in the requests dir and are named `s<N>_*.json`, where `N` is a
monotonic id chosen by `nextRequestNumber()` = max existing id + 1 over
`s<N>_request.json` / `_conv.json` and the retired `_solved.json` (so ids are
never reused within a run; the Python interface wipes them at startup so each
run restarts at `s0`). Every file is written to `*.tmp` and atomically renamed, so a reader
never sees a half-written file. Per request `N` the exchange is:

```
 C++  --s<N>_request.json-->  watcher
 C++  <--s<N>_reconfig.json--  watcher
 C++  --s<N>_conv.json (every solve that ran)-->  (results)
```

`s<N>_reconfig.json` is left in place after being read; its presence is how the
watcher recognizes an already-answered request, and `s<N>_conv.json`'s presence
marks the request fully consumed.

### `s<N>_request.json` — C++ → watcher (the surrogate)
The reduced operator the watcher uses to choose orderings. Sized by block count
and Krylov rank only (never the global problem size `n`).
```json
{
  "n_blocks": 3,
  "block_sizes": [8, 6, 4],
  "krylov_dim": 5,                 // curDim (Krylov vectors captured)
  "ranks": [r0, r1, r2],           // per-block truncated rank, <= krylov_dim
  "equation_ends": [0, r0, r0+r1, R],  // block boundaries in C_hat/b_hat, R=sum(ranks)
  "C_hat": [[...], ...],           // R x R reduced operator
  "b_hat": [...]                   // length-R reduced RHS
}
```

### `s<N>_reconfig.json` — watcher → C++ (the solves requested)
The watcher's answer: one row per solve it wants, in the order it wants them.
```json
{
  "request_id": 3,
  "solves": [
    {
      "ordering": [0, 0, 0],
      "surrogate_iters": 2,                          // null if no search ran
      "surrogate_flags": ["use_ordering", "opt_ordering"]
    },
    {
      "ordering": [0, 1, 2],
      "surrogate_iters": 4,
      "surrogate_flags": []
    }
  ]
}
```
An **ordering** is a restricted-growth vector: `ordering[k] = g` puts original
block `k` into new group `g` (group ids contiguous from 0). A single-member
group is kept as-is; a multi-member group is assembled into one monolithic
block and factored jointly. No ordering appears on two rows: a row carrying
several roles carries several flags.

Flags (`surrogate_flags`, a row may have any number):
- **`use_ordering`**: apply this ordering. It is the final solve and is not
  also run as a test. If several rows carry it, the first wins, with a warning.
- **`opt_ordering`**: the surrogate search's pick. Record only.
- **`exh_opt_ordering`**: reserved for an exhaustive search's pick. Accepted
  and recorded, never set by the watcher yet.

What C++ does with the rows:
- Every row except the `use_ordering` one is solved as a **test**, in order.
- The **final** solve is the `use_ordering` row if there is one. Otherwise it is
  the best test, re-solved: converged first, then least total (factor +
  iterate) wall time, and among tests that all failed, fewest iterations then
  smallest residual.
- No rows (or no answer before the timeout): no re-solve, and conv.json holds
  the initial row only.

### `s<N>_conv.json` — C++ output (every solve that ran)
One row per solve in execution order: the initial (the application's own
solve), the tests in reconfig order, the final. Every reconfig row comes back
with its `surrogate_iters` and `surrogate_flags`, so this file holds everything
the reconfig did, plus what actually happened.
```json
{
  "request_id": 3,
  "solves": [
    {
      "ordering": [0, 1, 2],
      "surrogate_iters": 4,
      "iters": 3,
      "type": "initial",
      "surrogate_flags": [],
      "converged": true,
      "initial_residual": 9.06,
      "final_residual": 5.1e-16,
      "factor_wall_time_sec": 0.0005,
      "iterate_wall_time_sec": 0.0022,
      "total_wall_time_sec": 0.0027
    },
    { "ordering": [0, 1, 2], ..., "type": "test", ... },
    { "ordering": [0, 0, 0], ..., "type": "final from selection mode use_ordering",
      "surrogate_flags": ["use_ordering", "opt_ordering"], ... }
  ]
}
```
- `type` is `"initial"`, `"test"`, or `"final from selection mode <mode>"`,
  with `<mode>` either `use_ordering` or `best_time`.
- The initial row's ordering is the application's own blocking,
  `[0, 1, ..., nb-1]`. Its `surrogate_iters` and `surrogate_flags` are copied
  from the reconfig row with that ordering when there is one, else `null` and
  `[]`. It ran the application's preconditioner, not necessarily the block
  Gauss-Seidel every test and final run.
- With no `use_ordering`, the best test appears twice, as a test and as the
  final, because no test keeps its solution vector.
- A number that is not finite (a NaN residual from a solve that broke down) is
  written `null`.
- All wall times are the cross-rank maximum (critical path), so they're
  identical on every rank and the best-time pick is deterministic.

`s<N>_solved.json`, which held the test sweep, is retired: conv.json carries
it.

## Residuals and the `converged` flag

`final_residual` in `conv.json` is the **explicitly
recomputed** relative residual `||b - A x|| / ||b||`, not Belos's
`achievedTol()`. Likewise `converged` requires both `Belos::Converged` and that
recomputed residual being within `kResidualSlack` of the solver tolerance.

This matters because Belos judges convergence from its recursively updated
(implicit) residual estimate, and cannot be asked for an explicit test here:
`BlockGmresSolMgr` forces the implicit test whenever `"Flexible Gmres"` is on,
and setting `"Explicit Residual Test"` switches off the flexible iterator this
hook requires. Since each block inverse is an inexact inner GMRES solve, the
preconditioner is non-stationary and that estimate can cross the tolerance while
the true residual does not. A candidate that converges falsely stops early, so
left unchecked it would report `converged` with a low iteration count, a small
residual and a short iterate time, winning the best-time pick. The bias also grows with merging, since a merged group is
larger and less well conditioned than the singletons it replaces.

## How the watcher picks the ordering

`wait_for_request.py` generates no orderings of its own. It hands the request's
surrogate to [`surrogate_search.py`](surrogate_search.py), a thin layer over
pyautoteko (expected as a sibling of this fork, the `$ROOT` layout in
trilinos-teko-pyfront's `SETUP.md`):

1. `C_hat` and `b_hat` are rebuilt as a pyautoteko `BlockMatrix` pair, blocked
   by `ranks` at the `equation_ends` boundaries.
2. pyautoteko's `RandomSearch` enumerates the orderings (every one of them when
   there are few enough: 541 at 5 blocks, about 1.4 s) or samples up to
   `MAX_SEARCH_COUNT` of them when there are not.
3. Each is scored by an FGMRES solve on the surrogate alone, at
   `SURROGATE_JITTER = 1e-4`, matching `kBlockSolveTol` so the surrogate is
   evaluated under the same inexactness the real block solves run at. No cost
   model of the real system exists yet, so ranking is on iterations alone
   (`RandomSearch`'s `unit_cost`).
4. The first row is the pick from `pick_opt_ordering`, flagged
   `opt_ordering`, and `use_ordering` too unless `SEND_USE_ORDERING` is off: with
   `threshold = (max_iters - min_iters) * 0.2 + min_iters`, the ordering with
   the FEWEST mergers among those at or below it, ties broken on fewer
   iterations then on the ordering itself.

`SEARCH_ON_STEPS` in the watcher decides which steps pay for a search at all.
A step is the request number `N` in `s<N>_request.json`, one adaptive solve,
numbered from 0 within a run. `None` (the default) searches every step, `[]`
searches none, and a list such as `[0, 3]` searches only those (any container
works, so `range(4)` is fine). A skipped step is still answered, with the same
fully merged ordering the unavailable-pyautoteko path uses, and the watcher
says which step it skipped and why.

`EMIT_TEST_ORDERINGS` in the watcher (default **off**,
`TEKO_WATCHER_EMIT_TEST_ORDERINGS`) sends every scored ordering as a row as
well, so the C++ side builds, solves and times each on the full system, and
conv.json puts each one's real `iters` beside its `surrogate_iters`. That is the
ground truth the surrogate is predicting, and it costs one real solve per
ordering. `trilinos-teko-pyfront/demo.py` turns it on.

`SEND_USE_ORDERING` (default **on**, `TEKO_WATCHER_SEND_USE_ORDERING`) decides
whether the pick carries `use_ordering`. Off, no row does, and C++ applies
whichever requested solve converged fastest. Both knobs are read from the
environment once, when the watcher starts, which the front end does on the
first `pyTeko` call.

If pyautoteko is not importable (missing clone, or a Python without
numpy/scipy) the watcher prints a warning naming the reason and the path it
looked in, and answers with the fully merged ordering `[0, 0, ..., 0]`, flagged
`use_ordering` whatever `SEND_USE_ORDERING` says. That still converges, so a run with a broken
watcher finishes instead of hanging or applying a preconditioner nobody chose.
