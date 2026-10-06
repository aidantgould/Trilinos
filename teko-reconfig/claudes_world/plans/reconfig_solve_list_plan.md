# Plan: reconfig.json and conv.json as lists of solves

**Status: built 2026-10-06** (record at the end). exh_opt_ordering reserved but never emitted; the demo
turns test orderings on globally; solved.json retired, run_comparison.py left alone.

## What changes

Today `s<N>_reconfig.json` is five named fields (`selection_mode`, `use_ordering`,
`opt_ordering`, `exh_opt_ordering`, `test_orderings`), `s<N>_conv.json` holds exactly two
solves without saying which orderings they used, and the sweep goes to a third file,
`s<N>_solved.json`. After this change both files are flat lists of solves, one row per
ordering, and conv.json is a superset of reconfig.json.

### s<N>_reconfig.json (watcher to C++)

```json
{
  "request_id": 3,
  "solves": [
    {"ordering": [0, 1, 2], "surrogate_iters": 6, "surrogate_flags": ["use_ordering", "opt_ordering"]},
    {"ordering": [0, 0, 0], "surrogate_iters": 1, "surrogate_flags": ["exh_opt_ordering"]},
    {"ordering": [0, 0, 1], "surrogate_iters": 3, "surrogate_flags": []}
  ]
}
```

- Every row is a solve the watcher is requesting. Field order: `ordering`,
  `surrogate_iters` (null when no search ran), `surrogate_flags` (a list, possibly
  empty, possibly several flags).
- Flags: `use_ordering` (apply this one), `opt_ordering` (the surrogate pick),
  `exh_opt_ordering` (reserved for an exhaustive search's pick: C++ accepts and
  records it, the watcher never sets it for now).
- `selection_mode` is gone. The rule on the C++ side:
  - A row flagged `use_ordering`: it is the final solve. Every other row is solved
    once as a test.
  - No row flagged `use_ordering`: every row is solved as a test, and the
    best one (converged first, then least total wall time, the old `best_time`
    comparator) is re-solved as the final.
  - More than one `use_ordering` row: the first wins, with a warning.
  - No rows at all, or a timeout: no re-solve. conv.json carries the initial solve only.

### s<N>_conv.json (C++ output)

```json
{
  "request_id": 3,
  "solves": [
    {"ordering": [0, 1, 2], "surrogate_iters": 6, "iters": 9,
     "type": "initial", "surrogate_flags": [], "converged": true,
     "initial_residual": 2.4, "final_residual": 3.1e-9,
     "factor_wall_time_sec": 0.01, "iterate_wall_time_sec": 0.02, "total_wall_time_sec": 0.03},
    {"ordering": [0, 0, 1], ..., "type": "test", "surrogate_flags": [], ...},
    {"ordering": [0, 0, 0], ..., "type": "test", "surrogate_flags": ["exh_opt_ordering"], ...},
    {"ordering": [0, 1, 2], ..., "type": "final from selection mode use_ordering",
     "surrogate_flags": ["use_ordering", "opt_ordering"], ...}
  ]
}
```

- Field order: `ordering`, `surrogate_iters`, `iters`, then the flags (`type`,
  `surrogate_flags`, `converged`), then residuals and timing.
- Row order is execution order: initial, the tests in reconfig order, the final.
- `type` is `"initial"`, `"test"`, or `"final from selection mode <mode>"`, where
  `<mode>` is `use_ordering` or `best_time`.
- The initial row's ordering is the application's own blocking, `[0, 1, ..., nb-1]`.
  Its `surrogate_iters` and `surrogate_flags` are copied from the reconfig row with that
  ordering if there is one, else null and `[]`. Its `converged` is Belos's verdict
  checked against the recomputed true residual, the same test every other row gets.
- Every reconfig row appears in conv.json with its `surrogate_iters` and
  `surrogate_flags`, so conv.json carries everything reconfig.json does.
- With no `use_ordering`, the best row appears twice, once as a test and once as the
  final (re-solved, because the sweep keeps no solution vector). That is what
  happened, so it is what gets recorded.
- `iterations` is renamed `iters`, matching `surrogate_iters`. The legacy
  `wall_time_sec` alias is dropped.

`s<N>_solved.json` is retired: conv.json now holds the sweep.

## Watcher (`wait_for_request.py`)

- `choose_ordering` returns the list of rows instead of a 3-tuple. Two knobs, each a
  module constant overridable by an environment variable read at startup (the same
  pattern as `TEKO_WATCHER_IDLE_TIMEOUT`):
  - `EMIT_TEST_ORDERINGS` / `TEKO_WATCHER_EMIT_TEST_ORDERINGS`: every scored ordering
    becomes a row (unchanged meaning, default off).
  - `SEND_USE_ORDERING` / `TEKO_WATCHER_SEND_USE_ORDERING`: whether the pick carries
    `use_ordering` (default on). Off means C++ picks by best time.
- Rows are merged by ordering, so a pick that is also a test row is one row with its
  flags, never two rows with the same ordering.
- The fallback (no search) answers one row, `[0, ..., 0]`, flagged `use_ordering`,
  `surrogate_iters` null.
- `surrogate_search.SearchResult` gains `iters_of(ordering)`, so every row can carry
  its surrogate iteration count.

## C++ (`Teko_KrylovReconfigIO.hpp`, `Teko_KrylovSurrogate.hpp`)

- `ReconfigResponse` becomes `{received, std::vector<RequestedSolve>}` with
  `RequestedSolve = {ordering, surrogate_iters (int, -1 for null), surrogate_flags
  (vector<string>)}`. The targeted parser grows a `parseSolvesField` that walks the
  `solves` array object by object, reusing `parseIntArrayField` plus new
  `parseIntOrNullField` and `parseStringArrayField`.
- Broadcast: every rank gets the orderings and the index of the `use_ordering` row
  (-1 if none). `surrogate_iters` and the flags stay on rank 0, which is the only rank
  that writes conv.json.
- `SolveStats` and `OrderingResult` merge into one `SolveRecord` with all the conv
  fields. `writeConvergenceJson(dir, id, records)` writes the list. `writeSolvedJson`
  is deleted.
- Selection as above. `solveOrdering` is unchanged apart from returning `SolveRecord`.

## Consumers

- `postprocess.py`: reads `solves`, plots the initial and final rows' iters and
  times per request as before.
- `ACTIVATION.md`: the format section rewritten, `selection_mode` text removed.
- `trilinos-teko-pyfront/python_front_end_interface.py`: no interface change. The
  watcher inherits the parent's environment, so settings made before the first
  `pyTeko` call reach it. The startup wipe keeps deleting stale `s*_solved.json`.
- `trilinos-teko-pyfront/demo.py`: sets `TEKO_WATCHER_EMIT_TEST_ORDERINGS=1` with
  `setdefault` before any solve, so every case sweeps its orderings with flags
  (3 at two blocks, 13 at three). `SEND_USE_ORDERING` stays on.

## Tests

- `claudes_world/tests/test_surrogate_search.py`: the `choose_ordering` and
  `write_reconfig` tests rewritten for rows. New: flag merging, `SEND_USE_ORDERING`
  off, env override, `exh_opt_ordering` semantics.
- A new `claudes_world/tests/test_reconfig_protocol.py` that runs `demo.py`
  end to end and checks every conv.json against its reconfig.json: field order,
  every reconfig row present, types, and the selection rule. It runs the demo twice,
  the second time with `TEKO_WATCHER_SEND_USE_ORDERING=0` so the best-time path
  runs too (the demo's `setdefault` lets the test's environment win). This is the only
  practical test of the C++ side, since the hook needs a live Belos solve.
- Rebuild: `build_trilinos.sh` (teko), then `build.sh`.

## Non-goals

- No change to `s<N>_request.json`.
- No change to how the surrogate picks `opt_ordering`.
- `pyautoteko/claudes_world/trilinos_comparison_experiment/run_comparison.py` reads
  the old conv/solved format, and is left that way until that study is rerun.
- No backward compatibility with old-format reconfig files: the watcher and the hook
  ship together.

## What was built (2026-10-06)

As planned, with these differences and additions:

- The flags field is `surrogate_flags` (plural), renamed at the owner's request
  mid-build, in both files.
- Broadcast is simpler than planned (orderings and `use_index` only, no flags
  bitmask), since only rank 0 writes conv.json.
- Non-finite doubles in conv.json are written `null`. Found by the end-to-end test:
  with test orderings on, Demo 2's singular case merges into `[0, 0]`, whose solve
  breaks down with a NaN residual, and `nan` is not JSON. The old `solved.json`
  writer had the same latent bug and never hit it because the demo never swept.
- The watcher writes reconfig.json with conv.json's layout (one field per line,
  arrays inline) rather than `json.dump(indent=2)`, which put every ordering entry on
  its own line.
- `choose_ordering` is renamed `choose_solves`, since it returns rows.
- The fallback row (no search ran) keeps `use_ordering` even with
  `SEND_USE_ORDERING` off, and carries no `opt_ordering`, since nothing was picked.
- `postprocess.py` plots the initial and final rows, relabelled "Initial" and
  "Final".
- The front end needed no change.

Verified: 23 watcher tests, 2 end-to-end protocol tests (the demo run with and without
`use_ordering`), the 5 nested warm-up tests, and the demo on 2 MPI ranks along both
paths (no hang, every case passes, the three-block final matches the single-rank
run's).
