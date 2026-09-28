# Plan: drive the watcher's ordering choice from pyautoteko's random search

**Built 2026-09-25.** Implemented as planned, with the four decisions below
taken as recommended except the test location. Verified end to end on the
recorded `requests/s0_request.json` (5 blocks, ranks [10, 11, 1, 11, 11]): 541
orderings scored in about 1.4 s, iteration counts 2 to 44, threshold 10.4, pick
`[0, 2, 1, 0, 2]` (3 groups, 7 iterations). The fallback path was verified by
running the watcher under a bare `python3` with no numpy: it warns and answers
`[0, 0, 0, 0, 0]`. 16 new tests plus 3 in pyautoteko, all passing. Two
additions beyond the original draft: `ACTIVATION.md` gained a "How the watcher
picks the ordering" section, and `SearchResult` exposes `iters_range` /
`orderings()` so the watcher's printout and the test-ordering list need no
knowledge of the results dict's shape.

`wait_for_request.py` currently answers every request with a hardcoded
`[0, 0, ..., 0, 1]` and its own `restricted_growth_strings` enumerator. It has
a surrogate in hand (`C_hat`, `b_hat`, `ranks`) and does nothing with it. This
plan replaces both halves: pyautoteko enumerates and samples the orderings, and
pyautoteko's FGMRES on the surrogate scores them.

Feasibility already checked against the real `requests/s0_request.json`
(N = 5, ranks [10, 11, 1, 11, 11], R = 44): 541 orderings evaluated in 1.35 s,
iteration counts spanning 2 to 44, and `C_hat`'s diagonal blocks exactly I.

## 1. `pyautoteko/random_search.py`: default cost handle

`RandomSearch.__init__`'s `cost_per_iter` becomes optional: passing `None`
(or omitting it) substitutes a module-level `unit_cost(ordering)` that returns
`1.0`, so `cost` equals `iters` and the ranking is by iteration count alone,
which is what a caller with no cost model for the real system wants. The
argument keeps its position, so all 40-odd existing call sites in `plotting/`,
`testing/`, `claudes_world/` and the test suite are untouched (decided
2026-09-25, over reordering to `(iters, N, cost_per_iter=None)`).

```python
def unit_cost(ordering):
    """Flat cost handle: every ordering costs 1 per iteration."""
    return 1.0

class RandomSearch:
    def __init__(self, cost_per_iter=None, iters=None, N=None,
                 max_search_count=600, ...):
        self.cost_per_iter = unit_cost if cost_per_iter is None else cost_per_iter
```

`iters` and `N` stay required in practice: they keep no usable default, and
omitting either raises. The watcher calls `RandomSearch(None, iters_handle,
n_blocks)`.

## 2. New module `teko-reconfig/surrogate_search.py`

Everything that knows about pyautoteko, so the watcher stays a watcher and the
search is testable without the polling loop.

```python
PYAUTOTEKO_DIR = Path(__file__).resolve().parents[1] / "pyautoteko"
SURROGATE_JITTER = 1e-4     # matches kBlockSolveTol, the production block solve
                            # tolerance; a fixed constant, not an env knob
SURROGATE_TOL    = 1e-8
MAX_SEARCH_COUNT = 600
OPT_RANGE_FRACTION = 0.2

available()                 -> bool          # pyautoteko importable
blocks_from_request(req)    -> (C_hat, b_hat)  # BlockMatrix pair, blocked by ranks
surrogate_iters(C_hat, b_hat, jitter=...)   -> callable(ordering) -> int
search(request)             -> SearchResult | None
pick_opt_ordering(results)  -> list[int]
```

- `blocks_from_request` slices the flat `C_hat` / `b_hat` at `equation_ends`
  into a `BlockMatrix` pair, one block per original block, sized by `ranks`.
- `surrogate_iters` returns a handle that builds
  `FGMRES(C_hat, b_hat, order=ordering, settings=...)` and returns `.iters`.
  Settings mirror `FGMRES.solve_reduced`: `precond="gs"`, `inner_tol=jitter`,
  `maxit` and `restart` capped at `C_hat.n`, `tol=SURROGATE_TOL`. The jitter is
  not modeling inexactness here, it is what makes distinct orderings
  distinguishable at all (see the wiki's `vtilde-from-z` and
  `FGMRES.solve_reduced`'s own docstring).
- `search` runs `RandomSearch(None, iters_handle, n_blocks)`, taking the unit
  cost from step 1 and returns a small dataclass: `results` (ordering tuple ->
  iters), `mode`, `opt_ordering`, `n_evaluated`.
- `pick_opt_ordering` implements the rule:
  `threshold = (max_iters - min_iters) * OPT_RANGE_FRACTION + min_iters`, keep
  every ordering with `iters <= threshold`, then take the one with the fewest
  mergers, that is the largest number of groups `len(set(ordering))`. Ties
  break on fewer iterations, then on the ordering tuple itself so the pick is
  deterministic.

## 3. `wait_for_request.py`

- Delete `restricted_growth_strings`, `make_test_orderings` and
  `MAX_BLOCKS_FOR_EXHAUSTIVE`. No ordering generation lives here any more.
- `make_ordering(n_blocks)` becomes the fallback only: `[0] * n_blocks`, the
  fully merged ordering.
- New setting `SEARCH_ON_STEPS = None` (added 2026-09-28): which steps the
  search runs on, a step being the request number `N` in `s<N>_request.json`.
  `None` searches every step, `[]` searches none, a list or any other container
  searches exactly those. A skipped step is answered with `fallback_ordering`,
  reported plainly rather than as a warning, since it is deliberate.
  `choose_ordering` takes the request id to apply it.
- New setting `EMIT_TEST_ORDERINGS = False` (default off). When True, the
  response's `test_orderings` is every ordering the search evaluated, so the
  C++ side builds, solves and times each on the full system into
  `s<N>_solved.json`. Bounded by `MAX_SEARCH_COUNT`, and expensive: 541 real
  solves at N = 5. When False, `test_orderings` is `[]`.
- `handle_request` calls `surrogate_search.search(request)`. On success
  `use_ordering = opt_ordering = ` the pick (decided 2026-09-25: the search
  drives the applied ordering, it is not record-only), and the printout reports the
  search mode, how many orderings were scored, the iteration range and the
  threshold. On failure (pyautoteko not importable, or the search raising) it
  prints a clearly marked warning naming the reason and the expected path, and
  falls back to `[0] * n_blocks` with `selection_mode = "chosen"` and no test
  orderings.
- `write_reconfig` takes the ordering, the opt ordering and the test orderings
  rather than recomputing them.

The fallback is deliberately the fully merged ordering: it is the one candidate
that is always valid whatever `n_blocks` is, and it is a direct solve of the
whole system, so a run with a broken watcher still converges rather than
silently testing a preconditioner nobody chose.

## 4. Tests

Run directly (`python test_x.py`), not through pytest: no pytest or pandas is
installed in the `trilinos-teko-pyfront/mypy` venv, which is the only
interpreter here with numpy and scipy, and it is the one the watcher is spawned
with. Every test file in pyautoteko already carries a `__main__` runner for
exactly this reason.

- `pyautoteko/claudes_world/tests/test_random_search.py`: the default cost
  handle (cost equals iters, `unit_cost` returns 1.0), and that an explicit
  `cost_per_iter` still wins.
- `teko-reconfig/claudes_world/tests/test_surrogate_search.py` (decided
  2026-09-25: in the fork, under `claudes_world`, matching pyautoteko's
  convention of keeping tests as working state), in pyautoteko's test style: `blocks_from_request` on a synthetic request
  (block sizes match `ranks`, merged matrix round-trips), `pick_opt_ordering`
  on a hand-built results dict where the threshold arithmetic and the
  fewest-mergers tie-break are checkable by hand, the empty and single-ordering
  edge cases, and one end-to-end `search` against the checked-in
  `requests/s0_request.json` if present (skipped otherwise, since
  `requests/` is gitignored).
- Watcher-level: `write_reconfig` payload shape with the search unavailable
  (fallback all zeros, `test_orderings` empty) and with a stubbed search.

## Non-goals

- No cost model for the real system. The dummy unit cost is a placeholder, so
  `cost` in the results is not meaningful yet and nothing ranks on it.
- `exh_opt_ordering` stays `[]`.
- No change to `selection_mode`, which stays `"chosen"`.
- No caching of the search across requests, and no reuse of one request's
  surrogate for another.
- No change to the C++ side.
