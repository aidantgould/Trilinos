# Plan: the watcher's search count as a knob, TEKO_WATCHER_MAX_SEARCH_COUNT

**Status: built 2026-10-20.** A bad value raises at startup; one commit, pushed. Built as planned except as noted under "As built".

## What changes

The number of orderings the surrogate search scores per request is today
`surrogate_search.MAX_SEARCH_COUNT = 600`, used as `search()`'s default because
`choose_solves` calls `surrogate_search.search(request)` without it. The
watcher gains its own module-level knob, read once at startup like
`EMIT_TEST_ORDERINGS` and `SEND_USE_ORDERING`, and passes it explicitly.

## Interface (`wait_for_request.py`)

```python
def env_count(name, default):
    """A positive int from the environment, read once at startup. Unset or
    empty gives default; anything else that is not a positive integer raises
    ValueError naming the variable and the value."""

# Orderings the surrogate search scores per request: RandomSearch's cap, which
# with the block count also picks its mode (exhaustive, shuffled, random).
# Overridden by TEKO_WATCHER_MAX_SEARCH_COUNT.
MAX_SEARCH_COUNT = env_count("TEKO_WATCHER_MAX_SEARCH_COUNT",
                             surrogate_search.MAX_SEARCH_COUNT)
```

Read at import, so a bad value stops the watcher before it answers anything:
`python wait_for_request.py` exits with the ValueError, and a watcher spawned
by pyTeko dies at once (its traceback on the parent's stderr) instead of
answering with a count nobody chose.

`choose_solves` calls `surrogate_search.search(request,
max_search_count=MAX_SEARCH_COUNT)`. The default stays defined once, in
`surrogate_search.py`.

The startup line names the count: `Watching ... (search count 600, until ...)`.

## Docs

- ACTIVATION.md: a row in the env-var table, and a sentence in "How the
  watcher picks the ordering" on what the count does to `RandomSearch`'s mode
  (below 541 at 5 blocks the search is no longer exhaustive) and, with test
  orderings on, to the number of real test solves.
- The watcher's docstring sample run: one line naming the variable.

## Tests (`claudes_world/tests/test_surrogate_search.py`)

- `env_count`: unset and empty give the default, a positive integer string
  parses (surrounding whitespace allowed), and `abc`, `0`, `-5`, `2.5` each
  raise ValueError naming the variable.
- `choose_solves` passes `MAX_SEARCH_COUNT` through: a stub `search` records
  its `max_search_count`, and a real search at a cap below the total runs in
  shuffled mode and scores exactly that many orderings.

## Non-goals

- No change to `RandomSearch` or to `random_trigger_factor`.
- No per-step count.

## As built

As planned, plus the watcher docstring's sample run naming the variable.
`test_surrogate_search.py`: 25 of 25 (two new). `test_reconfig_protocol.py`:
2 of 2. Checked by hand: `TEKO_WATCHER_MAX_SEARCH_COUNT=abc` stops the watcher
with `ValueError: TEKO_WATCHER_MAX_SEARCH_COUNT='abc' is not a positive
integer`, and `=40` shows as `search count 40` in the startup line.

One change from the draft test. A cap below the total does not always score
that many DISTINCT orderings: `RandomSearch.search` puts the fully merged and
unmerged orderings first without removing them from the shuffled list, so
either can come up twice inside the cap (at 3 blocks and a cap of 5, 4
distinct). The test asserts `n_evaluated <= cap`, that both extremes are
scored, and that every distinct scored ordering is emitted once. The
duplicate itself is left alone, per the non-goals; it costs one surrogate
solve, and with `EMIT_TEST_ORDERINGS` on no extra real solve, since rows are
merged by ordering.

**Fixed 2026-10-20 in pyautoteko.** `RandomSearch.search` now moves the
extremes to the front instead of adding them, so a cap of k walks k distinct
orderings. The same duplication had also cut exhaustive mode at
`max_search_count == total`: 539 of 541 orderings at 5 blocks (73 of 75 at 4),
which a `TEKO_WATCHER_MAX_SEARCH_COUNT` of 541 or 542 would have hit. The test
here asserts `n_evaluated == 5` again.

The sample run at the top of the watcher sets `TEKO_WATCHER_MAX_SEARCH_COUNT=40`
(added at the user's request) and says what a cap below the total gives up.
