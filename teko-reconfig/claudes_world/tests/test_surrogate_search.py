"""Tests for surrogate_search.py and the watcher's use of it. Run with pytest,
or directly:

    python test_surrogate_search.py

Needs pyautoteko (a sibling of the Trilinos fork) and its numpy/scipy for
everything but the fallback tests, which are exactly the case where it is
missing. Run under trilinos-teko-pyfront's mypy venv, the same interpreter the
watcher itself is spawned with.
"""

import json
import sys
from pathlib import Path

# the watcher and surrogate_search live two levels up, in teko-reconfig
RECONFIG_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RECONFIG_DIR))
import surrogate_search  # noqa: E402
import wait_for_request as watcher  # noqa: E402


def fake_request(n_blocks=3, ranks=(2, 3, 1), seed=0):
    """A request in the s<N>_request.json shape, with a diagonally dominant
    C_hat so the surrogate solve actually converges. Its diagonal blocks are
    the identity, as the real C_hat's are by construction."""
    import numpy as np
    rng = np.random.default_rng(seed)
    ends = [0]
    for r in ranks:
        ends.append(ends[-1] + r)
    R = ends[-1]
    C = 0.1 * rng.standard_normal((R, R))
    for k, r in enumerate(ranks):
        C[ends[k]:ends[k + 1], ends[k]:ends[k + 1]] = np.eye(r)
    return {
        "n_blocks": n_blocks,
        "block_sizes": [10 * r for r in ranks],
        "krylov_dim": max(ranks),
        "ranks": list(ranks),
        "equation_ends": ends,
        "C_hat": C.tolist(),
        "b_hat": rng.standard_normal(R).tolist(),
    }


def results_from(pairs):
    """A RandomSearch-shaped results dict from (ordering, iters) pairs."""
    return {tuple(o): {"iters": i, "cost_per_iter": 1.0, "cost": i}
            for o, i in pairs}


# ── blocks_from_request ───────────────────────────────────────────────────

def test_blocks_from_request_blocks_by_rank():
    import numpy as np
    req = fake_request()
    C_hat, b_hat = surrogate_search.blocks_from_request(req)
    assert C_hat.row_sizes == req["ranks"] and C_hat.col_sizes == req["ranks"]
    assert b_hat.row_sizes == req["ranks"]
    assert C_hat.n == sum(req["ranks"])
    # the merged surrogate is the flat matrix the request carried
    assert np.allclose(C_hat.merged, np.array(req["C_hat"]))
    assert np.allclose(b_hat.merged, np.array(req["b_hat"]))


def test_surrogate_iters_scores_an_ordering():
    req = fake_request()
    C_hat, b_hat = surrogate_search.blocks_from_request(req)
    iters = surrogate_search.surrogate_iters(C_hat, b_hat)
    n = iters((0, 1, 2))
    assert isinstance(n, int) and 0 < n <= C_hat.n


# ── pick_opt_ordering ─────────────────────────────────────────────────────

def test_pick_opt_takes_fewest_mergers_within_the_threshold():
    # iters 4 to 14, so threshold = (14 - 4) * 0.2 + 4 = 6. Of the three
    # orderings at or below it, (0, 1, 2) has the most groups.
    results = results_from([((0, 0, 0), 4),      # 1 group, the fastest
                            ((0, 0, 1), 5),      # 2 groups
                            ((0, 1, 2), 6),      # 3 groups, right on the line
                            ((0, 2, 1), 7),      # 3 groups, just outside
                            ((1, 0, 0), 14)])
    opt, threshold = surrogate_search.pick_opt_ordering(results)
    assert threshold == 6.0
    assert opt == [0, 1, 2]


def test_pick_opt_breaks_group_ties_on_iterations_then_tuple():
    results = results_from([((0, 1, 2), 5), ((0, 2, 1), 4), ((2, 1, 0), 4),
                            ((0, 0, 0), 4), ((1, 0, 0), 24)])
    # threshold = (24 - 4) * 0.2 + 4 = 8, so everything but the last is in.
    # Three orderings have 3 groups; two of those tie at 4 iterations, and the
    # smaller tuple wins.
    opt, threshold = surrogate_search.pick_opt_ordering(results)
    assert threshold == 8.0
    assert opt == [0, 2, 1]


def test_pick_opt_edge_cases():
    assert surrogate_search.pick_opt_ordering({}) == (None, 0.0)
    single = results_from([((0, 1), 7)])
    assert surrogate_search.pick_opt_ordering(single) == ([0, 1], 7.0)
    # a flat landscape: threshold is the common value, so every ordering is
    # within it and the fewest-mergers rule decides alone
    flat = results_from([((0, 0), 3), ((0, 1), 3), ((1, 0), 3)])
    assert surrogate_search.pick_opt_ordering(flat) == ([0, 1], 3.0)


def test_pick_opt_range_fraction_widens_the_field():
    results = results_from([((0, 0, 0), 4), ((0, 1, 2), 9), ((0, 0, 1), 14)])
    # at 0.2 the threshold is 6 and only the fully merged ordering qualifies
    assert surrogate_search.pick_opt_ordering(results)[0] == [0, 0, 0]
    # at 0.5 it is 9, which lets the unmerged ordering in and it wins
    assert surrogate_search.pick_opt_ordering(results, range_fraction=0.5)[0] == [0, 1, 2]


# ── search ────────────────────────────────────────────────────────────────

def test_search_scores_every_ordering_of_a_small_case():
    req = fake_request()
    result = surrogate_search.search(req)
    assert result.mode == "exhaustive"
    assert result.total == 13 and result.n_evaluated == 13   # Fubini(3)
    lo, hi = result.iters_range
    assert 0 < lo <= hi
    assert result.opt_ordering in result.orderings()
    assert all(isinstance(o, list) and sorted(set(o)) == list(range(max(o) + 1))
               for o in result.orderings())


def test_search_is_deterministic():
    req = fake_request(seed=7)
    a = surrogate_search.search(req)
    b = surrogate_search.search(req)
    assert a.opt_ordering == b.opt_ordering
    assert a.results == b.results


def test_search_on_a_recorded_request():
    # the real thing, if a request from a previous run is lying around
    # (requests/ is gitignored, so this is best effort). Capped at a handful of
    # orderings on purpose: this checks that a real request's shape parses and
    # searches, and a recorded surrogate can be large (a run at nb=160 leaves
    # R = 801, where a full 541-ordering search takes minutes).
    path = RECONFIG_DIR / "requests" / "s0_request.json"
    if not path.exists():
        return
    with open(path) as f:
        req = json.load(f)
    result = surrogate_search.search(req, max_search_count=4)
    assert 0 < result.n_evaluated <= 4
    assert len(result.opt_ordering) == req["n_blocks"]


# ── the watcher's use of it ───────────────────────────────────────────────

def knobs(emit=False, send_use=True):
    """Set EMIT_TEST_ORDERINGS and SEND_USE_ORDERING, and give back a function
    that restores their previous values.

    Set explicitly rather than assumed: they are knobs the owner flips between
    runs (and the environment can set), so a test that reads the module's
    current values tests the last edit rather than the behavior.
    """
    was = watcher.EMIT_TEST_ORDERINGS, watcher.SEND_USE_ORDERING
    watcher.EMIT_TEST_ORDERINGS, watcher.SEND_USE_ORDERING = emit, send_use

    def restore():
        watcher.EMIT_TEST_ORDERINGS, watcher.SEND_USE_ORDERING = was
    return restore


def fallback_rows(n_blocks):
    return [{"ordering": [0] * n_blocks, "surrogate_iters": None,
             "surrogate_flags": ["use_ordering"]}]


def test_choose_solves_sends_the_pick_flagged():
    req = fake_request()
    restore = knobs(emit=False, send_use=True)
    try:
        rows = watcher.choose_solves(req)
    finally:
        restore()
    assert len(rows) == 1
    (row,) = rows
    assert list(row) == ["ordering", "surrogate_iters", "surrogate_flags"]
    assert len(row["ordering"]) == req["n_blocks"]
    assert row["surrogate_flags"] == ["use_ordering", "opt_ordering"]
    assert isinstance(row["surrogate_iters"], int) and row["surrogate_iters"] > 0


def test_choose_solves_withholds_use_ordering_when_asked():
    req = fake_request()
    restore = knobs(emit=False, send_use=False)
    try:
        rows = watcher.choose_solves(req)
    finally:
        restore()
    assert [r["surrogate_flags"] for r in rows] == [["opt_ordering"]]


def test_choose_solves_emits_every_ordering_once_when_asked():
    req = fake_request()
    restore = knobs(emit=True, send_use=True)
    try:
        rows = watcher.choose_solves(req)
        result = surrogate_search.search(req)
    finally:
        restore()
    orderings = [tuple(r["ordering"]) for r in rows]
    assert len(rows) == 13                      # every ordering the search scored
    assert len(set(orderings)) == 13            # the pick is merged, not repeated
    assert rows[0]["surrogate_flags"] == ["use_ordering", "opt_ordering"]
    assert all(r["surrogate_flags"] == [] for r in rows[1:])
    for r in rows:                              # each carries its own prediction
        assert r["surrogate_iters"] == result.iters_of(r["ordering"])


def test_add_row_merges_flags_by_ordering():
    rows = {}
    watcher.add_row(rows, [0, 1], 4, ["opt_ordering"])
    watcher.add_row(rows, [0, 0], 2)
    watcher.add_row(rows, [0, 1], 4, ["use_ordering", "opt_ordering"])
    assert list(rows.values()) == [
        {"ordering": [0, 1], "surrogate_iters": 4,
         "surrogate_flags": ["opt_ordering", "use_ordering"]},
        {"ordering": [0, 0], "surrogate_iters": 2, "surrogate_flags": []},
    ]


def test_env_flag():
    import os
    name = "TEKO_WATCHER_TEST_ENV_FLAG"
    try:
        os.environ.pop(name, None)
        assert watcher.env_flag(name, True) is True
        assert watcher.env_flag(name, False) is False
        for falsy in ("", "0", "false", "FALSE"):
            os.environ[name] = falsy
            assert watcher.env_flag(name, True) is False
        for truthy in ("1", "yes", "true"):
            os.environ[name] = truthy
            assert watcher.env_flag(name, False) is True
    finally:
        os.environ.pop(name, None)


def test_search_on_step_gate():
    real = watcher.SEARCH_ON_STEPS
    try:
        watcher.SEARCH_ON_STEPS = None          # every step
        assert all(watcher.search_on_step(n) for n in (0, 1, 7))
        watcher.SEARCH_ON_STEPS = []            # no step
        assert not any(watcher.search_on_step(n) for n in (0, 1, 7))
        watcher.SEARCH_ON_STEPS = [0, 3]        # listed steps only
        assert watcher.search_on_step(0) and watcher.search_on_step(3)
        assert not watcher.search_on_step(1) and not watcher.search_on_step(4)
        watcher.SEARCH_ON_STEPS = range(2)      # any container works
        assert watcher.search_on_step(1) and not watcher.search_on_step(2)
    finally:
        watcher.SEARCH_ON_STEPS = real


def test_choose_solves_skips_steps_outside_the_list():
    req = fake_request()
    searched = []
    real_search, real_steps = surrogate_search.search, watcher.SEARCH_ON_STEPS

    def counting_search(request, **kwargs):
        searched.append(1)
        return real_search(request, **kwargs)

    surrogate_search.search = counting_search
    watcher.SEARCH_ON_STEPS = [1]
    restore = knobs(emit=False, send_use=True)
    try:
        skipped = watcher.choose_solves(req, request_id=0)
        ran = watcher.choose_solves(req, request_id=1)
    finally:
        surrogate_search.search = real_search
        watcher.SEARCH_ON_STEPS = real_steps
        restore()

    assert skipped == fallback_rows(3)          # fallback, and no search ran
    assert ran[0]["surrogate_iters"] is not None
    assert len(searched) == 1                   # only step 1 paid for a search


def test_choose_solves_empty_list_never_searches():
    req = fake_request()
    real_search, real_steps = surrogate_search.search, watcher.SEARCH_ON_STEPS

    def boom(request, **kwargs):
        raise AssertionError("the search must not run with SEARCH_ON_STEPS = []")

    surrogate_search.search = boom
    watcher.SEARCH_ON_STEPS = []
    try:
        for step in (0, 1, 2):
            assert watcher.choose_solves(req, request_id=step) == fallback_rows(3)
    finally:
        surrogate_search.search = real_search
        watcher.SEARCH_ON_STEPS = real_steps


def test_choose_solves_falls_back_when_pyautoteko_is_missing(monkeypatch=None):
    req = fake_request(n_blocks=5, ranks=(2, 2, 2, 2, 2))
    real_available = surrogate_search.available
    real_reason = surrogate_search.unavailable_reason
    surrogate_search.available = lambda: False
    surrogate_search.unavailable_reason = (
        lambda: "ModuleNotFoundError: No module named 'block_matrix'")
    # the fallback keeps use_ordering even with it withheld from picks
    restore = knobs(emit=True, send_use=False)
    try:
        rows = watcher.choose_solves(req)
    finally:
        surrogate_search.available = real_available
        surrogate_search.unavailable_reason = real_reason
        restore()
    assert rows == fallback_rows(5)


def test_choose_solves_falls_back_when_the_search_raises():
    req = fake_request()
    real = surrogate_search.search

    def boom(request, **kwargs):
        raise RuntimeError("no")

    surrogate_search.search = boom
    try:
        rows = watcher.choose_solves(req)
    finally:
        surrogate_search.search = real
    assert rows == fallback_rows(3)


def test_write_reconfig_payload(tmp_path=None):
    import tempfile
    rows = [watcher.solve_row([0, 1, 1], 3, ["use_ordering", "opt_ordering"]),
            watcher.solve_row([0, 0, 0], 1),
            watcher.solve_row([0, 1, 2])]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s4_reconfig.json"
        watcher.write_reconfig(rows, str(path), 4)
        text = path.read_text()
        payload = json.loads(text)
        assert list(payload) == ["request_id", "solves"]
        assert payload["request_id"] == 4
        assert payload["solves"] == [
            {"ordering": [0, 1, 1], "surrogate_iters": 3,
             "surrogate_flags": ["use_ordering", "opt_ordering"]},
            {"ordering": [0, 0, 0], "surrogate_iters": 1, "surrogate_flags": []},
            {"ordering": [0, 1, 2], "surrogate_iters": None, "surrogate_flags": []},
        ]
        assert list(payload["solves"][0]) == ["ordering", "surrogate_iters",
                                              "surrogate_flags"]
        assert '"ordering": [0, 1, 1]' in text  # arrays inline, as in conv.json
        assert not (Path(d) / "s4_reconfig.json.tmp").exists()


def test_write_reconfig_with_no_rows():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s0_reconfig.json"
        watcher.write_reconfig([], str(path), 0)
        assert json.loads(path.read_text()) == {"request_id": 0, "solves": []}


def test_iters_of():
    req = fake_request()
    result = surrogate_search.search(req)
    for ordering, r in result.results.items():
        assert result.iters_of(list(ordering)) == r["iters"]
    assert result.iters_of([0, 1, 2, 3]) is None    # never scored


def test_fallback_ordering():
    assert watcher.fallback_ordering(1) == [0]
    assert watcher.fallback_ordering(5) == [0, 0, 0, 0, 0]
    try:
        watcher.fallback_ordering(0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n_blocks = 0")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"{name} ok")
    print(f"\nall {len(tests)} tests passed")
