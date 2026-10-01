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

def emit_test_orderings(value):
    """Set EMIT_TEST_ORDERINGS and give back its previous value.

    Set explicitly rather than assumed: it is a knob the owner flips between
    runs, so a test that reads the module's current value tests the last edit
    rather than the behavior.
    """
    was = watcher.EMIT_TEST_ORDERINGS
    watcher.EMIT_TEST_ORDERINGS = value
    return was


def test_choose_ordering_uses_the_search():
    req = fake_request()
    was = emit_test_orderings(False)
    try:
        use, opt, tests = watcher.choose_ordering(req)
    finally:
        emit_test_orderings(was)
    assert use == opt and len(use) == req["n_blocks"]
    assert tests == []


def test_choose_ordering_emits_test_orderings_when_asked():
    req = fake_request()
    was = emit_test_orderings(True)
    try:
        _, _, tests = watcher.choose_ordering(req)
    finally:
        emit_test_orderings(was)
    assert len(tests) == 13     # every ordering the search scored


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


def test_choose_ordering_skips_steps_outside_the_list():
    req = fake_request()
    searched = []
    real_search, real_steps = surrogate_search.search, watcher.SEARCH_ON_STEPS

    def counting_search(request, **kwargs):
        searched.append(1)
        return real_search(request, **kwargs)

    surrogate_search.search = counting_search
    watcher.SEARCH_ON_STEPS = [1]
    try:
        skipped = watcher.choose_ordering(req, request_id=0)
        ran = watcher.choose_ordering(req, request_id=1)
    finally:
        surrogate_search.search = real_search
        watcher.SEARCH_ON_STEPS = real_steps

    assert skipped == ([0, 0, 0], None, [])     # fallback, and no search ran
    assert ran[0] != [0, 0, 0] or ran[1] is not None
    assert len(searched) == 1                   # only step 1 paid for a search


def test_choose_ordering_empty_list_never_searches():
    req = fake_request()
    real_search, real_steps = surrogate_search.search, watcher.SEARCH_ON_STEPS

    def boom(request, **kwargs):
        raise AssertionError("the search must not run with SEARCH_ON_STEPS = []")

    surrogate_search.search = boom
    watcher.SEARCH_ON_STEPS = []
    try:
        for step in (0, 1, 2):
            assert watcher.choose_ordering(req, request_id=step) == ([0, 0, 0], None, [])
    finally:
        surrogate_search.search = real_search
        watcher.SEARCH_ON_STEPS = real_steps


def test_choose_ordering_falls_back_when_pyautoteko_is_missing(monkeypatch=None):
    req = fake_request(n_blocks=5, ranks=(2, 2, 2, 2, 2))
    real_available = surrogate_search.available
    real_reason = surrogate_search.unavailable_reason
    surrogate_search.available = lambda: False
    surrogate_search.unavailable_reason = (
        lambda: "ModuleNotFoundError: No module named 'block_matrix'")
    try:
        use, opt, tests = watcher.choose_ordering(req)
    finally:
        surrogate_search.available = real_available
        surrogate_search.unavailable_reason = real_reason
    assert use == [0, 0, 0, 0, 0]
    assert opt is None and tests == []


def test_choose_ordering_falls_back_when_the_search_raises():
    req = fake_request()
    real = surrogate_search.search

    def boom(request, **kwargs):
        raise RuntimeError("no")

    surrogate_search.search = boom
    try:
        use, opt, tests = watcher.choose_ordering(req)
    finally:
        surrogate_search.search = real
    assert use == [0, 0, 0] and opt is None and tests == []


def test_write_reconfig_payload(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s0_reconfig.json"
        watcher.write_reconfig([0, 1, 1], str(path), opt_ordering=[0, 1, 1],
                               test_orderings=[[0, 0, 0], [0, 1, 1]])
        payload = json.loads(path.read_text())
        assert payload["selection_mode"] == "chosen"
        assert payload["use_ordering"] == [0, 1, 1]
        assert payload["opt_ordering"] == [0, 1, 1]
        assert payload["exh_opt_ordering"] == []
        assert payload["test_orderings"] == [[0, 0, 0], [0, 1, 1]]
        assert not (Path(d) / "s0_reconfig.json.tmp").exists()


def test_write_reconfig_defaults_opt_to_use():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s0_reconfig.json"
        watcher.write_reconfig([0, 0, 0], str(path))
        payload = json.loads(path.read_text())
        assert payload["opt_ordering"] == [0, 0, 0]
        assert payload["test_orderings"] == []


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
