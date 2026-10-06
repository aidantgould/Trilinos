"""End-to-end tests of the s<N>_reconfig.json / s<N>_conv.json protocol. Run
with pytest, or directly:

    python test_reconfig_protocol.py

Runs trilinos-teko-pyfront's demo.py (which sweeps every ordering of every
case as a test) in a fresh process against a temporary requests dir, then
checks each conv.json against the reconfig.json it answered. Twice: once as
the demo ships, where the watcher sends use_ordering, and once with
TEKO_WATCHER_SEND_USE_ORDERING=0, where C++ must pick the final by best time.
Needs libteko and teko_ext built from this tree, and pyautoteko beside it.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

PYFRONT = Path(__file__).resolve().parents[4] / "trilinos-teko-pyfront"
PYTHON = PYFRONT / "mypy" / "bin" / "python3"
TIMEOUT_SEC = 300

CONV_FIELDS = ["ordering", "surrogate_iters", "iters", "type", "surrogate_flags",
               "converged", "initial_residual", "final_residual",
               "factor_wall_time_sec", "iterate_wall_time_sec",
               "total_wall_time_sec"]
SURROGATE_FIELDS = ("ordering", "surrogate_iters", "surrogate_flags")


def run_demo(send_use_ordering):
    """Run demo.py and return {N: (reconfig, conv)} for every request."""
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ,
                   TEKO_RECONFIG_REQUESTS_DIR=d,
                   TEKO_WATCHER_SEND_USE_ORDERING="1" if send_use_ordering else "0",
                   OMP_NUM_THREADS="4")
        try:
            proc = subprocess.run([str(PYTHON), "demo.py"], cwd=PYFRONT, env=env,
                                  capture_output=True, text=True,
                                  timeout=TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"demo.py hung for {TIMEOUT_SEC} s")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "[FAIL]" not in proc.stdout, proc.stdout

        pairs = {}
        for conv_path in Path(d).glob("s*_conv.json"):
            n = int(conv_path.name[1:-len("_conv.json")])
            reconfig = json.loads((Path(d) / f"s{n}_reconfig.json").read_text())
            pairs[n] = (reconfig, json.loads(conv_path.read_text()))
        assert not list(Path(d).glob("s*_solved.json")), "solved.json is retired"
    assert pairs, "the demo wrote no conv.json"
    return pairs


def surrogate_part(row):
    return {k: row[k] for k in SURROGATE_FIELDS}


def better_by_time(a, b):
    """The C++ best-time comparator: converged first, then least total time,
    and between two failures fewer iterations, then the smaller residual."""
    if a["converged"] != b["converged"]:
        return a["converged"]
    if a["converged"]:
        return a["total_wall_time_sec"] < b["total_wall_time_sec"]
    if a["iters"] != b["iters"]:
        return a["iters"] < b["iters"]
    # null is a NaN in C++, and a comparison with NaN is false
    ra, rb = a["final_residual"], b["final_residual"]
    return ra is not None and rb is not None and ra < rb


def check_pair(n, reconfig, conv, send_use_ordering):
    requested = reconfig["solves"]
    rows = conv["solves"]
    assert conv["request_id"] == reconfig["request_id"] == n
    assert list(conv) == ["request_id", "solves"]
    for row in rows:
        assert list(row) == CONV_FIELDS, f"s{n}: field order {list(row)}"

    initial, tests, final = rows[0], rows[1:-1], rows[-1]
    nb = len(initial["ordering"])
    assert initial["type"] == "initial"
    assert initial["ordering"] == list(range(nb))
    same = [r for r in requested if r["ordering"] == initial["ordering"]]
    if same:
        assert surrogate_part(initial) == surrogate_part(same[0])
    else:
        assert initial["surrogate_iters"] is None and initial["surrogate_flags"] == []
    assert all(t["type"] == "test" for t in tests)

    use = [r for r in requested if "use_ordering" in r["surrogate_flags"]]
    assert bool(use) == send_use_ordering or len(requested) == 1
    if use:
        # use_ordering is respected and is not also run as a test
        assert final["type"] == "final from selection mode use_ordering"
        assert surrogate_part(final) == surrogate_part(use[0])
        expected_tests = [r for r in requested if r is not use[0]]
    else:
        assert final["type"] == "final from selection mode best_time"
        expected_tests = requested
        best = tests[0]
        for t in tests[1:]:
            if better_by_time(t, best):
                best = t
        assert surrogate_part(final) == surrogate_part(best)

    # every requested row comes back, in order, with its surrogate fields
    assert [surrogate_part(t) for t in tests] == [surrogate_part(r)
                                                  for r in expected_tests]
    for t in tests + [final]:
        assert len(t["ordering"]) == nb


def check_run(send_use_ordering):
    pairs = run_demo(send_use_ordering)
    for n, (reconfig, conv) in sorted(pairs.items()):
        check_pair(n, reconfig, conv, send_use_ordering)
    # the three-block case swept all 13 of its orderings
    assert any(len(r["solves"]) == 13 for r, _ in pairs.values())
    return pairs


def test_demo_with_use_ordering():
    pairs = check_run(send_use_ordering=True)
    assert all(c["solves"][-1]["type"].endswith("use_ordering")
               for _, c in pairs.values())


def test_demo_without_use_ordering():
    pairs = check_run(send_use_ordering=False)
    assert all(c["solves"][-1]["type"].endswith("best_time")
               for _, c in pairs.values())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
