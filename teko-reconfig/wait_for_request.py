"""
wait_for_request.py — continuously watches for reconfiguration requests
(s<N>_request.json, written by Teko::KrylovSurrogate::writeRequestJson) and
answers each one by writing s<N>_reconfig.json (read by
Teko::KrylovSurrogate::waitForOrdering).

The C++ side leaves s<N>_reconfig.json in place after reading it (it does not
delete it), and writes s<N>_conv.json once it has consumed the ordering. This
watcher simply keeps looping: it answers each new s<N>_request.json once and
goes back to watching for the next request.

The ordering it answers with comes from searching the surrogate the request
carries, in surrogate_search.py, which is a thin layer over pyautoteko's
RandomSearch and FGMRES. No orderings are enumerated or sampled here. Without
pyautoteko the watcher still answers, with the fully merged ordering and a
warning (see choose_ordering). SEARCH_ON_STEPS restricts which steps are
searched at all.

Runs until Ctrl-C or until RUN_SECONDS (default 300 = 5 minutes) have
elapsed, whichever comes first.

Normally you don't run this by hand — the front-end interface spawns it
automatically. To run it standalone (from its own directory):
    python wait_for_request.py

All three file types (s<N>_request.json, s<N>_reconfig.json, s<N>_conv.json)
live together in one directory. Defaults to the "requests" sibling of this
script; override with TEKO_RECONFIG_REQUESTS_DIR to match the C++ side's
default (kDefaultRequestsDir in Teko_KrylovSurrogate.hpp).
"""

import glob
import json
import os
import re
import time

import surrogate_search

REQUESTS_DIR = os.environ.get(
    "TEKO_RECONFIG_REQUESTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests"),
)
POLL_INTERVAL_SEC = 0.5
# Self-terminate after this many seconds with no request handled. Read from
# TEKO_WATCHER_IDLE_TIMEOUT (default 1000); <= 0 disables the idle timeout.
# This is only a safety net — when launched by the Python interface the
# watcher is normally stopped explicitly at interpreter exit.

IDLE_TIMEOUT_SEC = float(os.environ.get("TEKO_WATCHER_IDLE_TIMEOUT", "1000"))

REQUEST_RE = re.compile(r"s(\d+)_request\.json$")

# selection_mode written into the reconfig response, switching what the C++
# side returns after timing the test_orderings sweep:
#   "chosen"    — apply use_ordering (the sweep is still timed & recorded).
#   "best_conv" — apply the swept ordering with the best convergence.
#   "best_time" — apply the swept ordering with the best (factor+iterate) time.
SELECTION_MODE = "chosen"

# When True, every ordering the surrogate search scored is also sent as a
# test_ordering, so the C++ side builds, solves and times each one on the full
# system (results land in s<N>_solved.json). That is the ground truth the
# surrogate is predicting, and it is expensive: one real solve per ordering,
# 541 of them at 5 blocks. Off by default; the search itself runs either way.
EMIT_TEST_ORDERINGS = False

# Which steps the surrogate search runs on. A "step" is the request number N in
# s<N>_request.json: one adaptive solve, numbered from 0 within a run (the
# Python interface wipes the requests dir at startup, so each run restarts at
# s0).
#   None       — search on every step. The default.
#   []         — never search; every step is answered with fallback_ordering.
#   [0, 3]     — search on steps 0 and 3, fall back on every other step.
# Any container works, so range(4) or a set is fine too. A skipped step is
# still answered, just without paying for the search, which is the point: on a
# transient or multi-step run, one search early on can carry the whole run.
SEARCH_ON_STEPS = None


def search_on_step(request_id):
    """Whether the surrogate search runs on this step (see SEARCH_ON_STEPS)."""
    return SEARCH_ON_STEPS is None or request_id in SEARCH_ON_STEPS


def already_handled(request_id):
    # s<N>_conv.json is written (atomically) by the C++ side once it has
    # consumed s<N>_reconfig.json (or given up waiting for it). Its presence
    # means s<N>_request.json is a stale leftover from a previous run's
    # pyTeko() call, not a request this watcher should answer.
    return os.path.exists(os.path.join(REQUESTS_DIR, f"s{request_id}_conv.json"))


def fallback_ordering(n_blocks):
    """The ordering answered with when the surrogate search does not run.

    Everything merged into one group, i.e. a direct solve of the whole system.
    It is the one candidate that is valid at every block count, and it
    converges, so a run whose watcher lost pyautoteko still finishes rather
    than silently applying a preconditioner nobody chose. Also what a step
    outside SEARCH_ON_STEPS is answered with.
    """
    if n_blocks == 0:
        raise ValueError("n_blocks must be greater than zero")
    return [0] * n_blocks


def write_reconfig(ordering, path, opt_ordering=None, test_orderings=None):
    # Write to a temp file and atomically rename into place, so the C++ side's
    # fs::exists(path) poll never observes a partially-written file.
    #
    # Fields:
    #   selection_mode   — what the C++ side returns after the sweep (see
    #                      SELECTION_MODE). Written first.
    #   use_ordering     — the ordering applied in "chosen" mode (the field C++
    #                      always reads).
    #   opt_ordering     — the "optimal" ordering from the cheap surrogate
    #                      search. Defaults to a copy of use_ordering, which is
    #                      what the fallback path answers with.
    #   exh_opt_ordering — the optimal ordering from an exhaustive search.
    #                      Left blank for now.
    #   test_orderings   — candidate orderings to build/solve/time; results are
    #                      written by C++ to s<N>_solved.json. May be empty.
    tmp_path = path + ".tmp"
    payload = {
        "selection_mode": SELECTION_MODE,
        "use_ordering": list(ordering),
        "opt_ordering": list(ordering if opt_ordering is None else opt_ordering),
        "exh_opt_ordering": [],
        "test_orderings": list(test_orderings or []),
    }
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def choose_ordering(request, request_id=0):
    """Pick the ordering to answer a request with.

    Returns (use_ordering, opt_ordering, test_orderings). The ordering comes
    from searching the surrogate this request carries: pyautoteko enumerates
    (or samples) the orderings, scores each by an FGMRES solve on C_hat, and
    picks the one with the fewest mergers among those near the best iteration
    count (surrogate_search.pick_opt_ordering). No orderings are generated
    here.

    Three ways the search does not run, all of them answered with
    fallback_ordering: this step is outside SEARCH_ON_STEPS (deliberate, so it
    is reported plainly), pyautoteko is not importable, or the search itself
    raised (both of those warn). A warning rather than an exception: the C++
    side is blocked in waitForOrdering right now, and answering it badly beats
    not answering.
    """
    n_blocks = request["n_blocks"]

    if not search_on_step(request_id):
        print(f"  step {request_id} is not in SEARCH_ON_STEPS "
              f"({SEARCH_ON_STEPS}), so no search ran: answering with the "
              f"fully merged ordering.")
        return fallback_ordering(n_blocks), None, []

    if not surrogate_search.available():
        print(f"WARNING: pyautoteko is not importable, so no surrogate search "
              f"ran.\n"
              f"         reason:   {surrogate_search.unavailable_reason()}\n"
              f"         expected: {surrogate_search.PYAUTOTEKO_DIR}\n"
              f"         falling back to the fully merged ordering.")
        return fallback_ordering(n_blocks), None, []

    try:
        result = surrogate_search.search(request)
    except Exception as exc:
        print(f"WARNING: the surrogate search failed "
              f"({type(exc).__name__}: {exc}).\n"
              f"         falling back to the fully merged ordering.")
        return fallback_ordering(n_blocks), None, []

    lo, hi = result.iters_range
    print(f"  surrogate search: {result.n_evaluated} of {result.total} orderings "
          f"scored ({result.mode} mode)")
    print(f"  surrogate iters : {lo} to {hi}, threshold {result.threshold:.4g}")

    test_orderings = result.orderings() if EMIT_TEST_ORDERINGS else []
    return result.opt_ordering, result.opt_ordering, test_orderings


def handle_request(request_id):
    request_path = os.path.join(REQUESTS_DIR, f"s{request_id}_request.json")
    reconfig_path = os.path.join(REQUESTS_DIR, f"s{request_id}_reconfig.json")

    with open(request_path) as f:
        request = json.load(f)

    c_hat = request["C_hat"]            # single R x R matrix (list of rows)
    b_hat = request["b_hat"]            # single length-R vector
    n_rows = len(c_hat)
    n_cols = len(c_hat[0]) if c_hat else 0

    print(f"Received s{request_id}_request.json:")
    print(f"  n_blocks      = {request['n_blocks']}")
    print(f"  block_sizes   = {request['block_sizes']}")
    print(f"  krylov_dim    = {request['krylov_dim']}")
    print(f"  ranks         = {request['ranks']}")
    print(f"  equation_ends = {request['equation_ends']}")
    print(f"  C_hat shape   = {n_rows} x {n_cols}")
    print(f"  b_hat length  = {len(b_hat)}")

    ordering, opt_ordering, test_orderings = choose_ordering(request, request_id)
    write_reconfig(ordering, reconfig_path, opt_ordering=opt_ordering,
                   test_orderings=test_orderings)
    print(f"Wrote reconfig (mode={SELECTION_MODE}, use_ordering={ordering}, "
          f"{len(test_orderings)} test orderings) to {reconfig_path}")


def main():
    idle_desc = (f"idle timeout {IDLE_TIMEOUT_SEC:g}s"
                 if IDLE_TIMEOUT_SEC > 0 else "no idle timeout")
    print(f"Watching {REQUESTS_DIR} for s<N>_request.json "
          f"(until Ctrl-C, parent exit, or {idle_desc})...")

    answered = set()
    start = time.monotonic()
    last_activity = start
    initial_ppid = os.getppid()
    stop_reason = "Ctrl-C"
    try:
        while True:
            # Exit if our parent (the Python interface that launched us) has
            # gone away, so the watcher never orphans itself.
            if os.getppid() != initial_ppid:
                stop_reason = "parent exited"
                break
            # Idle timeout: stop if no request has been handled in a while.
            if IDLE_TIMEOUT_SEC > 0 and time.monotonic() - last_activity > IDLE_TIMEOUT_SEC:
                stop_reason = f"idle for {IDLE_TIMEOUT_SEC:g}s"
                break

            for path in glob.glob(os.path.join(REQUESTS_DIR, "s*_request.json")):
                m = REQUEST_RE.match(os.path.basename(path))
                if not m:
                    continue
                request_id = int(m.group(1))
                if request_id in answered:
                    continue
                if already_handled(request_id):
                    answered.add(request_id)
                    continue
                handle_request(request_id)
                answered.add(request_id)
                last_activity = time.monotonic()
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"

    elapsed = time.monotonic() - start
    print(f"Stopping ({stop_reason}) after {elapsed:.1f}s; "
          f"answered requests: {sorted(answered)}")


if __name__ == "__main__":
    main()
