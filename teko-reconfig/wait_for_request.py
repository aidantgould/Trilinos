"""
wait_for_request.py — continuously watches for reconfiguration requests
(s<N>_request.json, written by Teko::KrylovSurrogate::writeRequestJson) and
answers each one by writing s<N>_reconfig.json (read by
Teko::KrylovSurrogate::waitForOrdering).

The C++ side leaves s<N>_reconfig.json in place after reading it (it does not
delete it), and writes s<N>_conv.json once it has consumed the ordering. This
watcher simply keeps looping: it answers each new s<N>_request.json once and
goes back to watching for the next request.

It answers with a list of solves, one row per ordering, each carrying the
surrogate's iteration count for it and its flags (see write_reconfig). The
orderings come from searching the surrogate the request carries, in
surrogate_search.py, which is a thin layer over pyautoteko's RandomSearch and
FGMRES. No orderings are enumerated or sampled here. Without pyautoteko the
watcher still answers, with the fully merged ordering and a warning (see
choose_solves). SEARCH_ON_STEPS restricts which steps are searched at all.

Runs until Ctrl-C or until RUN_SECONDS (default 300 = 5 minutes) have
elapsed, whichever comes first.

Normally you don't run this by hand — the front-end interface spawns it
automatically. To run it standalone (from its own directory):
    python wait_for_request.py

All three file types (s<N>_request.json, s<N>_reconfig.json, s<N>_conv.json)
live together in one directory. Defaults to the "requests" sibling of this
script. The C++ side defaults elsewhere, to `teko-reconfig-requests` under the
directory the application was launched from (`defaultRequestsDir()` in
Teko_KrylovSurrogate.hpp), so the two agree only when
TEKO_RECONFIG_REQUESTS_DIR is exported to both. pyTeko() does that for you; when
driving a separate application, set it yourself. The hook prints the directory
it resolved.
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


def env_flag(name, default):
    """A boolean from the environment, read once at startup. Unset gives
    default. Empty, "0", "false" and "FALSE" are false and anything else is
    true: the same rule the C++ side applies to TEKO_ADAPTIVE_RECONFIG."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("", "0", "false", "FALSE")


# The flags a row of s<N>_reconfig.json can carry. A row may carry several.
#   use_ordering      the C++ side applies this ordering as the final solve.
#                     Without it on any row, C++ tests every row and applies
#                     the one that converged in the least wall time.
#   opt_ordering      the surrogate search's pick (pick_opt_ordering).
#   exh_opt_ordering  reserved for an exhaustive search's pick; C++ accepts and
#                     records it, nothing here sets it yet.
USE_ORDERING = "use_ordering"
OPT_ORDERING = "opt_ordering"
EXH_OPT_ORDERING = "exh_opt_ordering"

# When True, every ordering the surrogate search scored is also sent as a row,
# so the C++ side builds, solves and times each one on the full system as a
# test (one row each in s<N>_conv.json). That is the ground truth the surrogate
# is predicting, and it is expensive: one real solve per ordering, 541 of them
# at 5 blocks. Off by default; the search itself runs either way. Overridden by
# TEKO_WATCHER_EMIT_TEST_ORDERINGS.
EMIT_TEST_ORDERINGS = env_flag("TEKO_WATCHER_EMIT_TEST_ORDERINGS", False)

# When True, the surrogate's pick carries use_ordering, so C++ applies it. When
# False, no row does and C++ applies whichever requested solve performed best.
# The fallback ordering (no search ran) is always sent with use_ordering, since
# there is nothing to compare it against. Overridden by
# TEKO_WATCHER_SEND_USE_ORDERING.
SEND_USE_ORDERING = env_flag("TEKO_WATCHER_SEND_USE_ORDERING", True)

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


def solve_row(ordering, surrogate_iters=None, flags=()):
    """One row of s<N>_reconfig.json, in the field order C++ writes back in
    s<N>_conv.json: the ordering, the surrogate's iteration count for it (None,
    written null, when no search ran), and its flags."""
    return {"ordering": list(ordering),
            "surrogate_iters": surrogate_iters,
            "surrogate_flags": list(flags)}


def add_row(rows, ordering, surrogate_iters=None, flags=()):
    """Add a row to rows (a dict keyed by ordering tuple, in insertion order),
    or merge the flags into the row already there, so one ordering is never
    requested twice."""
    key = tuple(ordering)
    if key not in rows:
        rows[key] = solve_row(ordering, surrogate_iters)
    row = rows[key]
    for flag in flags:
        if flag not in row["surrogate_flags"]:
            row["surrogate_flags"].append(flag)
    return row


def write_reconfig(solves, path, request_id):
    # Write to a temp file and atomically rename into place, so the C++ side's
    # fs::exists(path) poll never observes a partially-written file.
    #
    # Fields:
    #   request_id — N, as in s<N>_conv.json.
    #   solves     — the rows (see solve_row). Each is solved by C++ and comes
    #                back as a row of s<N>_conv.json with the same ordering,
    #                surrogate_iters and surrogate_flags.
    #
    # Laid out like s<N>_conv.json: one field per line, arrays inline.
    tmp_path = path + ".tmp"
    rows = []
    for row in solves:
        fields = ",\n".join(f"      {json.dumps(k)}: {json.dumps(v)}"
                            for k, v in row.items())
        rows.append("    {\n" + fields + "\n    }")
    text = ("{\n"
            f'  "request_id": {json.dumps(request_id)},\n'
            '  "solves": [\n' + ",\n".join(rows) + ("\n" if rows else "") +
            "  ]\n"
            "}\n")
    with open(tmp_path, "w") as f:
        f.write(text)
    os.replace(tmp_path, path)


def choose_solves(request, request_id=0):
    """The solves to answer a request with, as a list of rows (solve_row).

    The orderings come from searching the surrogate this request carries:
    pyautoteko enumerates (or samples) the orderings, scores each by an FGMRES
    solve on C_hat, and picks the one with the fewest mergers among those near
    the best iteration count (surrogate_search.pick_opt_ordering). The pick is
    the first row, flagged opt_ordering, and use_ordering too when
    SEND_USE_ORDERING is on. With EMIT_TEST_ORDERINGS on, every other scored
    ordering follows as an unflagged row. No orderings are generated here.

    Three ways the search does not run, all of them answered with
    fallback_ordering: this step is outside SEARCH_ON_STEPS (deliberate, so it
    is reported plainly), pyautoteko is not importable, or the search itself
    raised (both of those warn). A warning rather than an exception: the C++
    side is blocked in waitForOrdering right now, and answering it badly beats
    not answering.
    """
    n_blocks = request["n_blocks"]
    fallback = [solve_row(fallback_ordering(n_blocks), flags=[USE_ORDERING])]

    if not search_on_step(request_id):
        print(f"  step {request_id} is not in SEARCH_ON_STEPS "
              f"({SEARCH_ON_STEPS}), so no search ran: answering with the "
              f"fully merged ordering.")
        return fallback

    if not surrogate_search.available():
        print(f"WARNING: pyautoteko is not importable, so no surrogate search "
              f"ran.\n"
              f"         reason:   {surrogate_search.unavailable_reason()}\n"
              f"         expected: {surrogate_search.PYAUTOTEKO_DIR}\n"
              f"         falling back to the fully merged ordering.")
        return fallback

    try:
        result = surrogate_search.search(request)
    except Exception as exc:
        print(f"WARNING: the surrogate search failed "
              f"({type(exc).__name__}: {exc}).\n"
              f"         falling back to the fully merged ordering.")
        return fallback

    lo, hi = result.iters_range
    print(f"  surrogate search: {result.n_evaluated} of {result.total} orderings "
          f"scored ({result.mode} mode)")
    print(f"  surrogate iters : {lo} to {hi}, threshold {result.threshold:.4g}")

    rows = {}
    pick_flags = [USE_ORDERING, OPT_ORDERING] if SEND_USE_ORDERING else [OPT_ORDERING]
    add_row(rows, result.opt_ordering, result.iters_of(result.opt_ordering),
            pick_flags)
    if EMIT_TEST_ORDERINGS:
        for ordering in result.orderings():
            add_row(rows, ordering, result.iters_of(ordering))
    return list(rows.values())


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

    solves = choose_solves(request, request_id)
    write_reconfig(solves, reconfig_path, request_id)
    flagged = ", ".join(f"{'+'.join(r['surrogate_flags'])}={r['ordering']}"
                        for r in solves if r["surrogate_flags"])
    print(f"Wrote reconfig ({len(solves)} solves; {flagged or 'none flagged'}) "
          f"to {reconfig_path}")


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
