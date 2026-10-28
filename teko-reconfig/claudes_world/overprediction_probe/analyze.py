"""Where does the surrogate overpredict, and why? Stage 2: read run.py's data.

    python analyze.py [cases.json]

The optional argument reads a partial index (request ids to cases) in place of
data/cases.json, which run.py writes only when it finishes.

For every test row, compares the watcher's surrogate_iters with the real
Trilinos count, flags clear overpredictions, and for each flagged ordering
re-scores the stored surrogate to separate the suspects:

  cap        surrogate_iters == R, the reduced solve's own iteration cap
  jitter     the same surrogate at jitter 0, and at 1e-4 under other noise seeds
  truncation per-block retained rank against krylov_dim (the Gram-route SVD
             drops a direction only below kGramSigmaFloor)
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "Trilinos" / "teko-reconfig"))
sys.path.insert(0, str(ROOT / "pyautoteko"))

import numpy as np  # noqa: E402
import surrogate_search as ss  # noqa: E402
from fgmres import FGMRES  # noqa: E402

DATA = HERE / "data"
NOISE_SEEDS = range(10)


def overpredicted(row):
    """Clearly over: at least 3 iterations and at least half the real count.
    Only a converged test has a real count: an unconverged one records the
    outer limit, a floor, and a surrogate above it is agreeing, not over."""
    if not row["converged"]:
        return False
    d = row["surrogate_iters"] - row["iters"]
    return d >= 3 and d >= 0.5 * row["iters"]


def rescore(C_hat, b_hat, ordering, tol, jitter, noise_seed=0):
    n = C_hat.n
    settings = dict(precond="gs", inner_tol=jitter, tol=tol, maxit=n, restart=n,
                    autosolve=True, noise_seed=noise_seed)
    return FGMRES(C_hat, b_hat, order=list(ordering), settings=settings).iters


def main():
    index = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "cases.json"
    cases = json.loads(index.read_text())
    diffs, flagged = Counter(), []
    for c in cases:
        n = c["request_id"]
        req = json.loads((DATA / "requests" / f"s{n}_request.json").read_text())
        conv = json.loads((DATA / "requests" / f"s{n}_conv.json").read_text())
        R = len(req["b_hat"])
        truncated = sum(req["krylov_dim"] - r for r in req["ranks"])
        tests = [r for r in conv["solves"] if r["type"] == "test"]
        over = [r for r in tests if overpredicted(r)]
        for r in tests:
            diffs[r["surrogate_iters"] - r["iters"]] += 1
        print(f"s{n:<3} seed {c['seed']:<4} nb {c['nb']:<3} m {req['krylov_dim']:<3} "
              f"ranks {req['ranks']} R {R:<4} dropped {truncated:<3} "
              f"real {min(r['iters'] for r in tests)}-{max(r['iters'] for r in tests)} "
              f"over {len(over)}/{len(tests)}")
        if over:
            flagged.append((c, req, over))

    print("\nsurrogate - real, all test rows:")
    print("  " + ", ".join(f"{d:+d}: {k}" for d, k in sorted(diffs.items())))

    print("\nflagged orderings, rescored on the stored surrogate:")
    print(f"{'case':<14}{'ordering':<18}{'real':>5}{'sur':>5}{'R':>5}"
          f"{'jit0':>6}{'seeds min-max':>15}")
    summary = Counter()
    for c, req, over in flagged:
        C_hat, b_hat = ss.blocks_from_request(req)
        tol = req.get("target_tol", ss.SURROGATE_TOL)
        R = C_hat.n
        for r in over:
            o = r["ordering"]
            j0 = rescore(C_hat, b_hat, o, tol, 0.0)
            seeded = [rescore(C_hat, b_hat, o, tol, ss.SURROGATE_JITTER, s)
                      for s in NOISE_SEEDS]
            print(f"s{c['request_id']:<3}nb{c['nb']:<3}sd{c['seed']:<4}"
                  f"{str(o):<18}{r['iters']:>5}{r['surrogate_iters']:>5}{R:>5}"
                  f"{j0:>6}{min(seeded):>8}-{max(seeded):<6}")
            summary["rows"] += 1
            summary["at cap"] += r["surrogate_iters"] >= R
            summary["fixed at jitter 0"] += not overpredicted(dict(r, surrogate_iters=j0))
            summary["fixed by best seed"] += not overpredicted(
                dict(r, surrogate_iters=min(seeded)))
    print("\n" + ", ".join(f"{k}: {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
