"""Every converged test of every case: the watcher's surrogate (jitter 1e-4)
against the same surrogate at jitter 0, both against the real count."""
import json, sys
import numpy as np
from analyze import rescore, ss, DATA
cases = json.loads(open(sys.argv[1]).read())
for c in cases:
    n = c["request_id"]
    req = json.loads((DATA / "requests" / f"s{n}_request.json").read_text())
    conv = json.loads((DATA / "requests" / f"s{n}_conv.json").read_text())
    C_hat, b_hat = ss.blocks_from_request(req)
    tol = req.get("target_tol", ss.SURROGATE_TOL)
    rows = [r for r in conv["solves"] if r["type"] == "test" and r["converged"]]
    real = np.array([r["iters"] for r in rows], float)
    jit = np.array([r["surrogate_iters"] for r in rows], float)
    j0 = np.array([rescore(C_hat, b_hat, r["ordering"], tol, 0.0) for r in rows], float)
    def stats(s):
        r = np.corrcoef(real, s)[0, 1] if real.std() and s.std() else float("nan")
        return f"mean diff {np.mean(s - real):+6.1f}  r {r:.3f}  over {np.sum((s - real >= 3) & (s - real >= 0.5 * real)):3d}"
    print(f"s{n} seed {c['seed']} ranks {req['ranks']} n={len(rows)}")
    print(f"   jitter 1e-4: {stats(jit)}")
    print(f"   jitter 0   : {stats(j0)}", flush=True)
