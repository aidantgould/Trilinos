"""s5 (seed 10, nb 9): rescore the overpredicted converged orderings on the
stored surrogate, at jitter 0 and at 1e-4 under a few noise seeds."""
import json, sys, time
from analyze import rescore, overpredicted, ss, DATA
n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
req = json.loads((DATA / "requests" / f"s{n}_request.json").read_text())
conv = json.loads((DATA / "requests" / f"s{n}_conv.json").read_text())
C_hat, b_hat = ss.blocks_from_request(req)
tol = req.get("target_tol", ss.SURROGATE_TOL)
over = sorted((r for r in conv["solves"] if r["type"] == "test" and overpredicted(r)),
              key=lambda r: r["iters"] - r["surrogate_iters"])[:8]
for r in over:
    t = time.time()
    j0 = rescore(C_hat, b_hat, r["ordering"], tol, 0.0)
    js = [rescore(C_hat, b_hat, r["ordering"], tol, 1e-4, s) for s in range(3)]
    print(r["ordering"], "real", r["iters"], "sur", r["surrogate_iters"], "jit0", j0,
          "seeds", js, f"{time.time() - t:.0f}s", flush=True)
