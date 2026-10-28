"""Where does the surrogate overpredict, and why? Stage 1: collect.

Runs pyautoteko synthetic cases through the real hook (pyTeko, the production
C_hat from the Gram-route SVD, the watcher's surrogate) with every ordering
emitted as a test, so each case leaves a request (C_hat, b_hat, ranks,
target_tol) and a conv.json pairing every ordering's surrogate_iters with the
real Trilinos count. analyze.py reads them.

    python run.py [n_cases] [nb ...]

Default: the first 12 seeds of cases.json at nb 9 and 13, dim 3, N = 5 (52
orderings per case). Output goes to data/, one requests dir per run, plus
data/cases.json mapping request ids to (seed, nb).
"""

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PYFRONT = ROOT / "trilinos-teko-pyfront"
DATA = HERE / "data"

# pyTeko reads these at import: a private requests dir, and every scored
# ordering run as a real test so each one gets a Trilinos count.
if DATA.exists():
    shutil.rmtree(DATA)
DATA.mkdir()
os.environ["TEKO_RECONFIG_REQUESTS_DIR"] = str(DATA / "requests")
os.environ["TEKO_WATCHER_EMIT_TEST_ORDERINGS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, str(PYFRONT))
from venv_bootstrap import ensure_venv  # noqa: E402

ensure_venv()

sys.path.insert(0, str(ROOT / "pyautoteko"))
from synthetic_case import SyntheticCase, CASES  # noqa: E402
from python_front_end_interface import pyTeko  # noqa: E402

DIM = 3
TOL = 1e-8
MAX_ITER = 200


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    nbs = [int(a) for a in sys.argv[2:]] or [9, 13]
    index = []
    for nb in nbs:
        for seed in CASES[:n_cases]:
            case = SyntheticCase(seed=seed, settings={"nb": nb, "dim": DIM})
            A, b = case.A.merged, case.b.merged
            pyTeko(A, b, "gs", block_sizes=list(case.nbd), tol=TOL, max_iter=MAX_ITER)
            index.append({"request_id": len(index), "seed": seed, "nb": nb, "dim": DIM})
            print(f"seed {seed} nb {nb}: request {len(index) - 1}", flush=True)
    (DATA / "cases.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
