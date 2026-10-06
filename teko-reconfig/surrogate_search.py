"""surrogate_search.py — choose a block ordering by searching pyautoteko's
surrogate, for wait_for_request.py.

Everything that knows about pyautoteko lives here, so the watcher stays a
watcher. The input is one parsed s<N>_request.json: the reduced operator C_hat
(R x R), the reduced right-hand side b_hat (length R), and the per-block ranks
that give their block boundaries. From that this module

  1. rebuilds C_hat / b_hat as a pyautoteko BlockMatrix pair, blocked by rank,
  2. scores an ordering by running FGMRES on that surrogate and reading off the
     iteration count (no full-system solve, no new operator applies),
  3. hands both to pyautoteko's RandomSearch, which owns the enumeration and
     sampling of orderings, and
  4. picks one out of the results (see pick_opt_ordering).

pyautoteko is expected as a sibling of the Trilinos fork, the same $ROOT layout
SETUP.md assumes for trilinos-teko-pyfront and trilinos-build. If it is not
importable (missing clone, or a bare Python without numpy/scipy), available()
returns False and the caller falls back; nothing here raises on import.
"""

import sys
from pathlib import Path

# pyautoteko sits beside the Trilinos fork; its modules are at its root and
# import each other by bare name, so that directory itself goes on sys.path
# (same resolution postprocess.py uses for the trilinos-teko-pyfront venv).
PYAUTOTEKO_DIR = Path(__file__).resolve().parents[2] / "pyautoteko"

# Inexactness applied to the surrogate's own block solves while searching.
# Matches kBlockSolveTol in Teko_KrylovReconfigPrec.hpp, the tolerance the real
# per-block inner GMRES solves run at. This is NOT modeling inexactness that
# happens in the surrogate (its diagonal blocks are already a fixed approximate
# factorization); it is what makes distinct orderings distinguishable at all.
# Unperturbed, orderings needing 54 to 100 real iterations all converge within
# one or two surrogate iterations of each other. See FGMRES.solve_reduced's
# docstring and the wiki's vtilde-from-z.
SURROGATE_JITTER = 1e-4
SURROGATE_TOL = 1e-8        # relative residual the surrogate solve stops at
MAX_SEARCH_COUNT = 600      # orderings scored per request, RandomSearch's cap

# Fraction of the observed iteration range that counts as "as good as the
# best". See pick_opt_ordering.
OPT_RANGE_FRACTION = 0.2

_import_error = None
try:
    if str(PYAUTOTEKO_DIR) not in sys.path:
        sys.path.insert(0, str(PYAUTOTEKO_DIR))
    import numpy as np
    from block_matrix import BlockMatrix
    from fgmres import FGMRES
    from random_search import RandomSearch
except Exception as exc:                                # pragma: no cover
    _import_error = exc


def available():
    """True when pyautoteko (and its numpy/scipy) imported cleanly."""
    return _import_error is None


def unavailable_reason():
    """Why available() is False, as a one-line string (or None if it is True)."""
    if _import_error is None:
        return None
    return f"{type(_import_error).__name__}: {_import_error}"


class SearchResult:
    """What one request's search found.

    results      ordering tuple -> {"iters", "cost_per_iter", "cost"}, exactly
                 RandomSearch's own results dict (cost is the placeholder unit
                 cost, so only iters means anything for now)
    mode         which mode RandomSearch ran in: exhaustive, shuffled, random
    total        orderings that exist for this block count
    opt_ordering the pick, as a plain list (see pick_opt_ordering)
    threshold    the iteration count opt_ordering had to beat
    """

    def __init__(self, results, mode, total, opt_ordering, threshold):
        self.results = results
        self.mode = mode
        self.total = total
        self.opt_ordering = opt_ordering
        self.threshold = threshold

    @property
    def n_evaluated(self):
        return len(self.results)

    @property
    def iters_range(self):
        its = [r["iters"] for r in self.results.values()]
        return (min(its), max(its)) if its else (0, 0)

    def orderings(self):
        """Every ordering evaluated, as plain lists (JSON-serializable)."""
        return [list(o) for o in self.results]

    def iters_of(self, ordering):
        """The surrogate iteration count of one ordering, or None if this
        search did not score it."""
        r = self.results.get(tuple(ordering))
        return None if r is None else int(r["iters"])


def blocks_from_request(request):
    """Slice a request's flat C_hat / b_hat into a BlockMatrix pair.

    equation_ends[k] is the first row/column of block k and equation_ends[k+1]
    one past its last, so block k is rank_k wide: the surrogate carries one
    block per original block, sized by retained rank rather than by degrees of
    freedom. That is the blocking every candidate ordering is applied to.
    """
    ends = request["equation_ends"]
    n_blocks = request["n_blocks"]
    C = np.asarray(request["C_hat"], dtype=float)
    b = np.asarray(request["b_hat"], dtype=float).reshape(-1)

    C_blocks = [[C[ends[i]:ends[i + 1], ends[j]:ends[j + 1]]
                 for j in range(n_blocks)] for i in range(n_blocks)]
    b_blocks = [b[ends[j]:ends[j + 1]] for j in range(n_blocks)]
    return BlockMatrix(blocks=C_blocks), BlockMatrix(blocks=b_blocks)


def surrogate_iters(C_hat, b_hat, jitter=SURROGATE_JITTER):
    """A handle taking an ordering and returning its surrogate iteration count.

    Settings mirror FGMRES.solve_reduced, which is how pyautoteko evaluates its
    own surrogate: block Gauss-Seidel, the jitter above, and maxit capped at
    C_hat's own dimension (exact arithmetic cannot need more, so an ordering
    that fails to resolve reports that dimension rather than some larger budget).
    """
    n = C_hat.n
    settings = dict(precond="gs", inner_tol=jitter, tol=SURROGATE_TOL,
                    maxit=n, restart=n, autosolve=True)

    def iters(ordering):
        return FGMRES(C_hat, b_hat, order=list(ordering), settings=settings).iters

    return iters


def pick_opt_ordering(results, range_fraction=OPT_RANGE_FRACTION):
    """The ordering to apply, out of a RandomSearch results dict.

    Take the iteration counts over the whole searched space and set

        threshold = (max_iters - min_iters) * range_fraction + min_iters,

    then among every ordering at or below that threshold pick the one with the
    FEWEST mergers, that is the most groups. Rationale: inside the top of the
    range the surrogate is not really separating these candidates, and a merged
    group is a bigger, worse conditioned block to factor, so the cheapest way
    to spend a tie is to merge as little as possible. This stands in for a cost
    model of the real system, which does not exist yet (the cost handle is the
    placeholder unit_cost).

    Ties break on fewer iterations, then on the ordering tuple itself, so the
    pick is deterministic. Returns a plain list; the empty results dict gives
    (None, 0.0).
    """
    if not results:
        return None, 0.0
    its = [r["iters"] for r in results.values()]
    threshold = (max(its) - min(its)) * range_fraction + min(its)
    within = [(o, r["iters"]) for o, r in results.items() if r["iters"] <= threshold]
    # most groups first, then fewest iterations, then the tuple for determinism
    best = min(within, key=lambda oi: (-len(set(oi[0])), oi[1], oi[0]))
    return list(best[0]), threshold


def search(request, max_search_count=MAX_SEARCH_COUNT, jitter=SURROGATE_JITTER,
           seed=0):
    """Search this request's surrogate and return a SearchResult.

    RandomSearch picks its own mode from the block count: every ordering when
    there are few enough (541 at n_blocks = 5, all scored in about a second),
    a shuffled or sampled subset when there are not. No cost handle is passed,
    so cost falls back to unit_cost and the ranking is on iterations alone.

    Raises if pyautoteko is not importable; call available() first.
    """
    if not available():
        raise RuntimeError(f"pyautoteko is not importable: {unavailable_reason()}")

    C_hat, b_hat = blocks_from_request(request)
    rs = RandomSearch(None, surrogate_iters(C_hat, b_hat, jitter=jitter),
                      request["n_blocks"], max_search_count=max_search_count,
                      seed=seed)
    opt_ordering, threshold = pick_opt_ordering(rs.results)
    return SearchResult(rs.results, rs.mode, rs.total_configs, opt_ordering,
                        threshold)
