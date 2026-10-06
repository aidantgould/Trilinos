"""Tests for the diagnostic branch's [TekoDiag] prints. Run with pytest, or
directly:

    python test_hook_diag.py

Needs the hook_diag binary, built against trilinos-build:

    cmake -S . -B build -G "Unix Makefiles" && cmake --build build

Each case runs the binary in a fresh process with TEKO_ADAPTIVE_RECONFIG
unset, so a hook that is reached prints its gate line as inert and returns
without waiting for a watcher. The cases are the ways an external app can miss
the hook (Pseudo Block GMRES, a flat operator) plus the one that reaches it,
and each test checks that the prints name what happened.
"""

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

BINARY = Path(__file__).resolve().parent / "build" / "hook_diag"
TIMEOUT_SEC = 60
GATES = ("TEKO_ADAPTIVE_RECONFIG", "TEKO_FACTOR_WARMUP", "TEKO_RECONFIG_REQUESTS_DIR")
DIAG_RE = re.compile(r"^\[TekoDiag (\w+)\] pid=\d+ (.*)$", re.MULTILINE)


@lru_cache(maxsize=None)
def run(solver, layout, *extra):
    """Run one case and return (stdout, stderr)."""
    env = {k: v for k, v in os.environ.items() if k not in GATES}
    proc = subprocess.run([str(BINARY), solver, layout, *extra], env=env, capture_output=True,
                          text=True, timeout=TIMEOUT_SEC, check=True)
    assert "SOLVE SOLVE_STATUS_CONVERGED" in proc.stdout, f"solve failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout, proc.stderr


def diag(solver, layout, *extra):
    """The [TekoDiag] lines of one case, as {id: [message, ...]}."""
    lines = {}
    for tag, msg in DIAG_RE.findall(run(solver, layout, *extra)[0]):
        lines.setdefault(tag, []).append(msg)
    return lines


def slot(msg):
    match = re.search(r"slot=(0x[0-9a-f]+)", msg)
    assert match, f"no slot address in: {msg}"
    return match.group(1)


def test_every_line_on_both_streams():
    for case in [("flex", "blocked"), ("pseudo", "flat")]:
        out, err = run(*case)
        assert DIAG_RE.findall(out) == DIAG_RE.findall(err)


def test_one_libbelos_copy_and_registration_ran():
    d = diag("flex", "blocked")
    assert len(d["L1"]) == 1
    assert "Teko registration ctor ran" in d["R1"][0] and "before=EMPTY" in d["R1"][0]
    assert "now SET" in d["H1"][0]


def test_flex_blocked_reaches_hook_through_one_slot():
    d = diag("flex", "blocked")
    assert "effective=yes" in d["B1"][0] and "opBlocked=yes" in d["B1"][0]
    assert "-> calling invoke" in d["B2"][0]
    assert "SET, calling it" in d["H2"][0]
    assert len({slot(d["L1"][0]), slot(d["R1"][0]), slot(d["H1"][0]), slot(d["H2"][0])}) == 1
    assert "hook inert" in run("flex", "blocked")[0]


def test_flex_flat_names_the_unblocked_operator():
    d = diag("flex", "flat")
    assert "opBlocked=no" in d["S1"][0]
    assert "effective=yes" in d["B1"][0] and "opBlocked=no" in d["B1"][0]
    assert "opBlocked=no" in d["B2"][0] and "NOT calling invoke" in d["B2"][0]
    assert "H2" not in d


def test_inner_block_gmres_cannot_crowd_out_the_outer_solve():
    # Every inner block solve is a non-flexible Block GMRES printing B1/B2 of
    # its own; the outer flexible solve's lines must still come through.
    d = diag("flex", "blocked", "inner-gmres")
    inner = [m for m in d["B2"] if "isFlexible=no" in m]
    assert inner, "inner Block GMRES solves printed no B2"
    assert any("effective=yes" in m and "opBlocked=yes" in m for m in d["B1"])
    assert any("isFlexible=yes" in m and "-> calling invoke" in m for m in d["B2"])
    assert "SET, calling it" in d["H2"][0]


def test_pseudo_never_reaches_block_gmres():
    for layout in ("flat", "blocked"):
        d = diag("pseudo", layout)
        assert 'Solver Type="Pseudo Block GMRES"' in d["S1"][0]
        assert "PseudoBlockGmresSolMgr::solve entered" in d["P1"][0]
        assert "B1" not in d and "H2" not in d


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
