"""Tests for the diagnostic branch's [TekoDiag] prints. Run with pytest, or
directly:

    python test_hook_diag.py

Needs the hook_diag binary, built against trilinos-build:

    cmake -S . -B build -G "Unix Makefiles" && cmake --build build

Each case runs the binary in a fresh process with TEKO_ADAPTIVE_RECONFIG
unset, so a hook that is reached prints its gate line as inert and returns
without waiting for a watcher. The cases are the ways an external app can miss
the hook (Pseudo Block GMRES, a flat operator) plus the ones that reach it
(Thyra-typed, and Tpetra-typed through the type-erased route), and each test
checks that the prints name what happened. One test runs the Tpetra route end
to end with the gate on and the watcher answering.
"""

import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARY = HERE / "build" / "hook_diag"
ROOT = HERE.parents[4]
WATCHER = ROOT / "Trilinos" / "teko-reconfig" / "wait_for_request.py"
PYTHON = ROOT / "trilinos-teko-pyfront" / "mypy" / "bin" / "python3"
TEKO_OBJ_DIR = (ROOT / "trilinos-build" / "packages" / "teko" / "src" / "CMakeFiles"
                / "teko.dir")
INVERSE_FACTORY_OBJ = TEKO_OBJ_DIR / "Teko_InverseFactory.cpp.o"
LINK_TXT = HERE / "build" / "CMakeFiles" / "hook_diag.dir" / "link.txt"
TIMEOUT_SEC = 60
GATES = ("TEKO_ADAPTIVE_RECONFIG", "TEKO_FACTOR_WARMUP", "TEKO_RECONFIG_REQUESTS_DIR")
DIAG_RE = re.compile(r"^\[TekoDiag (\w+)\] pid=\d+ (.*)$", re.MULTILINE)


def run_with(args, extra_env=None):
    """Run the binary with `args` and return (stdout, stderr)."""
    env = {k: v for k, v in os.environ.items() if k not in GATES}
    env.update(extra_env or {})
    proc = subprocess.run([str(BINARY), *args], env=env, capture_output=True,
                          text=True, timeout=TIMEOUT_SEC, check=True)
    assert "SOLVE SOLVE_STATUS_CONVERGED" in proc.stdout, f"solve failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout, proc.stderr


@lru_cache(maxsize=None)
def run(*args):
    """Run one case with every gate unset."""
    return run_with(args)


def parse(out):
    lines = {}
    for tag, msg in DIAG_RE.findall(out):
        lines.setdefault(tag, []).append(msg)
    return lines


def diag(*args):
    """The [TekoDiag] lines of one case, as {id: [message, ...]}."""
    return parse(run(*args)[0])


def true_residual(out):
    match = re.search(r"^TRUERES (\S+)$", out, re.MULTILINE)
    assert match, f"no TRUERES line:\n{out}"
    return float(match.group(1))


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
    assert "Teko registration ran via" in d["R1"][0] and "before=EMPTY" in d["R1"][0]
    assert "now SET" in d["H1"][0] and "now SET" in d["H3"][0]


def test_buildinverse_references_registration_for_static_links():
    # A static link keeps an object file only if something references it. The
    # free buildInverse must therefore reference ensureAdaptiveHookRegistered,
    # which lives in the registration object, so linking buildInverse drags it
    # (and its static constructor) in.
    syms = subprocess.run(["nm", "-C", str(INVERSE_FACTORY_OBJ)], capture_output=True,
                          text=True, check=True).stdout
    assert re.search(r"^\s+U Teko::KrylovSurrogate::ensureAdaptiveHookRegistered\(char const\*\)$",
                     syms, re.MULTILINE), "buildInverse does not reference the registration"


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


def test_static_libteko_still_registers():
    # The real thing: link hook_diag against a static libteko.a packed from
    # this build's object files, everything else shared as before. The link map
    # must show the registration object pulled in by buildInverse's reference,
    # and the run must register and reach the hook.
    with tempfile.TemporaryDirectory() as d:
        archive, exe, link_map = Path(d) / "libteko.a", Path(d) / "hook_diag_static", Path(d) / "link.map"
        subprocess.run(["ar", "rcs", str(archive), *map(str, TEKO_OBJ_DIR.rglob("*.o"))], check=True)
        cmd = LINK_TXT.read_text().strip()
        shared = re.search(r"\S*/libteko\.so[.\d]*", cmd).group(0)
        cmd = cmd.replace(shared, str(archive)).replace(
            "-o hook_diag ", f"-o {exe} -Wl,-Map,{link_map} ")
        subprocess.run(cmd, shell=True, cwd=BINARY.parent, check=True)

        assert "libteko" not in subprocess.run(["ldd", str(exe)], capture_output=True,
                                               text=True, check=True).stdout
        assert re.search(r"libteko\.a\(Teko_KrylovSurrogateInit\.cpp\.o\)\n\s+\S*"
                         r"libteko\.a\(Teko_InverseFactory\.cpp\.o\) "
                         r"\(Teko::KrylovSurrogate::ensureAdaptiveHookRegistered",
                         link_map.read_text()), "registration object not pulled in by buildInverse"

        env = {k: v for k, v in os.environ.items() if k not in GATES}
        out = subprocess.run([str(exe), "flex", "blocked"], env=env, capture_output=True,
                             text=True, timeout=TIMEOUT_SEC, check=True).stdout
        d_ = parse(out)
        assert "Teko registration ran via" in d_["R1"][0]
        assert "SET, calling it" in d_["H2"][0]


def test_tpetra_flex_reaches_the_erased_hook():
    out, _ = run("tpetra-flex")
    d = parse(out)
    assert "erased hook path" in d["B1"][0] and "InverseFactoryOperator" in d["B1"][0]
    assert any("-> calling invokeErased" in m for m in d["B4"])
    assert "SET, calling it" in d["H4"][0] and "Tpetra::MultiVector" in d["H4"][0]
    assert "types match" in d["K1"][0]
    assert any("rightPrec.getForwardOp()" in m and "accepted" in m for m in d["K2"])
    assert "blocks=2 sizes=[20,20]" in d["K3"][0]
    assert "hook inert" in out
    assert "no converged re-solve" in d["K4"][0]
    assert true_residual(out) < 1e-6


def test_tpetra_pseudo_never_reaches_a_hook():
    d = diag("tpetra-pseudo")
    assert "InverseFactoryOperator" in d["P1"][0]
    assert "B1" not in d and "H4" not in d


def test_tpetra_route_full_loop_with_watcher():
    # Gate on, watcher answering: the Tpetra-typed solve is reconfigured, the
    # re-solve's x lands in the application's own Tpetra vector, and the
    # request/conv files look like any other run's.
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, TEKO_RECONFIG_REQUESTS_DIR=d, TEKO_WATCHER_IDLE_TIMEOUT="120")
        watcher = subprocess.Popen([str(PYTHON), str(WATCHER)], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            out, err = run_with(["tpetra-flex"], {"TEKO_ADAPTIVE_RECONFIG": "1",
                                                  "TEKO_RECONFIG_REQUESTS_DIR": d})
        finally:
            watcher.terminate()
            watcher.wait(timeout=10)
        lines = parse(out)
        assert "hook ACTIVE" in out
        assert any("copied into the application's Tpetra LHS" in m for m in lines["K4"]), out + err
        assert true_residual(out) < 1e-6
        conv = json.loads((Path(d) / "s0_conv.json").read_text())
        final = conv["solves"][-1]
        assert final["type"].startswith("final") and final["converged"]
        assert len(final["ordering"]) == 2


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
