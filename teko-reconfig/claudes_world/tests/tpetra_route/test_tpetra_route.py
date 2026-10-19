"""Tests for the hook's reach from an external app: registration under a static
link, and the Thyra and Tpetra-typed routes into it. Run with pytest, or
directly:

    python test_tpetra_route.py

Needs the route_driver binary, built against trilinos-build:

    cmake -S . -B build -G "Unix Makefiles" && cmake --build build

Each case runs the binary in a fresh process. With TEKO_ADAPTIVE_RECONFIG unset,
a hook that is reached prints its gate line as inert and returns without
waiting for a watcher, so the gate line is the evidence that a route reaches
it. With the gate on and the watcher answering, which solve's x the
application got back is read off TRUERES, the driver's own explicit residual,
against the residuals conv.json records for solve 1 and the final solve.
"""

import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARY = HERE / "build" / "route_driver"
ROOT = HERE.parents[4]
WATCHER = ROOT / "Trilinos" / "teko-reconfig" / "wait_for_request.py"
PYTHON = ROOT / "trilinos-teko-pyfront" / "mypy" / "bin" / "python3"
TEKO_OBJ_DIR = (ROOT / "trilinos-build" / "packages" / "teko" / "src" / "CMakeFiles"
                / "teko.dir")
INVERSE_FACTORY_OBJ = TEKO_OBJ_DIR / "Teko_InverseFactory.cpp.o"
LINK_TXT = HERE / "build" / "CMakeFiles" / "route_driver.dir" / "link.txt"
TIMEOUT_SEC = 120
GATES = ("TEKO_ADAPTIVE_RECONFIG", "TEKO_FACTOR_WARMUP", "TEKO_RECONFIG_REQUESTS_DIR",
         "TEKO_RECONFIG_DRY_RUN")
ROUTES = ("thyra", "tpetra")
# The rescue cases stall solve 1 at this many outer iterations.
STALL_ITERS = 2


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k not in GATES}
    env.update(extra or {})
    return env


def run_binary(args, env, binary=BINARY):
    proc = subprocess.run([str(binary), *map(str, args)], env=env, capture_output=True,
                          text=True, timeout=TIMEOUT_SEC, check=True)
    return proc.stdout, proc.stderr


@lru_cache(maxsize=None)
def run_inert(route):
    """One case with every gate unset."""
    return run_binary([route], clean_env())


def run_with_watcher(args, extra_env=None):
    """Gate on, a watcher answering in a temporary requests dir. Returns
    (stdout, stderr, conv) with conv the s0_conv.json the run wrote."""
    with tempfile.TemporaryDirectory() as d:
        env = clean_env({"TEKO_RECONFIG_REQUESTS_DIR": d, "TEKO_WATCHER_IDLE_TIMEOUT": "120"})
        watcher = subprocess.Popen([str(PYTHON), str(WATCHER)], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            out, err = run_binary(args, dict(env, TEKO_ADAPTIVE_RECONFIG="1",
                                             **(extra_env or {})))
        finally:
            watcher.terminate()
            watcher.wait(timeout=10)
        conv = json.loads((Path(d) / "s0_conv.json").read_text())
    return out, err, conv


def true_residual(out):
    match = re.search(r"^TRUERES (\S+)$", out, re.MULTILINE)
    assert match, f"no TRUERES line:\n{out}"
    return float(match.group(1))


def solver_converged(out):
    match = re.search(r"^SOLVE (CONVERGED|UNCONVERGED)$", out, re.MULTILINE)
    assert match, f"no SOLVE line:\n{out}"
    return match.group(1) == "CONVERGED"


def same_residual(a, b):
    """Two explicit residuals of one x, computed by different code."""
    return abs(a - b) <= 1e-6 * max(abs(a), abs(b))


def test_buildinverse_references_registration_for_static_links():
    # A static link keeps an object file only if something references it. The
    # free buildInverse must therefore reference ensureAdaptiveHookRegistered,
    # which lives in the registration object, so linking buildInverse drags it
    # (and its static constructor) in.
    syms = subprocess.run(["nm", "-C", str(INVERSE_FACTORY_OBJ)], capture_output=True,
                          text=True, check=True).stdout
    assert re.search(r"^\s+U Teko::KrylovSurrogate::ensureAdaptiveHookRegistered\(char const\*\)$",
                     syms, re.MULTILINE), "buildInverse does not reference the registration"


def test_both_routes_reach_the_hook():
    for route in ROUTES:
        out, err = run_inert(route)
        assert "-> hook inert" in out and "-> hook inert" in err, f"{route}:\n{out}{err}"
        assert solver_converged(out) and true_residual(out) < 1e-6


def test_static_libteko_still_registers():
    # Link the driver against a static libteko.a packed from this build's
    # object files, everything else shared as before. The link map must show
    # the registration object pulled in by buildInverse's reference, and both
    # routes must still reach the hook.
    with tempfile.TemporaryDirectory() as d:
        archive, exe, link_map = Path(d) / "libteko.a", Path(d) / "route_static", Path(d) / "link.map"
        subprocess.run(["ar", "rcs", str(archive), *map(str, TEKO_OBJ_DIR.rglob("*.o"))], check=True)
        cmd = LINK_TXT.read_text().strip()
        shared = re.search(r"\S*/libteko\.so[.\d]*", cmd).group(0)
        cmd = cmd.replace(shared, str(archive)).replace(
            "-o route_driver ", f"-o {exe} -Wl,-Map,{link_map} ")
        subprocess.run(cmd, shell=True, cwd=BINARY.parent, check=True)

        assert "libteko" not in subprocess.run(["ldd", str(exe)], capture_output=True,
                                               text=True, check=True).stdout
        assert re.search(r"libteko\.a\(Teko_KrylovSurrogateInit\.cpp\.o\)\n\s+\S*"
                         r"libteko\.a\(Teko_InverseFactory\.cpp\.o\) "
                         r"\(Teko::KrylovSurrogate::ensureAdaptiveHookRegistered",
                         link_map.read_text()), "registration object not pulled in by buildInverse"
        for route in ROUTES:
            out, _ = run_binary([route], clean_env(), binary=exe)
            assert "-> hook inert" in out, f"static {route}:\n{out}"


def test_stalled_solve_is_rescued_into_the_applications_x():
    # Solve 1 stalls; the final solve converges and its x is what the
    # application holds: in the Thyra problem's own LHS, or copied back into the
    # application's Tpetra vector.
    for route in ROUTES:
        out, err, conv = run_with_watcher([route, STALL_ITERS])
        initial, final = conv["solves"][0], conv["solves"][-1]
        assert initial["type"] == "initial" and not initial["converged"], route
        assert final["type"].startswith("final") and final["converged"], route
        assert len(final["ordering"]) == 2
        assert solver_converged(out), f"{route}:\n{out}{err}"
        assert same_residual(true_residual(out), final["final_residual"]), route
        assert true_residual(out) < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
