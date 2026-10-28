"""Tests for the hook's reach from an external app: registration under a static
link, the Thyra and Tpetra-typed routes into it, and on both routes that a dry
run (TEKO_RECONFIG_DRY_RUN) leaves the application with solve 1. Run with
pytest, or directly:

    python test_tpetra_route.py

Needs the route_driver binary, built against trilinos-build:

    cmake -S . -B build -G "Unix Makefiles" && cmake --build build

Each case runs the binary in a fresh process. With TEKO_ADAPTIVE_RECONFIG unset,
a hook that is reached prints its gate line as inert and returns without
waiting for a watcher, so the gate line is the evidence that a route reaches
it. With the gate on and the watcher answering, which solve's x the
application got back is read off TRUERES, the driver's own explicit residual,
against the residuals conv.json records for solve 1 and the final solve.
The residual-target cases set the driver's ROUTE_* knobs the way a real
application varies them (a small b, a warm start, absolute residual scaling)
and check what conv.json and the request record against the driver's INITRES.
"""

import json
import math
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
# route_driver's kTol, and its system size: two fields of kNodes = 20.
TOL = 1e-8
N_ROWS = 40
# The residual cases' small right-hand side, about what the real application has.
SMALL_B = 1e-4


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
    (stdout, stderr, conv, request) with conv and request the s0_conv.json and
    s0_request.json the run wrote."""
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
        request = json.loads((Path(d) / "s0_request.json").read_text())
    return out, err, conv, request


def driver_residual(out, tag):
    match = re.search(rf"^{tag} (\S+)$", out, re.MULTILINE)
    assert match, f"no {tag} line:\n{out}"
    return float(match.group(1))


def true_residual(out):
    return driver_residual(out, "TRUERES")


def close(a, b, rel=1e-10):
    return abs(a - b) <= rel * max(abs(a), abs(b))


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
        out, err, conv, _ = run_with_watcher([route, STALL_ITERS])
        initial, final = conv["solves"][0], conv["solves"][-1]
        assert initial["type"] == "initial" and not initial["converged"], route
        assert final["type"].startswith("final") and final["converged"], route
        assert len(final["ordering"]) == 2
        assert solver_converged(out), f"{route}:\n{out}{err}"
        assert same_residual(true_residual(out), final["final_residual"]), route
        assert true_residual(out) < 1e-6
        assert conv["applied"] is True


def test_dry_run_keeps_the_stalled_solve_1():
    # The same stall with TEKO_RECONFIG_DRY_RUN: the final solve still runs and
    # converges in conv.json, but the application holds solve 1's x and Belos
    # reports solve 1's stall.
    for route in ROUTES:
        out, err, conv, _ = run_with_watcher([route, STALL_ITERS], {"TEKO_RECONFIG_DRY_RUN": "1"})
        initial, final = conv["solves"][0], conv["solves"][-1]
        assert "dry run: TEKO_RECONFIG_DRY_RUN set" in out and "not applied" in out, route
        assert not initial["converged"] and final["converged"], route
        assert conv["applied"] is False
        assert not solver_converged(out), f"{route}:\n{out}{err}"
        assert same_residual(true_residual(out), initial["final_residual"]), route
        assert not same_residual(true_residual(out), final["final_residual"]), route


def test_dry_run_keeps_a_converged_solve_1():
    for route in ROUTES:
        out, err, conv, _ = run_with_watcher([route], {"TEKO_RECONFIG_DRY_RUN": "1"})
        initial, final = conv["solves"][0], conv["solves"][-1]
        assert initial["converged"] and final["converged"], route
        assert conv["applied"] is False
        assert solver_converged(out), f"{route}:\n{out}{err}"
        assert same_residual(true_residual(out), initial["final_residual"]), route


def test_small_rhs_records_relative_residuals():
    # A right-hand side of norm ~1e-4 under Belos's default scaling. Every row
    # starts at relative residual 1, keeps ||b|| as its absolute one, and is
    # held to the raw tolerance, which is also what the surrogate is sent.
    b_norm = SMALL_B * math.sqrt(N_ROWS)
    for route in ROUTES:
        out, err, conv, request = run_with_watcher([route], {"ROUTE_B_SCALE": str(SMALL_B)})
        assert close(driver_residual(out, "INITRES"), 1.0), route
        assert close(request["target_tol"], TOL), route
        for row in conv["solves"]:
            assert close(row["initial_residual"], b_norm), f"{route}: {row}"
            assert close(row["initial_relative_residual"], 1.0), f"{route}: {row}"
            assert close(row["target_tol"], TOL), f"{route}: {row}"
            assert row["converged"], f"{route}: {row}"


def test_absolute_scaling_converts_the_target():
    # "None" scaling makes the tolerance absolute: the solves stop at a
    # relative residual of TOL / ||b||, and the verdicts must judge them
    # against that, not against TOL. This system converges in one or two
    # iterations, overshooting any target, so b is smaller here than SMALL_B:
    # at 1e-6 the one-iteration solves stop at about 1.2e-5, above the 1e-5 the
    # raw-tolerance verdict allowed.
    b_scale = 1e-6
    target = TOL / (b_scale * math.sqrt(N_ROWS))
    for route in ROUTES:
        out, err, conv, request = run_with_watcher(
            [route], {"ROUTE_B_SCALE": str(b_scale), "ROUTE_RES_SCALING": "None"})
        assert close(request["target_tol"], target), route
        assert solver_converged(out), f"{route}:\n{out}{err}"
        for row in conv["solves"]:
            assert close(row["target_tol"], target), f"{route}: {row}"
            assert row["converged"], f"{route}: {row}"
        # some solve really did stop short of the raw-tolerance verdict's bound
        assert any(row["final_residual"] > 1e3 * TOL for row in conv["solves"]), route
        assert conv["applied"] is True


def test_warm_start_records_solve_1s_own_initial_residual():
    # A nonzero x0: solve 1's initial residual is ||b - A x0||, which the driver
    # computes itself, and under the default scaling its target shrinks by the
    # same factor. The tests still start from zero.
    for route in ROUTES:
        out, err, conv, request = run_with_watcher([route], {"ROUTE_X0": "0.4"})
        init_rel = driver_residual(out, "INITRES")
        assert 0.0 < init_rel < 0.9, route
        b_norm = math.sqrt(N_ROWS)
        initial, rest = conv["solves"][0], conv["solves"][1:]
        assert close(initial["initial_relative_residual"], init_rel, 1e-8), f"{route}: {initial}"
        assert close(initial["initial_residual"], init_rel * b_norm, 1e-8), f"{route}: {initial}"
        assert close(initial["target_tol"], TOL * init_rel, 1e-8), f"{route}: {initial}"
        assert close(request["target_tol"], TOL), route
        for row in rest:
            assert close(row["initial_relative_residual"], 1.0), f"{route}: {row}"
            assert close(row["target_tol"], TOL), f"{route}: {row}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
