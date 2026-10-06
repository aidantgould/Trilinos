"""Tests for the factorization warm-up's gate and its safety under a nested
Teko preconditioner. Run with pytest, or directly:

    python test_warmup_nested.py

Needs the warmup_nested binary, built against trilinos-build:

    cmake -S . -B build -G "Unix Makefiles" && cmake --build build

Each case runs the binary in a fresh process, since both environment
variables are read once per process. The binary prints how many times its
top-level factory was asked to build: 2 when the warm-up ran, 1 when it did
not. A timeout counts as a failure, because the bug this guards against is a
hang (a nested buildInverse re-entering the warm-up's call_once).
"""

import os
import re
import subprocess
from pathlib import Path

BINARY = Path(__file__).resolve().parent / "build" / "warmup_nested"
TIMEOUT_SEC = 60
GATES = ("TEKO_ADAPTIVE_RECONFIG", "TEKO_FACTOR_WARMUP")


def builds_under(adaptive, warmup):
    """Run the binary with the two gates set as given (None means unset) and
    return the build count it prints."""
    env = {k: v for k, v in os.environ.items() if k not in GATES}
    for name, value in zip(GATES, (adaptive, warmup)):
        if value is not None:
            env[name] = value
    try:
        out = subprocess.run([str(BINARY)], env=env, capture_output=True,
                             text=True, timeout=TIMEOUT_SEC, check=True).stdout
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"hung for {TIMEOUT_SEC} s with TEKO_ADAPTIVE_RECONFIG={adaptive}, "
            f"TEKO_FACTOR_WARMUP={warmup}")
    match = re.search(r"^BUILDS (\d+)$", out, re.MULTILINE)
    assert match, f"no BUILDS line in output:\n{out}"
    return int(match.group(1))


def test_off_when_nothing_set():
    assert builds_under(None, None) == 1


def test_follows_adaptive_gate_when_warmup_unset():
    assert builds_under("1", None) == 2


def test_adaptive_gate_off_means_no_warmup():
    assert builds_under("0", None) == 1


def test_explicit_warmup_off_wins_over_adaptive():
    assert builds_under("1", "0") == 1


def test_explicit_warmup_on_without_adaptive():
    assert builds_under(None, "1") == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
