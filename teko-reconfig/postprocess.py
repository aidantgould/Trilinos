#!/usr/bin/env python3
"""Post-process Teko adaptive reconfiguration convergence files."""

import argparse
import json
import math
import re
from pathlib import Path
import os
import sys


def _ensure_mypy_venv():
    script_dir = Path(__file__).resolve().parent
    venv_dir = script_dir.parents[1] / "trilinos-teko-pyfront" / "mypy"
    venv_python = venv_dir / "bin" / "python3"

    if Path(sys.prefix).resolve() == venv_dir.resolve():
        return
    if not venv_python.exists():
        raise RuntimeError(f"Expected Python venv at {venv_python}")

    os.environ["VIRTUAL_ENV"] = str(venv_dir)
    os.environ["PATH"] = str(venv_dir / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.pop("PYTHONHOME", None)
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_ensure_mypy_venv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CONV_RE = re.compile(r"s(\d+)_conv\.json$")


def first_of_type(solves, prefix):
    """The first row of a conv.json whose type starts with prefix, or {}."""
    return next((row for row in solves if row.get("type", "").startswith(prefix)), {})


def load_convergence(requests_dir):
    """One row per request: the initial solve against the final one. The
    tests a request also ran sit between them in conv.json and are not
    plotted here."""
    rows = []
    for path in sorted(requests_dir.glob("s*_conv.json")):
        match = CONV_RE.match(path.name)
        if match is None:
            continue

        with path.open() as f:
            data = json.load(f)

        request_id = int(data.get("request_id", match.group(1)))
        solves = data.get("solves") or []
        initial = first_of_type(solves, "initial")
        final = first_of_type(solves, "final")
        rows.append(
            {
                "request_id": request_id,
                "initial_iters": initial.get("iters", math.nan),
                "final_iters": final.get("iters", math.nan),
                "initial_wall_time_sec": initial.get("total_wall_time_sec", math.nan),
                "final_wall_time_sec": final.get("total_wall_time_sec", math.nan),
            }
        )

    return sorted(rows, key=lambda row: row["request_id"])


def plot_convergence(rows, output_path):
    if not rows:
        raise RuntimeError("No s<N>_conv.json files found")

    request_ids = [row["request_id"] for row in rows]
    initial_iters = [row["initial_iters"] for row in rows]
    final_iters = [row["final_iters"] for row in rows]
    # a null time (a broken-down solve) plots as a gap
    initial_times = [math.nan if row["initial_wall_time_sec"] is None
                     else row["initial_wall_time_sec"] for row in rows]
    final_times = [math.nan if row["final_wall_time_sec"] is None
                   else row["final_wall_time_sec"] for row in rows]

    fig, (ax_iters, ax_time) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_iters.plot(request_ids, initial_iters, marker="o", label="Initial")
    ax_iters.plot(request_ids, final_iters, marker="s", label="Final")
    ax_iters.set_ylabel("Iterations")
    ax_iters.set_title("Teko Adaptive Reconfiguration")
    ax_iters.grid(True, alpha=0.3)
    ax_iters.legend()

    ax_time.plot(request_ids, initial_times, marker="o", label="Initial")
    ax_time.plot(request_ids, final_times, marker="s", label="Final")
    ax_time.set_xlabel("Request number")
    ax_time.set_ylabel("Total wall time (s)")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests-dir",
        type=Path,
        default=script_dir / "requests",
        help="Directory containing s<N>_conv.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "figs" / "reconfig_convergence.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    rows = load_convergence(args.requests_dir)
    plot_convergence(rows, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
