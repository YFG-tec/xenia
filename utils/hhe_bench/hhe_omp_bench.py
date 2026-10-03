"""Radiation-on rad_hhe_pe_1d timing/compare probe (C1b OpenMP).

Overrides the example RUN knobs (nx=256, stretch_factor=1.015, no plots)
and writes dumps under output/hhe_omp_bench/ (gitignored).

Usage (separate process per thread count; Kokkos reads OMP_NUM_THREADS
at first initialize):

    set OMP_NUM_THREADS=1
    python utils/hhe_bench/hhe_omp_bench.py --threads 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _paths import EXAMPLE, OUT_OMP as OUT  # noqa: E402

sys.path.insert(0, str(EXAMPLE))

import hhe_config  # noqa: E402
from hhe_config import RUN  # noqa: E402
from run import one_run  # noqa: E402

FLUIDS = ("HI", "HEI", "HII", "HEII")


def apply_knobs() -> None:
    RUN.nx = 256
    RUN.stretch = "log"
    RUN.stretch_factor = 1.015
    # Factory dt_max=10 is the nx=128 / stretch=1.025 hat. On this mesh
    # (inner dx/Rp 3.05e-3 vs 9.96e-3) dt=10 grows a startup mode:
    # resid ~1e27 and CFL collapse near t=2.5e3 s (hhe_nx256_fail_probe.py).
    # dt_max=5 kept resid O(1) and |vx| bounded through t=4.5e3 s
    # (hhe_nx256_hat_probe.py). Cap only; CFL formula unchanged.
    RUN.dt_max = 5.0
    RUN.tstop = 2000.0
    RUN.plots = False
    RUN.log_every = 100


def pack_fields(res: dict) -> dict[str, np.ndarray]:
    out = {
        "r": np.asarray(res["r"], dtype=np.float64),
        "P_e": np.asarray(res["final"]["P_e"], dtype=np.float64),
        "T_e": np.asarray(res["final"]["T_e"], dtype=np.float64),
        "flux": np.asarray(res["flux"], dtype=np.float64),
    }
    for name in FLUIDS:
        blk = res["final"][name]
        for key in ("n", "vx", "P", "T"):
            out[f"{name}_{key}"] = np.asarray(blk[key], dtype=np.float64)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--tstop",
        type=float,
        default=None,
        help="override RUN.tstop (default: factory 2e5 s)",
    )
    args = parser.parse_args()
    apply_knobs()
    if args.tstop is not None:
        RUN.tstop = float(args.tstop)
    OUT.mkdir(parents=True, exist_ok=True)

    omp = os.environ.get("OMP_NUM_THREADS", "")
    t0 = time.perf_counter()
    res = one_run(True)
    wall = time.perf_counter() - t0

    fields = pack_fields(res)
    npz = OUT / f"nt{args.threads}.npz"
    np.savez_compressed(npz, **fields)

    meta = {
        "threads_arg": args.threads,
        "OMP_NUM_THREADS": omp,
        "nx": int(RUN.nx),
        "stretch_factor": float(RUN.stretch_factor),
        "tstop": float(RUN.tstop),
        "dt_max": float(getattr(RUN, "dt_max", float("nan"))),
        "steps": int(res["steps"]),
        "t_final": float(res["t_final"]),
        "dt_final": float(res["dt_final"]),
        "resid_final": float(res["resid_final"]),
        "resid_max": float(res["resid_max"]),
        "wall_one_run_s": float(res["wall"]),
        "wall_process_s": float(wall),
        "npz": str(npz),
    }
    (OUT / f"nt{args.threads}.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
