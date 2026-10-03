"""Radiation-on rad_hhe_pe_1d CUDA wall-clock probe (C2).

Same knobs as hhe_omp_bench.py (nx=256, stretch=1.015, dt_max=5,
tstop=2000). Writes output/hhe_cuda_bench/ (gitignored).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from _paths import EXAMPLE, OUT_CUDA as OUT  # noqa: E402

sys.path.insert(0, str(EXAMPLE))

import xmf  # noqa: E402
from hhe_config import RUN  # noqa: E402
from run import one_run  # noqa: E402

FLUIDS = ("HI", "HEI", "HII", "HEII")


def apply_knobs() -> None:
    RUN.nx = 256
    RUN.stretch = "log"
    RUN.stretch_factor = 1.015
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
    apply_knobs()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"xmf={Path(xmf.__file__).resolve()}", flush=True)
    t0 = time.perf_counter()
    res = one_run(True)
    wall = time.perf_counter() - t0
    fields = pack_fields(res)
    npz = OUT / "cuda.npz"
    np.savez_compressed(npz, **fields)
    meta = {
        "backend": "cuda",
        "xmf": str(Path(xmf.__file__).resolve()),
        "nx": int(RUN.nx),
        "stretch_factor": float(RUN.stretch_factor),
        "tstop": float(RUN.tstop),
        "dt_max": float(RUN.dt_max),
        "steps": int(res["steps"]),
        "t_final": float(res["t_final"]),
        "dt_final": float(res["dt_final"]),
        "resid_final": float(res["resid_final"]),
        "resid_max": float(res["resid_max"]),
        "wall_one_run_s": float(res["wall"]),
        "wall_process_s": float(wall),
        "npz": str(npz),
    }
    (OUT / "cuda.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
