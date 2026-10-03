"""Example-private config for the 3-D two-fluid blast (D3).

Factory lives next to the example. PRIVATE: never imported by xmf.config,
never exported by `import xmf`.
"""
from __future__ import annotations

from typing import Any

from xmf.config import SCHEMA


class RUN:
    n = 64
    tstop = 0.06
    cfl = 0.3
    gamma = 1.4
    riemann = "tvdlf"
    reconstruction = "pcm"
    rk = "rk1"
    # Sedov-like: E=1 in a ball of radius 3.5 Δx; ambient p is tiny.
    energy = 1.0
    r0_cells = 3.5
    p_amb = 1.0e-6
    rho_light = 1.0
    rho_heavy = 4.0
    plots = True
    log_every = 20


def blast_3d(
    n: int | None = None,
    tstop: float | None = None,
) -> dict[str, Any]:
    nn = int(RUN.n if n is None else n)
    ts = float(RUN.tstop if tstop is None else tstop)
    return {
        "schema": SCHEMA,
        "backend": "serial",
        "gamma": RUN.gamma,
        "units": {"density": 1.0, "velocity": 1.0, "length": 1.0},
        "numerics": {
            "riemann": RUN.riemann,
            "reconstruction": RUN.reconstruction,
            "rk": RUN.rk,
        },
        "grid": {
            "geometry": "cartesian",
            "ndim": 3,
            "n": [nn, nn, nn],
            "stretch": "none",
            "x1min": -0.5,
            "x1max": 0.5,
            "x2min": -0.5,
            "x2max": 0.5,
            "x3min": -0.5,
            "x3max": 0.5,
            "nghost": 2,
        },
        "time": {"cfl": RUN.cfl, "tstop": ts, "dt_max": 0.0},
        "fluids": [
            {"name": "light", "mass": 1.0, "charge": 0, "kind": "neutral"},
            {"name": "heavy", "mass": 1.0, "charge": 0, "kind": "neutral"},
        ],
        "electron": {"mode": "off"},
        "sources": [],
        "constants": {},
        "radiation": {"enabled": False, "absorbers": []},
        "boundary": {"x1": "copy", "x2": "copy", "x3": "copy"},
        "ic": {"kind": "none"},
    }
