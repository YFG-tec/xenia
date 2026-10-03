"""Example-private config for 2-D flow past a circular cylinder.

Factory lives next to the example. PRIVATE: never imported by xmf.config.

Layout follows the classic circular-cylinder wake
(Braza, Chassaing & Ha Minh 1986, *JFM* 165, 79; Williamson 1996 review):
uniform stream, circular body, long downstream box. The kernel has no
physical ν (that is paper 3); the body is a no-slip mask pinned in the
example layer, and the wake Re is numerical (hllc + weno3).
"""
from __future__ import annotations

from typing import Any

from xmf.config import SCHEMA


class RUN:
    nx = 192
    ny = 128
    tstop = 16.0
    cfl = 0.35
    gamma = 1.4
    riemann = "hllc"
    reconstruction = "weno3"
    rk = "rk2"
    # Cylinder diameter D = 1 at the origin.
    D = 1.0
    x1min = -4.0
    x1max = 12.0
    x2min = -5.0
    x2max = 5.0
    rho0 = 1.0
    U_inf = 1.0
    Mach = 0.25
    # Braza-style trip: a short lift kick so shedding starts at once.
    trip_amp = 0.08
    trip_until = 2.0
    inlet_cells = 2
    plots = True
    log_every = 50
    snap_dt = 0.25


def sound_speed() -> float:
    return RUN.U_inf / RUN.Mach


def pressure() -> float:
    return RUN.rho0 * sound_speed() ** 2 / RUN.gamma


def cyl_2d(
    nx: int | None = None,
    ny: int | None = None,
    tstop: float | None = None,
) -> dict[str, Any]:
    nnx = int(RUN.nx if nx is None else nx)
    nny = int(RUN.ny if ny is None else ny)
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
            "ndim": 2,
            "n": [nnx, nny, 1],
            "stretch": "none",
            "x1min": RUN.x1min,
            "x1max": RUN.x1max,
            "x2min": RUN.x2min,
            "x2max": RUN.x2max,
            "x3min": -0.5,
            "x3max": 0.5,
            "nghost": 2,
        },
        "time": {"cfl": RUN.cfl, "tstop": ts, "dt_max": 0.0},
        "fluids": [
            {"name": "gas", "mass": 1.0, "charge": 0, "kind": "neutral"},
        ],
        "electron": {"mode": "off"},
        "sources": [],
        "constants": {},
        "radiation": {"enabled": False, "absorbers": []},
        "boundary": {"x1beg": "user", "x1end": "copy", "x2": "copy", "x3": "copy"},
        "ic": {"kind": "none"},
    }
