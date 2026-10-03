"""Example-private config for the 2-D two-fluid slab jet (D1).

Factory lives next to the example. PRIVATE: never imported by xmf.config,
never exported by `import xmf`.

Geometry and numbers follow the planar slab-jet KH line
(Ferrari, Trussoni & Massaglia 1982; Bodo et al. 1995): a light
beam with a tanh shear layer into a heavier ambient. Only the light
fluid is pushed. Empty sources — the rolls are the light beam's own
Kelvin–Helmholtz, not a drag term.
"""
from __future__ import annotations

from typing import Any

from xmf.config import SCHEMA


class RUN:
    nx = 192
    ny = 128
    tstop = 8.0
    cfl = 0.35
    gamma = 5.0 / 3.0
    riemann = "hllc"
    reconstruction = "weno3"
    rk = "rk2"
    # Beam radius and box in units of R (R = 1 below).
    R = 1.0
    x1min = 0.0
    x1max = 16.0
    x2min = -5.0
    x2max = 5.0
    # Light jet / heavy ambient; pressure-matched. M = Vj / cs_light.
    rho_light = 1.0
    rho_heavy = 4.0
    rho_light_amb = 0.05
    P0 = 0.6
    Mach = 2.0
    shear_a = 0.12
    pert_amp = 0.10
    pert_period = 2.0
    inlet_cells = 3
    # Collision switch (latter half of D1): "off" ships the empty sources
    # table (KH of the light beam alone); "on" couples the two fluids
    # with the same antisymmetric drag form as mf_cloud_2d (equal masses,
    # momentum exchange exact cell by cell). collision_tau is the heavy
    # fluid's drag time in code units inside the beam (n_light = 1 there).
    collision_tier = "off"
    collision_tau = 2.0
    plots = True
    log_every = 40


def jet_speed() -> float:
    cs = (RUN.gamma * RUN.P0 / RUN.rho_light) ** 0.5
    return RUN.Mach * cs


def jet_sources(tier: str) -> list[dict]:
    """The inter-fluid collision tapes of the switch. Off tier ships the
    empty table; on tier drags the heavy ambient after the beam with
    dv_light = -K n_heavy du, dv_heavy = +K n_light du -- exactly
    antisymmetric in momentum for the equal masses (cell by cell, no
    division, same form as mf_cloud_2d)."""
    if tier == "off":
        return []
    if tier != "on":
        raise ValueError(f"unknown collision tier {tier!r}; want off/on")
    K = 1.0 / RUN.collision_tau
    return [
        {
            "name": "drag_x",
            "expr": {
                "dvx_light": "-K * n_heavy * (vx_light - vx_heavy)",
                "dvx_heavy": "K * n_light * (vx_light - vx_heavy)",
            },
        },
        {
            "name": "drag_y",
            "expr": {
                "dvy_light": "-K * n_heavy * (vy_light - vy_heavy)",
                "dvy_heavy": "K * n_light * (vy_light - vy_heavy)",
            },
        },
    ]


def jet_2d(
    nx: int | None = None,
    ny: int | None = None,
    tstop: float | None = None,
    tier: str | None = None,
) -> dict[str, Any]:
    nnx = int(RUN.nx if nx is None else nx)
    nny = int(RUN.ny if ny is None else ny)
    ts = float(RUN.tstop if tstop is None else tstop)
    tr = str(RUN.collision_tier if tier is None else tier)
    sources = jet_sources(tr)
    constants = (
        {"K": 1.0 / RUN.collision_tau} if sources else {}
    )
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
            {"name": "light", "mass": 1.0, "charge": 0, "kind": "neutral"},
            {"name": "heavy", "mass": 1.0, "charge": 0, "kind": "neutral"},
        ],
        "electron": {"mode": "off"},
        "sources": sources,
        "constants": constants,
        "radiation": {"enabled": False, "absorbers": []},
        "boundary": {"x1beg": "user", "x1end": "copy", "x2": "copy", "x3": "copy"},
        "ic": {"kind": "none"},
    }
