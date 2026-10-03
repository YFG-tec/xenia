"""Example-private config for the 1-D two-fluid dustybox (F1).

Factory lives next to the example. PRIVATE: never imported by xmf.config,
never exported by `import xmf`.

Laibe & Price (2011, MNRAS 418:1491) dustybox: two fluids A, B of equal
mass and equal density in a periodic box, uniform pressure (no hyperbolic
driving), initial velocity offset Δu₀. Inter-fluid drag couples them:

    dv_A = -K n_B (v_A - v_B),   dv_B = +K n_A (v_A - v_B)

With n_A = n_B = 1 (raw density slot = 1 for mass_code = 1):

    d(Δu)/dt = -2K Δu   ⟹   Δu(t) = Δu₀ exp(-t/τ),   τ = 1/(2K)

Pure source-term test of the implicit 10×10 LU solve against an exact
ODE. The hyperbolic part does zero work (uniform P, uniform ρ).
"""
from __future__ import annotations

from typing import Any

from xmf.config import SCHEMA


class RUN:
    nx = 64
    tstop_tau = 3.0        # stop at 3τ (configurable per tier)
    cfl = 0.4
    gamma = 5.0 / 3.0
    riemann = "hllc"
    reconstruction = "plm"
    rk = "rk2"
    # Box [0, L].
    x1min = 0.0
    x1max = 1.0
    # Two fluids, equal mass, equal density.
    rho0 = 1.0
    P0 = 1.0
    du0 = 1.0              # initial velocity offset Δu₀ = v_A - v_B
    # Drag tiers: τ values (code units). K = 1/(2τ).
    tiers = {
        "weak": 10.0,       # τ = 10, t_stop = 30
        "mid": 1.0,         # τ = 1,  t_stop = 3
        "strong": 0.1,      # τ = 0.1, t_stop = 0.3
    }
    log_every = 20
    plots = True


def dustybox_sources(tau: float) -> list[dict[str, Any]]:
    """Drag tapes for a given τ. K = 1/(2τ); the antisymmetric form
    ensures exact cell-by-cell momentum conservation for equal masses."""
    K = 1.0 / (2.0 * tau)
    return [
        {
            "name": "drag_x",
            "expr": {
                "dvx_A": "-K * n_B * (vx_A - vx_B)",
                "dvx_B": "K * n_A * (vx_A - vx_B)",
            },
        },
    ]


def dustybox_1d(
    nx: int | None = None,
    tier: str = "mid",
    tstop: float | None = None,
) -> dict[str, Any]:
    nnx = int(RUN.nx if nx is None else nx)
    tau = RUN.tiers[tier]
    K = 1.0 / (2.0 * tau)
    ts = float(RUN.tstop_tau * tau if tstop is None else tstop)
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
            "ndim": 1,
            "n": [nnx, 1, 1],
            "stretch": "none",
            "x1min": RUN.x1min,
            "x1max": RUN.x1max,
            "x2min": -0.5,
            "x2max": 0.5,
            "x3min": -0.5,
            "x3max": 0.5,
            "nghost": 2,
        },
        "time": {"cfl": RUN.cfl, "tstop": ts, "dt_max": 0.0},
        "fluids": [
            {"name": "A", "mass": 1.0, "charge": 0, "kind": "neutral"},
            {"name": "B", "mass": 1.0, "charge": 0, "kind": "neutral"},
        ],
        "electron": {"mode": "off"},
        "sources": dustybox_sources(tau),
        "constants": {"K": K},
        "radiation": {"enabled": False, "absorbers": []},
        "boundary": {"x1": "periodic"},
        "ic": {"kind": "none"},
    }
