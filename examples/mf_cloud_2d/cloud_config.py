"""Example-private config for the 2-D two-fluid shock-dense-cloud (D2).

Factory lives next to the example. PRIVATE: never imported by xmf.config,
never exported by `import xmf`.

Klein, McKee & Colella (1994) cloud crushing lifted to two fluids: the
dense cloud is a DIFFERENT fluid. The shock exists only in fluid A (the
light wind); momentum reaches fluid B (the cloud) exclusively through the
inter-fluid collision tapes, so the collision coefficient K is the single
physical knob:

    K = 0      shock passes through, the cloud stays exactly at rest
    large K    the post-shock wind drags, deforms and strips the cloud

Drag tapes (equal masses, hence exactly antisymmetric in momentum,
cell by cell, no regression needed):

    dv_A = -K n_B (v_A - v_B),   dv_B = +K n_A (v_A - v_B)
    rho_A dv_A + rho_B dv_B = 0  (mass_code = 1: no T/F_b reads, so
                                  the phys port stays off and tape n_s
                                  is the raw density slot)

K is calibrated on the post-shock wind density rho_A1:
tau_c = 1 / (K rho_A1) is the cloud drag time; tiers are picked in units
of the cloud-crushing time t_cc = sqrt(chi) r_c / v_A1.

The incident shock is an initial condition (left of x_s = 0.5 the exact
Rankine-Hugoniot post-shock state moves right; post-shock flow is
supersonic, so no signal ever reaches the left face and the four `copy`
faces keep feeding the pristine reservoir). No user boundaries, no
pinning.
"""
from __future__ import annotations

import math
from typing import Any

from xmf.config import SCHEMA

TIERS = ("off", "weak", "mid", "strong")


class RUN:
    nx = 128
    ny = 64
    tstop_tcc = 3.0  # stop time in cloud-crushing times
    cfl = 0.4
    gamma = 5.0 / 3.0
    riemann = "hllc"
    # plm, not weno3: with weno3 the run dies at t ~= 1.2 t_cc when the
    # cloud is destroyed and stripped filaments go under-resolved (no
    # positivity guard; wind_vx non-finite). PLM survives all four tiers
    # to 3 t_cc at both 128x64 and 256x128. Same family as mf_jet_2d
    # (second order, hllc, rk2); the switch is recorded in the REPORT.
    reconstruction = "plm"
    rk = "rk2"
    # Box and shock / cloud geometry (code units).
    x1min = 0.0
    x1max = 4.0
    x2min = -1.0
    x2max = 1.0
    x_shock = 0.5       # post-shock state fills x < x_shock
    cloud_x = 1.5
    cloud_y = 0.0
    cloud_r = 0.25
    # Ambient wind (fluid A) and cloud (fluid B) states.
    rho_amb = 1.0
    P_amb = 1.0
    Mach = 10.0         # incident shock Mach number
    chi = 10.0          # cloud-to-ambient density contrast
    # Fluid-B background density outside the cloud. B fills the box at
    # this tenuous density with UNIFORM pressure P_amb (each fluid must
    # be in its own pressure equilibrium -- fluids do not share pressure,
    # so a cloud-pressure/floor-pressure jump would make B explode into
    # its own ambient). Uniform P_B + a rho jump = a stable contact,
    # exactly like the classical single-fluid cloud problem. 1e-2 keeps
    # B's background sound speed (~13) near the post-shock wind signal
    # (~17), so it does not cap dt.
    n_floor = 1.0e-2
    # Drag tiers: tau_c / t_cc (off = empty sources table).
    tau_over_tcc = {"off": None, "weak": 10.0, "mid": 1.0, "strong": 0.1}
    tier = "mid"
    plots = True
    log_every = 50


def post_shock() -> dict[str, float]:
    """Exact Rankine-Hugoniot post-shock state of the incident shock
    (rightward shock of Mach M into (rho_amb, 0, P_amb)) plus the derived
    cloud-crushing time t_cc = sqrt(chi) r_c / v_A1."""
    g = RUN.gamma
    m2 = RUN.Mach * RUN.Mach
    c0 = math.sqrt(g * RUN.P_amb / RUN.rho_amb)
    v_s = RUN.Mach * c0
    r = (g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0)          # rho1 / rho0
    p = 1.0 + 2.0 * g * (m2 - 1.0) / (g + 1.0)           # P1 / P0
    rho1 = RUN.rho_amb * r
    P1 = RUN.P_amb * p
    v1 = v_s * (1.0 - 1.0 / r)
    c1 = math.sqrt(g * P1 / rho1)
    t_cc = math.sqrt(RUN.chi) * RUN.cloud_r / v1
    return {
        "c0": c0,
        "v_shock": v_s,
        "rho_A1": rho1,
        "P_A1": P1,
        "v_A1": v1,
        "c_A1": c1,
        "t_cc": t_cc,
    }


def tier_K(tier: str) -> float:
    """Collision coefficient of a tier. tau_c = 1/(K rho_A1) is fixed in
    units of t_cc; K = 0 for the off tier (which ships an empty sources
    table instead of zero tapes)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; want {TIERS}")
    over = RUN.tau_over_tcc[tier]
    if over is None:
        return 0.0
    ps = post_shock()
    return 1.0 / (over * ps["t_cc"] * ps["rho_A1"])


def drag_sources(tier: str) -> list[dict[str, Any]]:
    """The inter-fluid collision tapes. Equal masses: the exchange is
    exactly antisymmetric in momentum (no division anywhere, and both
    sides vanish outside the cloud automatically)."""
    if tier == "off":
        return []
    return [
        {
            "name": "drag_x",
            "expr": {
                "dvx_wind": "-K * n_cloud * (vx_wind - vx_cloud)",
                "dvx_cloud": "K * n_wind * (vx_wind - vx_cloud)",
            },
        },
        {
            "name": "drag_y",
            "expr": {
                "dvy_wind": "-K * n_cloud * (vy_wind - vy_cloud)",
                "dvy_cloud": "K * n_wind * (vy_wind - vy_cloud)",
            },
        },
    ]


def cloud_2d(
    nx: int | None = None,
    ny: int | None = None,
    tstop: float | None = None,
    tier: str | None = None,
) -> dict[str, Any]:
    nnx = int(RUN.nx if nx is None else nx)
    nny = int(RUN.ny if ny is None else ny)
    tr = str(RUN.tier if tier is None else tier)
    ts = float(
        RUN.tstop_tcc * post_shock()["t_cc"] if tstop is None else tstop
    )
    K = tier_K(tr)
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
            {"name": "wind", "mass": 1.0, "charge": 0, "kind": "neutral"},
            {"name": "cloud", "mass": 1.0, "charge": 0, "kind": "neutral"},
        ],
        "electron": {"mode": "off"},
        "sources": drag_sources(tr),
        "constants": {"K": K},
        "radiation": {"enabled": False, "absorbers": []},
        "boundary": {
            "x1beg": "copy",
            "x1end": "copy",
            "x2": "copy",
            "x3": "copy",
        },
        "ic": {"kind": "none"},
    }
