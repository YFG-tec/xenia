"""Helpers to drive the installed xmf kernel from the evaluation example.

Public API stays setup → step → fields → finalize. Extra pack/unpack is
the same helper layer the A2/A4 contract tests already use.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

import xmf
from xmf._native import load
from xmf.config import AMU

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4
SLOTS = (SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P)
_PTR = ctypes.POINTER(ctypes.c_double)

MASS = 1.00794
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SOLVERS = ("hll", "hllc", "tvdlf")


def lib():
    return load()


def geom(run) -> tuple[int, int, int, int, int, int, int]:
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def axis(handle, direction: int) -> np.ndarray:
    ntot = lib().xmf_mesh_ncell(handle, direction)
    buf = np.empty(ntot, dtype=np.float64)
    assert lib().xmf_mesh_copy_x(handle, direction, buf.ctypes.data_as(_PTR)) == ntot
    return buf


def unpack_slot(handle, slot: int) -> np.ndarray:
    ncells = lib().xmf_state_ncells(handle)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib().xmf_state_unpack(handle, 0, slot, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def pack_slot(handle, slot: int, arr: np.ndarray) -> None:
    flat = np.ascontiguousarray(arr, dtype=np.float64)
    assert lib().xmf_state_pack(handle, 0, slot, flat.ctypes.data_as(_PTR)) == flat.size


def unpack_code(run) -> dict[str, np.ndarray]:
    """Interior fields in kernel code units, plus cell-centre axes."""
    ng, nx, ny, nz, nxt, nyt, nzt = geom(run)
    inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
    rho = unpack_slot(run._h, SLOT_N).reshape((nzt, nyt, nxt))[inner][0]
    vx = unpack_slot(run._h, SLOT_VX).reshape((nzt, nyt, nxt))[inner][0]
    vy = unpack_slot(run._h, SLOT_VY).reshape((nzt, nyt, nxt))[inner][0]
    vz = unpack_slot(run._h, SLOT_VZ).reshape((nzt, nyt, nxt))[inner][0]
    p = unpack_slot(run._h, SLOT_P).reshape((nzt, nyt, nxt))[inner][0]
    x = axis(run._h, 0)[ng : ng + nx]
    y = axis(run._h, 1)[ng : ng + ny] if ny > 1 else np.array([0.5])
    return {"rho": rho, "vx": vx, "vy": vy, "vz": vz, "p": p, "x": x, "y": y}


def pack_code(run, rho, vx, vy, vz, p) -> None:
    """Write full (ghost-included) blocks; then refill ghosts from face types."""
    ng, nx, ny, nz, nxt, nyt, nzt = geom(run)
    blocks = {
        SLOT_N: np.broadcast_to(np.asarray(rho, dtype=np.float64), (nzt, nyt, nxt)),
        SLOT_VX: np.broadcast_to(np.asarray(vx, dtype=np.float64), (nzt, nyt, nxt)),
        SLOT_VY: np.broadcast_to(np.asarray(vy, dtype=np.float64), (nzt, nyt, nxt)),
        SLOT_VZ: np.broadcast_to(np.asarray(vz, dtype=np.float64), (nzt, nyt, nxt)),
        SLOT_P: np.broadcast_to(np.asarray(p, dtype=np.float64), (nzt, nyt, nxt)),
    }
    for slot, arr in blocks.items():
        pack_slot(run._h, slot, np.ascontiguousarray(arr))
    lib().xmf_fill_bc(run._h)


def pack_from_xy(run, fn) -> None:
    """fn(x,y) → (rho, vx, vy, vz, p) on the full lattice including ghosts."""
    ng, nx, ny, nz, nxt, nyt, nzt = geom(run)
    xx = axis(run._h, 0)
    yy = axis(run._h, 1)
    rho = np.empty((nzt, nyt, nxt))
    vx = np.empty_like(rho)
    vy = np.empty_like(rho)
    vz = np.empty_like(rho)
    p = np.empty_like(rho)
    for j in range(nyt):
        for i in range(nxt):
            q = fn(float(xx[i]), float(yy[j]))
            rho[:, j, i] = q[0]
            vx[:, j, i] = q[1]
            vy[:, j, i] = q[2]
            vz[:, j, i] = q[3]
            p[:, j, i] = q[4]
    pack_code(run, rho, vx, vy, vz, p)


def cgs_from_code(run, code: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    u = run._cfg.units
    mass = run._cfg.mass
    n = code["rho"] * u.density / (mass * AMU)
    return {
        "n": n,
        "vx": code["vx"] * u.velocity,
        "vy": code["vy"] * u.velocity,
        "p": code["p"] * u.density * u.velocity * u.velocity,
        "x": code["x"],
        "y": code["y"],
    }


def base_cfg(
    *,
    n: list[int],
    ndim: int,
    riemann: str,
    tstop: float,
    x1: tuple[float, float],
    x2: tuple[float, float] = (0.0, 1.0),
    boundary: dict[str, str] | None = None,
    gamma: float = 5.0 / 3.0,
    cfl: float = 0.4,
    units: str = "astro",
    geometry: str = "cartesian",
) -> dict[str, Any]:
    cfg = xmf.shock_tube()
    cfg["gamma"] = gamma
    cfg["numerics"] = {"riemann": riemann, "reconstruction": "pcm", "rk": "rk1"}
    cfg["grid"]["n"] = list(n)
    cfg["grid"]["ndim"] = ndim
    cfg["grid"]["geometry"] = geometry
    cfg["grid"]["x1min"], cfg["grid"]["x1max"] = x1
    cfg["grid"]["x2min"], cfg["grid"]["x2max"] = x2
    cfg["time"] = {"cfl": cfl, "tstop": tstop, "dt_max": 0.0}
    cfg["electron"] = {"mode": "off"}
    cfg["sources"] = []
    if boundary:
        cfg["boundary"] = boundary
    if units == "code":
        cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    return cfg


def run_to_tstop(run) -> int:
    nstep = 0
    while run.t < run.tstop:
        run.step()
        nstep += 1
    return nstep


def l1(num, exact, scale: float) -> float:
    return float(np.mean(np.abs(num - exact)) / scale)


# ---------------------------------------------------------------------------
# Conservation helpers (B3 closed-box check)
# ---------------------------------------------------------------------------


def get_interior_slots(run, gamma: float):
    """Return interior rho, vx, vy, vz, P arrays (no ghosts) as (ny, nx)."""
    ng, nx, ny, nz, nxt, nyt, nzt = geom(run)
    shape = (nzt, nyt, nxt)
    sl = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
    rho = unpack_slot(run._h, SLOT_N).reshape(shape)[sl][0]
    vx = unpack_slot(run._h, SLOT_VX).reshape(shape)[sl][0]
    vy = unpack_slot(run._h, SLOT_VY).reshape(shape)[sl][0]
    vz = unpack_slot(run._h, SLOT_VZ).reshape(shape)[sl][0]
    p = unpack_slot(run._h, SLOT_P).reshape(shape)[sl][0]
    return rho, vx, vy, vz, p


def get_dv_2d(run):
    """Cell-volume outer product for a 2-D grid (interior cells only)."""
    ng, nx, ny, nz, nxt, nyt, nzt = geom(run)
    dv0 = np.empty(nxt, dtype=np.float64)
    lib().xmf_mesh_copy_dV(run._h, 0, dv0.ctypes.data_as(_PTR))
    dv1 = np.empty(nyt, dtype=np.float64)
    lib().xmf_mesh_copy_dV(run._h, 1, dv1.ctypes.data_as(_PTR))
    dv0_in = dv0[ng:ng + nx]
    dv1_in = dv1[ng:ng + ny]
    # 2-D cell volume = dV_1[i] * dV_2[j]; shape (ny, nx)
    return np.outer(dv1_in, dv0_in)


def compute_totals(run, gamma: float):
    """Total mass and total energy over interior cells.

    mass  = Σ ρ · dV
    energy = Σ [ P/(γ−1) + ½ρ(vx²+vy²+vz²) ] · dV
    """
    rho, vx, vy, vz, p = get_interior_slots(run, gamma)
    dv = get_dv_2d(run)
    mass = float(np.sum(rho * dv))
    energy = float(np.sum((p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)) * dv))
    return mass, energy
