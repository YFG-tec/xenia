"""2-D two-fluid slab jet (D1).

Light beam with a tanh shear layer into a heavier ambient; only the
light fluid is pushed. Inlet is pinned in the example layer (the IR
has no abs / piecewise). Writes output/fields.npz + summary.json.

    python examples/mf_jet_2d/run.py --no-plot
    python examples/mf_jet_2d/plot.py
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

import xmf
from jet_config import RUN, jet_2d, jet_speed

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
_PTR = ctypes.POINTER(ctypes.c_double)
FLUIDS = ("light", "heavy")


def lib():
    return xmf.api._lib()


def _axis(handle, direction: int) -> np.ndarray:
    ntot = lib().xmf_mesh_ncell(handle, direction)
    buf = np.empty(ntot, dtype=np.float64)
    got = lib().xmf_mesh_copy_x(handle, direction, buf.ctypes.data_as(_PTR))
    assert got == ntot
    return buf


def _pack_slot(handle, fluid: int, var: int, arr: np.ndarray) -> None:
    flat = np.ascontiguousarray(arr, dtype=np.float64)
    got = lib().xmf_state_pack(handle, fluid, var, flat.ctypes.data_as(_PTR))
    assert got == flat.size


def _unpack_rho(handle, fluid: int, shape: tuple[int, int, int]) -> np.ndarray:
    ncells = int(np.prod(shape))
    buf = np.empty(ncells, dtype=np.float64)
    got = lib().xmf_state_unpack(handle, fluid, 0, buf.ctypes.data_as(_PTR))
    assert got == ncells
    return buf.reshape(shape)


def _beam(y: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.tanh((np.abs(y) - RUN.R) / RUN.shear_a))


def pin_inlet(run, x: np.ndarray, y: np.ndarray) -> None:
    """Stamp the Bodo-style nozzle on the left strip, including ghosts."""
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
    w = ng + int(RUN.inlet_cells)
    sl = (slice(None), slice(None), slice(0, w))
    beam = np.broadcast_to(_beam(y)[None, :, None], (nzt, nyt, w)).copy()
    vj = jet_speed()
    phase = 2.0 * np.pi * float(run.t) / RUN.pert_period
    flap = np.broadcast_to(
        np.exp(-((y / RUN.R) ** 2))[None, :, None], (nzt, nyt, w)
    )
    ones = np.ones((nzt, nyt, w))
    rho_l = RUN.rho_light_amb + (RUN.rho_light - RUN.rho_light_amb) * beam
    vx_l = vj * beam
    vy_l = (RUN.pert_amp * vj) * np.sin(phase) * flap
    zero = np.zeros_like(rho_l)
    p0 = RUN.P0 * ones
    _pack_slot(run._h, 0, 0, _slot_overlay(run, 0, 0, (nzt, nyt, nxt), sl, rho_l))
    _pack_slot(run._h, 0, 1, _slot_overlay(run, 0, 1, (nzt, nyt, nxt), sl, vx_l))
    _pack_slot(run._h, 0, 2, _slot_overlay(run, 0, 2, (nzt, nyt, nxt), sl, vy_l))
    _pack_slot(run._h, 0, 3, _slot_overlay(run, 0, 3, (nzt, nyt, nxt), sl, zero))
    _pack_slot(run._h, 0, 4, _slot_overlay(run, 0, 4, (nzt, nyt, nxt), sl, p0))
    rho_h = RUN.rho_heavy * ones
    _pack_slot(run._h, 1, 0, _slot_overlay(run, 1, 0, (nzt, nyt, nxt), sl, rho_h))
    _pack_slot(run._h, 1, 1, _slot_overlay(run, 1, 1, (nzt, nyt, nxt), sl, zero))
    _pack_slot(run._h, 1, 2, _slot_overlay(run, 1, 2, (nzt, nyt, nxt), sl, zero))
    _pack_slot(run._h, 1, 3, _slot_overlay(run, 1, 3, (nzt, nyt, nxt), sl, zero))
    _pack_slot(run._h, 1, 4, _slot_overlay(run, 1, 4, (nzt, nyt, nxt), sl, p0))


def _slot_overlay(
    run,
    fluid: int,
    var: int,
    shape: tuple[int, int, int],
    sl: tuple,
    patch: np.ndarray,
) -> np.ndarray:
    buf = np.empty(int(np.prod(shape)), dtype=np.float64)
    got = lib().xmf_state_unpack(run._h, fluid, var, buf.ctypes.data_as(_PTR))
    assert got == buf.size
    arr = buf.reshape(shape)
    arr[sl] = patch
    return arr


def pack_ic(run) -> dict[str, float]:
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
    x = _axis(run._h, 0)
    y = _axis(run._h, 1)
    z = _axis(run._h, 2)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    vj = jet_speed()
    beam = _beam(yy)
    zero = np.zeros_like(yy)
    p0 = np.full_like(yy, RUN.P0)
    rho_l = RUN.rho_light_amb + (RUN.rho_light - RUN.rho_light_amb) * beam * (
        xx <= (x[ng] + RUN.inlet_cells * (x[ng + 1] - x[ng]))
    )
    _pack_slot(run._h, 0, 0, rho_l)
    _pack_slot(run._h, 0, 1, vj * beam * (xx <= (x[ng] + RUN.inlet_cells * (x[ng + 1] - x[ng]))))
    _pack_slot(run._h, 0, 2, zero)
    _pack_slot(run._h, 0, 3, zero)
    _pack_slot(run._h, 0, 4, p0)
    _pack_slot(run._h, 1, 0, np.full_like(yy, RUN.rho_heavy))
    _pack_slot(run._h, 1, 1, zero)
    _pack_slot(run._h, 1, 2, zero)
    _pack_slot(run._h, 1, 3, zero)
    _pack_slot(run._h, 1, 4, p0)
    pin_inlet(run, x, y)
    lib().xmf_fill_bc(run._h)
    pin_inlet(run, x, y)
    return {
        "Vj": vj,
        "cs_light": vj / RUN.Mach,
        "R": RUN.R,
        "Mach": RUN.Mach,
        "nu": RUN.rho_heavy / RUN.rho_light,
    }


def unpack_code(run) -> dict[str, np.ndarray]:
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
    inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
    out: dict[str, np.ndarray] = {
        "x": _axis(run._h, 0)[ng : ng + nx],
        "y": _axis(run._h, 1)[ng : ng + ny],
    }
    fld = run.fields_full()
    for name in FLUIDS:
        blk = fld[name]
        out[f"{name}_vx"] = blk["vx"][0]
        out[f"{name}_vy"] = blk["vy"][0]
        out[f"{name}_P"] = blk["P"][0]
        rho = _unpack_rho(run._h, FLUIDS.index(name), (nzt, nyt, nxt))[inner][0]
        out[f"{name}_rho"] = rho.copy()
    return out


def save_fields(code: dict[str, np.ndarray], meta: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fields.npz"
    np.savez_compressed(path, **code)
    (out / "summary.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--tstop", type=float, default=None)
    parser.add_argument(
        "--tier",
        default=None,
        choices=("off", "on"),
        help="collision switch (off = empty sources, on = drag tapes)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    tier = str(RUN.collision_tier if args.tier is None else args.tier)
    cfg = jet_2d(nx=args.nx, ny=args.ny, tstop=args.tstop, tier=tier)
    out = OUT if tier == "off" else OUT / tier
    nx, ny, _ = cfg["grid"]["n"]
    print(f"xmf={Path(xmf.__file__).resolve()}", flush=True)
    print(
        f"ndim=2  n={nx}x{ny}  Vj={jet_speed():.3f}  "
        f"M={RUN.Mach}  nu={RUN.rho_heavy / RUN.rho_light:.1f}  "
        f"collision={tier}  K={float(cfg['constants'].get('K', 0.0)):.4g}  "
        f"tstop={cfg['time']['tstop']}",
        flush=True,
    )
    run = xmf.setup(cfg)
    try:
        x = _axis(run._h, 0)
        y = _axis(run._h, 1)
        ic = pack_ic(run)
        t0 = time.perf_counter()
        steps = 0
        while run.t < run.tstop:
            lib().xmf_fill_bc(run._h)
            pin_inlet(run, x, y)
            run.step()
            steps += 1
            if RUN.log_every and steps % RUN.log_every == 0:
                print(
                    f"step {steps}  t={run.t:.4e}  dt={run.dt:.3e}",
                    flush=True,
                )
        wall = time.perf_counter() - t0
        code = unpack_code(run)
        if not all(np.isfinite(code[f"{n}_rho"]).all() for n in FLUIDS):
            raise RuntimeError("non-finite density")
        du = np.sqrt(
            (code["light_vx"] - code["heavy_vx"]) ** 2
            + (code["light_vy"] - code["heavy_vy"]) ** 2
        )
        meta = {
            "nx": nx,
            "ny": ny,
            "riemann": RUN.riemann,
            "reconstruction": RUN.reconstruction,
            "steps": steps,
            "t": float(run.t),
            "dt": float(run.dt),
            "wall_s": wall,
            "collision_tier": tier,
            "K": float(cfg["constants"].get("K", 0.0)),
            "du_max": float(np.max(du)),
            "vy_light_max": float(np.max(np.abs(code["light_vy"]))),
            **ic,
            "xmf": str(Path(xmf.__file__).resolve()),
        }
        dump = save_fields(code, meta, out)
        print(json.dumps(meta, indent=2), flush=True)
        print(f"wrote {dump}", flush=True)
        if RUN.plots and not args.no_plot:
            import sys

            if str(HERE) not in sys.path:
                sys.path.insert(0, str(HERE))
            from plot import draw

            draw(out)
    finally:
        run.close()


if __name__ == "__main__":
    main()
