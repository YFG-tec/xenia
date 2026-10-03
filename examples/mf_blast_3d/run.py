"""3-D two-fluid Sedov-like blast (D3 demo).

Advances and writes `output/fields.npz` + `output/summary.json`.
Figures live in `plot.py` and can be redrawn without stepping:

    python examples/mf_blast_3d/run.py --no-plot
    python examples/mf_blast_3d/plot.py
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

import xmf
from blast_config import RUN, blast_3d

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


def pack_ic(run) -> dict[str, float]:
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
    x = _axis(run._h, 0)
    y = _axis(run._h, 1)
    z = _axis(run._h, 2)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dx = float(x[ng + 1] - x[ng])
    r0 = RUN.r0_cells * dx
    vol = (4.0 / 3.0) * np.pi * r0**3
    p_in = (RUN.gamma - 1.0) * RUN.energy / vol
    inside = (xx * xx + yy * yy + zz * zz) <= r0 * r0
    p = np.where(inside, p_in, RUN.p_amb)
    zero = np.zeros_like(p)
    rhos = (RUN.rho_light, RUN.rho_heavy)
    for f, rho0 in enumerate(rhos):
        rho = np.full_like(p, rho0)
        _pack_slot(run._h, f, 0, rho)
        _pack_slot(run._h, f, 1, zero)
        _pack_slot(run._h, f, 2, zero)
        _pack_slot(run._h, f, 3, zero)
        _pack_slot(run._h, f, 4, p)
    lib().xmf_fill_bc(run._h)
    return {"dx": dx, "r0": r0, "p_in": float(p_in), "p_amb": RUN.p_amb}


def unpack_code(run) -> dict[str, np.ndarray]:
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
    inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
    out: dict[str, np.ndarray] = {
        "x": _axis(run._h, 0)[ng : ng + nx],
        "y": _axis(run._h, 1)[ng : ng + ny],
        "z": _axis(run._h, 2)[ng : ng + nz],
    }
    fld = run.fields_full()
    for name in FLUIDS:
        blk = fld[name]
        out[f"{name}_n"] = blk["n"]
        out[f"{name}_vx"] = blk["vx"]
        out[f"{name}_vy"] = blk["vy"]
        out[f"{name}_vz"] = blk["vz"]
        out[f"{name}_P"] = blk["P"]
        rho = _unpack_rho(run._h, FLUIDS.index(name), (nzt, nyt, nxt))[inner]
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
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--tstop", type=float, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    nn = int(RUN.n if args.n is None else args.n)
    cfg = blast_3d(n=nn, tstop=args.tstop)
    print(f"xmf={Path(xmf.__file__).resolve()}", flush=True)
    print(
        f"ndim=3  n={nn}^3  lines/sweep={nn*nn}  "
        f"fluids=2  sources=[]  tstop={cfg['time']['tstop']}",
        flush=True,
    )
    run = xmf.setup(cfg)
    try:
        ic = pack_ic(run)
        t0 = time.perf_counter()
        steps = 0
        while run.t < run.tstop:
            run.step()
            steps += 1
            if RUN.log_every and steps % RUN.log_every == 0:
                print(
                    f"step {steps}  t={run.t:.4e}  dt={run.dt:.3e}",
                    flush=True,
                )
        wall = time.perf_counter() - t0
        code = unpack_code(run)
        vmax = (
            float(np.max(np.abs(code["light_vx"]))),
            float(np.max(np.abs(code["light_vy"]))),
            float(np.max(np.abs(code["light_vz"]))),
        )
        if not all(np.isfinite(code[f"{n}_rho"]).all() for n in FLUIDS):
            raise RuntimeError("non-finite density")
        if min(vmax) <= 0.0:
            raise RuntimeError(f"expected all three speeds nonzero, got {vmax}")
        meta = {
            "n": nn,
            "riemann": RUN.riemann,
            "ncells": nn * nn * nn,
            "nlines": nn * nn,
            "steps": steps,
            "t": float(run.t),
            "dt": float(run.dt),
            "wall_s": wall,
            "us_per_cell_step": 1.0e6 * wall / max(steps * nn**3, 1),
            "vmax_xyz": vmax,
            "rho_minmax_light": [
                float(np.min(code["light_rho"])),
                float(np.max(code["light_rho"])),
            ],
            **ic,
            "xmf": str(Path(xmf.__file__).resolve()),
        }
        dump = save_fields(code, meta, OUT)
        print(json.dumps(meta, indent=2), flush=True)
        print(f"wrote {dump}", flush=True)
        if RUN.plots and not args.no_plot:
            import sys

            if str(HERE) not in sys.path:
                sys.path.insert(0, str(HERE))
            from plot import draw

            draw(OUT)
    finally:
        run.close()


if __name__ == "__main__":
    main()
