"""2-D flow past a circular cylinder.

Uniform stream, no-slip disk pinned in the example layer (the kernel
has no solid body and no physical ν). Writes output/fields.npz and output/snaps.npz.

    python examples/cyl_2d/run.py --no-plot
    python examples/cyl_2d/plot.py --anim
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

import xmf
from cyl_config import RUN, cyl_2d, pressure, sound_speed

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
_PTR = ctypes.POINTER(ctypes.c_double)


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


def _unpack_slot(handle, fluid: int, var: int, shape: tuple[int, int, int]) -> np.ndarray:
    buf = np.empty(int(np.prod(shape)), dtype=np.float64)
    got = lib().xmf_state_unpack(handle, fluid, var, buf.ctypes.data_as(_PTR))
    assert got == buf.size
    return buf.reshape(shape)


def _grid(run) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    shape = (nz + 2 * ng, ny + 2 * ng, nx + 2 * ng)
    x = _axis(run._h, 0)
    y = _axis(run._h, 1)
    z = _axis(run._h, 2)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return xx, yy, zz, shape


def _masks(xx: np.ndarray, yy: np.ndarray, ng: int) -> tuple[np.ndarray, np.ndarray]:
    rad = 0.5 * RUN.D
    body = (xx * xx + yy * yy) <= rad * rad
    inlet = np.zeros(xx.shape, dtype=bool)
    inlet[:, :, : ng + int(RUN.inlet_cells)] = True
    inlet[body] = False
    return body, inlet


def _build_pin_vals(shape, body, inlet, trip: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the C2sup4 pin mask + values for the disk + inlet.

    Returns (mask, vals) where mask is int32 [nztot,nytot,nxtot] and vals
    is float64 [nfluid=1, 5, nztot,nytot,nxtot]. The kernel applies them
    automatically after every sweep and ghost fill -- no per-step call
    from the example layer.
    """
    p0 = pressure()
    combined = body | inlet
    mask = np.zeros(shape, dtype=np.int32)
    mask[combined] = 1
    vals = np.zeros((1, 5) + shape, dtype=np.float64)
    # Disk: no-slip, ambient density and pressure.
    vals[0, 0][body] = RUN.rho0
    vals[0, 1][body] = 0.0
    vals[0, 2][body] = 0.0
    vals[0, 3][body] = 0.0
    vals[0, 4][body] = p0
    # Inlet: uniform stream (trip gives a transient vy kick).
    vals[0, 0][inlet] = RUN.rho0
    vals[0, 1][inlet] = RUN.U_inf
    vals[0, 2][inlet] = trip
    vals[0, 3][inlet] = 0.0
    vals[0, 4][inlet] = p0
    return mask, vals


def pack_ic(run) -> dict[str, float]:
    xx, yy, _, shape = _grid(run)
    body, inlet = _masks(xx, yy, run._cfg.ng)
    p0 = pressure()
    rho = np.full(shape, RUN.rho0)
    vx = np.full(shape, RUN.U_inf)
    vy = np.full(shape, RUN.trip_amp * RUN.U_inf)
    vz = np.zeros(shape)
    pre = np.full(shape, p0)
    rho[body] = RUN.rho0
    vx[body] = 0.0
    vy[body] = 0.0
    pre[body] = p0
    for var, arr in enumerate((rho, vx, vy, vz, pre)):
        _pack_slot(run._h, 0, var, arr)
    lib().xmf_fill_bc(run._h)
    # C2sup4: load the disk + inlet pin once. The kernel applies it
    # automatically after every sweep and ghost fill.
    mask, vals = _build_pin_vals(shape, body, inlet, RUN.trip_amp * RUN.U_inf)
    run.set_pin(mask, vals)
    return {
        "D": RUN.D,
        "U_inf": RUN.U_inf,
        "Mach": RUN.Mach,
        "cs": sound_speed(),
        "P0": p0,
        "n_body": int(np.count_nonzero(body)),
    }


def unpack_code(run) -> dict[str, np.ndarray]:
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
    inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
    fld = run.fields_full()["gas"]
    rho = _unpack_slot(run._h, 0, 0, (nzt, nyt, nxt))[inner]
    return {
        "x": _axis(run._h, 0)[ng : ng + nx],
        "y": _axis(run._h, 1)[ng : ng + ny],
        "rho": rho[0].copy(),
        "vx": fld["vx"][0],
        "vy": fld["vy"][0],
        "P": fld["P"][0],
    }


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
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    cfg = cyl_2d(nx=args.nx, ny=args.ny, tstop=args.tstop)
    nx, ny, _ = cfg["grid"]["n"]
    print(f"xmf={Path(xmf.__file__).resolve()}", flush=True)
    print(
        f"ndim=2  n={nx}x{ny}  U={RUN.U_inf}  M={RUN.Mach}  "
        f"D={RUN.D}  tstop={cfg['time']['tstop']}",
        flush=True,
    )
    run = xmf.setup(cfg)
    try:
        xx, yy, _, shape = _grid(run)
        body, inlet = _masks(xx, yy, run._cfg.ng)
        ic = pack_ic(run)
        t0 = time.perf_counter()
        steps = 0
        snap_t: list[float] = []
        snap_vx: list[np.ndarray] = []
        snap_vy: list[np.ndarray] = []
        t_next = 0.0
        trip_active = True
        while run.t < run.tstop:
            # C2sup4: reload the pin once when the trip expires (vy -> 0).
            if trip_active and run.t >= RUN.trip_until:
                mask, vals = _build_pin_vals(shape, body, inlet, 0.0)
                run.set_pin(mask, vals)
                trip_active = False
            run.step()
            steps += 1
            if run.t + 1.0e-15 >= t_next:
                frame = unpack_code(run)
                snap_t.append(float(run.t))
                snap_vx.append(np.asarray(frame["vx"], dtype=np.float32))
                snap_vy.append(np.asarray(frame["vy"], dtype=np.float32))
                t_next += float(RUN.snap_dt)
            if RUN.log_every and steps % RUN.log_every == 0:
                print(
                    f"step {steps}  t={run.t:.4e}  dt={run.dt:.3e}",
                    flush=True,
                )
        wall = time.perf_counter() - t0
        code = unpack_code(run)
        if not np.isfinite(code["rho"]).all():
            raise RuntimeError("non-finite density")
        meta = {
            "nx": nx,
            "ny": ny,
            "riemann": RUN.riemann,
            "reconstruction": RUN.reconstruction,
            "steps": steps,
            "t": float(run.t),
            "dt": float(run.dt),
            "wall_s": wall,
            "vy_max": float(np.max(np.abs(code["vy"]))),
            "speed_max": float(
                np.max(np.sqrt(code["vx"] ** 2 + code["vy"] ** 2))
            ),
            **ic,
            "xmf": str(Path(xmf.__file__).resolve()),
        }
        dump = save_fields(code, meta, OUT)
        snap_path = OUT / "snaps.npz"
        np.savez_compressed(
            snap_path,
            t=np.asarray(snap_t, dtype=np.float64),
            vx=np.stack(snap_vx, axis=0),
            vy=np.stack(snap_vy, axis=0),
            x=code["x"],
            y=code["y"],
        )
        print(json.dumps(meta, indent=2), flush=True)
        print(f"wrote {dump} and {snap_path} ({len(snap_t)} frames)", flush=True)
        if RUN.plots and not args.no_plot:
            import sys

            if str(HERE) not in sys.path:
                sys.path.insert(0, str(HERE))
            from plot import draw

            draw(OUT, anim=True)
    finally:
        run.close()


if __name__ == "__main__":
    main()
