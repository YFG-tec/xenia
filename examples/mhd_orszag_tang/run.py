"""2-D Orszag–Tang vortex (ideal MHD, CT induction, periodic domain).

Standard MHD benchmark (Orszag & Tang 1979, JFM 95:119). Code units:
gamma = 5/3 on [0, 2 pi]^2 with periodic boundaries,

    rho = gamma^2 / (4 pi),   P = 5 / (12 pi),
    u   = (-sin y, sin x, 0),
    B   = (-sin y, sin 2x, 0) / sqrt(4 pi).

One ion species (q = 1, w_s = 1), electrons off; magnetic field on, so the
induction path is the M5 CT update (edge E = -u_+ x B, face-centred B).
Numerics match the tests/test_m5_ct.py nail: hll / plm / rk3, cfl = 0.4.

Advances and writes `output/fields.npz` + `output/summary.json`.
Default tstop is π (the usual published snapshot); the M5 contract
nail stays t = 0.5 (`tests/test_m5_ct.py`). `--animate` dumps ~60
time-spaced frames (density + magnetic pressure) and writes a GIF.

    python examples/mhd_orszag_tang/run.py --n 128 --animate
    python examples/mhd_orszag_tang/plot.py
    python examples/mhd_orszag_tang/animate.py
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import xmf
from xmf.config import AMU, SCHEMA

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
_PTR = ctypes.POINTER(ctypes.c_double)

GAMMA = 5.0 / 3.0
RHO0 = GAMMA * GAMMA / (4.0 * math.pi)
P0 = 5.0 / (12.0 * math.pi)
B0 = 1.0 / math.sqrt(4.0 * math.pi)


def lib():
    return xmf.api._lib()


def ot_config(
    n: int, tstop: float, backend: str = "serial", mhd: bool = True
) -> dict:
    L = 2.0 * math.pi
    return {
        "schema": SCHEMA,
        "backend": backend,
        "gamma": GAMMA,
        "units": {"density": 1.0, "velocity": 1.0, "length": 1.0},
        "numerics": {"riemann": "hll", "reconstruction": "plm", "rk": "rk3"},
        "grid": {
            "geometry": "cartesian",
            "ndim": 2,
            "n": [n, n, 1],
            "stretch": "none",
            "x1min": 0.0,
            "x1max": L,
            "x2min": 0.0,
            "x2max": L,
        },
        "time": {"cfl": 0.4, "tstop": tstop, "dt_max": 0.0},
        "fluids": [{"name": "HII", "mass": 1.0, "charge": 1, "kind": "ion"}],
        "electron": {"mode": "off"},
        "sources": [],
        "constants": {},
        "radiation": {"enabled": False, "absorbers": []},
        "boundary": {"x1": "periodic", "x2": "periodic", "x3": "none"},
        "mhd": bool(mhd),
        "ic": {
            "kind": "riemann",
            "x0": L / 2.0,
            "left": {"n": RHO0, "v": 0.0, "p": P0},
            "right": {"n": RHO0, "v": 0.0, "p": P0},
        },
    }


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


def pack_ic(run) -> None:
    """Orszag–Tang initial state: uniform rho/P, vortex velocity, sheared B."""
    p = run._cfg
    ng = p.ng
    ntot = tuple(n + 2 * ng for n in p.n)
    xc = _axis(run._h, 0)
    yc = _axis(run._h, 1)

    # Fluid five slots (cell-centred, flat k,j,i with total extents).
    j_idx, i_idx = np.meshgrid(
        np.arange(ntot[1]), np.arange(ntot[0]), indexing="ij"
    )
    vx = np.tile(-np.sin(yc[j_idx]), (ntot[2], 1))
    vy = np.tile(np.sin(xc[i_idx]), (ntot[2], 1))
    rho = np.full_like(vx, RHO0)
    prs = np.full_like(vx, P0)
    vz = np.zeros_like(vx)
    _pack_slot(run._h, 0, 0, rho)
    _pack_slot(run._h, 0, 1, vx)
    _pack_slot(run._h, 0, 2, vy)
    _pack_slot(run._h, 0, 3, vz)
    _pack_slot(run._h, 0, 4, prs)
    if not run._cfg.extra.get("mhd"):
        return

    # Face-centred B: B1 on x1-faces (k, j, i+1), B2 on x2-faces (i, k, j+1),
    # B3 on x3-faces (j, i, k+1). Same layout as tests/test_m5_ct.py.
    b1 = np.zeros((ntot[2], ntot[1], ntot[0] + 1), dtype=np.float64)
    for j in range(ntot[1]):
        b1[:, j, :] = -math.sin(yc[j]) * B0
    b2 = np.zeros((ntot[0], ntot[2], ntot[1] + 1), dtype=np.float64)
    for i in range(ntot[0]):
        b2[i, :, :] = math.sin(2.0 * xc[i]) * B0
    b3 = np.zeros((ntot[1], ntot[0], ntot[2] + 1), dtype=np.float64)
    for comp, blk in ((0, b1), (1, b2), (2, b3)):
        flat = np.ascontiguousarray(blk.ravel())
        got = lib().xmf_mag_pack(run._h, comp, flat.ctypes.data_as(_PTR))
        assert got == flat.size


def total_energy(run) -> float:
    """Fluid + magnetic energy (code units; fields_full is interior-only)."""
    p = run._cfg
    fld = run.fields_full()
    H = fld["HII"]
    mass = p.fluids[0].mass * AMU
    rho = H["n"] * mass / p.units.density
    vx = H["vx"] / p.units.velocity
    vy = H["vy"] / p.units.velocity
    vz = H["vz"] / p.units.velocity
    prs = H["P"] / (p.units.density * p.units.velocity**2)
    bscale = math.sqrt(p.units.density) * p.units.velocity
    ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    ie = prs / (GAMMA - 1.0)
    me = 0.0
    if "B1" in fld:
        b1 = fld["B1"] / bscale
        b2 = fld["B2"] / bscale
        b3 = fld["B3"] / bscale
        me = 0.5 * (b1 * b1 + b2 * b2 + b3 * b3)
    return float(np.sum(ke + ie + me))


def unpack_code(run) -> dict[str, np.ndarray]:
    p = run._cfg
    ng = p.ng
    nx, ny = p.n[0], p.n[1]
    fld = run.fields_full()  # interior (nz, ny, nx) arrays, physical units
    H = fld["HII"]
    mass = p.fluids[0].mass * AMU
    bscale = math.sqrt(p.units.density) * p.units.velocity
    out = {
        "x": _axis(run._h, 0)[ng : ng + nx],
        "y": _axis(run._h, 1)[ng : ng + ny],
        "rho": (H["n"] * mass / p.units.density).copy(),
        "vx": (H["vx"] / p.units.velocity).copy(),
        "vy": (H["vy"] / p.units.velocity).copy(),
        "P": (H["P"] / (p.units.density * p.units.velocity**2)).copy(),
        "B1": (
            (fld["B1"] / bscale).copy()
            if "B1" in fld
            else np.zeros_like(H["n"])
        ),
        "B2": (
            (fld["B2"] / bscale).copy()
            if "B2" in fld
            else np.zeros_like(H["n"])
        ),
        "B3": (
            (fld["B3"] / bscale).copy()
            if "B3" in fld
            else np.zeros_like(H["n"])
        ),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument(
        "--tstop",
        type=float,
        default=math.pi,
        help="final time (default π, the usual published OT snapshot; "
        "the contract nail is t=0.5)",
    )
    parser.add_argument(
        "--backend",
        choices=["serial", "host"],
        default="serial",
        help="recorded in the config; on this Kokkos build (Serial+OpenMP) "
        "the default execution space is already OpenMP -- OMP_NUM_THREADS "
        "is the real thread lever",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--frames-every",
        type=int,
        default=0,
        metavar="K",
        help="capture a snapshot every K steps into frames.npz; 0 disables "
        "unless --frame-dt or --animate is set",
    )
    parser.add_argument(
        "--frame-dt",
        type=float,
        default=0.0,
        metavar="DT",
        help="capture a snapshot every DT code-time units (preferred for "
        "animation). Ignored when <= 0",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="write ~60 time-spaced frames and build orszag_tang.gif "
        "after the run (implies --frame-dt = tstop/60 if unset)",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--no-mhd",
        action="store_true",
        help="same vortex IC but magnetic gate closed — OpenMP hydro only "
        "(occupancy control; not the OT contract)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        metavar="N",
        help="stop after N steps even if t < tstop (0 = run to tstop). "
        "Bench uses this so 1000² occupancy does not need a 0.1π wall",
    )
    args = parser.parse_args()
    frame_dt = float(args.frame_dt)
    if args.animate and frame_dt <= 0.0 and args.frames_every <= 0:
        frame_dt = args.tstop / 60.0
    want_frames = frame_dt > 0.0 or args.frames_every > 0

    mhd_on = not args.no_mhd
    cfg = ot_config(args.n, args.tstop, args.backend, mhd=mhd_on)
    out = Path(args.out)
    omp = os.environ.get("OMP_NUM_THREADS", "")
    print(f"xmf={Path(xmf.__file__).resolve()}", flush=True)
    print(
        f"ndim=2  n={args.n}^2  riemann=hll  recon=plm  rk=rk3  "
        f"tstop={args.tstop}  BC=periodic  mhd={'on' if mhd_on else 'off'}  "
        f"backend={args.backend}  OMP_NUM_THREADS={omp or '(default)'}",
        flush=True,
    )
    run = xmf.setup(cfg)
    try:
        pack_ic(run)
        divb0 = float(run.divb_max)
        e0 = total_energy(run)
        t0 = time.perf_counter()
        steps = 0
        max_divb = divb0
        frames_t: list[float] = []
        frames_rho: list[np.ndarray] = []
        frames_magp: list[np.ndarray] = []
        next_frame_t = 0.0

        def capture() -> None:
            # Interior rho + magnetic pressure (code units) for animate.py.
            code = unpack_code(run)
            frames_t.append(float(run.t))
            frames_rho.append(code["rho"][0].astype(np.float32))
            magp = 0.5 * (
                code["B1"][0] ** 2 + code["B2"][0] ** 2 + code["B3"][0] ** 2
            )
            frames_magp.append(magp.astype(np.float32))

        if want_frames:
            capture()
            next_frame_t = frame_dt
        while run.t < run.tstop:
            if args.max_steps > 0 and steps >= args.max_steps:
                break
            run.step()
            steps += 1
            # Port D: divb_max pulls B to the host mirror. Sample with the
            # progress print, not every step — otherwise the timed loop is
            # measuring the report path, not induction.
            if want_frames:
                by_step = args.frames_every > 0 and steps % args.frames_every == 0
                by_time = frame_dt > 0.0 and run.t + 1e-15 >= next_frame_t
                if by_step or by_time:
                    capture()
                    if by_time:
                        while next_frame_t <= run.t + 1e-15:
                            next_frame_t += frame_dt
            if steps % 20 == 0:
                divb = float(run.divb_max)
                max_divb = max(max_divb, divb)
                print(
                    f"step {steps}  t={run.t:.4e}  dt={run.dt:.3e}  "
                    f"divb_max={divb:.3e}",
                    flush=True,
                )
        if want_frames and (not frames_t or frames_t[-1] < run.t - 1e-12):
            capture()
        wall = time.perf_counter() - t0

        code = unpack_code(run)
        if not np.isfinite(code["rho"]).all():
            raise RuntimeError("non-finite density")
        e1 = total_energy(run)
        drift = abs(e1 - e0) / abs(e0)
        l1_rho = float(np.mean(np.abs(code["rho"] - RHO0)))
        l1_p = float(np.mean(np.abs(code["P"] - P0)))
        me = 0.5 * (code["B1"] ** 2 + code["B2"] ** 2 + code["B3"] ** 2)
        l1_me = float(np.mean(np.abs(me - 0.5 * B0 * B0)))

        meta = {
            "n": args.n,
            "tstop": args.tstop,
            "backend": args.backend,
            "mhd": mhd_on,
            "max_steps": args.max_steps,
            "riemann": "hll",
            "reconstruction": "plm",
            "rk": "rk3",
            "cfl": 0.4,
            "steps": steps,
            "t": float(run.t),
            "wall_s": wall,
            "us_per_step": (wall / steps * 1e6) if steps else 0.0,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
            "divb_initial": divb0,
            "divb_max": max_divb,
            "divb_final": float(run.divb_max),
            "energy_initial": e0,
            "energy_final": e1,
            "energy_drift": drift,
            "l1_rho": l1_rho,
            "l1_p": l1_p,
            "l1_me": l1_me,
            "ic": {"rho0": RHO0, "p0": P0, "B0": B0, "gamma": GAMMA},
            "xmf": str(Path(xmf.__file__).resolve()),
        }
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "fields.npz", **code)
        if frames_rho:
            np.savez_compressed(
                out / "frames.npz",
                t=np.asarray(frames_t, dtype=np.float64),
                rho=np.stack(frames_rho),
                magp=np.stack(frames_magp),
                x=code["x"],
                y=code["y"],
            )
            print(f"wrote {out / 'frames.npz'} ({len(frames_t)} frames)", flush=True)
            if args.animate:
                if str(HERE) not in sys.path:
                    sys.path.insert(0, str(HERE))
                from animate import make

                make(out, out / "orszag_tang.gif")
        (out / "summary.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(meta, indent=2), flush=True)
        print(f"wrote {out / 'fields.npz'}", flush=True)
        if not args.no_plot:
            if str(HERE) not in sys.path:
                sys.path.insert(0, str(HERE))
            from plot import draw

            draw(out)
    finally:
        run.close()


if __name__ == "__main__":
    main()
