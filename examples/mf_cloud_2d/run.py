"""2-D two-fluid shock-dense-cloud interaction (D2).

Klein, McKee & Colella (1994) cloud crushing lifted to two fluids: the
shock is an initial condition in fluid A (wind); the dense cloud is fluid
B and is dragged only through the collision tapes. K tiers: off / weak /
mid / strong (see cloud_config.RUN.tau_over_tcc). All four faces are
`copy` -- the post-shock flow is supersonic, so the left reservoir stays
pristine and feeds the inflow. Writes output/fields_<tier>.npz +
summary_<tier>.json.

    python examples/mf_cloud_2d/run.py --tier off --no-plot
    python examples/mf_cloud_2d/run.py --tier strong --nx 256 --ny 128
    python examples/mf_cloud_2d/plot.py --tiers off mid strong
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

import xmf
from cloud_config import RUN, cloud_2d, post_shock, tier_K

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
_PTR = ctypes.POINTER(ctypes.c_double)
FLUIDS = ("wind", "cloud")


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
    ncells = int(np.prod(shape))
    buf = np.empty(ncells, dtype=np.float64)
    got = lib().xmf_state_unpack(handle, fluid, var, buf.ctypes.data_as(_PTR))
    assert got == ncells
    return buf.reshape(shape)


def pack_ic(run) -> None:
    """Shock as an initial condition: fluid A carries the exact
    Rankine-Hugoniot post-shock state left of x_shock; fluid B is the
    dense cloud patch (pressure-matched) on a same-temperature floor."""
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
    x = _axis(run._h, 0)
    y = _axis(run._h, 1)
    z = _axis(run._h, 2)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    ps = post_shock()
    shocked = xx < RUN.x_shock
    in_cloud = (xx - RUN.cloud_x) ** 2 + (yy - RUN.cloud_y) ** 2 < (
        RUN.cloud_r**2
    )
    zero = np.zeros_like(xx)
    # Fluid A (wind): ambient ahead, post-shock behind x_shock.
    rho_w = np.where(shocked, ps["rho_A1"], RUN.rho_amb)
    vx_w = np.where(shocked, ps["v_A1"], 0.0)
    P_w = np.where(shocked, ps["P_A1"], RUN.P_amb)
    _pack_slot(run._h, 0, 0, rho_w)
    _pack_slot(run._h, 0, 1, vx_w)
    _pack_slot(run._h, 0, 2, zero)
    _pack_slot(run._h, 0, 3, zero)
    _pack_slot(run._h, 0, 4, P_w)
    # Fluid B (cloud): dense patch on a tenuous same-fluid background.
    # P_B is UNIFORM = P_amb: fluids do not share pressure, so each must
    # be in its own equilibrium; the rho jump at the cloud edge is a
    # stable contact (the classical cloud-crushing construction).
    rho_c = np.where(in_cloud, RUN.chi * RUN.rho_amb, RUN.n_floor)
    P_c = np.full_like(xx, RUN.P_amb)
    _pack_slot(run._h, 1, 0, rho_c)
    _pack_slot(run._h, 1, 1, zero)
    _pack_slot(run._h, 1, 2, zero)
    _pack_slot(run._h, 1, 3, zero)
    _pack_slot(run._h, 1, 4, P_c)
    lib().xmf_fill_bc(run._h)


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
    for f, name in enumerate(FLUIDS):
        for var, tag in ((0, "rho"), (1, "vx"), (2, "vy"), (4, "P")):
            blk = _unpack_slot(run._h, f, var, (nzt, nyt, nxt))
            out[f"{name}_{tag}"] = blk[inner][0].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--tstop", type=float, default=None)
    parser.add_argument(
        "--tier",
        default=None,
        choices=("off", "weak", "mid", "strong"),
        help="collision tier (tau_c in units of t_cc; off = empty sources)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    tier = str(RUN.tier if args.tier is None else args.tier)
    cfg = cloud_2d(nx=args.nx, ny=args.ny, tstop=args.tstop, tier=tier)
    nx, ny, _ = cfg["grid"]["n"]
    ps = post_shock()
    print(f"xmf={Path(xmf.__file__).resolve()}", flush=True)
    print(
        f"ndim=2  n={nx}x{ny}  tier={tier}  K={tier_K(tier):.4e}  "
        f"M={RUN.Mach}  chi={RUN.chi}  t_cc={ps['t_cc']:.4e}  "
        f"tstop={cfg['time']['tstop']:.4e} ({cfg['time']['tstop'] / ps['t_cc']:.2f} t_cc)",
        flush=True,
    )
    run = xmf.setup(cfg)
    try:
        pack_ic(run)
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
        for name in FLUIDS:
            for tag in ("rho", "vx", "vy", "P"):
                arr = code[f"{name}_{tag}"]
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"non-finite {name}_{tag}")
            if (code[f"{name}_rho"] <= 0.0).any():
                raise RuntimeError(f"non-positive {name}_rho")
            if (code[f"{name}_P"] <= 0.0).any():
                raise RuntimeError(f"non-positive {name}_P")
        du = np.sqrt(
            (code["wind_vx"] - code["cloud_vx"]) ** 2
            + (code["wind_vy"] - code["cloud_vy"]) ** 2
        )
        v_cloud = np.sqrt(code["cloud_vx"] ** 2 + code["cloud_vy"] ** 2)
        dx = code["x"][1] - code["x"][0]
        dy = code["y"][1] - code["y"][0]
        meta = {
            "tier": tier,
            "K": tier_K(tier),
            "nx": nx,
            "ny": ny,
            "riemann": RUN.riemann,
            "reconstruction": RUN.reconstruction,
            "rk": RUN.rk,
            "cfl": RUN.cfl,
            "steps": steps,
            "t": float(run.t),
            "t_cc": ps["t_cc"],
            "t_over_tcc": float(run.t) / ps["t_cc"],
            "dt": float(run.dt),
            "wall_s": wall,
            "du_max": float(np.max(du)),
            "v_cloud_max": float(np.max(v_cloud)),
            "mass_cloud": float(np.sum(code["cloud_rho"]) * dx * dy),
            "mass_wind": float(np.sum(code["wind_rho"]) * dx * dy),
            "mach": RUN.Mach,
            "chi": RUN.chi,
            **{k: float(v) for k, v in ps.items() if k not in ("tier",)},
            "xmf": str(Path(xmf.__file__).resolve()),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        dump = OUT / f"fields_{tier}.npz"
        np.savez_compressed(dump, **code)
        (OUT / f"summary_{tier}.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(meta, indent=2), flush=True)
        print(f"wrote {dump}", flush=True)
        if RUN.plots and not args.no_plot:
            import sys

            if str(HERE) not in sys.path:
                sys.path.insert(0, str(HERE))
            from plot import draw_tier

            draw_tier(OUT, tier)
    finally:
        run.close()


if __name__ == "__main__":
    main()
