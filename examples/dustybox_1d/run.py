"""1-D two-fluid dustybox (F1).

Laibe & Price (2011, MNRAS 418:1491) dustybox: uniform periodic box,
two fluids of equal mass and equal density, uniform pressure, initial
velocity offset Δu₀. Drag couples them with timescale τ; analytic
solution Δu(t) = Δu₀ exp(-t/τ). Pure source-term test.

    python examples/dustybox_1d/run.py
    python examples/dustybox_1d/run.py --tier weak --nx 128
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

import xmf
from dustybox_config import RUN, dustybox_1d

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
_PTR = ctypes.POINTER(ctypes.c_double)
FLUIDS = ("A", "B")
TIERS = ("weak", "mid", "strong")
NX_SWEEP = (64, 128)


def lib():
    return xmf.api._lib()


def _pack_slot(handle, fluid: int, var: int, arr: np.ndarray) -> None:
    flat = np.ascontiguousarray(arr, dtype=np.float64)
    got = lib().xmf_state_pack(handle, fluid, var, flat.ctypes.data_as(_PTR))
    assert got == flat.size


def analytic_du(t: float, tau: float, du0: float) -> float:
    return du0 * np.exp(-t / tau)


def run_one(nx: int, tier: str) -> dict:
    """Run one (nx, tier) combination and return error diagnostics."""
    cfg = dustybox_1d(nx=nx, tier=tier)
    tau = RUN.tiers[tier]
    du0 = RUN.du0
    ts = cfg["time"]["tstop"]
    nx_g = cfg["grid"]["n"][0]

    print(f"  nx={nx_g}  tier={tier}  τ={tau}  K={cfg['constants']['K']:.4e}  "
          f"tstop={ts:.4e}", flush=True)

    run = xmf.setup(cfg)
    try:
        # Pack initial conditions: uniform ρ=1, P=P0, v_A = +Δu₀/2, v_B = -Δu₀/2
        ng = run._cfg.ng
        nxtot = nx_g + 2 * ng
        nytot = 1 + 2 * ng
        nztot = 1 + 2 * ng
        shape = (nztot, nytot, nxtot)

        rho = np.full(shape, RUN.rho0, dtype=np.float64)
        P = np.full(shape, RUN.P0, dtype=np.float64)
        vx_A = np.full(shape, +du0 / 2.0, dtype=np.float64)
        vx_B = np.full(shape, -du0 / 2.0, dtype=np.float64)
        zero = np.zeros(shape, dtype=np.float64)

        # Pack fluid A: slots 0(rho), 1(vx), 2(vy), 3(vz), 4(P)
        _pack_slot(run._h, 0, 0, rho)
        _pack_slot(run._h, 0, 1, vx_A)
        _pack_slot(run._h, 0, 2, zero)
        _pack_slot(run._h, 0, 3, zero)
        _pack_slot(run._h, 0, 4, P)
        # Pack fluid B
        _pack_slot(run._h, 1, 0, rho)
        _pack_slot(run._h, 1, 1, vx_B)
        _pack_slot(run._h, 1, 2, zero)
        _pack_slot(run._h, 1, 3, zero)
        _pack_slot(run._h, 1, 4, P)

        # Fill ghost zones (periodic wraps)
        lib().xmf_fill_bc(run._h)

        # Sample at t = τ, 2τ, 3τ
        sample_times = [tau, 2.0 * tau, 3.0 * tau]
        samples = []
        t0 = time.perf_counter()
        steps = 0

        for t_target in sample_times:
            while run.t < t_target - 1e-14 * tau:
                run.step()
                steps += 1
                if RUN.log_every and steps % RUN.log_every == 0:
                    print(f"    step {steps}  t={run.t:.4e}  dt={run.dt:.3e}",
                          flush=True)

            # Read fields
            fld = run.fields_full()
            vx_a = fld["A"]["vx"][0, 0, :]  # shape (nx,)
            vx_b = fld["B"]["vx"][0, 0, :]
            du_num = vx_a - vx_b  # per-cell Δu
            du_ana = analytic_du(run.t, tau, du0)
            err = np.abs(du_num - du_ana)
            L1 = float(np.mean(err))
            Linf = float(np.max(err))
            du_mean = float(np.mean(du_num))
            samples.append({
                "t": float(run.t),
                "t_over_tau": float(run.t / tau),
                "du_ana": du_ana,
                "du_num_mean": du_mean,
                "du_num_min": float(np.min(du_num)),
                "du_num_max": float(np.max(du_num)),
                "L1": L1,
                "Linf": Linf,
                "steps": steps,
            })

        wall = time.perf_counter() - t0

        # Final check: all fields finite
        fld = run.fields_full()
        for name in FLUIDS:
            for key in ("n", "vx", "P"):
                if not np.isfinite(fld[name][key]).all():
                    raise RuntimeError(f"non-finite {name}.{key}")

        return {
            "nx": nx_g,
            "tier": tier,
            "tau": tau,
            "K": cfg["constants"]["K"],
            "du0": du0,
            "tstop": ts,
            "samples": samples,
            "wall_s": wall,
            "total_steps": steps,
        }
    finally:
        run.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="dustybox 1D drag benchmark")
    parser.add_argument("--nx", type=int, default=None,
                        help="single resolution override (skips sweep)")
    parser.add_argument("--tier", default=None, choices=TIERS,
                        help="single tier override (skips sweep)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"xmf={Path(xmf.__file__).resolve()}", flush=True)

    tiers = [args.tier] if args.tier else list(TIERS)
    nx_list = [args.nx] if args.nx else list(NX_SWEEP)

    all_results = []
    for tier in tiers:
        print(f"\n=== tier={tier}  τ={RUN.tiers[tier]} ===", flush=True)
        for nx in nx_list:
            result = run_one(nx, tier)
            all_results.append(result)

    # Print summary table
    print("\n" + "=" * 90, flush=True)
    print(f"{'tier':>6} {'τ':>6} {'nx':>5} "
          f"{'t/τ':>5} {'L1':>12} {'L∞':>12} {'du_ana':>10} {'du_num':>10} {'steps':>6}",
          flush=True)
    print("-" * 90, flush=True)
    for r in all_results:
        for s in r["samples"]:
            print(f"{r['tier']:>6} {r['tau']:>6.1f} {r['nx']:>5} "
                  f"{s['t_over_tau']:>5.1f} {s['L1']:>12.4e} {s['Linf']:>12.4e} "
                  f"{s['du_ana']:>10.6f} {s['du_num_mean']:>10.6f} {s['steps']:>6}",
                  flush=True)

    # Convergence check: for each tier at t=3τ, L1 should decrease with nx
    print("\n--- Convergence assertion (t=3τ) ---", flush=True)
    for tier in tiers:
        tier_results = [r for r in all_results if r["tier"] == tier]
        if len(tier_results) >= 2:
            L1_coarse = tier_results[0]["samples"][-1]["L1"]
            L1_fine = tier_results[1]["samples"][-1]["L1"]
            converged = L1_fine < L1_coarse
            print(f"  {tier}: L1(nx={tier_results[0]['nx']})={L1_coarse:.4e}  "
                  f"L1(nx={tier_results[1]['nx']})={L1_fine:.4e}  "
                  f"converged={'YES' if converged else 'NO'}", flush=True)
            if not converged:
                print(f"  WARNING: {tier} did NOT converge!", flush=True)

    # Save results
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "dustybox_results.json"
    out_path.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)

    # Optional plot
    if RUN.plots and not args.no_plot:
        try:
            from plot import draw
            draw(all_results, OUT)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
