"""A1 convergence: single sound wave L1 vs N for pcm+rk1, plm+rk2, weno3+rk3.

Setup
-----
- 1-D Cartesian, periodic boundaries, domain [0, 1]
- Single fluid, gamma = 5/3, uniform background (rho0, P0) = (1, 1)
- Small-amplitude right-going sound wave: delta rho, delta v, delta P all
  in phase (linearised Euler eigenvector)
- t_stop = L / cs  (one sound-crossing period; the wave returns to its
  initial position under periodic BCs)
- Error measured against the initial condition (exact = IC at t = t_stop)

Tiers
-----
    pcm + rk1   (nominal spatial order 1)
    plm + rk2   (nominal spatial order 2)
    weno3 + rk3 (nominal spatial order >= 2 on smooth data)

Assertions
----------
    PCM   order in [0.95, 1.05]
    PLM   order in [1.8,  2.1]
    WENO3 order >= 1.8

Usage
-----
    python utils/convergence/run.py

Writes convergence.txt and convergence.png under the same directory.
"""
from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import numpy as np

import xmf
from xmf._native import load as _load_lib

HERE = Path(__file__).resolve().parent

# --- physical setup ---------------------------------------------------------
GAMMA = 5.0 / 3.0
RHO0 = 1.0
P0 = 1.0
V0 = 0.0
AMP = 1.0e-7
LBOX = 1.0
CS = math.sqrt(GAMMA * P0 / RHO0)
TSTOP = LBOX / CS  # one sound-crossing period
CFL = 0.4

NS = (32, 64, 128, 256, 512)
TIERS = (
    {"reconstruction": "pcm",   "rk": "rk1", "riemann": "hllc", "label": "pcm+rk1"},
    {"reconstruction": "plm",   "rk": "rk2", "riemann": "hllc", "label": "plm+rk2"},
    {"reconstruction": "weno3", "rk": "rk3", "riemann": "hllc", "label": "weno3+rk3"},
)

_PTR = ctypes.POINTER(ctypes.c_double)


# --- kernel helpers (same layer as examples/hd_eval/kernel.py) ---------------
def _lib():
    return _load_lib()


def _axis(handle, direction: int) -> np.ndarray:
    ntot = _lib().xmf_mesh_ncell(handle, direction)
    buf = np.empty(ntot, dtype=np.float64)
    got = _lib().xmf_mesh_copy_x(handle, direction, buf.ctypes.data_as(_PTR))
    assert got == ntot
    return buf


def _pack_slot(handle, fluid: int, var: int, arr: np.ndarray) -> None:
    flat = np.ascontiguousarray(arr, dtype=np.float64)
    got = _lib().xmf_state_pack(handle, fluid, var, flat.ctypes.data_as(_PTR))
    assert got == flat.size


def _unpack_slot(handle, fluid: int, var: int, ncells: int) -> np.ndarray:
    buf = np.empty(ncells, dtype=np.float64)
    got = _lib().xmf_state_unpack(handle, fluid, var, buf.ctypes.data_as(_PTR))
    assert got == ncells
    return buf


def _make_cfg(n: int, recon: str, rk: str, riemann: str) -> dict:
    return {
        "schema": 1,
        "backend": "serial",
        "gamma": GAMMA,
        "units": {"density": 1.0, "velocity": 1.0, "length": 1.0},
        "numerics": {
            "riemann": riemann,
            "reconstruction": recon,
            "rk": rk,
        },
        "grid": {
            "geometry": "cartesian",
            "ndim": 1,
            "n": [n, 1, 1],
            "stretch": "none",
            "x1min": 0.0,
            "x1max": LBOX,
            "nghost": 2,
        },
        "time": {"cfl": CFL, "tstop": TSTOP, "dt_max": 0.0},
        "fluids": [{"name": "gas", "mass": 1.0, "charge": 0, "kind": "neutral"}],
        "electron": {"mode": "off"},
        "sources": [],
        "constants": {},
        "radiation": {"enabled": False, "absorbers": []},
        "boundary": {"x1": "periodic"},
        "ic": {"kind": "none"},
    }


def _sound_wave(x: np.ndarray, t: float) -> dict[str, np.ndarray]:
    """Right-going sound wave: exact linearised Euler solution at time t.

    rho(x,t) = rho0 + A sin(k(x - cs t))
    v(x,t)   = v0 + (cs/rho0) A sin(k(x - cs t))
    P(x,t)   = P0 + cs^2 A sin(k(x - cs t))

    With periodic BCs the argument wraps modulo LBOX.
    """
    k = 2.0 * math.pi / LBOX
    phase = np.sin(k * (x - CS * t))
    rho = RHO0 + AMP * phase
    vx = V0 + (CS / RHO0) * AMP * phase
    p = P0 + CS * CS * AMP * phase
    return {"rho": rho, "vx": vx, "p": p}


def _run_one(n: int, tier: dict) -> dict:
    """Run one (N, tier) pair. Returns dict with L1/Linf for rho, vx, P."""
    cfg = _make_cfg(n, tier["reconstruction"], tier["rk"], tier["riemann"])
    run = xmf.setup(cfg)
    p = run._cfg
    ng = p.ng
    nxt = n + 2 * ng
    nyt = 1 + 2 * ng
    nzt = 1 + 2 * ng
    ncells = nxt * nyt * nzt

    # Build initial conditions on the full lattice (t = 0).
    x = _axis(run._h, 0)  # length nxt, includes ghosts
    ic = _sound_wave(x, 0.0)
    # Broadcast to 3-D shape (nzt, nyt, nxt) -- 1-D slab.
    rho3 = np.broadcast_to(ic["rho"][None, None, :], (nzt, nyt, nxt)).copy()
    vx3 = np.broadcast_to(ic["vx"][None, None, :], (nzt, nyt, nxt)).copy()
    zero3 = np.zeros((nzt, nyt, nxt), dtype=np.float64)
    p3 = np.broadcast_to(ic["p"][None, None, :], (nzt, nyt, nxt)).copy()

    _pack_slot(run._h, 0, 0, rho3)  # rho
    _pack_slot(run._h, 0, 1, vx3)   # vx
    _pack_slot(run._h, 0, 2, zero3)  # vy
    _pack_slot(run._h, 0, 3, zero3)  # vz
    _pack_slot(run._h, 0, 4, p3)    # P

    # Fill ghosts from periodic BCs.
    _lib().xmf_fill_bc(run._h)

    # Advance to t_stop.
    nstep = 0
    while run.t < TSTOP - 1.0e-14:
        run.step()
        nstep += 1

    t_actual = run.t  # may slightly overshoot TSTOP

    # Read final interior fields.
    rho_f = _unpack_slot(run._h, 0, 0, ncells).reshape((nzt, nyt, nxt))
    vx_f = _unpack_slot(run._h, 0, 1, ncells).reshape((nzt, nyt, nxt))
    p_f = _unpack_slot(run._h, 0, 4, ncells).reshape((nzt, nyt, nxt))
    rho_f_inner = rho_f[ng, ng, ng:ng + n].copy()
    vx_f_inner = vx_f[ng, ng, ng:ng + n].copy()
    p_f_inner = p_f[ng, ng, ng:ng + n].copy()

    run.close()

    # Exact solution at the actual final time (on interior cell centres).
    x_inner = x[ng:ng + n]
    exact = _sound_wave(x_inner, t_actual)

    # Errors against the exact linearised Euler solution.
    err_rho = np.abs(rho_f_inner - exact["rho"])
    err_vx = np.abs(vx_f_inner - exact["vx"])
    err_p = np.abs(p_f_inner - exact["p"])

    return {
        "n": n,
        "nstep": nstep,
        "t_final": t_actual,
        "L1_rho": float(np.mean(err_rho)),
        "Linf_rho": float(np.max(err_rho)),
        "L1_vx": float(np.mean(err_vx)),
        "Linf_vx": float(np.max(err_vx)),
        "L1_p": float(np.mean(err_p)),
        "Linf_p": float(np.max(err_p)),
    }


def _fit_order(ns, errors):
    """Least-squares fit: error = C * N^(-p). Returns (p, log_C).

    log(error) = -p * log(N) + log(C), so the order p = -slope.
    """
    log_n = np.log(np.asarray(ns, dtype=np.float64))
    log_e = np.log(np.asarray(errors, dtype=np.float64))
    mask = np.isfinite(log_e)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    A = np.vstack([log_n[mask], np.ones(mask.sum())]).T
    sol, _, _, _ = np.linalg.lstsq(A, log_e[mask], rcond=None)
    return float(-sol[0]), float(sol[1])


def _pairwise_orders(ns, errors):
    """Consecutive refinement orders: order_i = log(e_i/e_{i+1}) / log(N_{i+1}/N_i)."""
    orders = []
    for i in range(len(ns) - 1):
        if errors[i] > 0 and errors[i + 1] > 0:
            o = math.log(errors[i] / errors[i + 1]) / math.log(ns[i + 1] / ns[i])
            orders.append(o)
        else:
            orders.append(float("nan"))
    return orders


def _run_one_cfl(n: int, tier: dict, cfl: float) -> dict:
    """Same as _run_one but with a custom CFL value."""
    cfg = _make_cfg(n, tier["reconstruction"], tier["rk"], tier["riemann"])
    cfg["time"]["cfl"] = cfl
    run = xmf.setup(cfg)
    p = run._cfg
    ng = p.ng
    nxt = n + 2 * ng
    nyt = 1 + 2 * ng
    nzt = 1 + 2 * ng
    ncells = nxt * nyt * nzt

    x = _axis(run._h, 0)
    ic = _sound_wave(x, 0.0)
    rho3 = np.broadcast_to(ic["rho"][None, None, :], (nzt, nyt, nxt)).copy()
    vx3 = np.broadcast_to(ic["vx"][None, None, :], (nzt, nyt, nxt)).copy()
    zero3 = np.zeros((nzt, nyt, nxt), dtype=np.float64)
    p3 = np.broadcast_to(ic["p"][None, None, :], (nzt, nyt, nxt)).copy()

    _pack_slot(run._h, 0, 0, rho3)
    _pack_slot(run._h, 0, 1, vx3)
    _pack_slot(run._h, 0, 2, zero3)
    _pack_slot(run._h, 0, 3, zero3)
    _pack_slot(run._h, 0, 4, p3)
    _lib().xmf_fill_bc(run._h)

    nstep = 0
    while run.t < TSTOP - 1.0e-14:
        run.step()
        nstep += 1

    t_actual = run.t

    rho_f = _unpack_slot(run._h, 0, 0, ncells).reshape((nzt, nyt, nxt))
    vx_f = _unpack_slot(run._h, 0, 1, ncells).reshape((nzt, nyt, nxt))
    p_f = _unpack_slot(run._h, 0, 4, ncells).reshape((nzt, nyt, nxt))
    rho_f_inner = rho_f[ng, ng, ng:ng + n].copy()
    vx_f_inner = vx_f[ng, ng, ng:ng + n].copy()
    p_f_inner = p_f[ng, ng, ng:ng + n].copy()

    run.close()

    x_inner = x[ng:ng + n]
    exact = _sound_wave(x_inner, t_actual)
    err_rho = np.abs(rho_f_inner - exact["rho"])
    err_vx = np.abs(vx_f_inner - exact["vx"])
    err_p = np.abs(p_f_inner - exact["p"])

    return {
        "n": n,
        "nstep": nstep,
        "t_final": t_actual,
        "cfl": cfl,
        "L1_rho": float(np.mean(err_rho)),
        "Linf_rho": float(np.max(err_rho)),
        "L1_vx": float(np.mean(err_vx)),
        "Linf_vx": float(np.max(err_vx)),
        "L1_p": float(np.mean(err_p)),
        "Linf_p": float(np.max(err_p)),
    }


def main() -> None:
    lines = []
    lines.append("# A1 Convergence: Sound Wave L1 vs N")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- gamma = {GAMMA:.10f} (5/3)")
    lines.append(f"- background: rho0 = {RHO0}, P0 = {P0}, v0 = {V0}")
    lines.append(f"- amplitude A = {AMP}")
    lines.append(f"- sound speed cs = sqrt(gamma P0/rho0) = {CS:.6f}")
    lines.append(f"- domain L = {LBOX}, t_stop = L/cs = {TSTOP:.6f} (one sound-crossing period)")
    lines.append(f"- CFL = {CFL}")
    lines.append(f"- Riemann solver = hllc")
    lines.append(f"- boundary = periodic (x1)")
    lines.append(f"- resolutions N = {NS}")
    lines.append(f"- error = |numerical - exact linear wave at actual t_final|")
    lines.append("")

    all_results = {}
    fitted_orders = {}

    for tier in TIERS:
        label = tier["label"]
        print(f"[A1] tier={label}", flush=True)
        results = []
        for n in NS:
            rec = _run_one(n, tier)
            results.append(rec)
            print(
                f"  N={n:4d}  nstep={rec['nstep']:5d}  t={rec['t_final']:.6f}  "
                f"L1(rho)={rec['L1_rho']:.3e}  Linf(rho)={rec['Linf_rho']:.3e}  "
                f"L1(vx)={rec['L1_vx']:.3e}  L1(P)={rec['L1_p']:.3e}",
                flush=True,
            )
        all_results[label] = results

        # Fit convergence order from L1(rho).
        ns_arr = [r["n"] for r in results]
        l1_rho = [r["L1_rho"] for r in results]
        linf_rho = [r["Linf_rho"] for r in results]
        l1_vx = [r["L1_vx"] for r in results]
        l1_p = [r["L1_p"] for r in results]

        order_l1_rho, _ = _fit_order(ns_arr, l1_rho)
        order_linf_rho, _ = _fit_order(ns_arr, linf_rho)
        order_l1_vx, _ = _fit_order(ns_arr, l1_vx)
        order_l1_p, _ = _fit_order(ns_arr, l1_p)

        fitted_orders[label] = {
            "L1_rho": order_l1_rho,
            "Linf_rho": order_linf_rho,
            "L1_vx": order_l1_vx,
            "L1_p": order_l1_p,
        }

        lines.append(f"## Tier: {label}")
        lines.append("")
        lines.append("| N | nstep | t_final | L1(rho) | Linf(rho) | L1(vx) | Linf(vx) | L1(P) | Linf(P) |")
        lines.append("|---|-------|---------|---------|-----------|--------|----------|-------|---------|")
        for r in results:
            lines.append(
                f"| {r['n']} | {r['nstep']} | {r['t_final']:.6f} | {r['L1_rho']:.4e} | {r['Linf_rho']:.4e} "
                f"| {r['L1_vx']:.4e} | {r['Linf_vx']:.4e} | {r['L1_p']:.4e} | {r['Linf_p']:.4e} |"
            )
        lines.append("")
        lines.append(f"- Fitted order (L1 rho):     {order_l1_rho:.3f}")
        lines.append(f"- Fitted order (Linf rho):   {order_linf_rho:.3f}")
        lines.append(f"- Fitted order (L1 vx):      {order_l1_vx:.3f}")
        lines.append(f"- Fitted order (L1 P):       {order_l1_p:.3f}")
        # Pairwise orders (diagnostic: shows per-doubling convergence).
        pw_rho = _pairwise_orders(ns_arr, l1_rho)
        pw_str = ", ".join(f"{o:.2f}" for o in pw_rho)
        lines.append(f"- Pairwise orders (L1 rho):  {pw_str}")
        lines.append("")

    # --- time-refinement check at fixed N=512 --------------------------------
    lines.append("## Time-refinement check (N=512, CFL halved)")
    lines.append("")
    lines.append("Run N=512 at CFL=0.2 (half the default). If temporal error dominates,")
    lines.append("the error should drop by ~2^(RK order) when CFL is halved.")
    lines.append("")

    n_ref = 512
    lines.append("| tier | CFL | L1(rho) | ratio | expected (if temporal dominates) |")
    lines.append("|------|-----|---------|-------|----------------------------------|")
    for tier in TIERS:
        label = tier["label"]
        rk_order = {"rk1": 1, "rk2": 2, "rk3": 3}[tier["rk"]]
        # Get the CFL=0.4 result from the main sweep.
        rec_default = next(r for r in all_results[label] if r["n"] == n_ref)
        # Run at CFL=0.2.
        rec_half = _run_one_cfl(n_ref, tier, 0.2)
        ratio = rec_default["L1_rho"] / max(rec_half["L1_rho"], 1.0e-300)
        expected = 2.0 ** rk_order
        lines.append(
            f"| {label} | 0.4 | {rec_default['L1_rho']:.4e} | | |"
        )
        lines.append(
            f"| {label} | 0.2 | {rec_half['L1_rho']:.4e} | {ratio:.2f} | {expected:.0f} |"
        )
        print(
            f"  [dt-check] {label}: CFL=0.4 L1={rec_default['L1_rho']:.3e}, "
            f"CFL=0.2 L1={rec_half['L1_rho']:.3e}, ratio={ratio:.2f} (expect ~{expected} if temporal)",
            flush=True,
        )
    lines.append("")

    # --- assertions ---------------------------------------------------------
    lines.append("## Assertions")
    lines.append("")

    assertions = []
    for label, (lo, hi) in [
        ("pcm+rk1",   (0.95, 1.05)),
        ("plm+rk2",   (1.8,  2.1)),
        ("weno3+rk3", (1.8,  float("inf"))),
    ]:
        order = fitted_orders[label]["L1_rho"]
        if hi == float("inf"):
            passed = order >= lo
            spec = f">= {lo}"
        else:
            passed = lo <= order <= hi
            spec = f"[{lo}, {hi}]"
        status = "PASS" if passed else "FAIL"
        assertions.append((label, order, spec, status))
        lines.append(f"- {label}: order(L1 rho) = {order:.3f}, spec = {spec} -> **{status}**")

    lines.append("")

    # --- conclusion ---------------------------------------------------------
    lines.append("## Conclusion")
    lines.append("")
    all_pass = all(a[3] == "PASS" for a in assertions)
    if all_pass:
        lines.append("All tiers pass their convergence order assertions.")
    else:
        passed_list = [a[0] for a in assertions if a[3] == "PASS"]
        failed = [a[0] for a in assertions if a[3] != "PASS"]
        if passed_list:
            lines.append(f"Passed: {', '.join(passed_list)}.")
        if failed:
            lines.append(f"Failed: {', '.join(failed)}. See tables and time-refinement check above.")
    lines.append("")

    # Write report.
    report = "\n".join(lines)
    (HERE / "convergence.txt").write_text(report, encoding="utf-8")
    print(report)

    # --- plot (optional, matplotlib may not be available) -------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for label, res in all_results.items():
            ns_arr = [r["n"] for r in res]
            l1_rho = [r["L1_rho"] for r in res]
            linf_rho = [r["Linf_rho"] for r in res]
            axes[0].loglog(ns_arr, l1_rho, "o-", label=label)
            axes[1].loglog(ns_arr, linf_rho, "o-", label=label)

        # Reference slopes.
        nref = np.array([32, 512], dtype=float)
        for slope, ls in [(1, ":"), (2, "--"), (3, "-.")]:
            ref = 1.0e-3 * (nref / 32.0) ** (-slope)
            axes[0].loglog(nref, ref, ls, color="grey", alpha=0.5, label=f"O(N^{{-{slope}}})")
            axes[1].loglog(nref, ref, ls, color="grey", alpha=0.5, label=f"O(N^{{-{slope}}})")

        axes[0].set_xlabel("N")
        axes[0].set_ylabel("L1(rho)")
        axes[0].set_title("L1 convergence")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, which="both", ls=":", alpha=0.3)

        axes[1].set_xlabel("N")
        axes[1].set_ylabel("Linf(rho)")
        axes[1].set_title("Linf convergence")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, which="both", ls=":", alpha=0.3)

        fig.tight_layout()
        fig.savefig(HERE / "convergence.png", dpi=150)
        plt.close(fig)
        lines.append("![convergence](convergence.png)")
        (HERE / "convergence.txt").write_text("\n".join(lines), encoding="utf-8")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
