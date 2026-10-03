"""Radiation-driven two-fluid (HI/HII) escape with a pressure-only
electron block on a 1-D spherical grid (B2 example).

Builds the setup through the hpe_config.hpe_1d() factory (the
example-PRIVATE config module in this directory -- never part of
xmf.config, never exported by `import xmf`; full cgs: every unit factor
is 1, so code values ARE cgs values), packs the bifluid exponential
n(r) = n_beg exp(-(r-R_p)/H_ic) initial state plus the electron
pressure P_e = n_HII kB T_beg through the xmf_state_pack /
xmf_state_write_pe helper seams, advances the radiation-on run to
t_stop, and writes the figures plus a plain-text number block into
output/. The no-heating (radiation-off) comparison is no longer a
default evaluation.

Usage:
    python examples/rad_h_pe_1d/run.py

Knobs (nx, tstop, stretch, n_beam, plots, log_every) live in the
pinned RUN object at the top of hpe_config.py. There is no CLI.

The SED table and the photo-ionisation cross sections are REFERENCED
from ../rad_hd_1d/data/ (the SSH reference tables shipped with the A5
example); this directory deliberately holds no copy of its own.
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

import xmf
from hpe_config import RUN, hpe_1d
from xmf.config import KB

HERE = os.path.dirname(os.path.abspath(__file__))
# Reference the A5 data directory; never copy the tables.
DATA = os.path.join(os.path.dirname(HERE), "rad_hd_1d", "data")
OUT = os.path.join(HERE, "output")
DPTR = ctypes.POINTER(ctypes.c_double)


def lib():
    return xmf.api._lib()


def mdot_g_s(n, v, r, mass_g):
    """Spherical mass-loss rate 4π m n r^2 v_r, g/s."""
    return (4.0 * np.pi) * mass_g * n * r * r * v


def snapshot(run, r: np.ndarray) -> dict:
    """Interior radial profiles (cgs). T = P / (n k_B); the electron
    pressure block rides along as its own profile."""
    fld = run.fields_full()
    out = {"r": r.copy(), "P_e": fld["P_e"][0, 0, :].copy()}
    for name in ("HI", "HII"):
        n = fld[name]["n"][0, 0, :].copy()
        p = fld[name]["P"][0, 0, :].copy()
        out[name] = {
            "n": n,
            "vx": fld[name]["vx"][0, 0, :].copy(),
            "P": p,
            "T": p / (np.maximum(n, 1e-300) * KB),
        }
    # Quasi-neutral read: the only charged fluid is HII, so n_e = n_HII.
    out["T_e"] = out["P_e"] / (np.maximum(out["HII"]["n"], 1e-300) * KB)
    return out


def pack_ic(run, cfg) -> None:
    """Bifluid n(r) = n_beg*frac*exp(-(r-R_p)/H_ic), v = 0,
    P = n*kB*T_beg, and the electron pressure P_e(r) = n_HII(r)*kB*T_beg
    (quasi-neutral IC: n_e = n_HII).

    Packed by radius over every block (ghosts included); P_e through the
    xmf_state_write_pe seam (same block length, ghosts included). The
    x1beg user_bc tapes assign the fluid base to ghosts only; P_e ghosts
    fall back to the existing copy @1 fill."""
    consts = cfg["constants"]
    m_g = consts["m_HI_g"]
    r_p = consts["R_p"]
    h_ic = consts["H_ic"]
    n_min = consts["n_ic_min"]
    ng = cfg["grid"]["nghost"]
    nxtot = cfg["grid"]["n"][0] + 2 * ng
    ntot = (1 + 2 * ng) * (1 + 2 * ng) * nxtot
    xc = np.empty(nxtot, dtype=np.float64)
    got = lib().xmf_mesh_copy_x(run._h, 0, xc.ctypes.data_as(DPTR))
    assert got == nxtot
    n_hii_prof = None
    for f, frac in enumerate((1.0, consts["ion_frac"])):
        n = consts["n_beg"] * frac * np.exp(-(xc - r_p) / h_ic)
        n = np.maximum(n, n_min * frac)
        if f == 1:
            n_hii_prof = n
        blocks = (
            (0, n * m_g),
            (1, np.zeros_like(n)),
            (2, np.zeros_like(n)),
            (3, np.zeros_like(n)),
            (4, n * KB * consts["T_beg"]),
        )
        for var, arr in blocks:
            full = np.tile(arr, (1 + 2 * ng) * (1 + 2 * ng))
            got = lib().xmf_state_pack(run._h, f, var,
                                       full.ctypes.data_as(DPTR))
            assert got == ntot
    # Electron pressure block: P_e = n_HII kB T_beg (n_e = n_HII).
    pe = n_hii_prof * KB * consts["T_beg"]
    full = np.tile(pe, (1 + 2 * ng) * (1 + 2 * ng))
    got = lib().xmf_state_write_pe(run._h, full.ctypes.data_as(DPTR))
    assert got == ntot


def one_run(radiation: bool) -> dict:
    label = "on" if radiation else "off"
    cfg = hpe_1d(
        DATA,
        radiation=radiation,
        nx=RUN.nx,
        n_beam=RUN.n_beam,
        tstop=RUN.tstop,
        stretch=RUN.stretch,
        stretch_factor=RUN.stretch_factor,
    )
    run = xmf.setup(cfg)
    assert run.num_states == 5 * len(cfg["fluids"]) + 1, (
        "B2 open-electron contract: N = 5*nfluid + 1 (P_e last)"
    )
    pack_ic(run, cfg)
    r_p = cfg["constants"]["R_p"]
    xc = np.empty(RUN.nx + 4, dtype=np.float64)
    lib().xmf_mesh_copy_x(run._h, 0, xc.ctypes.data_as(DPTR))
    r = xc[2 : 2 + RUN.nx]
    dxf = np.empty(RUN.nx + 4, dtype=np.float64)
    lib().xmf_mesh_copy_dx(run._h, 0, dxf.ctypes.data_as(DPTR))
    ic = snapshot(run, r)

    hist = {"t": [], "dt": [], "resid": [], "ion_frac_base": [],
            "vx_hi_top": [], "vx_hii_top": [],
            "pe_base": [], "pe_top": []}
    t_wall = time.time()
    steps = 0
    while run.t < RUN.tstop:
        run.step()
        steps += 1
        if RUN.log_every and (
            steps % RUN.log_every == 0 or run.t >= RUN.tstop
        ):
            print(
                f"[{label}] step {steps}  t = {run.t:.4e} s  "
                f"dt = {run.dt:.3e} s  resid = {run.source_resid:.2e}",
                flush=True,
            )
        if steps % 10 == 1 or run.t >= RUN.tstop:
            fld = run.fields_full()
            n1 = fld["HI"]["n"][0, 0, :]
            n2 = fld["HII"]["n"][0, 0, :]
            pe = fld["P_e"][0, 0, :]
            hist["t"].append(run.t)
            hist["dt"].append(run.dt)
            hist["resid"].append(run.source_resid)
            hist["ion_frac_base"].append(n2[0] / max(n1[0], 1e-300))
            hist["vx_hi_top"].append(fld["HI"]["vx"][0, 0, -1])
            hist["vx_hii_top"].append(fld["HII"]["vx"][0, 0, -1])
            hist["pe_base"].append(pe[0])
            hist["pe_top"].append(pe[-1])
    wall = time.time() - t_wall

    final = snapshot(run, r)
    fld = run.fields_full()
    flx = run.flux()  # (nbeam, nz, ny, nx) interior, transmitted flux
    resid_final = run.source_resid
    dt_final = run.dt
    run.close()

    out = {
        "label": label,
        "steps": steps,
        "wall": wall,
        "t_final": float(np.asarray(hist["t"])[-1]),
        "dx_full": dxf.copy(),
        "dt_final": dt_final,
        "resid_final": resid_final,
        "resid_max": float(np.max(np.asarray(hist["resid"]))),
        "r": r,
        "ic": ic,
        "final": final,
        "hist": {k: np.asarray(v) for k, v in hist.items()},
        "fields": fld,
        "flux": flx,
        "r_p": r_p,
        "cfg_constants": cfg["constants"],
        "nx": RUN.nx,
        "n_beam": RUN.n_beam,
        "tstop": RUN.tstop,
        "stretch": RUN.stretch,
        "stretch_factor": RUN.stretch_factor,
    }
    print(f"[{label}] {steps} steps, t = {out['t_final']:.4e} s, "
          f"wall = {wall:.1f} s, final dt = {dt_final:.3e} s, "
          f"source_resid(final/max) = {resid_final:.2e} / "
          f"{out['resid_max']:.2e}")
    return out


def write_profile_table(path: str, res: dict) -> None:
    """r, IC and final n/v/T/P for both fluids plus P_e (cgs)."""
    r = res["r"]
    ic, fn = res["ic"], res["final"]
    header = (
        "r_cm  r_over_Rp  "
        "nHI_ic  nHII_ic  vxHI_ic  vxHII_ic  THI_ic  THII_ic  Pe_ic  "
        "nHI  nHII  vxHI  vxHII  THI  THII  PHI  PHII  Pe\n"
    )
    rows = [header]
    rp = res["r_p"]
    for i in range(r.size):
        rows.append(
            f"{r[i]:.8e} {r[i]/rp:.6e} "
            f"{ic['HI']['n'][i]:.6e} {ic['HII']['n'][i]:.6e} "
            f"{ic['HI']['vx'][i]:.6e} {ic['HII']['vx'][i]:.6e} "
            f"{ic['HI']['T'][i]:.6e} {ic['HII']['T'][i]:.6e} "
            f"{ic['P_e'][i]:.6e} "
            f"{fn['HI']['n'][i]:.6e} {fn['HII']['n'][i]:.6e} "
            f"{fn['HI']['vx'][i]:.6e} {fn['HII']['vx'][i]:.6e} "
            f"{fn['HI']['T'][i]:.6e} {fn['HII']['T'][i]:.6e} "
            f"{fn['HI']['P'][i]:.6e} {fn['HII']['P'][i]:.6e} "
            f"{fn['P_e'][i]:.6e}\n"
        )
    with open(path, "w") as fh:
        fh.writelines(rows)


def number_block(on: dict) -> str:
    lines = ["# rad_h_pe_1d number block (written by run.py)"]
    ic = on["ic"]
    c = on["cfg_constants"]
    g_m = c["G"] * c["M_p"]
    tide_coef = 3.0 * c["G"] * c["M_star"] / (c["a"] ** 3)
    r0, r1 = on["r"][0], on["r"][-1]
    a_net0 = -g_m / (r0 * r0) + tide_coef * r0
    a_net1 = -g_m / (r1 * r1) + tide_coef * r1
    lines += [
        "[model]",
        "  fluids HI(q=0)/HII(q=1); electron = pressure-only, "
        "share = by-charge, shock_heat = ions-only (pinned)",
        "  photoionisation heating -> dP_e (direct write onto the P_e "
        "block); recombination reads derived n_e",
        "  HI-HII resonance collision coef = 2.67e-10*T^0.5*"
        "(1-0.083*log10(T))^2, T = 0.5*(T_HI+T_HII); net dn = 0",
        "  HII-e Coulomb heat: nu = 0.0299*(T_e+T_floor)^{-3/2}, "
        "dP_HII = -dP_e, no dvx (yardstick HII momentum never <-> ELCI)",
        f"  dvx = -G*M_p/r**2 + 3*G*M_star*r/a**3  "
        f"(M_star={c['M_star']:.4e} g, a={c['a']:.4e} cm)",
        "[ic]",
        "  n_HI(r)  = n_beg * exp(-(r-R_p)/H_ic),  n_beg = 1e13 cm^-3, "
        f"H_ic = {c['H_ic']/c['R_p']:.3f} R_p,  n_ic_min = {c['n_ic_min']:.1e}",
        "  n_HII(r) = ion_frac * n_HI,    ion_frac = 1e-3",
        "  T = 1500 K both fluids, v = 0, P = n k_B T, "
        "P_e = n_HII k_B T (quasi-neutral)",
        f"  n_HI   base/top = {ic['HI']['n'][0]:.6e} / {ic['HI']['n'][-1]:.6e} cm^-3",
        f"  P_e    base/top = {ic['P_e'][0]:.6e} / {ic['P_e'][-1]:.6e} erg/cm^3",
        f"  a_net  base/top = {a_net0:.4e} / {a_net1:.4e} cm/s^2",
        f"  x1beg = user (ghosts: n,v=0,P via user_bc tapes; P_e copy@1); "
        f"x1end = user old-tree outflow",
    ]
    for res in (on,):
        fn = res["final"]
        n1, n2 = fn["HI"]["n"], fn["HII"]["n"]
        v1, v2 = fn["HI"]["vx"], fn["HII"]["vx"]
        pe = fn["P_e"]
        lines += [
            f"[{res['label']} final]",
            f"  steps = {res['steps']}, t_final = {res['t_final']:.6e} s, "
            f"wall = {res['wall']:.1f} s",
            f"  dt_final = {res['dt_final']:.6e} s",
            f"  source_resid final = {res['resid_final']:.3e}, "
            f"max over history = {res['resid_max']:.3e}",
            f"  n_HI   base/top = {n1[0]:.6e} / {n1[-1]:.6e} cm^-3",
            f"  n_HII  base/top = {n2[0]:.6e} / {n2[-1]:.6e} cm^-3",
            f"  ion frac base/top = {n2[0]/max(n1[0],1e-300):.6e} / "
            f"{n2[-1]/max(n1[-1],1e-300):.6e}",
            f"  vx_HI  min/max = {v1.min():.6e} / {v1.max():.6e} cm/s",
            f"  vx_HII min/max = {v2.min():.6e} / {v2.max():.6e} cm/s",
            f"  T_HI   base/top = {fn['HI']['T'][0]:.4e} / {fn['HI']['T'][-1]:.4e} K",
            f"  T_HII  base/top = {fn['HII']['T'][0]:.4e} / {fn['HII']['T'][-1]:.4e} K",
            f"  T_e    base/top = {fn['T_e'][0]:.4e} / {fn['T_e'][-1]:.4e} K",
            f"  P_HII  base/top = {fn['HII']['P'][0]:.6e} / {fn['HII']['P'][-1]:.6e} erg/cm^3",
            f"  P_e    base/top = {pe[0]:.6e} / {pe[-1]:.6e} erg/cm^3",
            f"  P_e    min/max  = {pe.min():.6e} / {pe.max():.6e} erg/cm^3",
        ]
    lines.append(f"nx = {RUN.nx}, n_beam = {RUN.n_beam}, "
                 f"tstop = {RUN.tstop:.6e}")
    dxf = on["dx_full"] / on["r_p"]
    dact = dxf[2 : 2 + RUN.nx]
    lines.append(
        f"grid x1: stretch = {RUN.stretch}, factor = {RUN.stretch_factor}; "
        f"dx/R_p first/last = {dact[0]:.6e} / {dact[-1]:.6e}, "
        f"ratio = {dact[-1] / dact[0]:.4f}"
    )
    # Spherical mass-loss rate Mdot_s = 4π m_s n_s r^2 v_r (g/s); the
    # hydrogen sum is the conserved flux. Quote r > 4.5 R_p (outside the
    # Hill radius).
    m1 = c["m_HI_g"]
    m2 = c["m_HII_g"]
    lines.append("[mass flux  4 pi m n r^2 v_r]")
    r_hill = 4.5 * on["r_p"]
    for res in (on,):
        r = res["r"]
        n1, v1 = res["final"]["HI"]["n"], res["final"]["HI"]["vx"]
        n2, v2 = res["final"]["HII"]["n"], res["final"]["HII"]["vx"]
        md1, md2 = mdot_g_s(n1, v1, r, m1), mdot_g_s(n2, v2, r, m2)
        mdt = md1 + md2
        mask = r > r_hill
        mean = float(np.mean(mdt[mask]))
        scatter = float(np.std(mdt[mask]) / max(abs(mean), 1e-300))
        lines.append(
            f"  {res['label']}: Mdot_HI+Mdot_HII mid/top = "
            f"{mdt[len(r)//2]:.6e} / {mdt[-1]:.6e} g/s; "
            f"r>4.5 R_p mean = {mean:.6e} g/s, std/|mean| = {scatter:.4e}"
        )
    return "\n".join(lines) + "\n"


def make_plots(on: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return
    rr_on = on["r"] / on["r_p"]
    ic = on["ic"]

    # 1. densities + velocities (IC overlay + radiation-on final)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    ax[0].plot(rr_on, ic["HI"]["n"], color="0.6", ls=":", label="HI IC")
    ax[0].plot(rr_on, ic["HII"]["n"], color="0.6", ls="-.", label="HII IC")
    for res, c in ((on, "C0"),):
        rr = res["r"] / res["r_p"]
        ax[0].plot(rr, res["fields"]["HI"]["n"][0, 0, :], c=c,
                   label=f"HI {res['label']}")
        ax[0].plot(rr, res["fields"]["HII"]["n"][0, 0, :], c=c, ls="--",
                   label=f"HII {res['label']}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("r / R_p")
    ax[0].set_ylabel("n, cm^-3")
    ax[0].legend(fontsize=8)
    ax[0].set_title("number densities")
    for res, c in ((on, "C0"),):
        rr = res["r"] / res["r_p"]
        ax[1].plot(rr, res["fields"]["HI"]["vx"][0, 0, :], c=c,
                   label=f"HI {res['label']}")
        ax[1].plot(rr, res["fields"]["HII"]["vx"][0, 0, :], c=c, ls="--",
                   label=f"HII {res['label']}")
    ax[1].axhline(0.0, color="0.5", lw=0.8)
    ax[1].set_xlabel("r / R_p")
    ax[1].set_ylabel("v_r, cm/s")
    ax[1].legend(fontsize=8)
    ax[1].set_title("radial velocities")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "profiles.png"), dpi=140)
    plt.close(fig)

    # 2. pressures + P_e (the B2 block) + ion fraction
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for res, c in ((on, "C0"),):
        rr = res["r"] / res["r_p"]
        f = res["fields"]
        ax[0].plot(rr, f["HII"]["P"][0, 0, :], c=c,
                   label=f"P_HII {res['label']}")
        ax[0].plot(rr, f["HI"]["P"][0, 0, :], c=c, ls="--",
                   label=f"P_HI {res['label']}")
        ax[0].plot(rr, f["P_e"][0, 0, :], c=c, ls=":", lw=1.8,
                   label=f"P_e {res['label']}")
    ax[0].plot(rr_on, ic["P_e"], color="0.6", ls="-.", label="P_e IC")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("r / R_p")
    ax[0].set_ylabel("P, erg/cm^3")
    ax[0].legend(fontsize=7)
    ax[0].set_title("pressures incl. electron block")
    for res, c in ((on, "C0"),):
        rr = res["r"] / res["r_p"]
        f = res["fields"]
        n1 = f["HI"]["n"][0, 0, :]
        n2 = f["HII"]["n"][0, 0, :]
        ax[1].plot(rr, n2 / np.maximum(n1, 1e-300), c=c,
                   label=res["label"])
    ax[1].set_yscale("log")
    ax[1].set_xlabel("r / R_p")
    ax[1].set_ylabel("n_HII / n_HI")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pressure_ionfrac.png"), dpi=140)
    plt.close(fig)

    # 3. transmitted flux (on run)
    flx = on["flux"][:, 0, 0, :]
    finf_tot = float(np.sum(flx[:, -1]))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(rr_on, np.sum(flx, axis=0) / finf_tot, c="C0",
            label="total transmitted / F_inf")
    for b, c in ((0, "C1"), (flx.shape[0] // 2, "C2"),
                 (flx.shape[0] - 1, "C3")):
        fb_inf = flx[b, -1]
        if fb_inf > 0:
            ax.plot(rr_on, flx[b] / fb_inf, c=c, ls="--",
                    label=f"beam {b}")
    ax.set_yscale("log")
    ax.set_xlabel("r / R_p")
    ax.set_ylabel("F / F_inf")
    ax.legend(fontsize=8)
    ax.set_title("inward scan attenuation (radiation on)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flux_attenuation.png"), dpi=140)
    plt.close(fig)

    # 4. time history: dt, residual, base ion fraction, P_e base/top
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
    for res, c in ((on, "C0"),):
        h = res["hist"]
        ax[0].plot(h["t"], h["dt"], c=c, label=res["label"])
        ax[1].plot(h["t"], h["resid"], c=c, label=res["label"])
        ax[2].plot(h["t"], h["ion_frac_base"], c=c, label=res["label"])
        ax[3].plot(h["t"], h["pe_base"], c=c, label=f"base {res['label']}")
        ax[3].plot(h["t"], h["pe_top"], c=c, ls="--",
                   label=f"top {res['label']}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("t, s")
    ax[0].set_ylabel("dt, s")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("t, s")
    ax[1].set_ylabel("source_resid")
    ax[2].set_xlabel("t, s")
    ax[2].set_ylabel("n_HII/n_HI at R_p")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("t, s")
    ax[3].set_ylabel("P_e, erg/cm^3")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "history.png"), dpi=140)
    plt.close(fig)

    # 5. spherical mass-loss rate
    m1 = on["cfg_constants"]["m_HI_g"]
    m2 = on["cfg_constants"]["m_HII_g"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r_hill = 4.5
    for res, c in ((on, "C0"),):
        r = res["r"]
        rr = r / res["r_p"]
        n1, v1 = res["final"]["HI"]["n"], res["final"]["HI"]["vx"]
        n2, v2 = res["final"]["HII"]["n"], res["final"]["HII"]["vx"]
        md1, md2 = mdot_g_s(n1, v1, r, m1), mdot_g_s(n2, v2, r, m2)
        mdt = md1 + md2
        ax[0].plot(rr, md1, c=c, ls="--", label=f"HI {res['label']}")
        ax[0].plot(rr, md2, c=c, ls=":", label=f"HII {res['label']}")
        ax[0].plot(rr, mdt, c=c, lw=1.8, label=f"HI+HII {res['label']}")
        mask = rr >= r_hill
        ax[1].plot(rr[mask], mdt[mask], c=c, lw=1.8,
                   label=f"HI+HII {res['label']}")
    ax[0].axhline(0.0, color="0.5", lw=0.8)
    ax[0].axvline(r_hill, color="0.5", ls=":", lw=0.8)
    ax[0].set_yscale("symlog", linthresh=1e8)
    ax[1].axhline(0.0, color="0.5", lw=0.8)
    ax[0].set_xlabel("r / R_p")
    ax[1].set_xlabel("r / R_p")
    ax[0].set_ylabel(r"$\dot{M}$, g/s")
    ax[1].set_ylabel(r"$\dot{M}$, g/s")
    ax[0].set_title(r"$\dot{M}_s = 4\pi m_s n_s r^2 v_r$  (symlog)")
    ax[1].set_title(r"outflow $r > 4.5 R_p$")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mass_flux.png"), dpi=140)
    plt.close(fig)
    print("plots written to", OUT)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    on = one_run(True)
    block = number_block(on)
    with open(os.path.join(OUT, "numbers.txt"), "w") as fh:
        fh.write(block)
    write_profile_table(os.path.join(OUT, "profiles_on.txt"), on)
    print(block)
    if RUN.plots:
        make_plots(on)


if __name__ == "__main__":
    main()
