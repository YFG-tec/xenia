"""Radiation-driven four-fluid (HI/HEI/HII/HEII) escape with a
pressure-only electron block on a 1-D spherical grid (B3 example, full
Xing+2023 source table).

Builds the setup through the hhe_config.hhe_1d() factory (the
example-PRIVATE config module in this directory -- never part of
xmf.config, never exported by `import xmf`; full cgs: every unit factor
is 1, so code values ARE cgs values), packs the four-fluid exponential
n_s(r) = n_beg frac_s exp(-(r-R_p)/H_ic) initial state plus the
electron pressure P_e = (n_HII + n_HEII) kB T_beg through the
xmf_state_pack / xmf_state_write_pe helper seams, advances the
radiation-on run to t_stop, and writes the figures plus a plain-text
number block into output/. The no-heating (radiation-off) comparison
is no longer a default evaluation.

Usage:
    python examples/rad_hhe_pe_1d/run.py

Knobs (nx, tstop, stretch, plots, log_every) live in the pinned RUN
object at the top of hhe_config.py; n_beam is always the full 53-beam
table. There is no CLI.

The SED table and the photo-ionisation cross sections (sigma_H AND
sigma_He) are REFERENCED from ../rad_hd_1d/data/ (the SSH reference
tables shipped with the A5 example); this directory deliberately holds
no copy of its own.
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

import xmf
from hhe_config import RUN, hhe_1d
from xmf.config import AMU, KB

HERE = os.path.dirname(os.path.abspath(__file__))
# Reference the A5 data directory; never copy the tables.
DATA = os.path.join(os.path.dirname(HERE), "rad_hd_1d", "data")
OUT = os.path.join(HERE, "output")
DPTR = ctypes.POINTER(ctypes.c_double)

FLUIDS = ("HI", "HEI", "HII", "HEII")


def lib():
    return xmf.api._lib()


def mdot_g_s(n, v, r, mass_g):
    """Spherical mass-loss rate 4 pi m n r^2 v_r, g/s."""
    return (4.0 * np.pi) * mass_g * n * r * r * v


def snapshot(run, r: np.ndarray) -> dict:
    """Interior radial profiles (cgs). T = P / (n k_B); the electron
    pressure block rides along as its own profile. n_e = n_HII + n_HEII
    (charge-weighted read, both ions q = 1)."""
    fld = run.fields_full()
    out = {"r": r.copy(), "P_e": fld["P_e"][0, 0, :].copy()}
    for name in FLUIDS:
        n = fld[name]["n"][0, 0, :].copy()
        p = fld[name]["P"][0, 0, :].copy()
        out[name] = {
            "n": n,
            "vx": fld[name]["vx"][0, 0, :].copy(),
            "P": p,
            "T": p / (np.maximum(n, 1e-300) * KB),
        }
    ne = out["HII"]["n"] + out["HEII"]["n"]
    out["T_e"] = out["P_e"] / (np.maximum(ne, 1e-300) * KB)
    return out


def pack_ic(run, cfg) -> None:
    """Four-fluid n_s(r) = n_beg*frac_s*exp(-(r-R_p)/H_ic), v = 0,
    P_s = n_s kB T_beg (HEI carries the locked HE_SH_COEF = 1.0), and
    the electron pressure P_e(r) = (n_HII + n_HEII) kB T_beg.

    Packed by radius over every block (ghosts included); P_e through the
    xmf_state_write_pe seam. The x1beg user_bc tapes assign the fluid
    base to ghosts only; P_e ghosts fall back to the copy @1 fill."""
    consts = cfg["constants"]
    r_p = consts["R_p"]
    h_ic = consts["H_ic"]
    n_min = consts["n_ic_min"]
    t_beg = consts["T_beg"]
    he_sh = consts["HE_SH_COEF"]
    ng = cfg["grid"]["nghost"]
    nxtot = cfg["grid"]["n"][0] + 2 * ng
    ntot = (1 + 2 * ng) * (1 + 2 * ng) * nxtot
    xc = np.empty(nxtot, dtype=np.float64)
    got = lib().xmf_mesh_copy_x(run._h, 0, xc.ctypes.data_as(DPTR))
    assert got == nxtot
    env = np.exp(-(xc - r_p) / h_ic)
    # (fluid index, fraction, mass amu, IC temperature factor)
    species = (
        (0, consts["frac_HI"], consts["m_HI_amu"], 1.0),
        (1, consts["frac_HEI"], consts["m_HEI_amu"], he_sh),
        (2, consts["frac_HII"], consts["m_HII_amu"], 1.0),
        (3, consts["frac_HEII"], consts["m_HEII_amu"], 1.0),
    )
    n_ion = np.zeros_like(xc)
    for f, frac, mass_amu, t_fac in species:
        n = consts["n_beg"] * frac * env
        n = np.maximum(n, n_min * frac)
        if f in (2, 3):
            n_ion += n
        blocks = (
            (0, n * mass_amu * AMU),
            (1, np.zeros_like(n)),
            (2, np.zeros_like(n)),
            (3, np.zeros_like(n)),
            (4, n * KB * t_beg * t_fac),
        )
        for var, arr in blocks:
            full = np.tile(arr, (1 + 2 * ng) * (1 + 2 * ng))
            got = lib().xmf_state_pack(run._h, f, var,
                                       full.ctypes.data_as(DPTR))
            assert got == ntot
    # Electron pressure block: P_e = (n_HII + n_HEII) kB T_beg.
    pe = n_ion * KB * t_beg
    full = np.tile(pe, (1 + 2 * ng) * (1 + 2 * ng))
    got = lib().xmf_state_write_pe(run._h, full.ctypes.data_as(DPTR))
    assert got == ntot


def one_run(radiation: bool) -> dict:
    label = "on" if radiation else "off"
    cfg = hhe_1d(
        DATA,
        radiation=radiation,
        nx=RUN.nx,
        tstop=RUN.tstop,
        stretch=RUN.stretch,
        stretch_factor=RUN.stretch_factor,
    )
    run = xmf.setup(cfg)
    assert run.num_states == 5 * len(cfg["fluids"]) + 1, (
        "B3 open-electron contract: N = 5*nfluid + 1 (P_e last) = 21"
    )
    pack_ic(run, cfg)
    r_p = cfg["constants"]["R_p"]
    xc = np.empty(RUN.nx + 4, dtype=np.float64)
    lib().xmf_mesh_copy_x(run._h, 0, xc.ctypes.data_as(DPTR))
    r = xc[2 : 2 + RUN.nx]
    dxf = np.empty(RUN.nx + 4, dtype=np.float64)
    lib().xmf_mesh_copy_dx(run._h, 0, dxf.ctypes.data_as(DPTR))
    ic = snapshot(run, r)

    hist = {"t": [], "dt": [], "resid": [], "ion_frac_h_base": [],
            "ion_frac_he_base": [], "vx_hii_top": [], "vx_heii_top": [],
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
            n_hi = fld["HI"]["n"][0, 0, :]
            n_hei = fld["HEI"]["n"][0, 0, :]
            n_hii = fld["HII"]["n"][0, 0, :]
            n_heii = fld["HEII"]["n"][0, 0, :]
            pe = fld["P_e"][0, 0, :]
            hist["t"].append(run.t)
            hist["dt"].append(run.dt)
            hist["resid"].append(run.source_resid)
            hist["ion_frac_h_base"].append(n_hii[0] / max(n_hi[0], 1e-300))
            hist["ion_frac_he_base"].append(n_heii[0] / max(n_hei[0], 1e-300))
            hist["vx_hii_top"].append(fld["HII"]["vx"][0, 0, -1])
            hist["vx_heii_top"].append(fld["HEII"]["vx"][0, 0, -1])
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
    """r, IC and final n/vx/T/P for all four fluids plus P_e (cgs)."""
    r = res["r"]
    ic, fn = res["ic"], res["final"]
    header = "r_cm  r_over_Rp"
    for s in FLUIDS:
        header += f"  n{s}_ic  vx{s}_ic  T{s}_ic"
    header += "  Pe_ic"
    for s in FLUIDS:
        header += f"  n{s}  vx{s}  T{s}  P{s}"
    header += "  Pe\n"
    rows = [header]
    rp = res["r_p"]
    for i in range(r.size):
        line = f"{r[i]:.8e} {r[i]/rp:.6e}"
        for s in FLUIDS:
            line += (f" {ic[s]['n'][i]:.6e} {ic[s]['vx'][i]:.6e} "
                     f"{ic[s]['T'][i]:.6e}")
        line += f" {ic['P_e'][i]:.6e}"
        for s in FLUIDS:
            line += (f" {fn[s]['n'][i]:.6e} {fn[s]['vx'][i]:.6e} "
                     f"{fn[s]['T'][i]:.6e} {fn[s]['P'][i]:.6e}")
        line += f" {fn['P_e'][i]:.6e}\n"
        rows.append(line)
    with open(path, "w") as fh:
        fh.writelines(rows)


def number_block(on: dict) -> str:
    lines = ["# rad_hhe_pe_1d number block (written by run.py)"]
    ic = on["ic"]
    c = on["cfg_constants"]
    g_m = c["G"] * c["M_p"]
    tide_coef = 3.0 * c["G"] * c["M_star"] / (c["a"] ** 3)
    r0, r1 = on["r"][0], on["r"][-1]
    a_net0 = -g_m / (r0 * r0) + tide_coef * r0
    a_net1 = -g_m / (r1 * r1) + tide_coef * r1
    lines += [
        "[model]",
        "  fluids HI/HEI(q=0) + HII/HEII(q=1); electron = pressure-only, "
        "share = by-charge, shock_heat = ions-only (pinned)",
        "  n_e = n_HII + n_HEII; v_e = (n_HII v_HII + n_HEII v_HEII)/n_e "
        "(charge-weighted, spelled in tapes; no vx_e read name)",
        "  full Xing table: photoionisation (H+He) + recombination + "
        "e-impact ionisation + CX1/CX2 + Ly-alpha cooling, one summed "
        "chemistry pair per element before the CHE pressure / velocity "
        "inheritance",
        "  collisions: HI-HII / HEI-HEII resonance, HI-HEI / HII-HEII "
        "unlike-pair, ion-e Coulomb 0.0299 T_e^{-3/2} with the frictional "
        "heat (2/3) m_e n_ion n_e (v_ion - v_e)^2 written back; dP_e "
        "Coulomb bands are the exact negation of the ion side; no dvx_e",
        f"  HE_RATIO = {c['HE_RATIO']}, HE_SH_COEF = {c['HE_SH_COEF']}, "
        f"n_beg = {c['n_beg']:.1e} cm^-3 (H+He total)",
        f"  dvx = -G*M_p/r**2 + 3*G*M_star*r/a**3  "
        f"(M_star={c['M_star']:.4e} g, a={c['a']:.4e} cm)",
        "[ic]",
        "  n_s(r) = n_beg frac_s exp(-(r-R_p)/H_ic), floored; "
        f"H_ic = {c['H_ic']/c['R_p']:.3f} R_p, n_ic_min = {c['n_ic_min']:.1e}",
        f"  frac_HI = {c['frac_HI']:.6f}, frac_HEI = {c['frac_HEI']:.6f}, "
        f"frac_HII = {c['frac_HII']:.3e}, frac_HEII = {c['frac_HEII']:.3e}",
        "  T = 1500 K all fluids, v = 0, P = n k_B T, "
        "P_e = (n_HII + n_HEII) k_B T (quasi-neutral)",
        f"  n_HI   base/top = {ic['HI']['n'][0]:.6e} / {ic['HI']['n'][-1]:.6e} cm^-3",
        f"  n_HEI  base/top = {ic['HEI']['n'][0]:.6e} / {ic['HEI']['n'][-1]:.6e} cm^-3",
        f"  P_e    base/top = {ic['P_e'][0]:.6e} / {ic['P_e'][-1]:.6e} erg/cm^3",
        f"  a_net  base/top = {a_net0:.4e} / {a_net1:.4e} cm/s^2",
        f"  x1beg = user (ghosts: n,v=0,P via user_bc tapes; P_e copy@1); "
        f"x1end = user old-tree outflow",
    ]
    for res in (on,):
        fn = res["final"]
        pe = fn["P_e"]
        lines += [
            f"[{res['label']} final]",
            f"  steps = {res['steps']}, t_final = {res['t_final']:.6e} s, "
            f"wall = {res['wall']:.1f} s",
            f"  dt_final = {res['dt_final']:.6e} s",
            f"  source_resid final = {res['resid_final']:.3e}, "
            f"max over history = {res['resid_max']:.3e}",
        ]
        for s in FLUIDS:
            n, v, t = fn[s]["n"], fn[s]["vx"], fn[s]["T"]
            lines.append(
                f"  n_{s:4s} base/top = {n[0]:.6e} / {n[-1]:.6e} cm^-3; "
                f"vx min/max = {v.min():.4e} / {v.max():.4e} cm/s; "
                f"T base/top = {t[0]:.4e} / {t[-1]:.4e} K"
            )
        ne = fn["HII"]["n"] + fn["HEII"]["n"]
        lines += [
            f"  ion frac H base/top = "
            f"{fn['HII']['n'][0]/max(fn['HI']['n'][0],1e-300):.6e} / "
            f"{fn['HII']['n'][-1]/max(fn['HI']['n'][-1],1e-300):.6e}",
            f"  ion frac He base/top = "
            f"{fn['HEII']['n'][0]/max(fn['HEI']['n'][0],1e-300):.6e} / "
            f"{fn['HEII']['n'][-1]/max(fn['HEI']['n'][-1],1e-300):.6e}",
            f"  n_e    base/top = {ne[0]:.6e} / {ne[-1]:.6e} cm^-3",
            f"  T_e    base/top = {fn['T_e'][0]:.4e} / {fn['T_e'][-1]:.4e} K",
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
    # Spherical mass-loss rate Mdot = sum_s 4 pi m_s n_s r^2 v_r (g/s);
    # quote r > 4.5 R_p (outside the Hill radius).
    masses_g = {
        "HI": c["m_HI_amu"] * AMU, "HEI": c["m_HEI_amu"] * AMU,
        "HII": c["m_HII_amu"] * AMU, "HEII": c["m_HEII_amu"] * AMU,
    }
    lines.append("[mass flux  sum_s 4 pi m_s n_s r^2 v_r]")
    r_hill = 4.5 * on["r_p"]
    for res in (on,):
        r = res["r"]
        mdt = np.zeros_like(r)
        for s in FLUIDS:
            mdt += mdot_g_s(res["final"][s]["n"], res["final"][s]["vx"],
                            r, masses_g[s])
        mask = r > r_hill
        mean = float(np.mean(mdt[mask]))
        scatter = float(np.std(mdt[mask]) / max(abs(mean), 1e-300))
        lines.append(
            f"  {res['label']}: Mdot mid/top = "
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
    styles = {"HI": "-", "HEI": "--", "HII": "-.", "HEII": ":"}
    for s in FLUIDS:
        ax[0].plot(rr_on, ic[s]["n"], color="0.6", ls=styles[s],
                   label=f"{s} IC")
    for res, c in ((on, "C0"),):
        rr = res["r"] / res["r_p"]
        for s in FLUIDS:
            ax[0].plot(rr, res["fields"][s]["n"][0, 0, :], c=c,
                       ls=styles[s], label=f"{s} {res['label']}")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("r / R_p")
    ax[0].set_ylabel("n, cm^-3")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("number densities")
    for res, c in ((on, "C0"),):
        rr = res["r"] / res["r_p"]
        for s in ("HII", "HEII"):
            ax[1].plot(rr, res["fields"][s]["vx"][0, 0, :], c=c,
                       ls=styles[s], label=f"{s} {res['label']}")
    ax[1].axhline(0.0, color="0.5", lw=0.8)
    ax[1].set_xlabel("r / R_p")
    ax[1].set_ylabel("v_r, cm/s")
    ax[1].legend(fontsize=8)
    ax[1].set_title("ion radial velocities")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "profiles.png"), dpi=140)
    plt.close(fig)

    # 2. pressures + P_e + ion fractions
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for res, c in ((on, "C0"),):
        rr = res["r"] / res["r_p"]
        f = res["fields"]
        ax[0].plot(rr, f["HII"]["P"][0, 0, :], c=c,
                   label=f"P_HII {res['label']}")
        ax[0].plot(rr, f["HEII"]["P"][0, 0, :], c=c, ls="--",
                   label=f"P_HEII {res['label']}")
        ax[0].plot(rr, f["P_e"][0, 0, :], c=c, ls=":", lw=1.8,
                   label=f"P_e {res['label']}")
    ax[0].plot(rr_on, ic["P_e"], color="0.6", ls="-.", label="P_e IC")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("r / R_p")
    ax[0].set_ylabel("P, erg/cm^3")
    ax[0].legend(fontsize=7)
    ax[0].set_title("ion + electron pressures")
    for res, c in ((on, "C0"),):
        rr = res["r"] / res["r_p"]
        f = res["fields"]
        for s, neut, ls in (("HII", "HI", "-"), ("HEII", "HEI", "--")):
            n_ion = f[s]["n"][0, 0, :]
            n_neu = f[neut]["n"][0, 0, :]
            ax[1].plot(rr, n_ion / np.maximum(n_neu, 1e-300), c=c, ls=ls,
                       label=f"{s}/{neut} {res['label']}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("r / R_p")
    ax[1].set_ylabel("ion fraction")
    ax[1].legend(fontsize=7)
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
    ax.set_title("inward scan attenuation, H+He absorbers (radiation on)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "flux_attenuation.png"), dpi=140)
    plt.close(fig)

    # 4. time history: dt, residual, base ion fractions, P_e base/top
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
    for res, c in ((on, "C0"),):
        h = res["hist"]
        ax[0].plot(h["t"], h["dt"], c=c, label=res["label"])
        ax[1].plot(h["t"], h["resid"], c=c, label=res["label"])
        ax[2].plot(h["t"], h["ion_frac_h_base"], c=c, label="H " + res["label"])
        ax[2].plot(h["t"], h["ion_frac_he_base"], c=c, ls="--",
                   label="He " + res["label"])
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
    ax[2].set_ylabel("ion fraction at R_p")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("t, s")
    ax[3].set_ylabel("P_e, erg/cm^3")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "history.png"), dpi=140)
    plt.close(fig)

    # 5. spherical mass-loss rate (all four fluids)
    c = on["cfg_constants"]
    masses_g = {
        "HI": c["m_HI_amu"] * AMU, "HEI": c["m_HEI_amu"] * AMU,
        "HII": c["m_HII_amu"] * AMU, "HEII": c["m_HEII_amu"] * AMU,
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r_hill = 4.5
    for res, c_ in ((on, "C0"),):
        r = res["r"]
        rr = r / res["r_p"]
        mdt = np.zeros_like(r)
        for s in FLUIDS:
            md = mdot_g_s(res["final"][s]["n"], res["final"][s]["vx"],
                          r, masses_g[s])
            mdt += md
            ax[0].plot(rr, md, c=c_, ls=styles[s], label=f"{s} {res['label']}")
        ax[0].plot(rr, mdt, c=c_, lw=1.8, label=f"total {res['label']}")
        mask = rr >= r_hill
        ax[1].plot(rr[mask], mdt[mask], c=c_, lw=1.8,
                   label=f"total {res['label']}")
    ax[0].axhline(0.0, color="0.5", lw=0.8)
    ax[0].axvline(r_hill, color="0.5", ls=":", lw=0.8)
    ax[0].set_yscale("symlog", linthresh=1e8)
    ax[1].axhline(0.0, color="0.5", lw=0.8)
    ax[0].set_xlabel("r / R_p")
    ax[1].set_xlabel("r / R_p")
    ax[0].set_ylabel(r"$\dot{M}$, g/s")
    ax[1].set_ylabel(r"$\dot{M}$, g/s")
    ax[0].set_title(r"$\dot{M} = \sum_s 4\pi m_s n_s r^2 v_r$  (symlog)")
    ax[1].set_title(r"outflow $r > 4.5 R_p$")
    ax[0].legend(fontsize=6, ncol=2)
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
