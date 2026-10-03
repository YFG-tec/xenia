"""Example-private config for the rad_hhe_pe_1d setup (B3 four-fluid
Xing+2023 factory).

The HI/HEI/HII/HEII + pressure-only-electron factory lives here, next
to the example that uses it. PRIVATE by contract: it is never imported
by xmf.config and never exported by `import xmf` -- only run.py (this
directory) and the factory contract test load it, by absolute path.
KB/AMU/SCHEMA are imported from xmf.config, never redefined here.

Differences from the B2 rad_h_pe_1d factory (same SED table, same XUV
rescale, same gravity+tide, same nx/stretch/tstop knobs):
  * SECOND absorber: radiation.absorbers = ["HI", "HEI"] with one sigma
    row per absorber (the B3 appended multi-absorber port; sigma_He read
    from the same photoIon_sigma.dat as sigma_H). The scan's per-beam
    optical depth is tau_b = (sigma_H,b n_HI + sigma_He,b n_HEI) ds.
  * Four fluids HI/HEI/HII/HEII after the yardstick
    SSH_220804/mmd_model_initialize.c masses (HEII loses one electron
    mass: 4.002053). n_e = n_HII + n_HEII (both q = 1) and the electron
    velocity is the charge-weighted mean spelled out in the tapes -- no
    vx_e read name, no fifth electron momentum fluid.
  * FULL yardstick source table (SSH_220804/mmd_model_src.c), one
    chemistry sum per element pair before the CHE pressure / velocity
    inheritance (never a CHE per channel):
      ion_H  = photoion_HI + CX1 + e-impact_H,  rec_H  = recomb_HII + CX2
      ion_He = photoion_HEI + CX2 + e-impact_He, rec_He = recomb_HEII + CX1
    CX1: HI + HEII -> HII + HEI, k = 1.25e-15 (300/T)^{-0.25};
    CX2: HII + HEI -> HI + HEII, k = 1.75e-11 (300/T)^{0.75}
    exp(-128000/T); T is the mass-weighted TMP_REDUCE of the pair
    (mmd.h:73). The old-tree startup gate MIN(1, exp(g_time-20)) is
    LOCKED to 1 (tstop = 2e5 s >> 20 s; the IR has no time read).
  * Elastic / resonance / ion Coulomb collisions (:152-208): HI-HII and
    HEI-HEII resonance, HI-HEI and HII-HEII unlike-pair, ion-electron
    Coulomb 0.0299 T_e^{-3/2}. Momentum exchange for the four fluid
    pairs only (amu divisor, antisymmetric velocity differences);
    HII-e / HEII-e carry NO dvx_* band -- their pressure band is the
    full COL_PRS_TRANS with the charge-weighted v_e, and the frictional
    heat (2/3) m_e n_ion n_e (v_ion - v_e)^2 is written back into the
    pressure bands (B2 omitted it: single ion makes v_e = v_ion).
    dP_e bands are the exact negation of the ion-side Coulomb bands,
    same divisor. G_DT is never carried over (the kernel multiplies its
    own dt).
  * IC: n_beg = 1e13 cm^-3 is the H+He TOTAL (old NDT_IBEG):
      n_HI = n_beg (1-Y),  n_HEI = n_beg Y,
      n_HII = 1e-3 n_HI,   n_HEII = 1e-3 n_HEI,
    Y = HE_RATIO = 0.0776441 (p_HD209458b.ini), HE_SH_COEF = 1.0;
    P_e,ic = (n_HII + n_HEII) kB T_beg, T_beg = 1500 K; envelope = B2
    thin exponential + floor.
"""
from __future__ import annotations

import math
from typing import Any

from xmf.config import AMU, KB, SCHEMA

# ---- Pinned run knobs (edit here; run.py has no CLI) ---------------------


class RUN:
    nx = 128
    n_beam = 53
    tstop = 2.0e5
    stretch = "log"  # "none" | "log"
    stretch_factor = 1.025  # log: e.g. 1.025
    # Stability cap for the four-fluid chemistry set: the source step is
    # ONE linearised backward-Euler Newton (the yardstick get_du scheme,
    # mmd_integrate.c), and the CX / momentum-loading / T-activated
    # channels grow on a ~10 s scale during the startup free-fall
    # transient. Empirically dt ~ 18 s crashes the step (xmf_step -1) and
    # dt ~ 22 s (the bare CFL dt at nx = 128) diverges; 10 s runs through
    # with bounded fields. The CFL formula itself stays cfl = 0.4.
    dt_max = 10.0
    plots = True
    log_every = 100  # 0 = silent
    # Evaluation is radiation ON only. The factory still accepts
    # radiation=False for contract tests; run.py does not launch it.


# ---- Reference numbers of the HD 209458b setup (all cgs) ------------------
H_PLANET = 6.62606876e-27  # Planck constant, erg s (SSH CONST_h)
C_LIGHT = 2.99792458e10    # speed of light, cm/s (SSH CONST_c)
R_JUP = 7.1492e9           # Jupiter radius, cm
M_JUP = 1.8982e30          # Jupiter mass, g
M_SUN = 2.0e33             # solar mass, g (SSH CONST_Msun)
AU = 1.49597892e13         # astronomical unit, cm (SSH CONST_au)
EV_ERG = 1.602176463158e-12  # electron volt in erg (SSH CONST_eV)
AMU_H = 1.00794            # hydrogen mass, amu
AMU_HE = 4.002602          # neutral helium mass, amu
AMU_HEII = 4.002053        # He+ mass, amu (one electron less)
M_E_AMU = 5.4858e-4        # electron mass, amu (mmd_model_initialize.c:29)


def _read_sed(path) -> tuple[list[float], list[float]]:
    """SSH SED file: header lines, then 'DATA:', a column header line and
    rows of (wavelength [A], flux)."""
    wavelengths: list[float] = []
    fluxes: list[float] = []
    data = False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not data:
                if stripped.startswith("DATA:"):
                    data = True
                continue
            if not stripped or stripped.startswith("wavelength"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            wavelengths.append(float(parts[0]))
            fluxes.append(float(parts[1]))
    if not wavelengths:
        raise ValueError(f"no DATA rows found in SED file {path}")
    return wavelengths, fluxes


def _read_sigma(path, key: str = "sigma_H", count: int = 53) -> list[float]:
    """SSH photo-ionisation cross-section file: a 'key' line followed by
    indented continuation lines of comma-separated values."""
    values: list[float] = []
    collecting = False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if collecting:
                if stripped.startswith("sigma_") or not stripped:
                    break
                for tok in stripped.split(","):
                    tok = tok.strip()
                    if tok:
                        values.append(float(tok))
            elif stripped.startswith(key):
                rest = stripped[len(key):]
                for tok in rest.split(","):
                    tok = tok.strip()
                    if tok:
                        values.append(float(tok))
                collecting = True
    if len(values) < count:
        raise ValueError(f"{key}: found {len(values)} values in {path}, want {count}")
    return values[:count]


def _x1end_grid_rate(
    x1min: float, x1max: float, nx: int, stretch: str, stretch_factor: float | None
) -> float:
    """Old-tree outflow_BC (SSH mmd_model_IC_BC.c:383):
    grid_rate = x[IEND] / x[IEND-1], the last two *active* cell centres.
    Same geometric-face recipe as xmf.config._log_faces (not imported:
    the factory stays a private file)."""
    nx = int(nx)
    if stretch in ("log", "x1") and stretch_factor is not None and float(stretch_factor) != 1.0:
        f = float(stretch_factor)
        total = x1max - x1min
        dx0 = total * (f - 1.0) / (f**nx - 1.0)
        faces = [x1min + dx0 * (f**k - 1.0) / (f - 1.0) for k in range(nx + 1)]
        faces[0] = x1min
        faces[-1] = x1max
    else:
        dx = (x1max - x1min) / float(nx)
        faces = [x1min + dx * k for k in range(nx + 1)]
    xc_end = 0.5 * (faces[-1] + faces[-2])
    xc_prev = 0.5 * (faces[-2] + faces[-3])
    return xc_end / xc_prev


def _x1end_outflow_tapes(fluids: tuple[str, ...]) -> dict[str, str]:
    """Yardstick IEND userdef outflow, written as assignment tapes (all
    four fluids; B2 supplement convention).

    For each fluid (SSH :388-390):
      n  = n@1**(1+g) / n@2**g
      P  = P@1**(1+g) / P@2**g
      vx = MAX(0, (1+g)*vx@1 - g*vx@2)
    A3 froze max/min opcodes, so MAX(0, v) is the value-only identity
    0.5*(v + (v*v)**0.5). vy/vz stay unwritten -> copy @1. P_e ghosts
    stay the existing copy@1 fill.
    """
    g = "x1end_grid_rate"
    out: dict[str, str] = {}
    for name in fluids:
        vext = f"((1.0 + {g}) * vx_{name}@1 - {g} * vx_{name}@2)"
        out[f"n_{name}"] = f"n_{name}@1**(1.0 + {g}) / n_{name}@2**{g}"
        out[f"P_{name}"] = f"P_{name}@1**(1.0 + {g}) / P_{name}@2**{g}"
        out[f"vx_{name}"] = f"0.5 * (({vext}) + (({vext}) * ({vext}))**0.5)"
    return out


def hhe_1d(
    data_dir,
    *,
    radiation: bool = True,
    nx: int | None = None,
    tstop: float | None = None,
    stretch: str | None = None,
    stretch_factor: float | None = None,
    dt_max: float | None = None,
) -> dict[str, Any]:
    """Radiation-driven four-fluid (HI/HEI/HII/HEII) escape with a
    pressure-only electron block on a 1-D spherical grid
    r in [R_p, 10 R_p], full cgs (every unit is 1, so code values ARE
    cgs values). n_beam is ALWAYS the full 53-beam table (two absorbers
    share the beam grid; there is no single-beam fallback).

    data_dir must hold the SSH SED table (hd209458b-1163.dat) and the
    photo-ionisation cross sections (photoIon_sigma.dat, keys sigma_H
    AND sigma_He) -- the example REFERENCES ../rad_hd_1d/data/ and never
    copies it. SED rescale is exactly the A5 xuv_initialize form.

    radiation=False is a factory path for contract tests, not a default
    evaluation: the beam table stays loaded with zero boundary fluxes
    (F_b = 0, vacuum limit) while every source tape stays on. The outer
    face falls back to default copy-outflow (B2 supplement split).

    The initial / floor state is packed by the caller through
    xmf_state_pack + xmf_state_write_pe; ic.kind is 'none'.
    """
    nx = RUN.nx if nx is None else int(nx)
    tstop = RUN.tstop if tstop is None else float(tstop)
    stretch = RUN.stretch if stretch is None else str(stretch)
    if stretch_factor is None:
        stretch_factor = RUN.stretch_factor
    dt_max = RUN.dt_max if dt_max is None else float(dt_max)

    from pathlib import Path

    data = Path(data_dir)
    lam, sed = _read_sed(data / "hd209458b-1163.dat")
    sigma_h_all = _read_sigma(data / "photoIon_sigma.dat", "sigma_H", len(lam))
    sigma_he_all = _read_sigma(data / "photoIon_sigma.dat", "sigma_He", len(lam))

    r_p = 1.36 * R_JUP
    m_p = 0.69 * M_JUP
    m_star = 1.119 * M_SUN
    a_orb = 0.04707 * AU
    x1min, x1max = r_p, 10.0 * r_p
    x1end_grid_rate = _x1end_grid_rate(x1min, x1max, nx, stretch, stretch_factor)
    gamma = 5.0 / 3.0
    n_beg = 1.0e13      # H+He TOTAL at R_p, cm^-3 (old NDT_IBEG)
    he_ratio = 0.0776441  # locked to p_HD209458b.ini
    he_sh_coef = 1.0      # locked to p_HD209458b.ini (IC T multiplier of He)
    ion_frac = 1.0e-3     # n_HII/n_HI = n_HEII/n_HEI at the base
    t_beg = 1500.0        # K, all fluids and the IC electron pressure
    eta = 0.20            # heating efficiency
    xuv_factor = 0.25
    r_orbit = 0.04707     # AU
    xuv_tot_flux = 4.716
    si_504 = 0.89

    hnu_all = [H_PLANET * C_LIGHT / w * 1.0e8 for w in lam]

    # XUV_RESIZE: total flux -> xuv_tot_flux, sub-504 A share -> si_504.
    flux_tot = sum(sed)
    flux_504 = sum(s for w, s in zip(lam, sed) if w < 504.0)
    share_504 = flux_504 / flux_tot
    resized = []
    for w, s in zip(lam, sed):
        scale = xuv_tot_flux / flux_tot
        if w < 504.0:
            scale *= si_504 / share_504
        else:
            scale *= (1.0 - si_504) / (1.0 - share_504)
        resized.append(s * scale)
    planet_scale = xuv_factor / (r_orbit * r_orbit)
    finf_all = [s * planet_scale for s in resized]

    if RUN.n_beam != len(lam):
        raise ValueError(f"n_beam must be {len(lam)} (full table)")
    nb = len(lam)
    sigma_h, sigma_he, hnu, finf = sigma_h_all, sigma_he_all, hnu_all, finf_all

    # Fractions of the H+He total at the base (locked IC recipe).
    frac_hi = 1.0 - he_ratio
    frac_hei = he_ratio
    frac_hii = ion_frac * frac_hi
    frac_heii = ion_frac * frac_hei

    constants: dict[str, Any] = {
        "G": 6.6726e-8,
        "M_p": m_p,
        "M_star": m_star,
        "a": a_orb,
        "eta": eta,
        "n_floor": 1.0e-30,
        "T_floor": 1.0,   # K; every rate law floors its temperature here
        "gm1": gamma - 1.0,
        "kB_cgs": KB,     # rate constants written in cgs (yardstick form)
        "eV": EV_ERG,
        # Gram masses (CHE / COL friction terms carry ATOM_MASS = amu*CONST_amu).
        "m_HI_g": AMU_H * AMU,
        "m_HEI_g": AMU_HE * AMU,
        "m_HII_g": AMU_H * AMU,
        "m_HEII_g": AMU_HEII * AMU,
        "m_e_g": M_E_AMU * AMU,
        # amu numbers (yardstick momentum / pressure divisors).
        "m_HI_amu": AMU_H,
        "m_HEI_amu": AMU_HE,
        "m_HII_amu": AMU_H,
        "m_HEII_amu": AMU_HEII,
        "m_e_amu": M_E_AMU,
        # Recombination (yardstick :106-107): alpha (300/T_e)^0.64 n_ion n_e.
        "alpha_rec_H": 4.0e-12,
        "alpha_rec_He": 4.6e-12,
        "T_rec_ref": 300.0,
        "rec_exp": 0.64,
        # Charge exchange (yardstick :84-94; startup gate locked to 1).
        "k_cx1": 1.25e-15,
        "k_cx2": 1.75e-11,
        "cx1_exp": -0.25,
        "cx2_exp": 0.75,
        "T_cx2_act": 128000.0,
        # Electron-impact ionisation (yardstick :100-103).
        "e_imp_H": 2.91e-8,
        "e_imp_He": 1.75e-8,
        "chi_H_num": 13.6 * EV_ERG,   # 13.6 eV in erg
        "chi_He_num": 24.6 * EV_ERG,  # 24.6 eV in erg
        "chi_den_fac": 1.5,           # chi = E_ion / (1.5 kB T_e)
        "chi_H_den_add": 0.232,
        "chi_He_den_add": 0.180,
        # Ly-alpha cooling (yardstick :78): -7.5e-19 n_HI n_e exp(-118348/T_e).
        "cool_coef": 7.5e-19,
        "cool_T_act": 118348.0,
        # Elastic / resonance collisions (yardstick :152-162).
        "k_coll_HH": 2.67e-10,
        "coll_beta_HH": 0.083,
        "k_coll_HeHe": 3.50e-10,
        "coll_beta_HeHe": 0.093,
        "k_coll_HHe_den": 1.04e18,
        "coll_HHe_exp": 0.732,
        "amu_g": AMU,               # CONST_amu, HI-HEI coefficient divisor
        "k_coll_ii": 1.15,
        "inv_log10": 1.0 / math.log(10.0),
        # Ion-electron Coulomb (yardstick :164-166; no G_DT).
        "k_coul": 0.0299,
        # IC / floor parameters (the caller packs the state by radius).
        "R_p": r_p,
        "n_beg": n_beg,
        "HE_RATIO": he_ratio,
        "HE_SH_COEF": he_sh_coef,
        "frac_HI": frac_hi,
        "frac_HEI": frac_hei,
        "frac_HII": frac_hii,
        "frac_HEII": frac_heii,
        "T_beg": t_beg,
        "n_HI_beg": n_beg * frac_hi,
        "n_HEI_beg": n_beg * frac_hei,
        "n_HII_beg": n_beg * frac_hii,
        "n_HEII_beg": n_beg * frac_heii,
        "P_HI_beg": n_beg * frac_hi * KB * t_beg,
        "P_HEI_beg": n_beg * frac_hei * he_sh_coef * KB * t_beg,
        "P_HII_beg": n_beg * frac_hii * KB * t_beg,
        "P_HEII_beg": n_beg * frac_heii * KB * t_beg,
        "P_e_beg": n_beg * (frac_hii + frac_heii) * KB * t_beg,
        # IC envelope: n(r) = n_f,beg exp(-(r-R_p)/H_ic), floored (A5 form).
        "H_ic": 0.04 * r_p,
        "n_ic_min": 1.0e6,  # cm^-3, applied per fraction
        # Old-tree IEND outflow (mmd_model_IC_BC.c:383): x[IEND]/x[IEND-1].
        "x1end_grid_rate": x1end_grid_rate,
    }
    for b in range(nb):
        constants[f"sigma_H_{b}"] = sigma_h[b]
        constants[f"sigma_He_{b}"] = sigma_he[b]
        constants[f"hnu_{b}"] = hnu[b]

    # ---- Source table (full yardstick mmd_model_src.c) ---------------------
    # 53-way beam sums are expanded: the frozen opcode set has no sum
    # opcode, so each beam contributes its own term. Squares are written
    # as explicit X*X products (the seeded forward-AD POW refuses bases
    # <= 0 and velocity / log-term differences cross zero). T^{-3/2} is
    # spelled 1/((T)**0.5*(T)); log10(T) is log(T)*inv_log10.
    photo_h_sum = " + ".join(f"sigma_H_{b} * F_{b} / hnu_{b}" for b in range(nb))
    photo_he_sum = " + ".join(f"sigma_He_{b} * F_{b} / hnu_{b}" for b in range(nb))
    heat_h_sum = " + ".join(f"sigma_H_{b} * F_{b}" for b in range(nb))
    heat_he_sum = " + ".join(f"sigma_He_{b} * F_{b}" for b in range(nb))
    photo_h = f"n_HI * ({photo_h_sum})"
    photo_he = f"n_HEI * ({photo_he_sum})"
    heat_tot = f"n_HI * ({heat_h_sum}) + n_HEI * ({heat_he_sum})"

    # Mass-weighted reduced temperatures TMP_REDUCE(f1,f2) =
    # (T_f1 m_f2 + T_f2 m_f1)/(m_f1 + m_f2) (mmd.h:73), spelled with the
    # derived T reads. Every rate floors at T_floor.
    t_cx1 = ("(T_HI * m_HEII_amu + T_HEII * m_HI_amu) "
             "/ (m_HI_amu + m_HEII_amu)")           # HI + HEII pair (CX1)
    t_cx2 = ("(T_HII * m_HEI_amu + T_HEI * m_HII_amu) "
             "/ (m_HII_amu + m_HEI_amu)")           # HII + HEI pair (CX2)
    t_hhe = ("(T_HI * m_HEI_amu + T_HEI * m_HI_amu) "
             "/ (m_HI_amu + m_HEI_amu)")            # HI-HEI collision
    t_ii = ("(T_HII * m_HEII_amu + T_HEII * m_HII_amu) "
            "/ (m_HII_amu + m_HEII_amu)")           # HII-HEII collision
    cx1_rate = f"k_cx1 * (T_rec_ref / (({t_cx1}) + T_floor))**cx1_exp"
    cx2_rate = (
        f"k_cx2 * (T_rec_ref / (({t_cx2}) + T_floor))**cx2_exp "
        f"* exp(-T_cx2_act / (({t_cx2}) + T_floor))"
    )
    cx1 = f"({cx1_rate}) * n_HI * n_HEII"
    cx2 = f"({cx2_rate}) * n_HII * n_HEI"

    # Electron-impact ionisation (T_e chain-derived, never a constant).
    te_floor = "(T_e + T_floor)"
    chi_h = f"chi_H_num / (chi_den_fac * kB_cgs * {te_floor})"
    chi_he = f"chi_He_num / (chi_den_fac * kB_cgs * {te_floor})"
    eimp_h_rate = (
        f"e_imp_H * ({chi_h})**0.39 / (chi_H_den_add + ({chi_h})) "
        f"* exp(-({chi_h}))"
    )
    eimp_he_rate = (
        f"e_imp_He * ({chi_he})**0.35 / (chi_He_den_add + ({chi_he})) "
        f"* exp(-({chi_he}))"
    )
    eimp_h = f"({eimp_h_rate}) * n_HI * n_e"
    eimp_he = f"({eimp_he_rate}) * n_HEI * n_e"

    # Recombination reads the DERIVED n_e (quasi-neutral read port).
    rec_h_rate = f"alpha_rec_H * (T_rec_ref / {te_floor})**rec_exp"
    rec_he_rate = f"alpha_rec_He * (T_rec_ref / {te_floor})**rec_exp"
    rec_h_photo = f"({rec_h_rate}) * n_HII * n_e"
    rec_he_photo = f"({rec_he_rate}) * n_HEII * n_e"

    # One chemistry sum per element pair (photo + CX + e-impact), after
    # the yardstick :113-144 bookkeeping: ion_ndt[HI] vs rec_ndt[HII],
    # ion_ndt[HEI] vs rec_ndt[HEII].
    ion_h = f"({photo_h}) + ({cx1}) + ({eimp_h})"
    rec_h = f"({rec_h_photo}) + ({cx2})"
    ion_he = f"({photo_he}) + ({cx2}) + ({eimp_he})"
    rec_he = f"({rec_he_photo}) + ({cx1})"

    # CHE_PRS_TRANS(f1,f2,nloss,nget) = -nloss/n1 P1 + nget/n2 P2
    # + (1/3) nget m_f1_g (v2 - v1)^2 (yardstick :2-4), floored divisors.
    def che_prs_trans(f1: str, f2: str, nloss: str, nget: str) -> str:
        dv = f"(vx_{f2} - vx_{f1})"
        return (
            f"-({nloss}) / (n_{f1} + n_floor) * P_{f1}"
            f" + ({nget}) / (n_{f2} + n_floor) * P_{f2}"
            f" + (1.0 / 3.0) * ({nget}) * m_{f1}_g * ({dv} * {dv})"
        )

    # Collision rates (yardstick :152-166, no G_DT). Resonance pairs use
    # the mean temperature, unlike pairs the reduced temperature; all
    # floored. log10(T) = log(T)*inv_log10, squares explicit X*X.
    t_hh = f"(0.5 * (T_HI + T_HII) + T_floor)"
    log_hh = f"(1.0 - coll_beta_HH * log({t_hh}) * inv_log10)"
    nu_hh = f"k_coll_HH * ({t_hh})**0.5 * ({log_hh} * {log_hh})"
    t_hehe = f"(0.5 * (T_HEI + T_HEII) + T_floor)"
    log_hehe = f"(1.0 - coll_beta_HeHe * log({t_hehe}) * inv_log10)"
    nu_hehe = f"k_coll_HeHe * ({t_hehe})**0.5 * ({log_hehe} * {log_hehe})"
    t_hhe_f = f"(({t_hhe}) + T_floor)"
    nu_hhe = f"kB_cgs * ({t_hhe_f}) / (k_coll_HHe_den * ({t_hhe_f})**coll_HHe_exp * amu_g)"
    t_ii_f = f"(({t_ii}) + T_floor)"
    nu_ii = f"k_coll_ii / (({t_ii_f})**0.5 * ({t_ii_f}))"
    nu_e = f"k_coul / ({te_floor}**0.5 * {te_floor})"

    # Charge-weighted electron velocity, spelled out (no vx_e read name):
    # v_e = (n_HII v_HII + n_HEII v_HEII) / n_e.
    v_e = "(n_HII * vx_HII + n_HEII * vx_HEII) / (n_e + n_floor)"

    # COL_PRS_TRANS(f1,f2) = 2 (n1 P2 - n2 P1)
    # + (2/3) m_f2_g n1 n2 (v1 - v2)^2 (yardstick :6-9).
    def col_prs_trans(f1: str, f2: str) -> str:
        dv = f"(vx_{f1} - vx_{f2})"
        return (
            f"2.0 * (n_{f1} * P_{f2} - n_{f2} * P_{f1})"
            f" + (2.0 / 3.0) * m_{f2}_g * n_{f1} * n_{f2} * ({dv} * {dv})"
        )

    # Ion-electron Coulomb bands: full COL_PRS_TRANS with v_e (B2 dropped
    # the friction term; with two ions u_e != u_ion so the yardstick
    # (2/3) m_e n_ion n_e (v_ion - v_e)^2 frictional heat MUST ride along).
    def col_prs_trans_e(fion: str) -> str:
        dv = f"(vx_{fion} - ({v_e}))"
        return (
            f"2.0 * (n_{fion} * P_e - n_e * P_{fion})"
            f" + (2.0 / 3.0) * m_e_g * n_{fion} * n_e * ({dv} * {dv})"
        )

    coul_hii = f"({nu_e}) / (m_HII_amu + m_e_amu) * ({col_prs_trans_e('HII')})"
    coul_heii = f"({nu_e}) / (m_HEII_amu + m_e_amu) * ({col_prs_trans_e('HEII')})"

    sources = [
        # Chemistry, one summed pair per element (net H / He nuclei
        # conserved by construction: the four dn tapes are antisymmetric
        # pairwise, HI<->HII and HEI<->HEII).
        {"name": "chem_HI", "expr": {"dn_HI": f"-({ion_h}) + ({rec_h})"}},
        {"name": "chem_HII", "expr": {"dn_HII": f"({ion_h}) - ({rec_h})"}},
        {"name": "chem_HEI", "expr": {"dn_HEI": f"-({ion_he}) + ({rec_he})"}},
        {"name": "chem_HEII", "expr": {"dn_HEII": f"({ion_he}) - ({rec_he})"}},
        # Chemistry velocity inheritance (yardstick :115,:124,:133,:142),
        # summed rates, floored divisors.
        {
            "name": "drag_HI",
            "expr": {"dvx_HI": f"({rec_h}) / (n_HI + n_floor) * (vx_HII - vx_HI)"},
        },
        {
            "name": "drag_HII",
            "expr": {"dvx_HII": f"({ion_h}) / (n_HII + n_floor) * (vx_HI - vx_HII)"},
        },
        {
            "name": "drag_HEI",
            "expr": {"dvx_HEI": f"({rec_he}) / (n_HEI + n_floor) * (vx_HEII - vx_HEI)"},
        },
        {
            "name": "drag_HEII",
            "expr": {"dvx_HEII": f"({ion_he}) / (n_HEII + n_floor) * (vx_HEI - vx_HEII)"},
        },
        # CHE pressure transfer, one band per fluid with the summed rates.
        {"name": "che_HI", "expr": {"dP_HI": che_prs_trans("HI", "HII", ion_h, rec_h)}},
        {"name": "che_HII", "expr": {"dP_HII": che_prs_trans("HII", "HI", rec_h, ion_h)}},
        {"name": "che_HEI", "expr": {"dP_HEI": che_prs_trans("HEI", "HEII", ion_he, rec_he)}},
        {"name": "che_HEII", "expr": {"dP_HEII": che_prs_trans("HEII", "HEI", rec_he, ion_he)}},
        # Photoionisation heating + Ly-alpha cooling -> dP_e directly
        # (yardstick :148, (gamma-1)(eta*heating + cooling)).
        {"name": "photoheat_Pe", "expr": {"dP_e": f"gm1 * eta * ({heat_tot})"}},
        {
            "name": "cool_Pe",
            "expr": {
                "dP_e": (
                    "gm1 * (-cool_coef * n_HI * n_e * "
                    f"exp(-cool_T_act / {te_floor}))"
                )
            },
        },
        # Elastic / resonance collisions: momentum exchange (amu divisor,
        # antisymmetric differences -> total momentum conserved).
        {
            "name": "coll_mom_HI",
            "expr": {
                "dvx_HI": (
                    f"({nu_hh}) / m_HI_amu * n_HII * (vx_HII - vx_HI)"
                    f" + ({nu_hhe}) / m_HI_amu * n_HEI * (vx_HEI - vx_HI)"
                )
            },
        },
        {
            "name": "coll_mom_HEI",
            "expr": {
                "dvx_HEI": (
                    f"({nu_hehe}) / m_HEI_amu * n_HEII * (vx_HEII - vx_HEI)"
                    f" + ({nu_hhe}) / m_HEI_amu * n_HI * (vx_HI - vx_HEI)"
                )
            },
        },
        {
            "name": "coll_mom_HII",
            "expr": {
                "dvx_HII": (
                    f"({nu_hh}) / m_HII_amu * n_HI * (vx_HI - vx_HII)"
                    f" + ({nu_ii}) / m_HII_amu * n_HEII * (vx_HEII - vx_HII)"
                )
            },
        },
        {
            "name": "coll_mom_HEII",
            "expr": {
                "dvx_HEII": (
                    f"({nu_hehe}) / m_HEII_amu * n_HEI * (vx_HEI - vx_HEII)"
                    f" + ({nu_ii}) / m_HEII_amu * n_HII * (vx_HII - vx_HEII)"
                )
            },
        },
        # Elastic / resonance collisions: internal energy + friction heat
        # (COL_PRS_TRANS), net dn = 0.
        {
            "name": "coll_prs_HI",
            "expr": {
                "dP_HI": (
                    f"({nu_hh}) / (m_HI_amu + m_HII_amu) * ({col_prs_trans('HI', 'HII')})"
                    f" + ({nu_hhe}) / (m_HI_amu + m_HEI_amu) * ({col_prs_trans('HI', 'HEI')})"
                )
            },
        },
        {
            "name": "coll_prs_HEI",
            "expr": {
                "dP_HEI": (
                    f"({nu_hehe}) / (m_HEI_amu + m_HEII_amu) * ({col_prs_trans('HEI', 'HEII')})"
                    f" + ({nu_hhe}) / (m_HEI_amu + m_HI_amu) * ({col_prs_trans('HEI', 'HI')})"
                )
            },
        },
        {
            "name": "coll_prs_HII",
            "expr": {
                "dP_HII": (
                    f"({nu_hh}) / (m_HII_amu + m_HI_amu) * ({col_prs_trans('HII', 'HI')})"
                    f" + ({nu_ii}) / (m_HII_amu + m_HEII_amu) * ({col_prs_trans('HII', 'HEII')})"
                )
            },
        },
        {
            "name": "coll_prs_HEII",
            "expr": {
                "dP_HEII": (
                    f"({nu_hehe}) / (m_HEII_amu + m_HEI_amu) * ({col_prs_trans('HEII', 'HEI')})"
                    f" + ({nu_ii}) / (m_HEII_amu + m_HII_amu) * ({col_prs_trans('HEII', 'HII')})"
                )
            },
        },
        # Ion-electron Coulomb heat exchange (yardstick :192,:204 ion
        # side): full COL_PRS_TRANS with the charge-weighted v_e, so the
        # frictional heat (2/3) m_e n_ion n_e (v_ion - v_e)^2 rides along.
        # NO dvx_* band (yardstick ion momentum never exchanges with ELCI).
        {"name": "coul_prs_HII", "expr": {"dP_HII": coul_hii}},
        {"name": "coul_prs_HEII", "expr": {"dP_HEII": coul_heii}},
        {
            # Electron side: exact negation of the two ion-side Coulomb
            # bands, same divisors -> the Coulomb exchange conserves
            # P_HII + P_HEII + P_e (antisymmetric by construction).
            "name": "coul_prs_e",
            "expr": {"dP_e": f"-(({coul_hii}) + ({coul_heii}))"},
        },
        {
            "name": "gravity",
            "expr": {"dvx": "-G * M_p / r**2 + 3 * G * M_star * r / a**3"},
        },
    ]

    ng = 2
    # x1beg ghosts: closed assignment of the planetary base, all four
    # fluids (tapes write ghost layers only; P_e ghosts fall back to the
    # copy @1 fill).
    user_bc = {
        "x1beg": {
            "n_HI": "n_HI_beg",
            "vx_HI": "0.0",
            "vy_HI": "0.0",
            "vz_HI": "0.0",
            "P_HI": "P_HI_beg",
            "n_HEI": "n_HEI_beg",
            "vx_HEI": "0.0",
            "vy_HEI": "0.0",
            "vz_HEI": "0.0",
            "P_HEI": "P_HEI_beg",
            "n_HII": "n_HII_beg",
            "vx_HII": "0.0",
            "vy_HII": "0.0",
            "vz_HII": "0.0",
            "P_HII": "P_HII_beg",
            "n_HEII": "n_HEII_beg",
            "vx_HEII": "0.0",
            "vy_HEII": "0.0",
            "vz_HEII": "0.0",
            "P_HEII": "P_HEII_beg",
        },
    }
    # On: old-tree log extrapolation for all four fluids. Off: default
    # copy-outflow (a draining column inverts the last two cells and the
    # power law explodes; B2 supplement split).
    if radiation:
        user_bc["x1end"] = _x1end_outflow_tapes(("HI", "HEI", "HII", "HEII"))
        x1end_kind = "user"
    else:
        x1end_kind = "outflow"

    return {
        "schema": SCHEMA,
        "backend": "serial",
        "gamma": gamma,
        "units": {"density": 1.0, "velocity": 1.0, "length": 1.0},
        "numerics": {"riemann": "hll", "reconstruction": "plm", "rk": "rk2"},
        "grid": {
            "geometry": "spherical",
            "ndim": 1,
            "n": [nx, 1, 1],
            "stretch": stretch,
            **(
                {"stretch_factor": stretch_factor}
                if stretch_factor is not None
                else {}
            ),
            "x1min": x1min,
            "x1max": x1max,
            "nghost": ng,
        },
        "time": {"cfl": 0.4, "tstop": float(tstop), "dt_max": float(dt_max)},
        "fluids": [
            {"name": "HI", "mass": AMU_H, "charge": 0, "kind": "neutral"},
            {"name": "HEI", "mass": AMU_HE, "charge": 0, "kind": "neutral"},
            {"name": "HII", "mass": AMU_H, "charge": 1, "kind": "ion"},
            {"name": "HEII", "mass": AMU_HEII, "charge": 1, "kind": "ion"},
        ],
        "electron": {
            "mode": "pressure-only",
            "share": "by-charge",
            "shock_heat": "ions-only",
        },
        "sources": sources,
        "constants": constants,
        "radiation": {
            "enabled": bool(radiation),
            "absorbers": ["HI", "HEI"],
            "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
            "bins": {
                "n": nb,
                "sigma": [list(sigma_h), list(sigma_he)],
                "hnu": hnu,
                "flux_inf": finf if radiation else [0.0] * nb,
            },
        },
        "boundary": {"x1beg": "user", "x1end": x1end_kind},
        "user_bc": user_bc,
        "ic": {"kind": "none"},
    }
