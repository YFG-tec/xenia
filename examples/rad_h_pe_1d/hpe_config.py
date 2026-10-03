"""Example-private config for the rad_h_pe_1d setup (B2 ion + P_e factory).

The HI/HII + pressure-only-electron factory lives here, next to the
example that uses it. PRIVATE by contract: it is never imported by
xmf.config and never exported by `import xmf` -- only run.py (this
directory) and the factory contract test load it, by absolute path.
KB/AMU/SCHEMA are imported from xmf.config, never redefined here.

Differences from the A5 rad_hd_1d factory (same SED/sigma tables, same
XUV rescale, same gravity+tide, same user x1beg base):
  * electron = {"mode": "pressure-only", "share": "by-charge",
    "shock_heat": "ions-only"} -- one separate P_e block, never a fluid
    slot; the hydro shock-heating share tier is pinned explicitly.
  * Photoionisation heating goes to dP_e (B2 appended IR write name,
    directly onto the electron block), not dP_HII.
  * Recombination reads the DERIVED n_e, never n_HII*n_HII.
  * HI-HII resonance collision after the yardstick
    SSH_220804/mmd_model_src.c:155-156:
      coef = 2.67e-10 * T^0.5 * (1 - 0.083*log10(T))^2,
      T = 0.5*(T_HI + T_HII), net dn = 0; momentum and internal energy
    are exchanged (COL_PRS_TRANS structure, squares written as explicit
    X*X products so the forward-AD tape machine never sees a POW base
    crossing zero).
  * HII-electron Coulomb heat exchange (B2 supplement) after the
    yardstick SSH_220804/mmd_model_src.c:164,192,207:
      nu = 0.0299 * T_e^{-3/2} (the yardstick multiplies G_DT; the IR
      does NOT -- the kernel multiplies its own dt),
      Delta = 2*(n_HII*P_e - n_e*P_HII) (heat-exchange part of
      COL_PRS_TRANS only: the single-ion u_e = u_HII makes the friction
      term identically zero, so it is omitted),
      dP_HII = nu/(m_HII + m_e)*Delta, dP_e = nu/(m_e + m_HII)*(-Delta)
    -- antisymmetric, same divisor, total ion+electron pressure exactly
    conserved. NO dvx_* band: the yardstick HII momentum (:187-188)
    exchanges with HI/HEII only, never with ELCI. Net dn = 0. Electron
    mass 5.4858e-4 amu after mmd_model_initialize.c:29.
"""
from __future__ import annotations

import math
from typing import Any

from xmf.config import AMU, KB, SCHEMA

# ---- Pinned run knobs (edit here; run.py has no CLI) ---------------------
# Flat ConfigDict-style pin. A class body, not @dataclass: tests load this
# file via importlib without registering it in sys.modules, and dataclass
# looks the module up. pytest may still pass nx/n_beam/tstop into hpe_1d
# to override a single field.


class RUN:
    nx = 128
    n_beam = 53
    tstop = 2.0e5
    stretch = "log"  # "none" | "log"
    stretch_factor = 1.025  # log: e.g. 1.025
    plots = True
    log_every = 100  # 0 = silent
    # Evaluation is radiation ON only. The factory still accepts
    # radiation=False for contract tests; run.py does not launch it.


# ---- Reference numbers of the HD 209458b setup (all cgs) ------------------
# They live here in named constants -- the C kernel receives only the
# compiled tapes and the beam table, never a physical formula or a
# species name.
H_PLANET = 6.62606876e-27  # Planck constant, erg s (SSH CONST_h)
C_LIGHT = 2.99792458e10    # speed of light, cm/s (SSH CONST_c)
R_JUP = 7.1492e9           # Jupiter radius, cm
M_JUP = 1.8982e30          # Jupiter mass, g
M_SUN = 2.0e33             # solar mass, g (SSH CONST_Msun)
AU = 1.49597892e13         # astronomical unit, cm (SSH CONST_au)
AMU_H = 1.00794            # hydrogen mass, amu


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
    """Yardstick IEND userdef outflow, written as assignment tapes.

    For each fluid (SSH :388-390):
      n  = n@1**(1+g) / n@2**g
      P  = P@1**(1+g) / P@2**g
      vx = MAX(0, (1+g)*vx@1 - g*vx@2)
    A3 froze max/min opcodes, so MAX(0, v) is the value-only identity
    0.5*(v + (v*v)**0.5). Both ghost layers use the same @1/@2 (the two
    last active cells); the old IEND_BOX cascaded the second ghost off
    the first, which user_bc cannot do. vy/vz stay unwritten → copy @1.
    P_e is not a user_bc write name; its ghosts stay the existing copy@1.
    """
    g = "x1end_grid_rate"
    out: dict[str, str] = {}
    for name in fluids:
        vext = f"((1.0 + {g}) * vx_{name}@1 - {g} * vx_{name}@2)"
        # Exact yardstick. A drained (radiation-off) column can invert
        # n@1/n@2 and the power law then blows the ghosts -- that is an
        # accepted tradeoff of this BC, not a kernel bug.
        out[f"n_{name}"] = f"n_{name}@1**(1.0 + {g}) / n_{name}@2**{g}"
        out[f"P_{name}"] = f"P_{name}@1**(1.0 + {g}) / P_{name}@2**{g}"
        out[f"vx_{name}"] = f"0.5 * (({vext}) + (({vext}) * ({vext}))**0.5)"
    return out


def hpe_1d(
    data_dir,
    *,
    radiation: bool = True,
    nx: int | None = None,
    n_beam: int | None = None,
    tstop: float | None = None,
    stretch: str | None = None,
    stretch_factor: float | None = None,
) -> dict[str, Any]:
    """Radiation-driven two-fluid (HI/HII) escape with a pressure-only
    electron block on a 1-D spherical grid r in [R_p, 10 R_p], full cgs
    (every unit is 1, so code values ARE cgs values).

    data_dir must hold the SSH SED table (hd209458b-1163.dat) and the
    photo-ionisation cross sections (photoIon_sigma.dat) -- the example
    REFERENCES ../rad_hd_1d/data/ and never copies it. SED rescale is
    exactly the A5 xuv_initialize form (XUV_RESIZE to XUV_TOT_FLUX with
    the sub-504 A share SPECTRAL_INDEX_504A, then XUV_FACTOR/R_Orbit^2);
    hnu_b = h*c/lambda_b.

    radiation=False is a factory path for contract tests, not a default
    evaluation: the beam table stays loaded but its boundary fluxes are
    zeroed by normalize(), so F_b = 0 while recombination, CHE pressure
    transfer, the HI-HII resonance collision, the HII-electron Coulomb
    heat exchange and gravity+tide stay on. The outer face falls back
    to default copy-outflow (see x1end split below).

    Electron closure (B2): mode pressure-only, share by-charge,
    shock_heat tier pinned to ions-only (Tóth convention: no tier is
    physical truth -- all three are switchable model options; this
    example keeps every hydro shock increment in the ion pressures and
    lets the photoionisation heating enter P_e through the dP_e source
    tape). P_e ghosts fall back to the existing copy @1 fill; the user
    x1beg tapes assign fluid slots only. x1end is split: radiation ON
    uses the old-tree userdef outflow (power-law n/P + MAX(0,v) via
    user_bc); radiation OFF uses the default copy-outflow (log
    extrapolation is unfriendly to a draining column).

    The initial / floor state is packed by the caller through
    xmf_state_pack + xmf_state_write_pe (P_e,ic = n_HII,ic kB T_beg);
    ic.kind is 'none'.

    nx / n_beam / tstop / stretch / stretch_factor default to RUN at the
    top of this module. Passing a value still overrides that field (used
    by tests).
    """
    nx = RUN.nx if nx is None else int(nx)
    n_beam = RUN.n_beam if n_beam is None else int(n_beam)
    tstop = RUN.tstop if tstop is None else float(tstop)
    stretch = RUN.stretch if stretch is None else str(stretch)
    if stretch_factor is None:
        stretch_factor = RUN.stretch_factor

    from pathlib import Path

    data = Path(data_dir)
    lam, sed = _read_sed(data / "hd209458b-1163.dat")
    sigma_all = _read_sigma(data / "photoIon_sigma.dat", "sigma_H", len(lam))

    r_p = 1.36 * R_JUP
    m_p = 0.69 * M_JUP
    m_star = 1.119 * M_SUN
    a_orb = 0.04707 * AU
    x1min, x1max = r_p, 10.0 * r_p
    x1end_grid_rate = _x1end_grid_rate(x1min, x1max, nx, stretch, stretch_factor)
    gamma = 5.0 / 3.0
    n_beg = 1.0e13     # n_HI(R_p), cm^-3
    ion_frac = 1.0e-3  # n_HII / n_HI (= n_e / n_HI at the base)
    t_beg = 1500.0     # K, both fluids (and the IC electron pressure)
    eta = 0.20         # heating efficiency
    xuv_factor = 0.25
    r_orbit = 0.04707  # AU
    xuv_tot_flux = 4.716
    si_504 = 0.89

    # hnu_b = h*c/lambda_b with lambda in Angstrom (xuv_initialize: *1e8).
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
    # Planet scaling at the orbit.
    planet_scale = xuv_factor / (r_orbit * r_orbit)
    finf_all = [s * planet_scale for s in resized]

    if n_beam == len(lam):
        sigma, hnu, finf = sigma_all, hnu_all, finf_all
    elif n_beam == 1:
        # Single-beam fallback: sigma_eff and hnu_eff preserve BOTH the
        # ionisation rate and the heating at the unattenuated boundary.
        f_tot = sum(finf_all)
        heat = sum(s * f for s, f in zip(sigma_all, finf_all))
        gam = sum(s * f / e for s, f, e in zip(sigma_all, finf_all, hnu_all))
        sigma = [heat / f_tot]
        hnu = [heat / gam]
        finf = [f_tot]
    else:
        raise ValueError(f"n_beam must be {len(lam)} (full table) or 1 (fallback)")
    nb = len(sigma)

    constants: dict[str, Any] = {
        "G": 6.6726e-8,
        "M_p": m_p,
        "M_star": m_star,
        "a": a_orb,
        "eta": eta,
        "n_floor": 1.0e-30,
        "alpha_rec": 4.0e-12,  # recombination prefactor, cm^3/s
        "T_rec_ref": 300.0,
        "T_floor": 1.0,  # K; keeps (Tref/T)**k finite when P hits 1e-30
        "rec_exp": 0.64,
        "gm1": gamma - 1.0,
        "m_HI_g": AMU_H * AMU,   # gram masses (COL pressure transfer term)
        "m_HII_g": AMU_H * AMU,
        # HI-HII resonance collision yardstick (SSH mmd_model_src.c:155-156):
        # coef = k_coll*T^0.5*(1 - coll_beta*log10(T))^2, T = 0.5*(T_HI+T_HII).
        # log10(T) is written log(T)*inv_log10 (the tape set has only log),
        # the square as an explicit X*X product (the seeded forward-AD POW
        # refuses bases <= 0 and (1 - beta*log10 T) crosses zero).
        "k_coll": 2.67e-10,   # cm^3/s
        "coll_beta": 0.083,
        "inv_log10": 1.0 / math.log(10.0),
        "m_HI_amu": AMU_H,    # amu numbers, yardstick momentum/pressure divisors
        "m_HII_amu": AMU_H,
        # HII-electron Coulomb exchange yardstick
        # (SSH mmd_model_src.c:164 coef, :192/:207 pressure transfer):
        # nu = k_coul * T_e^{-3/2}; the yardstick multiplies G_DT, the IR
        # does not (the kernel multiplies its own dt). Electron amu mass
        # after SSH mmd_model_initialize.c:29.
        "k_coul": 0.0299,
        "m_e_amu": 5.4858e-4,
        # IC / floor parameters (the caller packs the state by radius)
        "R_p": r_p,
        "n_beg": n_beg,
        "n_HII_beg": n_beg * ion_frac,
        "P_HI_beg": n_beg * KB * t_beg,
        "P_HII_beg": n_beg * ion_frac * KB * t_beg,
        "P_e_beg": n_beg * ion_frac * KB * t_beg,  # n_e = n_HII at the base
        "ion_frac": ion_frac,
        "T_beg": t_beg,
        # IC envelope: n(r) = n_beg*exp(-(r-R_p)/H_ic), floored (A5 form).
        "H_ic": 0.04 * r_p,
        "n_ic_min": 1.0e6,  # cm^-3
        # Old-tree IEND outflow (mmd_model_IC_BC.c:383): x[IEND]/x[IEND-1].
        "x1end_grid_rate": x1end_grid_rate,
    }
    for b in range(nb):
        constants[f"sigma_{b}"] = sigma[b]
        constants[f"hnu_{b}"] = hnu[b]

    # 53-way sums are expanded here: the frozen opcode set has no sum
    # opcode, so each beam contributes its own term. Velocity-squared
    # terms are written as explicit products (never **2): the seeded AD
    # refuses POW bases <= 0, and (v2 - v1) crosses zero at rest.
    photo_terms = " + ".join(f"sigma_{b} * F_{b} / hnu_{b}" for b in range(nb))
    heat_terms = " + ".join(f"sigma_{b} * F_{b}" for b in range(nb))
    gamma_ion = f"n_HI * ({photo_terms})"
    heat_vol = f"n_HI * ({heat_terms})"
    # B2: recombination reads the DERIVED n_e (quasi-neutral read port),
    # never n_HII*n_HII.
    recomb = (
        "alpha_rec * (T_rec_ref / (T_HII + T_floor))**rec_exp * n_HII * n_e"
    )
    # HI-HII resonance collision rate (yardstick coefficient).
    t_mid = "(0.5 * (T_HI + T_HII))"
    log_term = f"(1.0 - coll_beta * log({t_mid}) * inv_log10)"
    coll_rate = f"k_coll * ({t_mid})**0.5 * ({log_term} * {log_term})"
    dv_che = "(vx_HII - vx_HI) * (vx_HII - vx_HI)"
    dv_che_rev = "(vx_HI - vx_HII) * (vx_HI - vx_HII)"
    # COL_PRS_TRANS structure of the yardstick: 2*(n_f1*P_f2 - n_f2*P_f1)
    # + (2/3)*m_f2*n_f1*n_f2*(v_f1 - v_f2)^2, divided by (m_f1 + m_f2) amu.
    col_hi = (
        f"2.0 * (n_HI * P_HII - n_HII * P_HI) "
        f"+ (2.0 / 3.0) * m_HII_g * n_HI * n_HII * ({dv_che_rev})"
    )
    col_hii = (
        f"2.0 * (n_HII * P_HI - n_HI * P_HII) "
        f"+ (2.0 / 3.0) * m_HI_g * n_HII * n_HI * ({dv_che})"
    )
    # HII-electron Coulomb heat exchange (yardstick mmd_model_src.c:164:
    # coef[HII][ELCI] = 0.0299*T_e^{-1.5}*G_DT -- the G_DT factor is NOT
    # carried over: the IR kernel multiplies its own dt). T_e is the
    # derived read P_e/(n_e kB); the T_floor guard + the 1/(sqrt(T)*T)
    # spelling mirror the resonance band (the seeded forward-AD POW
    # refuses bases <= 0, so no **(-1.5)). The yardstick COL_PRS_TRANS
    # (:5-8) friction term (2/3)*m*n1*n2*(v1-v2)^2 is OMITTED: with a
    # single ion the electron block shares u_e = u_HII, so (v_HII-v_e)^2
    # is identically zero; only the heat-exchange part Delta remains.
    coul_T = "(T_e + T_floor)"
    coul_rate = f"k_coul / (({coul_T})**0.5 * ({coul_T}))"
    coul_delta = "2.0 * (n_HII * P_e - n_e * P_HII)"
    coul_dP_HII = f"({coul_rate}) / (m_HII_amu + m_e_amu) * ({coul_delta})"
    sources = [
        {"name": "photoion_HI", "expr": {"dn_HI": f"-({gamma_ion})"}},
        {"name": "photoion_HII", "expr": {"dn_HII": gamma_ion}},
        {
            # B2: the photoionisation heating enters the electron pressure
            # block DIRECTLY through the appended dP_e write name -- never
            # through a fluid P slot.
            "name": "photoheat_Pe",
            "expr": {"dP_e": f"gm1 * eta * ({heat_vol})"},
        },
        {"name": "recomb_HI", "expr": {"dn_HI": recomb}},
        {"name": "recomb_HII", "expr": {"dn_HII": f"-({recomb})"}},
        {
            "name": "drag_HI",
            "expr": {"dvx_HI": f"({recomb}) / (n_HI + n_floor) * (vx_HII - vx_HI)"},
        },
        {
            "name": "drag_HII",
            "expr": {
                "dvx_HII": f"({gamma_ion}) / (n_HII + n_floor) * (vx_HI - vx_HII)"
            },
        },
        {
            "name": "che_HI",
            "expr": {
                "dP_HI": (
                    f"-({gamma_ion}) / (n_HI + n_floor) * P_HI"
                    f" + ({recomb}) / (n_HII + n_floor) * P_HII"
                    f" + (1.0 / 3.0) * ({recomb}) * m_HI_g * {dv_che}"
                )
            },
        },
        {
            "name": "che_HII",
            "expr": {
                "dP_HII": (
                    f"-({recomb}) / (n_HII + n_floor) * P_HII"
                    f" + ({gamma_ion}) / (n_HI + n_floor) * P_HI"
                    f" + (1.0 / 3.0) * ({gamma_ion}) * m_HII_g * {dv_che_rev}"
                )
            },
        },
        {
            # Resonance collision: net dn = 0; momentum exchanged with the
            # yardstick amu divisor, total momentum conserved (equal-mass
            # species, antisymmetric velocity differences).
            "name": "coll_mom_HI",
            "expr": {
                "dvx_HI": f"({coll_rate}) / m_HI_amu * n_HII * (vx_HII - vx_HI)"
            },
        },
        {
            "name": "coll_mom_HII",
            "expr": {
                "dvx_HII": f"({coll_rate}) / m_HII_amu * n_HI * (vx_HI - vx_HII)"
            },
        },
        {
            # Internal-energy exchange + frictional heating (COL_PRS_TRANS).
            "name": "coll_prs_HI",
            "expr": {
                "dP_HI": f"({coll_rate}) / (m_HI_amu + m_HII_amu) * ({col_hi})"
            },
        },
        {
            "name": "coll_prs_HII",
            "expr": {
                "dP_HII": f"({coll_rate}) / (m_HI_amu + m_HII_amu) * ({col_hii})"
            },
        },
        {
            # Coulomb heat exchange, ion side (yardstick :192: HII PRS +=
            # coef[HII][ELCI]/(m_HII+m_e)*COL_PRS_TRANS(HII,ELCI)). Net
            # dn = 0; NO dvx_* band -- the yardstick HII momentum (:187-188)
            # never exchanges with ELCI.
            "name": "coul_prs_HII",
            "expr": {"dP_HII": coul_dP_HII},
        },
        {
            # Coulomb heat exchange, electron side (yardstick :207 wrote
            # the antisymmetric COL_PRS_TRANS(ELCI,HII) into the ELCI
            # slot-0 block; we write the appended dP_e block with the same
            # bookkeeping). Exact negation of the ion-side expression,
            # same divisor -> P_HII + P_e conserved bit for bit.
            "name": "coul_prs_e",
            "expr": {"dP_e": f"-({coul_dP_HII})"},
        },
        {
            "name": "gravity",
            "expr": {"dvx": "-G * M_p / r**2 + 3 * G * M_star * r / a**3"},
        },
    ]

    ng = 2
    # x1beg ghosts: closed assignment of the planetary base (A5 shape).
    # Tapes write ghost layers only -- they never pin the first interior
    # cell and never touch the P_e block (its ghosts fall back to the
    # existing copy @1 fill).
    user_bc = {
        "x1beg": {
            "n_HI": "n_beg",
            "vx_HI": "0.0",
            "vy_HI": "0.0",
            "vz_HI": "0.0",
            "P_HI": "P_HI_beg",
            "n_HII": "n_HII_beg",
            "vx_HII": "0.0",
            "vy_HII": "0.0",
            "vz_HII": "0.0",
            "P_HII": "P_HII_beg",
        },
    }
    # On: old-tree log extrapolation. Off: default copy-outflow -- a
    # draining column inverts the last two cells and the power law
    # explodes; that tradeoff is accepted by using two different faces.
    if radiation:
        user_bc["x1end"] = _x1end_outflow_tapes(("HI", "HII"))
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
        "time": {"cfl": 0.4, "tstop": float(tstop), "dt_max": 0.0},
        "fluids": [
            {"name": "HI", "mass": AMU_H, "charge": 0, "kind": "hd"},
            {"name": "HII", "mass": AMU_H, "charge": 1, "kind": "hd"},
        ],
        # B2 electron closure: one separate P_e block. shock_heat is the
        # hydro shock-heating share tier, pinned explicitly (ions-only:
        # every Godunov increment stays in the ion pressures; the
        # photoionisation heating enters P_e through the dP_e tape).
        "electron": {
            "mode": "pressure-only",
            "share": "by-charge",
            "shock_heat": "ions-only",
        },
        "sources": sources,
        "constants": constants,
        "radiation": {
            "enabled": bool(radiation),
            "absorbers": ["HI"],
            "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
            "bins": {
                "n": nb,
                "sigma": sigma,
                "hnu": hnu,
                "flux_inf": finf if radiation else [0.0] * nb,
            },
        },
        "boundary": {"x1beg": "user", "x1end": x1end_kind},
        "user_bc": user_bc,
        "ic": {"kind": "none"},
    }
