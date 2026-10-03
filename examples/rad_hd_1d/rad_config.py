"""Example-private config for the rad_hd_1d setup (A5 exam factory).

The HD209458b radiation-HD factory lives here, next to the example that
uses it; the public pool (xmf.config) keeps only the shared schema,
natural constants, enums and the generic shock_tube/normalize frontend.
KB/AMU/SCHEMA are imported from xmf.config, never redefined here.
"""
from __future__ import annotations

from typing import Any

from xmf.config import AMU, KB, SCHEMA

# ---- Pinned run knobs (edit here; run.py has no CLI) ---------------------
# Flat ConfigDict-style pin. A class body, not @dataclass: tests load this
# file via importlib without registering it in sys.modules, and dataclass
# looks the module up. pytest may still pass nx/n_beam/tstop into
# hh_rad_1d to override a single field.


class RUN:
    nx = 128
    n_beam = 53
    tstop = 2.0e3
    stretch = "log"  # "none" | "log"
    stretch_factor = 1.025  # log: e.g. 1.025
    plots = True
    log_every = 100  # 0 = silent

# ---- A5 radiation-HD factory ---------------------------------------------
# Reference numbers of the HD 209458b setup (all cgs). They live here in
# named constants -- the C kernel receives only the compiled tapes and the
# beam table, never a physical formula or a species name.
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


def hh_rad_1d(
    data_dir,
    *,
    radiation: bool = True,
    nx: int | None = None,
    n_beam: int | None = None,
    tstop: float | None = None,
    stretch: str | None = None,
    stretch_factor: float | None = None,
) -> dict[str, Any]:
    """Radiation-driven two-fluid (HI/HII) escape setup on a 1-D spherical
    grid r in [R_p, 10 R_p] in full cgs (every unit is 1, so code values
    ARE cgs values; UNIT_LENGTH = R_p is deliberately NOT used).

    data_dir must hold the SSH SED table (hd209458b-1163.dat) and the
    photo-ionisation cross sections (photoIon_sigma.dat). The SED is
    scaled exactly like xuv_initialize: XUV_RESIZE reshapes it to
    XUV_TOT_FLUX with the sub-504 A share SPECTRAL_INDEX_504A, then the
    planet scaling XUV_FACTOR/R_Orbit^2 is applied. hnu_b = h*c/lambda_b.

    radiation=False builds the off-radiation comparison: the beam table
    stays loaded but its boundary fluxes are zeroed by normalize(), so
    F_b = 0 while recombination, CHE pressure transfer and
    gravity+tide stay on. n_beam=1 selects the single-beam fallback whose
    effective sigma
    and hnu reproduce the full-table ionisation rate AND heating at the
    unattenuated boundary.

    Gravity is the SSH tidal-cost form
    dvx = -G*M_p/r**2 + 3*G*M_star*r/a**3 (M_star = 1.119 M_sun,
    a = 0.04707 AU). The Hill radius sits near 4 R_p, so the outer
    half of [R_p, 10 R_p] is unbound. Recombination uses
    T_HII + T_floor (1 K) so (Tref/T)**k stays finite when the
    hydro pressure floor (1e-30) would otherwise drive T → 0.

    The initial / floor state is packed by the caller through
    xmf_state_pack (two fluids by radius, never squeezed into a flat
    rho_l); ic.kind is 'none'. pack_ic uses a thin exponential
    n ∝ exp(-(r-R_p)/H_ic) so the envelope sits under the wind
    except near r = R_p. The x1beg user face assigns the base
    (n, v=0, P) to ghosts only (analytic user_bc tapes); the first
    interior cell is not pinned.

    stretch selects the x1 interface factory (A1 stretch supplement;
    Python-only -- the kernel only ever sees the resulting faces):
    'none' (default) keeps the equidistant exam baseline; 'log'
    stretches geometrically with grid.stretch_factor > 1 crowding
    cells at the planet (inner-dense). For nx=128 a factor of 1.025
    puts the first cell at ~0.01 R_p (~1/4 of the IC scale height
    H_ic = 0.04 R_p, which a uniform 128 cell at 0.070 R_p does not
    resolve) and the last at ~0.23 R_p; the width ratio is ~23.
    Ghost widths equal the adjacent active width, so inner ghosts sit
    at R_p - dx0 and R_p - 2 dx0, safely above r = 0.

    nx / n_beam / tstop / stretch / stretch_factor default to RUN
    at the top of this module. Passing a value still overrides that
    field (used by tests).
    """
    from pathlib import Path

    nx = RUN.nx if nx is None else int(nx)
    n_beam = RUN.n_beam if n_beam is None else int(n_beam)
    tstop = RUN.tstop if tstop is None else float(tstop)
    stretch = RUN.stretch if stretch is None else str(stretch)
    if stretch_factor is None:
        stretch_factor = RUN.stretch_factor

    data = Path(data_dir)
    lam, sed = _read_sed(data / "hd209458b-1163.dat")
    sigma_all = _read_sigma(data / "photoIon_sigma.dat", "sigma_H", len(lam))

    r_p = 1.36 * R_JUP
    m_p = 0.69 * M_JUP
    m_star = 1.119 * M_SUN
    a_orb = 0.04707 * AU
    gamma = 5.0 / 3.0
    n_beg = 1.0e13     # n_HI(R_p), cm^-3
    ion_frac = 1.0e-3  # n_HII / n_HI
    t_beg = 1500.0     # K, both fluids
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
        # ionisation rate and the heating at the unattenuated boundary:
        # sigma_eff = H/F_tot, hnu_eff = H/Gamma.
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
        "m_HI_g": AMU_H * AMU,   # gram masses for the CHE momentum term
        "m_HII_g": AMU_H * AMU,
        # IC / floor parameters (the caller packs the state by radius)
        "R_p": r_p,
        "n_beg": n_beg,
        "n_HII_beg": n_beg * ion_frac,
        "P_HI_beg": n_beg * KB * t_beg,
        "P_HII_beg": n_beg * ion_frac * KB * t_beg,
        "ion_frac": ion_frac,
        "T_beg": t_beg,
        # IC envelope: n(r) = n_beg*exp(-(r-R_p)/H_ic), floored.
        # The old (R_p/r)^2 pack sat ~2–7 orders above the t=1e6 wind
        # for r ≳ 1.1 R_p; flux conservation then spends the run
        # dumping excess mass. H_ic = 0.04 R_p stays under that wind
        # everywhere except the pinned r = R_p cell.
        "H_ic": 0.04 * r_p,
        "n_ic_min": 1.0e6,  # cm^-3; 1 or 10 is too rarefied on this grid
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
    recomb = "alpha_rec * (T_rec_ref / (T_HII + T_floor))**rec_exp * n_HII * n_HII"
    dv_che = "(vx_HII - vx_HI) * (vx_HII - vx_HI)"
    dv_che_rev = "(vx_HI - vx_HII) * (vx_HI - vx_HII)"
    sources = [
        {"name": "photoion_HI", "expr": {"dn_HI": f"-({gamma_ion})"}},
        {"name": "photoion_HII", "expr": {"dn_HII": gamma_ion}},
        {"name": "photoheat_HII", "expr": {"dP_HII": f"gm1 * eta * ({heat_vol})"}},
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
            "name": "gravity",
            "expr": {"dvx": "-G * M_p / r**2 + 3 * G * M_star * r / a**3"},
        },
    ]

    ng = 2
    # x1beg ghosts: closed assignment of the planetary base. Tapes write
    # ghost layers only -- they do not pin the first interior cell (that
    # was the old floor, and next to the thin exponential IC it put a
    # 1e13 / ~vacuum jump on every step). Unwritten slots copy @1.
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
        }
    }

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
            "x1min": r_p,
            "x1max": 10.0 * r_p,
            "nghost": ng,
        },
        "time": {"cfl": 0.4, "tstop": float(tstop), "dt_max": 0.0},
        "fluids": [
            {"name": "HI", "mass": AMU_H, "charge": 0, "kind": "hd"},
            {"name": "HII", "mass": AMU_H, "charge": 1, "kind": "hd"},
        ],
        "electron": {"mode": "off"},
        "sources": sources,
        "constants": constants,
        "radiation": {
            "enabled": bool(radiation),
            "absorbers": ["HI"],
            "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
            # Off-radiation comparison: the table stays loaded (the ion
            # tapes keep reading F_b) but the boundary fluxes are zero,
            # so F_b = 0 everywhere while recombination, CHE and gravity
            # stay on.
            "bins": {
                "n": nb,
                "sigma": sigma,
                "hnu": hnu,
                "flux_inf": finf if radiation else [0.0] * nb,
            },
        },
        "boundary": {"x1beg": "user", "x1end": "outflow"},
        "user_bc": user_bc,
        "ic": {"kind": "none"},
    }
