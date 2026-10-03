"""Case specs: Riemann cross-matrix and three classic 2D HD tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import xmf
from xmf.config import AMU, KB

from kernel import (
    MASS,
    base_cfg,
    cgs_from_code,
    l1,
    pack_from_xy,
    run_to_tstop,
    unpack_code,
)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from utils.riemann import exact_hi  # noqa: E402

GAMMA_TUBE = 5.0 / 3.0
UD, UV = 1.0e-12, 1.0e6

SOD_NL, SOD_TL = 1.0e13, 1.0e4
SOD_NR, SOD_TR = 1.25e12, 8.0e3
SOD_PL = SOD_NL * KB * SOD_TL
SOD_PR = SOD_NR * KB * SOD_TR
SOD_X0 = 5.5

TORO2_N = UD / (MASS * AMU)
TORO2_V = 2.0e6
TORO2_P = 0.4


def _tube_scales(n_ex, v_ex, p_ex, n_left: float):
    return (
        n_left,
        max(float(np.max(np.abs(v_ex))), 1.0),
        float(np.max(p_ex)),
    )


def _l1_triple(n, v, p, n_ex, v_ex, p_ex, n_left: float):
    sc = _tube_scales(n_ex, v_ex, p_ex, n_left)
    return {
        "L1_n": l1(n, n_ex, sc[0]),
        "L1_v": l1(v, v_ex, sc[1]),
        "L1_p": l1(p, p_ex, sc[2]),
    }


# ---------------------------------------------------------------------------
# Cross matrix: extruded 1D Toro problems
# ---------------------------------------------------------------------------


def make_1d_a(riemann: str) -> dict:
    cfg = xmf.shock_tube()
    cfg["numerics"]["riemann"] = riemann
    return cfg


def make_1d_b(riemann: str) -> dict:
    cfg = xmf.shock_tube()
    cfg["numerics"]["riemann"] = riemann
    cfg["grid"]["x1min"], cfg["grid"]["x1max"] = -1.0, 1.0
    cfg["boundary"] = {"x1": "copy"}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.0,
        "left": {"n": TORO2_N, "v": -TORO2_V, "p": TORO2_P},
        "right": {"n": TORO2_N, "v": +TORO2_V, "p": TORO2_P},
    }
    return cfg


def make_2d_a(riemann: str) -> dict:
    cfg = xmf.shock_tube()
    cfg["numerics"]["riemann"] = riemann
    cfg["grid"]["n"] = [128, 48, 1]
    cfg["grid"]["ndim"] = 2
    cfg["grid"]["x2min"], cfg["grid"]["x2max"] = 0.0, 1.0
    cfg["boundary"] = {"x1": "reflect", "x2": "copy"}
    return cfg


def make_2d_b(riemann: str) -> dict:
    cfg = xmf.shock_tube()
    cfg["numerics"]["riemann"] = riemann
    cfg["grid"]["n"] = [48, 128, 1]
    cfg["grid"]["ndim"] = 2
    cfg["grid"]["x2min"], cfg["grid"]["x2max"] = 1.0, 10.0
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    cfg["ic"]["x0"] = 0.0
    return cfg


def run_1d_a(riemann: str) -> dict:
    cfg = make_1d_a(riemann)
    run = xmf.setup(cfg)
    try:
        nstep = run_to_tstop(run)
        hi = run.fields["HI"]
        n_ex, v_ex, p_ex = exact_hi(
            hi.x, run.t, SOD_X0, SOD_NL, SOD_NR, 0.0, 0.0, SOD_PL, SOD_PR,
            gamma=GAMMA_TUBE,
        )
        rec = _l1_triple(hi.n, hi.v, hi.p, n_ex, v_ex, p_ex, SOD_NL)
        rec.update(
            case="1D-A",
            title="Sod along x1 (Toro)",
            riemann=riemann,
            t=float(run.t),
            nstep=nstep,
            coord=hi.x,
            num=(hi.n, hi.v, hi.p),
            exact=(n_ex, v_ex, p_ex),
            x0=SOD_X0,
            xlab=r"$x_1$",
        )
        return rec
    finally:
        run.close()


def run_1d_b(riemann: str) -> dict:
    cfg = make_1d_b(riemann)
    run = xmf.setup(cfg)
    try:
        nstep = run_to_tstop(run)
        hi = run.fields["HI"]
        n_ex, v_ex, p_ex = exact_hi(
            hi.x, run.t, 0.0, TORO2_N, TORO2_N, -TORO2_V, TORO2_V,
            TORO2_P, TORO2_P, gamma=GAMMA_TUBE,
        )
        rec = _l1_triple(hi.n, hi.v, hi.p, n_ex, v_ex, p_ex, TORO2_N)
        rec.update(
            case="1D-B",
            title="Toro 2 double rarefaction along x1",
            riemann=riemann,
            t=float(run.t),
            nstep=nstep,
            coord=hi.x,
            num=(hi.n, hi.v, hi.p),
            exact=(n_ex, v_ex, p_ex),
            x0=0.0,
            xlab=r"$x_1$",
        )
        return rec
    finally:
        run.close()


def run_2d_a(riemann: str) -> dict:
    cfg = make_2d_a(riemann)
    run = xmf.setup(cfg)
    try:
        nstep = run_to_tstop(run)
        code = unpack_code(run)
        cgs = cgs_from_code(run, code)
        x, y = cgs["x"], cgs["y"]
        n_ex, v_ex, p_ex = exact_hi(
            x, run.t, SOD_X0, SOD_NL, SOD_NR, 0.0, 0.0, SOD_PL, SOD_PR,
            gamma=GAMMA_TUBE,
        )
        nm, vm, pm = cgs["n"].mean(axis=0), cgs["vx"].mean(axis=0), cgs["p"].mean(axis=0)
        rec = _l1_triple(nm, vm, pm, n_ex, v_ex, p_ex, SOD_NL)
        jdev = float(np.max(np.abs(cgs["n"] - cgs["n"].mean(axis=0, keepdims=True)))) / SOD_NL
        rec.update(
            case="2D-A",
            title="Sod extruded along x1 (2D field + lineout)",
            riemann=riemann,
            t=float(run.t),
            nstep=nstep,
            coord=x,
            num=(nm, vm, pm),
            exact=(n_ex, v_ex, p_ex),
            x0=SOD_X0,
            xlab=r"$x_1$",
            field2d=cgs["n"],
            vx2d=cgs["vx"],
            vy2d=cgs["vy"],
            x=x,
            y=y,
            jdev=jdev,
            vn_name="v_x",
        )
        return rec
    finally:
        run.close()


def run_2d_b(riemann: str) -> dict:
    cfg = make_2d_b(riemann)
    run = xmf.setup(cfg)
    try:
        y0 = 5.5
        n2r = MASS * AMU / run._cfg.units.density

        def ic(x, y):
            left = y < y0
            n = SOD_NL if left else SOD_NR
            p = SOD_PL if left else SOD_PR
            return (n * n2r, 0.0, 0.0, 0.0, p)

        pack_from_xy(run, ic)
        nstep = run_to_tstop(run)
        code = unpack_code(run)
        cgs = cgs_from_code(run, code)
        x, y = cgs["x"], cgs["y"]
        n_ex, v_ex, p_ex = exact_hi(
            y, run.t, y0, SOD_NL, SOD_NR, 0.0, 0.0, SOD_PL, SOD_PR,
            gamma=GAMMA_TUBE,
        )
        nm, vm, pm = cgs["n"].mean(axis=1), cgs["vy"].mean(axis=1), cgs["p"].mean(axis=1)
        rec = _l1_triple(nm, vm, pm, n_ex, v_ex, p_ex, SOD_NL)
        idev = float(np.max(np.abs(cgs["n"] - cgs["n"].mean(axis=1, keepdims=True)))) / SOD_NL
        rec.update(
            case="2D-B",
            title="Sod along x2 (2D field + lineout)",
            riemann=riemann,
            t=float(run.t),
            nstep=nstep,
            coord=y,
            num=(nm, vm, pm),
            exact=(n_ex, v_ex, p_ex),
            x0=y0,
            xlab=r"$x_2$",
            field2d=cgs["n"],
            vx2d=cgs["vx"],
            vy2d=cgs["vy"],
            x=x,
            y=y,
            idev=idev,
            vn_name="v_y",
            max_vy=float(np.max(np.abs(cgs["vy"]))),
        )
        return rec
    finally:
        run.close()


RIEMANN_RUNNERS = {
    "1D-A": run_1d_a,
    "1D-B": run_1d_b,
    "2D-A": run_2d_a,
    "2D-B": run_2d_b,
}


# ---------------------------------------------------------------------------
# Classic 2D HD (literature setups; no simple exact 2D Riemann solution)
# ---------------------------------------------------------------------------


def _classic_base(riemann: str, n: int, tstop: float, gamma: float) -> dict:
    return base_cfg(
        n=[n, n, 1],
        ndim=2,
        riemann=riemann,
        tstop=tstop,
        x1=(0.0, 1.0),
        x2=(0.0, 1.0),
        boundary={"x1": "copy", "x2": "copy"},
        gamma=gamma,
        cfl=0.4,
        units="code",
    )


def run_riemann2d(riemann: str, n: int = 128) -> dict:
    """Lax & Liu (1998) Configuration 3; also Kurganov & Tadmor (2002).

    Four-shock 2D Riemann on the unit square, γ = 1.4, t = 0.3.
    Quadrants (x>0.5, y>0.5) / (x<0.5, y>0.5) / (x<0.5, y<0.5) / (x>0.5, y<0.5):
      (1.5, 0, 0, 1.5), (0.5323, 1.206, 0, 0.3),
      (0.138, 1.206, 1.206, 0.029), (0.5323, 0, 1.206, 0.3).
    """
    cfg = _classic_base(riemann, n, 0.3, 1.4)
    run = xmf.setup(cfg)
    try:
        def ic(x, y):
            right, up = x >= 0.5, y >= 0.5
            if right and up:
                return (1.5, 0.0, 0.0, 0.0, 1.5)
            if (not right) and up:
                return (0.5323, 1.206, 0.0, 0.0, 0.3)
            if (not right) and (not up):
                return (0.138, 1.206, 1.206, 0.0, 0.029)
            return (0.5323, 0.0, 1.206, 0.0, 0.3)

        pack_from_xy(run, ic)
        nstep = run_to_tstop(run)
        code = unpack_code(run)
        return {
            "case": "C-R2D",
            "title": "2D Riemann Config. 3 (Lax & Liu 1998)",
            "cite": "Lax & Liu, SIAM J. Sci. Comput. 19, 319 (1998); "
                    "Kurganov & Tadmor, Numer. Methods PDEs 18, 584 (2002)",
            "riemann": riemann,
            "t": float(run.t),
            "nstep": nstep,
            "gamma": 1.4,
            **code,
        }
    finally:
        run.close()


def run_sedov(riemann: str, n: int = 128) -> dict:
    """2D cylindrical Sedov–Taylor blast (Sedov 1959).

    Ambient ρ=1, p=1e-6; E=1 deposited in r < 3.5 Δx at the origin of
    [-0.5, 0.5]². First-order PCM+RK1 will smear the shell; the check is
    whether the shock stays circular (direction-split isotropy).
    """
    cfg = base_cfg(
        n=[n, n, 1],
        ndim=2,
        riemann=riemann,
        tstop=0.1,
        x1=(-0.5, 0.5),
        x2=(-0.5, 0.5),
        boundary={"x1": "copy", "x2": "copy"},
        gamma=1.4,
        cfl=0.3,
        units="code",
    )
    run = xmf.setup(cfg)
    try:
        dx = 1.0 / n
        r0 = 3.5 * dx
        energy = 1.0
        p_in = (1.4 - 1.0) * energy / (np.pi * r0 * r0)
        p_out = 1.0e-6

        def ic(x, y):
            r2 = x * x + y * y
            return (1.0, 0.0, 0.0, 0.0, p_in if r2 <= r0 * r0 else p_out)

        pack_from_xy(run, ic)
        nstep = run_to_tstop(run)
        code = unpack_code(run)
        x, y = code["x"], code["y"]
        xx, yy = np.meshgrid(x, y, indexing="xy")
        # unpack is (ny, nx) with j = y, i = x  — rho[j, i]
        rr = np.sqrt(xx * xx + yy * yy)
        rho = code["rho"]
        # shock radius vs angle: radius of max |∂ρ/∂r| in 16 sectors
        theta = np.arctan2(yy, xx)
        radii = []
        for k in range(16):
            lo, hi = -np.pi + k * np.pi / 8, -np.pi + (k + 1) * np.pi / 8
            m = (theta >= lo) & (theta < hi) & (rr > 2.0 * r0)
            if not np.any(m):
                continue
            rbin = rr[m]
            dbin = rho[m]
            order = np.argsort(rbin)
            rbin, dbin = rbin[order], dbin[order]
            if len(rbin) < 8:
                continue
            drho = np.abs(np.gradient(dbin, rbin))
            radii.append(float(rbin[int(np.argmax(drho))]))
        radii = np.asarray(radii, dtype=float)
        circ = float(np.std(radii) / np.mean(radii)) if len(radii) else float("nan")
        return {
            "case": "C-SEDOV",
            "title": "2D Sedov–Taylor blast (cylindrical)",
            "cite": "Sedov, Similarity and Dimensional Methods in Mechanics (1959); "
                    "standard code-paper blast, e.g. Fryxell et al. 2000 (FLASH)",
            "riemann": riemann,
            "t": float(run.t),
            "nstep": nstep,
            "gamma": 1.4,
            "shock_R": float(np.mean(radii)) if len(radii) else float("nan"),
            "circularity": circ,
            "p_in": float(p_in),
            **code,
        }
    finally:
        run.close()


def run_kh(riemann: str, n: int = 128) -> dict:
    """Kelvin–Helmholtz shear layer, McNally et al. (2012) style.

    Two interfaces at y = 0.25 and 0.75; ρ = 2 / 1, |Δv_x| = 1, p = 2.5,
    γ = 5/3, single-mode v_y seed. No periodic face in this kernel: x
    boundaries are copy, so the run is stopped at t = 0.8 before the
    primary roll-up hits the sides. PCM+RK1 is very dissipative — this
    case is a dissipation diagnostic, not a high-Re KH paper figure.
    """
    cfg = _classic_base(riemann, n, 0.8, 5.0 / 3.0)
    run = xmf.setup(cfg)
    try:
        def ic(x, y):
            # smooth two-stream profile (McNally+2012 eq. 1–4, L=1)
            w0 = 0.05
            rho = 1.0 + 0.5 * (
                np.tanh((y - 0.25) / w0) - np.tanh((y - 0.75) / w0)
            )
            vx = 0.5 * (
                np.tanh((y - 0.25) / w0) - np.tanh((y - 0.75) / w0) - 1.0
            )
            vy = 0.01 * np.sin(4.0 * np.pi * x) * (
                np.exp(-((y - 0.25) / 0.2) ** 2)
                + np.exp(-((y - 0.75) / 0.2) ** 2)
            )
            return (float(rho), float(vx), float(vy), 0.0, 2.5)

        pack_from_xy(run, ic)
        nstep = run_to_tstop(run)
        code = unpack_code(run)
        # kinetic energy in the transverse velocity as a cheap growth proxy
        ke_y = float(np.mean(0.5 * code["rho"] * code["vy"] ** 2))
        return {
            "case": "C-KH",
            "title": "Kelvin–Helmholtz shear layer",
            "cite": "McNally, Lyra & Passy, ApJS 201, 18 (2012); "
                    "Robertson et al., MNRAS 401, 2463 (2010)",
            "riemann": riemann,
            "t": float(run.t),
            "nstep": nstep,
            "gamma": 5.0 / 3.0,
            "ke_y": ke_y,
            **code,
        }
    finally:
        run.close()


CLASSIC_RUNNERS = {
    "C-R2D": run_riemann2d,
    "C-SEDOV": run_sedov,
    "C-KH": run_kh,
}


# ---------------------------------------------------------------------------
# Axisymmetric 2D: cylindrical (R,z) and spherical (r,θ). n_φ = 1.
# The A4 operator still updates with dt/dx, not A/dV. These cases lock the
# IC / BC / figures; the metric operator is a later slice.
# ---------------------------------------------------------------------------


def run_cyl_rz(riemann: str = "hllc", n: int = 96,
               conservation: bool = False) -> dict:
    """Cylindrical (R, z) Sedov, φ-symmetric (n_φ = 1).

    Point blast at the origin of the meridional plane. With a metric
    operator this is the 3-D spherical Sedov (R ∝ t^{2/5}); with the
    present Cartesian sweep it is the 2-D planar Sedov (R ∝ t^{1/2}).
    Axis R = 0 is reflect.

    When *conservation* is True, all boundaries become reflect (closed box)
    and total mass / energy are recorded every 10 steps.
    """
    gamma = 1.4
    if conservation:
        bnd = {"x1beg": "reflect", "x1end": "reflect", "x2": "reflect"}
    else:
        bnd = {"x1beg": "reflect", "x1end": "copy", "x2": "copy"}
    cfg = base_cfg(
        n=[n, 2 * n, 1],
        ndim=2,
        riemann=riemann,
        tstop=0.1,
        x1=(0.0, 0.5),
        x2=(-0.5, 0.5),
        boundary=bnd,
        gamma=gamma,
        cfl=0.3,
        units="code",
        geometry="cylindrical",
    )
    run = xmf.setup(cfg)
    try:
        dx = 0.5 / n
        r0 = 3.5 * dx
        p_in = (gamma - 1.0) * 1.0 / (np.pi * r0 * r0)
        p_out = 1.0e-6

        def ic(R, z):
            return (1.0, 0.0, 0.0, 0.0, p_in if (R * R + z * z) <= r0 * r0 else p_out)

        pack_from_xy(run, ic)

        # Conservation recording (B3)
        cons_history = []
        if conservation:
            from kernel import compute_totals
            m0, e0 = compute_totals(run, gamma)
            cons_history.append((0, 0.0, m0, e0))
            record_every = 10

        nstep = 0
        while run.t < run.tstop:
            run.step()
            nstep += 1
            if conservation and nstep % record_every == 0:
                from kernel import compute_totals
                m, e = compute_totals(run, gamma)
                cons_history.append((nstep, float(run.t), m, e))

        # Always record final state for conservation
        if conservation:
            from kernel import compute_totals as _ct
            m_final, e_final = _ct(run, gamma)
            if not cons_history or cons_history[-1][0] != nstep:
                cons_history.append((nstep, float(run.t), m_final, e_final))

        code = unpack_code(run)
        R, z = code["x"], code["y"]
        RR, ZZ = np.meshgrid(R, z, indexing="xy")
        rr = np.sqrt(RR * RR + ZZ * ZZ)
        rho = code["rho"]
        theta = np.arctan2(ZZ, RR)
        radii = []
        for k in range(16):
            lo, hi = -np.pi + k * np.pi / 8, -np.pi + (k + 1) * np.pi / 8
            m = (theta >= lo) & (theta < hi) & (rr > 2.0 * r0)
            if not np.any(m):
                continue
            rbin = rr[m]
            dbin = rho[m]
            order = np.argsort(rbin)
            rbin, dbin = rbin[order], dbin[order]
            if len(rbin) < 8:
                continue
            # unique radii to avoid np.gradient divide-by-zero on duplicates
            rbin, inv = np.unique(rbin, return_index=True)
            dbin = dbin[inv]
            if len(rbin) < 8:
                continue
            drho = np.abs(np.gradient(dbin, rbin))
            radii.append(float(rbin[int(np.argmax(drho))]))
        radii = np.asarray(radii, dtype=float)
        circ = float(np.std(radii) / np.mean(radii)) if len(radii) else float("nan")
        rec = {
            "case": "G-CYL",
            "title": "Cylindrical (R,z) Sedov, φ-symmetric",
            "cite": "axisymmetric Sedov in cylindrical coordinates; "
                    "metric operator would recover the 3-D spherical blast",
            "riemann": riemann,
            "geometry": "cylindrical",
            "t": float(run.t),
            "nstep": nstep,
            "gamma": gamma,
            "shock_R": float(np.mean(radii)) if len(radii) else float("nan"),
            "circularity": circ,
            "max_vphi": float(np.max(np.abs(code["vz"]))),
            **code,
        }
        if conservation:
            rec["conservation"] = cons_history
            rec["boundary"] = "reflect-all (closed box)"
        return rec
    finally:
        run.close()


def run_sph_rth(riemann: str = "hllc", nr: int = 96, nth: int = 48,
                conservation: bool = False) -> dict:
    """Spherical (r, θ) radial blast, φ-symmetric (n_φ = 1).

    High-pressure ball at the origin on r ∈ [0.15, 1.15], θ ∈ [0, π].
    A correct spherical operator dilutes along r (A ∝ r²); the present
    Cartesian sweep evolves each θ-row as a 1-D Cartesian tube. Axis
    θ = 0, π are reflect. Diagnostic: density variation across θ.

    When *conservation* is True, r boundaries become reflect (closed box:
    inner + outer walls) and total mass / energy are recorded every 10 steps.
    """
    gamma = 1.4
    if conservation:
        bnd = {"x1": "reflect", "x2": "reflect"}
    else:
        bnd = {"x1": "copy", "x2": "reflect"}
    cfg = base_cfg(
        n=[nr, nth, 1],
        ndim=2,
        riemann=riemann,
        tstop=0.12,
        x1=(0.15, 1.15),
        x2=(0.0, np.pi),
        boundary=bnd,
        gamma=gamma,
        cfl=0.3,
        units="code",
        geometry="spherical",
    )
    run = xmf.setup(cfg)
    try:
        r_disc = 0.45

        def ic(r, th):
            inside = r < r_disc
            return (1.0 if inside else 0.125, 0.0, 0.0, 0.0, 1.0 if inside else 0.1)

        pack_from_xy(run, ic)

        # Conservation recording (B3)
        cons_history = []
        if conservation:
            from kernel import compute_totals
            m0, e0 = compute_totals(run, gamma)
            cons_history.append((0, 0.0, m0, e0))
            record_every = 10

        nstep = 0
        while run.t < run.tstop:
            run.step()
            nstep += 1
            if conservation and nstep % record_every == 0:
                from kernel import compute_totals
                m, e = compute_totals(run, gamma)
                cons_history.append((nstep, float(run.t), m, e))

        # Always record final state
        if conservation:
            from kernel import compute_totals as _ct
            m_final, e_final = _ct(run, gamma)
            if not cons_history or cons_history[-1][0] != nstep:
                cons_history.append((nstep, float(run.t), m_final, e_final))

        code = unpack_code(run)
        rho = code["rho"]
        mean = rho.mean(axis=0, keepdims=True)
        theta_dev = float(np.max(np.abs(rho - mean) / (mean + 1e-30)))
        rec = {
            "case": "G-SPH",
            "title": "Spherical (r,θ) radial blast, φ-symmetric",
            "cite": "2-D axisymmetric spherical blast; radial Sod-like ball "
                    "(Toro / code-paper spherical shock-tube family)",
            "riemann": riemann,
            "geometry": "spherical",
            "t": float(run.t),
            "nstep": nstep,
            "gamma": gamma,
            "r_disc": r_disc,
            "theta_dev": theta_dev,
            "max_vth": float(np.max(np.abs(code["vy"]))),
            "max_vphi": float(np.max(np.abs(code["vz"]))),
            **code,
        }
        if conservation:
            rec["conservation"] = cons_history
            rec["boundary"] = "reflect-all (closed box)"
        return rec
    finally:
        run.close()


GEOM_RUNNERS = {
    "G-CYL": run_cyl_rz,
    "G-SPH": run_sph_rth,
}
