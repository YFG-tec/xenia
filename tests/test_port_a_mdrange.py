"""Port A contract tests: MDRange parallel skeleton + MHD payload.

Verification of the MDRange<Rank<3>> iteration space conversion for all four
sweep kernels. Tests are independent of M-block (M1-M6) and existing Kokkos
serial tests.

Initial values and boundaries are stated in each test docstring. Conservation
regression is only asserted on the IC+BC stated here; reflect/closed box does
NOT assert momentum conservation (wall impulse); periodic box asserts mass
conservation (discrete FV exact) and energy boundedness.

Pin values source: measured 2026-08-31 on the post-MDRange-conversion code
(probe: .tmp/port_a_probe_flat.py). Upper bounds = measured x 1.5 rounded
up (same convention as M4/M5).

Measured values (2026-08-31):
  2D contact (HLLC, 32x8x1, 20 steps): drift = 0.0, L1(n) change = 0.0,
    max|vx| = 0.0 (discrete steady state, bitwise).
  3D contact (HLL, 32x4x4, 20 steps): drift = -2.43e-2 (0.086 cells,
    dx1 = 0.281).
  Conservation (periodic Sod, 32x8x1, 10 steps): mass_drift = 0.0 (exact),
    energy_drift = 1.56e-16 (machine epsilon).
  Degenerate axis (1D vs 2D ny=1, 64x1x1, 20 steps): all 5 slots bitwise
    identical, t and dt bitwise identical.
  2D MHD OT smoke (16x16, 5 steps): max_divb = 3.58e-11.

Test list:
  * 2D contact discontinuity (mag off): density step + u=0 + grad P=0,
    HLLC (discrete steady state), contact position drift == 0 bitwise.
  * 3D contact discontinuity (mag off): thin 3D (32x4x4) same IC, HLL.
  * 2D idle component: vz stays bitwise 0 in 2D Sod along x1 (HLL, 20 steps).
  * 1D idle components: vy/vz stay bitwise 0 in 1D Sod (advance to tstop).
  * 2D Sod conservation (periodic box): total mass exactly conserved (0.0),
    energy drift < 1e-14 (machine epsilon).
  * Degenerate axis: 1D (ndim=1) and 2D-ny=1 (ndim=2) produce bitwise
    identical state after 20 steps; vz stays 0 in both.
  * 2D MHD smoke: small Orszag-Tang (16x16), MHD on, 5 steps, divB < 1e-10.

Prohibition self-check (docs/TODO.md S-A + SESSION.md):
  - No pluto.ini read; no XUV_FACTOR==0 sentinel; not single-component only.
  - Five fluid slots (n, vx, vy, vz, P) + B1/B2/B3 for MHD tests.
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU, KB

_PTR = ctypes.POINTER(ctypes.c_double)

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

ION_FLUIDS = [{"name": "HII", "mass": 1.0, "charge": 1, "kind": "ion"}]

# Contact discontinuity parameters (same convention as test_hydro.py)
N_L, N_R = 1.0e13, 1.25e12
P0 = 13.806505  # dyn/cm^2, same both sides => grad P = 0 bitwise
X0 = 5.5        # midpoint of default shock_tube domain [1, 10]


@pytest.fixture(scope="module")
def lib():
    """Shared native library."""
    return load()


# --- Helpers ------------------------------------------------------------------


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _unpack(lib, run, f, var):
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, f, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _pack(lib, run, f, var, arr):
    data = np.ascontiguousarray(arr, dtype=np.float64)
    ncells = lib.xmf_state_ncells(run._h)
    assert data.size == ncells
    assert lib.xmf_state_pack(run._h, f, var, data.ctypes.data_as(_PTR)) == ncells


def _ptr(a):
    return a.ctypes.data_as(_PTR)


def _ntot_of(run):
    n = run._cfg.n
    ng = run._cfg.ng
    return tuple(n[d] + 2 * ng for d in range(3))


def _len_b(ntot, comp):
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return (ntot[comp] + 1) * ntot[d1] * ntot[d2]


def _pack_b(lib, h, comp, blk):
    got = lib.xmf_mag_pack(h, comp, _ptr(np.ascontiguousarray(blk.ravel())))
    assert got == blk.size, f"xmf_mag_pack B[{comp}] failed ({got})"


def _iso_position(x_int, n_int, mid):
    """Density mid-isosurface position: first crossing by linear interpolation."""
    below = np.where(n_int < mid)[0]
    assert len(below) > 0, "profile should contain values below mid"
    i = int(below[0])
    assert i > 0, "isosurface should not be at leftmost cell"
    frac = (mid - n_int[i - 1]) / (n_int[i] - n_int[i - 1])
    return float(x_int[i - 1] + frac * (x_int[i] - x_int[i - 1]))


# --- 1. 2D contact discontinuity does not move (mag off) ---------------------


def test_2d_contact_discontinuity_does_not_move(lib):
    """2D density step + u=0 + grad P=0 (mag off): contact should not move.

    IC: Riemann step at x0=5.5, left (n=N_L, v=0, p=P0), right (n=N_R, v=0, p=P0).
    BC: copy (all directions).
    Numerics: hllc (discrete steady state for stationary contact).
    Grid: 32x8x1, ndim=2.
    Steps: 20.

    Assertion (2026-08-31 measured: all zero, discrete steady state):
      - Contact isosurface drift == 0 (bitwise).
      - L1(n) change == 0 (density does not evolve at all).
      - max|vx| == 0 (no spurious flow generated).
    HLLC is a discrete steady state for SM=0: mass and energy fluxes at the
    contact interface are zero, so density should not evolve at all.
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [32, 8, 1]
    cfg["grid"]["ndim"] = 2
    cfg["numerics"]["riemann"] = "hllc"
    cfg["boundary"] = {"x1": "copy", "x2": "copy"}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": X0,
        "left": {"n": N_L, "v": 0.0, "p": P0},
        "right": {"n": N_R, "v": 0.0, "p": P0},
    }
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        n2r = run._cfg.mass * AMU / run._cfg.units.density

        n_before = _unpack(lib, run, 0, SLOT_N).reshape(nzt, nyt, nxt)

        for _ in range(20):
            run.step()

        n_after = _unpack(lib, run, 0, SLOT_N).reshape(nzt, nyt, nxt)

        # Average over y (j direction) to get 1D profile
        row_b = n_before[ng, ng:ng + ny, ng:ng + nx].mean(axis=0)
        row_a = n_after[ng, ng:ng + ny, ng:ng + nx].mean(axis=0)
        x = np.empty(nxt, dtype=np.float64)
        lib.xmf_mesh_copy_x(run._h, 0, _ptr(x))
        x_inner = x[ng:ng + nx]

        mid = 0.5 * (N_L * n2r + N_R * n2r)
        iso0 = _iso_position(x_inner, row_b, mid)
        iso1 = _iso_position(x_inner, row_a, mid)

        # HLLC discrete steady state: drift, L1 change, vx all bitwise 0
        assert iso1 == iso0, (
            f"2D HLLC contact drift = {iso1 - iso0:+.3e} (should be bitwise 0)"
        )
        assert np.array_equal(row_a, row_b), (
            f"2D HLLC contact: density evolved (should be bitwise frozen)"
        )
        vx = _unpack(lib, run, 0, SLOT_VX)
        assert np.all(vx == 0.0), (
            f"2D HLLC contact: max|vx|={float(np.max(np.abs(vx))):.3e} "
            f"(should be bitwise 0, no spurious flow)"
        )
    finally:
        run.close()


# --- 2. Thin 3D contact discontinuity does not move (mag off) -----------------


def test_3d_contact_discontinuity_does_not_move(lib):
    """Thin 3D density step + u=0 + grad P=0 (mag off): contact does not move.

    IC: same Riemann step at x0=5.5.
    BC: copy (all directions).
    Numerics: hll (allows smearing, but center should not drift > 1 cell).
    Grid: 32x4x4, ndim=3.
    Steps: 20.

    Assertion (2026-08-31 measured: drift = -2.43e-2 = 0.086 cells):
      contact drift < 1 cell (HLL smears but center holds, generous bound).
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [32, 4, 4]
    cfg["grid"]["ndim"] = 3
    cfg["boundary"] = {"x1": "copy", "x2": "copy", "x3": "copy"}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": X0,
        "left": {"n": N_L, "v": 0.0, "p": P0},
        "right": {"n": N_R, "v": 0.0, "p": P0},
    }
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        dx1 = (run._cfg.x1max - run._cfg.x1min) / nx
        n2r = run._cfg.mass * AMU / run._cfg.units.density

        n_before = _unpack(lib, run, 0, SLOT_N).reshape(nzt, nyt, nxt)

        for _ in range(20):
            run.step()

        n_after = _unpack(lib, run, 0, SLOT_N).reshape(nzt, nyt, nxt)

        # Average over y and z for 1D profile
        row_b = n_before[ng:ng + nz, ng:ng + ny, ng:ng + nx].mean(axis=(0, 1))
        row_a = n_after[ng:ng + nz, ng:ng + ny, ng:ng + nx].mean(axis=(0, 1))
        x = np.empty(nxt, dtype=np.float64)
        lib.xmf_mesh_copy_x(run._h, 0, _ptr(x))
        x_inner = x[ng:ng + nx]

        mid = 0.5 * (N_L * n2r + N_R * n2r)
        iso0 = _iso_position(x_inner, row_b, mid)
        iso1 = _iso_position(x_inner, row_a, mid)

        assert abs(iso1 - iso0) <= 1.0 * dx1, (
            f"3D contact drift {iso1 - iso0:+.3e} > 1 cell ({dx1:.3e})"
        )
        # Step should still be distinguishable
        assert float(np.max(row_a)) > mid > float(np.min(row_a)), (
            "contact should still be visible after 20 steps"
        )
    finally:
        run.close()


# --- 3. 2D idle component vz stays bitwise 0 ---------------------------------


def test_2d_idle_vz_remains_zero(lib):
    """2D Sod along x1 (mag off): vz (slot 3) stays bitwise 0 in full block.

    IC: Sod Riemann step at x0=5.5, left (N_L, 0, P_L), right (N_R, 0, P_R).
    BC: copy x1, reflect x2.
    Numerics: hll/pcm/rk1 (default shock_tube).
    Grid: 32x8x1, ndim=2.
    Steps: 20.

    In 2D (x1-x2 plane), vz is always transverse to both sweep directions:
    - x1 sweep: vz is transverse, only advected with mass flux (zero IC => 0)
    - x2 sweep: vz is transverse, same logic
    Hence vz must stay bitwise 0 in the full block (including ghosts).
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [32, 8, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    run = xmf.setup(cfg)
    try:
        for _ in range(20):
            run.step()

        vz = _unpack(lib, run, 0, SLOT_VZ)
        assert np.all(vz == 0.0), (
            f"2D vz should be bitwise 0, max|vz|={float(np.max(np.abs(vz))):.3e}"
        )
    finally:
        run.close()


# --- 4. 1D idle components vy/vz stay bitwise 0 ------------------------------


def test_1d_idle_vy_vz_remain_zero(lib):
    """1D Sod (mag off): vy (slot 2) and vz (slot 3) stay bitwise 0.

    IC: default shock_tube() Riemann problem.
    BC: default (reflect x1).
    Numerics: hll/pcm/rk1 (default).
    Grid: 256x1x1 (default shock_tube).
    Steps: advance to tstop.

    In 1D, vy and vz are always transverse to the x1 sweep. Zero IC => zero
    transverse flux => bitwise 0 forever. Including ghost cells.
    """
    cfg = xmf.shock_tube()
    run = xmf.setup(cfg)
    try:
        run.advance()
        assert run.t >= run.tstop

        for var, name in ((SLOT_VY, "vy"), (SLOT_VZ, "vz")):
            buf = _unpack(lib, run, 0, var)
            assert np.all(buf == 0.0), (
                f"1D {name} should be bitwise 0 (incl. ghosts), "
                f"max|{name}|={float(np.max(np.abs(buf))):.3e}"
            )
    finally:
        run.close()


# --- 5. 2D Sod conservation in periodic box -----------------------------------


def test_2d_sod_conservation_periodic(lib):
    """2D periodic box with Sod Riemann IC: total mass conserved, energy bounded.

    IC: Sod step at x0=5.5, left (N_L, v=0, P_L=N_L*KB*1e4),
        right (N_R, v=0, P_R=N_R*KB*8e3).
    BC: periodic x1, periodic x2.
    Numerics: hll/pcm/rk1 (default).
    Grid: 32x8x1, ndim=2.
    Steps: 10.

    Conservation assertions (2026-08-31 measured):
      - Total mass sum(rho*dV) is exactly conserved (drift == 0.0) by discrete
        FV with periodic BC (telescoping fluxes).
      - Total energy drift < 1e-14 (measured 1.56e-16 = machine epsilon;
        Godunov + RK1 is conservative to roundoff for this smooth-enough IC).
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [32, 8, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "periodic", "x2": "periodic"}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": X0,
        "left": {"n": N_L, "v": 0.0, "p": N_L * KB * 1.0e4},
        "right": {"n": N_R, "v": 0.0, "p": N_R * KB * 8.0e3},
    }
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
        gamma = run._cfg.gamma

        def compute_totals():
            rho = _unpack(lib, run, 0, SLOT_N).reshape(nzt, nyt, nxt)
            vx = _unpack(lib, run, 0, SLOT_VX).reshape(nzt, nyt, nxt)
            vy = _unpack(lib, run, 0, SLOT_VY).reshape(nzt, nyt, nxt)
            vz = _unpack(lib, run, 0, SLOT_VZ).reshape(nzt, nyt, nxt)
            P = _unpack(lib, run, 0, SLOT_P).reshape(nzt, nyt, nxt)
            rho_i = rho[inner]
            mass = float(np.sum(rho_i))
            ke = 0.5 * rho_i * (vx[inner]**2 + vy[inner]**2 + vz[inner]**2)
            ie = P[inner] / (gamma - 1.0)
            energy = float(np.sum(ke + ie))
            return mass, energy

        mass0, energy0 = compute_totals()

        for _ in range(10):
            run.step()

        mass1, energy1 = compute_totals()

        # Mass: exact conservation (measured drift = 0.0 bitwise)
        assert mass1 == mass0, (
            f"Mass not exactly conserved: M0={mass0!r}, M1={mass1!r}, "
            f"diff={mass1 - mass0:.3e}"
        )

        # Energy: machine epsilon level (measured 1.56e-16)
        energy_drift = abs(energy1 - energy0) / energy0
        assert energy_drift < 1e-14, (
            f"Energy drift {energy_drift:.3e} > 1e-14 "
            f"(E0={energy0:.6e}, E1={energy1:.6e})"
        )
    finally:
        run.close()


# --- 6. Degenerate axis: 1D == 2D (ny=1) bitwise -----------------------------


def test_degenerate_axis_1d_matches_2d_ny1(lib):
    """1D (ndim=1) and 2D (ndim=2, ny=1) produce bitwise identical evolution.

    IC: Sod Riemann step at x0=5.5 (default shock_tube).
    BC: reflect x1 (default); copy x2 (2D only).
    Numerics: hll/pcm/rk1 (default).
    Grid: nx=64, ny=1 (2D) / ny=1 (1D).
    Steps: 20.

    MDRange skeleton: n_d=1 => t0 or t1 = 1, same kernel invocation.
    The y-sweep is skipped when ny=1 (nline_y = 1 <= 1). The x-sweep and CFL
    computation should be identical. All 5 slots and t/dt should match bitwise.
    """
    results = {}
    for ndim in (1, 2):
        cfg = xmf.shock_tube()
        cfg["grid"]["n"] = [64, 1, 1]
        cfg["grid"]["ndim"] = ndim
        if ndim == 2:
            cfg["boundary"] = {"x1": "reflect", "x2": "copy"}
        run = xmf.setup(cfg)
        try:
            for _ in range(20):
                run.step()
            slots = [_unpack(lib, run, 0, s).copy() for s in range(5)]
            results[ndim] = (run.t, run.dt, slots)
        finally:
            run.close()

    t1, dt1, s1 = results[1]
    t2, dt2, s2 = results[2]

    assert t1 == t2, f"t differs: 1D={t1!r} vs 2D-ny1={t2!r}"
    assert dt1 == dt2, f"dt differs: 1D={dt1!r} vs 2D-ny1={dt2!r}"

    for slot, name in enumerate(("n", "vx", "vy", "vz", "P")):
        assert np.array_equal(s1[slot], s2[slot]), (
            f"1D vs 2D-ny1 {name}: max|diff|="
            f"{float(np.max(np.abs(s1[slot] - s2[slot]))):.3e}"
        )

    # vz stays 0 in both (degenerate z-axis)
    assert np.all(s1[SLOT_VZ] == 0.0), "1D vz should be 0"
    assert np.all(s2[SLOT_VZ] == 0.0), "2D-ny1 vz should be 0"


# --- 7. 2D MHD smoke: small OT, CT still works under MDRange -----------------


def test_2d_mhd_small_ot_smoke(lib):
    """Small Orszag-Tang (16x16) MHD smoke: CT still works under MDRange.

    IC: OT vortex on [0, 2pi]^2, rho = gamma^2/(4pi), P = 5/(12pi),
        u = (-sin y, sin x, 0), B = (-sin y, sin 2x, 0) / sqrt(4pi).
    BC: periodic x1, periodic x2.
    Numerics: hll/plm/rk3, cfl=0.4.
    Grid: 16x16x1, ndim=2.
    MHD: on, 1 ion (HII, q=1, w_s=1), electron off.
    Steps: 5.

    Assertion (2026-08-31 measured: max_divb = 3.58e-11 over 5 steps):
      divB stays < 1e-10 (measured x 1.5 rounded up; CT keeps divergence at
      machine noise level). This pins that the MHD payload (M4 load embedded
      in MDRange skeleton) is correctly wired and CT (which is NOT converted
      to MDRange in Port A) still operates correctly on the updated B field.
    """
    gamma = 5.0 / 3.0
    rho = gamma * gamma / (4.0 * math.pi)
    P = 5.0 / (12.0 * math.pi)
    B0 = 1.0 / math.sqrt(4.0 * math.pi)
    nx = ny = 16
    L = 2.0 * math.pi

    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [nx, ny, 1]
    cfg["grid"]["ndim"] = 2
    cfg["grid"]["x1min"] = 0.0
    cfg["grid"]["x1max"] = L
    cfg["grid"]["x2min"] = 0.0
    cfg["grid"]["x2max"] = L
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "plm", "rk": "rk3"}
    cfg["gamma"] = gamma
    cfg["time"] = {"cfl": 0.4, "tstop": 0.5, "dt_max": 0.0}
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg["electron"] = {"mode": "off"}
    cfg["boundary"] = {"x1": "periodic", "x2": "periodic", "x3": "none"}
    cfg["mhd"] = True
    cfg["ic"] = {
        "kind": "riemann",
        "x0": L / 2.0,
        "left": {"n": rho, "v": 0.0, "p": P},
        "right": {"n": rho, "v": 0.0, "p": P},
    }

    run = xmf.setup(cfg)
    try:
        h = run._h
        ng = run._cfg.ng
        ntot = _ntot_of(run)

        # Read cell-centre coordinates
        nc0 = lib.xmf_mesh_ncell(h, 0)
        nc1 = lib.xmf_mesh_ncell(h, 1)
        xc0 = np.empty(nc0, dtype=np.float64)
        xc1 = np.empty(nc1, dtype=np.float64)
        lib.xmf_mesh_copy_x(h, 0, _ptr(xc0))
        lib.xmf_mesh_copy_x(h, 1, _ptr(xc1))

        # Pack fluid IC: rho uniform, vx = -sin(y), vy = sin(x), vz = 0, P uniform
        ncells = lib.xmf_state_ncells(h)
        rho_arr = np.full(ncells, rho, dtype=np.float64)
        vx_arr = np.empty(ncells, dtype=np.float64)
        vy_arr = np.empty(ncells, dtype=np.float64)
        vz_arr = np.zeros(ncells, dtype=np.float64)
        P_arr = np.full(ncells, P, dtype=np.float64)

        for idx in range(ncells):
            i = idx % ntot[0]
            j = (idx // ntot[0]) % ntot[1]
            x = xc0[i]
            y = xc1[j]
            vx_arr[idx] = -math.sin(y)
            vy_arr[idx] = math.sin(x)

        lib.xmf_state_pack(h, 0, 0, _ptr(rho_arr))
        lib.xmf_state_pack(h, 0, 1, _ptr(vx_arr))
        lib.xmf_state_pack(h, 0, 2, _ptr(vy_arr))
        lib.xmf_state_pack(h, 0, 3, _ptr(vz_arr))
        lib.xmf_state_pack(h, 0, 4, _ptr(P_arr))

        # Pack B IC: B1 = -sin(y)*B0, B2 = sin(2x)*B0, B3 = 0
        b1 = np.zeros((ntot[2], ntot[1], ntot[0] + 1), dtype=np.float64)
        for k in range(ntot[2]):
            for j in range(ntot[1]):
                y = xc1[j]
                for i in range(ntot[0] + 1):
                    b1[k, j, i] = -math.sin(y) * B0

        b2 = np.zeros((ntot[0], ntot[2], ntot[1] + 1), dtype=np.float64)
        for k in range(ntot[2]):
            for j in range(ntot[1] + 1):
                for i in range(ntot[0]):
                    x = xc0[i]
                    b2[i, k, j] = math.sin(2.0 * x) * B0

        b3 = np.zeros((ntot[1], ntot[0], ntot[2] + 1), dtype=np.float64)

        _pack_b(lib, h, 0, b1)
        _pack_b(lib, h, 1, b2)
        _pack_b(lib, h, 2, b3)

        # Initial divB check
        divb0 = run.divb_max
        assert divb0 < 1e-12, f"initial divb_max={divb0:.3e} (expect machine noise)"

        # Run 5 steps
        max_divb = divb0
        for step in range(5):
            run.step()
            d = run.divb_max
            if d > max_divb:
                max_divb = d

        # divB should stay < 1e-10 (measured 3.58e-11 x 1.5 rounded up)
        assert max_divb < 1e-10, (
            f"max_divb={max_divb:.3e} > 1e-10 after 5 steps "
            f"(CT may be broken under MDRange skeleton)"
        )
        assert run.t > 0.0, "simulation should have advanced in time"
    finally:
        run.close()
