"""Port C contract tests: device CFL bit-wise reference + fluid pin bit-wise.

Card: docs/SESSION.md = docs/cmd/PROMPT_PORT_C.md, "交什么" §4.

Tests:
  1a. CFL device bit-wise, HD gate CLOSED, non-uniform state (3D).
  1b. CFL device bit-wise, MHD gate OPEN, non-uniform state + B (3D).
  2a. Pin sentinel bit-wise over multiple steps, interior + ghost mask.
  2b. Pin does NOT touch P_e or B (bit-wise vs no-pin control).
  3.  Pin + user floor coexistence (no cross-talk).
  4.  Clear regression: never-loaded vs set-then-cleared, 1D empty table
      bit-wise identical (compare against test_kokkos_serial pins without
      touching them).
  5.  Pit-1 regression: test_dt_reasonable_3d_hd is in the suite and green.
  6.  Run.set_pin Python shape validation.

Host-path note (Windows OpenMP):
  On the Windows OpenMP build the default memory space IS HostSpace, so
  there is no separate storage to cross-check against. The device CFL
  path (compute_dt_device, parallel_reduce + Max<double>) is the only
  path exercised once run->dev is attached. The host serial loop is
  reachable only before xmf_dev_attach (i.e. during setup's own
  compute_dt call) or in a host-only build. The bit-wise reference
  below therefore nails the DEVICE path; the host path cannot be forced
  through any public lever (confirmed: no config key, no env var, no
  ctypes export toggles run->dev).

Grid note for bit-wise CFL:
  Power-of-2 active cell counts per direction (e.g. 8, 4, 4) are used
  so that dx = 1/n is exactly representable in IEEE 754 (0.125, 0.25)
  and the kernel's ghost-padding arithmetic (xf[i+1] - xf[i]) yields
  the EXACT same double for every cell. With non-power-of-2 counts
  (e.g. ny=6), the 1/6 face spacing is not exact and the ghost padding
  propagation produces up to 4 distinct dx values at ULP level, making
  bit-wise equality between Python and the kernel ill-defined (the two
  paths read the same array but the per-cell term s/dx lands on
  slightly different operands -- 1 ULP noise, not a formula bug).

Symbol binding note:
  The xmf_bc_set_pin binding gap in _native.py (once missing
  argtypes/restype, which caused an OverflowError on Windows 64-bit
  when ctypes defaulted to c_int for the void* handle) has been
  filled in _native.py load() (same treatment as xmf_bc_set_user).
  All tests below go through the public Run.set_pin port.

Conventions (docs/TODO.md §A):
  - One block per file.
  - 1D (nx2=nx3=1) AND thin 2D/3D covered.
  - No pluto.ini read, no XUV_FACTOR==0 sentinel, all five slots
    asserted (never VX1 alone).
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU

_PTR = ctypes.POINTER(ctypes.c_double)
_IPTR = ctypes.POINTER(ctypes.c_int)

GAMMA = 5.0 / 3.0
RHO_FLOOR = 1.0e-30
P_FLOOR = 1.0e-30
S_FLOOR = 1.0e-12

# Slot indices matching XMF_SLOT_* in the C kernel.
RHO, VX, VY, VZ, P = 0, 1, 2, 3, 4
SLOT_NAMES = ("n", "vx", "vy", "vz", "P")


# ---------------------------------------------------------------------------
# CFL reference helpers (Python re-implementation of cfl_cell_signal_sum)
# ---------------------------------------------------------------------------


def _bidx(comp, ntot, i, j, k):
    """Staggered face index of bf[comp], matching host_bidx / cfl_bidx.

    c[3] = {i, j, k}; d1 = (comp+1)%3; d2 = (comp+2)%3.
    n1 = ntot[d1] if d1 in {1,2} else nxtot -- see C.
    Returns (((c[d2] * n1) + c[d1]) * (nd+1)) + c[comp].
    """
    d1 = (comp + 1) % 3
    d2 = (comp + 2) % 3
    n1 = ntot[0] if d1 == 0 else ntot[1] if d1 == 1 else ntot[2]
    nd = ntot[0] if comp == 0 else ntot[1] if comp == 1 else ntot[2]
    c = [i, j, k]
    return ((c[d2] * n1) + c[d1]) * (nd + 1) + c[comp]


def _cell_b_center(b_blk, ntot, k, j, i):
    """Cell-centred B by face averaging, matching cell_b_center."""
    bc = np.empty(3, dtype=np.float64)
    for comp in range(3):
        c0 = _bidx(comp, ntot, i, j, k)
        ci = _bidx(comp, ntot, i + (1 if comp == 0 else 0),
                   j + (1 if comp == 1 else 0),
                   k + (1 if comp == 2 else 0))
        bc[comp] = 0.5 * (b_blk[comp][c0] + b_blk[comp][ci])
    return bc


def _cfl_reference(ncells, nd, dx, nfluid, charge, mass_code, gamma, ele,
                   mag, fld_all, pe, b_blk, ntot):
    """Python re-implementation of cfl_cell_signal_sum + max reduction.

    fld_all : ndarray (nfluid, 5, ncells) -- state in [f][slot][cell]
              flat layout, same as xmf_state_pack.
    pe      : ndarray (ncells,) or None.
    b_blk   : list of 3 flat arrays, or None when mag == 0.
    Returns inv_max (double) -- same quantity compute_dt divides cfl by.
    """
    inv_max = 0.0
    nxtot, nytot, nztot = ntot
    for k in range(nd[2]):
        for j in range(nd[1]):
            for i in range(nd[0]):
                ig = i + 2  # ng = 2
                jg = j + 2
                kg = k + 2
                c = (kg * nytot + jg) * nxtot + ig
                # cell_ne
                ne = 0.0
                if ele or mag:
                    for f in range(nfluid):
                        ne += charge[f] * fld_all[f, RHO, c] / mass_code[f]
                # cell_b_center
                bc = np.zeros(3, dtype=np.float64)
                if mag:
                    bc = _cell_b_center(b_blk, ntot, kg, jg, ig)
                # B3 gate
                nchg = 0
                ue = np.zeros(3, dtype=np.float64)
                rhoe = 0.0
                pe_use = 0.0
                if ele:
                    for f in range(nfluid):
                        if charge[f] > 0.0:
                            nchg += 1
                    if nchg >= 2:
                        ne = 0.0
                        mx = my = mz = 0.0
                        for f in range(nfluid):
                            if charge[f] == 0.0:
                                continue
                            ns = (charge[f] * fld_all[f, RHO, c]
                                  / mass_code[f])
                            ne += ns
                            mx += ns * fld_all[f, VX, c]
                            my += ns * fld_all[f, VY, c]
                            mz += ns * fld_all[f, VZ, c]
                        # cell_ionized
                        ionized = False
                        for f in range(nfluid):
                            if (charge[f] > 0.0
                                    and fld_all[f, RHO, c] > RHO_FLOOR):
                                ionized = True
                                break
                        if ionized:
                            ue[0] = mx / ne
                            ue[1] = my / ne
                            ue[2] = mz / ne
                            for f in range(nfluid):
                                if charge[f] == 0.0:
                                    continue
                                ns = (charge[f] * fld_all[f, RHO, c]
                                      / mass_code[f])
                                rhoe += (ns / ne) * fld_all[f, RHO, c]
                            pe_use = pe[c]
                inv = 0.0
                for d in range(3):
                    if nd[d] <= 1:
                        continue
                    var = VX + d
                    idx = (i, j, k)[d]
                    s = 0.0
                    for f in range(nfluid):
                        fn = fld_all[f, RHO, c]
                        fv = fld_all[f, var, c]
                        fp = fld_all[f, P, c]
                        puse = fp
                        if ele:
                            # cell_w
                            if (charge[f] <= 0.0 or ne <= 0.0
                                    or not any(
                                        charge[g] > 0.0
                                        and fld_all[g, RHO, c] > RHO_FLOOR
                                        for g in range(nfluid))):
                                w = 0.0
                            else:
                                w = (charge[f] * fn / mass_code[f] / ne)
                            puse = fp + w * pe[c]
                        if fn <= RHO_FLOOR or puse <= P_FLOOR:
                            continue
                        if mag and charge[f] > 0.0:
                            # cell_w (recompute as C does)
                            if (charge[f] <= 0.0 or ne <= 0.0
                                    or not any(
                                        charge[g] > 0.0
                                        and fld_all[g, RHO, c] > RHO_FLOOR
                                        for g in range(nfluid))):
                                w = 0.0
                            else:
                                w = (charge[f] * fn / mass_code[f] / ne)
                            if w < 0.0:
                                w = 0.0
                            if w > 1.0:
                                w = 1.0
                            b2 = w * (bc[0]**2 + bc[1]**2 + bc[2]**2) / fn
                            bn = bc[d]
                            bn2 = w * bn * bn / fn
                            a2 = gamma * puse / fn
                            if b2 <= 0.0:
                                cf2v = a2
                            else:
                                ssum = a2 + b2
                                disc = ssum * ssum - 4.0 * a2 * bn2
                                if disc < 0.0:
                                    disc = 0.0
                                cf2v = 0.5 * (ssum + math.sqrt(disc))
                            sig = abs(fv) + math.sqrt(cf2v)
                            if sig > s:
                                s = sig
                        else:
                            cs = math.sqrt(gamma * puse / fn)
                            if abs(fv) + cs > s:
                                s = abs(fv) + cs
                    if rhoe > RHO_FLOOR and pe_use > P_FLOOR:
                        ue_d = ue[d]
                        se = abs(ue_d) + math.sqrt(gamma * pe_use / rhoe)
                        if se > s:
                            s = se
                    if s <= S_FLOOR:
                        s = S_FLOOR
                    inv += s / dx[d][idx]
                if inv > inv_max:
                    inv_max = inv
    return inv_max


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ptr(a):
    return a.ctypes.data_as(_PTR)


def _iptr(a):
    return a.ctypes.data_as(_IPTR)


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _unpack_state(lib, h, nfluid, ncells):
    """Unpack every fluid's five slots. Returns (nfluid, 5, ncells)."""
    out = np.empty((nfluid, 5, ncells), dtype=np.float64)
    for f in range(nfluid):
        for s in range(5):
            buf = np.empty(ncells, dtype=np.float64)
            got = lib.xmf_state_unpack(h, f, s, _ptr(buf))
            assert got == ncells
            out[f, s] = buf
    return out


def _pack_state(lib, h, f, var, data):
    """Pack one fluid/slot block. data must have size == ncells."""
    got = lib.xmf_state_pack(h, f, var, _ptr(np.ascontiguousarray(data)))
    assert got == data.size


def _unpack_b_all(lib, h, ntot):
    """Unpack three B blocks (flat). Returns list of 3 flat arrays."""
    out = []
    for comp in range(3):
        d1, d2 = (comp + 1) % 3, (comp + 2) % 3
        nd = ntot[comp]
        n1 = ntot[d1]
        n2 = ntot[d2]
        nb = (nd + 1) * n1 * n2
        buf = np.empty(nb, dtype=np.float64)
        got = lib.xmf_mag_unpack(h, comp, _ptr(buf))
        assert got == nb
        out.append(buf)
    return out


def _pack_b(lib, h, comp, data):
    got = lib.xmf_mag_pack(h, comp, _ptr(np.ascontiguousarray(data)))
    assert got == data.size


def _copy_dx(lib, h, nd_tot):
    """Per-direction cell widths including ghosts. Returns list of 3."""
    out = []
    for d in range(3):
        buf = np.empty(nd_tot[d], dtype=np.float64)
        got = lib.xmf_mesh_copy_dx(h, d, _ptr(buf))
        assert got == nd_tot[d]
        out.append(buf)
    return out


# ---------------------------------------------------------------------------
# 1a. CFL device bit-wise: HD gate CLOSED, non-uniform state
# ---------------------------------------------------------------------------


def _cfl_hd_cfg():
    """3D HD box (gate closed). Power-of-2 counts for uniform dx.
    units=1 so code == physical."""
    cfg = xmf.shock_tube()
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["grid"]["n"] = [8, 4, 4]
    cfg["grid"]["ndim"] = 3
    cfg["grid"]["x1min"] = 0.0
    cfg["grid"]["x1max"] = 1.0
    cfg["grid"]["x2min"] = 0.0
    cfg["grid"]["x2max"] = 1.0
    cfg["grid"]["x3min"] = 0.0
    cfg["grid"]["x3max"] = 1.0
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm",
                       "rk": "rk1"}
    cfg["time"] = {"cfl": 0.4, "tstop": 10.0, "dt_max": 0.0}
    cfg["boundary"] = {"x1": "periodic", "x2": "periodic",
                       "x3": "periodic"}
    cfg["ic"] = {"kind": "riemann", "x0": 0.5,
                 "left": {"n": 1.0, "v": 0.0, "p": 1.0},
                 "right": {"n": 1.0, "v": 0.0, "p": 1.0}}
    return cfg


def test_cfl_device_bitwise_hd():
    """Device CFL bit-wise reference, HD gate closed, non-uniform state.

    Pack non-uniform density, three-component velocity and pressure
    (no B, electron off). Save the post-fill state, step once, compare
    run.dt against Python's per-cell re-implementation of
    cfl_cell_signal_sum + max reduction.

    On Windows OpenMP the device memory space IS HostSpace, so the
    device path's arithmetic is standard double. The Python reference
    uses the same operations in the same order; both produce IEEE 754
    results with no FMA (MinGW -O2 default does not contract across
    statements). The max reduction is order-independent.
    """
    lib = load()
    cfg = _cfl_hd_cfg()
    run = xmf.setup(cfg)
    try:
        h = run._h
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        ncells = lib.xmf_state_ncells(h)
        nfluid = lib.xmf_state_nfluid(h)
        assert nfluid == 1

        # Non-uniform density, velocity, pressure (code units).
        rng = np.random.RandomState(42)
        rho = 1.0 + 0.3 * rng.randn(ncells)
        rho = np.maximum(rho, 0.1)
        vx = 0.3 * rng.randn(ncells)
        vy = 0.2 * rng.randn(ncells)
        vz = 0.1 * rng.randn(ncells)
        prs = 1.0 + 0.2 * rng.randn(ncells)
        prs = np.maximum(prs, 0.1)
        _pack_state(lib, h, 0, RHO, rho)
        _pack_state(lib, h, 0, VX, vx)
        _pack_state(lib, h, 0, VY, vy)
        _pack_state(lib, h, 0, VZ, vz)
        _pack_state(lib, h, 0, P, prs)

        # Fill ghosts (device path).
        assert lib.xmf_fill_bc(h) == 0

        # Save the device state the CFL will see.
        fld_saved = _unpack_state(lib, h, nfluid, ncells)
        dx = _copy_dx(lib, h, (nxt, nyt, nzt))

        # Step: compute_dt runs on device from fld_saved.
        run.step()

        # Python reference.
        charge = np.zeros(nfluid, dtype=np.float64)
        mass_code = np.ones(nfluid, dtype=np.float64)
        inv_max = _cfl_reference(
            ncells, (nx, ny, nz), dx, nfluid, charge, mass_code,
            GAMMA, ele=False, mag=False,
            fld_all=fld_saved, pe=None, b_blk=None,
            ntot=(nxt, nyt, nzt))
        assert inv_max > 0.0, "inv_max must be positive (non-trivial state)"
        dt_ref = cfg["time"]["cfl"] / inv_max
        assert run.dt == dt_ref, (
            f"HD CFL bit-wise: run.dt={run.dt!r} != "
            f"dt_ref={dt_ref!r} (diff={run.dt - dt_ref:.3e})"
        )
    finally:
        run.close()


# ---------------------------------------------------------------------------
# 1b. CFL device bit-wise: MHD gate OPEN, non-uniform state + B
# ---------------------------------------------------------------------------


def _cfl_mhd_cfg():
    """3D MHD box (gate open). Single ion fluid, electron off.
    Power-of-2 counts."""
    cfg = xmf.shock_tube()
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["grid"]["n"] = [8, 4, 4]
    cfg["grid"]["ndim"] = 3
    cfg["grid"]["x1min"] = 0.0
    cfg["grid"]["x1max"] = 1.0
    cfg["grid"]["x2min"] = 0.0
    cfg["grid"]["x2max"] = 1.0
    cfg["grid"]["x3min"] = 0.0
    cfg["grid"]["x3max"] = 1.0
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm",
                       "rk": "rk1"}
    cfg["time"] = {"cfl": 0.4, "tstop": 10.0, "dt_max": 0.0}
    cfg["boundary"] = {"x1": "periodic", "x2": "periodic",
                       "x3": "periodic"}
    cfg["mhd"] = True
    cfg["fluids"] = [{"name": "HII", "mass": 1.0, "charge": 1,
                      "kind": "ion"}]
    cfg["electron"] = {"mode": "off"}
    cfg["ic"] = {"kind": "riemann", "x0": 0.5,
                 "left": {"n": 1.0, "v": 0.0, "p": 1.0},
                 "right": {"n": 1.0, "v": 0.0, "p": 1.0}}
    return cfg


def test_cfl_device_bitwise_mhd():
    """Device CFL bit-wise reference, MHD gate open, non-uniform state + B.

    Single ion fluid (charge=1), non-uniform rho/v/P and non-uniform
    three-component B. The fast-magnetosonic formula with face-averaged
    B is exercised. Same bit-wise contract as 1a.
    """
    lib = load()
    cfg = _cfl_mhd_cfg()
    run = xmf.setup(cfg)
    try:
        h = run._h
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        ncells = lib.xmf_state_ncells(h)
        nfluid = lib.xmf_state_nfluid(h)
        assert nfluid == 1

        rng = np.random.RandomState(137)
        rho = 1.0 + 0.3 * rng.randn(ncells)
        rho = np.maximum(rho, 0.1)
        vx = 0.3 * rng.randn(ncells)
        vy = 0.2 * rng.randn(ncells)
        vz = 0.1 * rng.randn(ncells)
        prs = 1.0 + 0.2 * rng.randn(ncells)
        prs = np.maximum(prs, 0.1)
        _pack_state(lib, h, 0, RHO, rho)
        _pack_state(lib, h, 0, VX, vx)
        _pack_state(lib, h, 0, VY, vy)
        _pack_state(lib, h, 0, VZ, vz)
        _pack_state(lib, h, 0, P, prs)

        # Non-uniform B (face-centred blocks).
        for comp in range(3):
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            nd = nxt if comp == 0 else nyt if comp == 1 else nzt
            n1 = nxt if d1 == 0 else nyt if d1 == 1 else nzt
            n2 = nxt if d2 == 0 else nyt if d2 == 1 else nzt
            nb = (nd + 1) * n1 * n2
            base = (0.4, 0.6, 0.8)[comp]
            b = base + 0.1 * rng.randn(nb)
            _pack_b(lib, h, comp, b)

        # Fill ghosts.
        assert lib.xmf_fill_bc(h) == 0

        # Save state + B.
        fld_saved = _unpack_state(lib, h, nfluid, ncells)
        b_saved = _unpack_b_all(lib, h, (nxt, nyt, nzt))
        dx = _copy_dx(lib, h, (nxt, nyt, nzt))

        run.step()

        charge = np.array([1.0], dtype=np.float64)
        mass_code = np.array([1.0], dtype=np.float64)
        inv_max = _cfl_reference(
            ncells, (nx, ny, nz), dx, nfluid, charge, mass_code,
            GAMMA, ele=False, mag=True,
            fld_all=fld_saved, pe=None, b_blk=b_saved,
            ntot=(nxt, nyt, nzt))
        assert inv_max > 0.0, "inv_max must be positive (MHD state)"
        dt_ref = cfg["time"]["cfl"] / inv_max
        assert run.dt == dt_ref, (
            f"MHD CFL bit-wise: run.dt={run.dt!r} != "
            f"dt_ref={dt_ref!r} (diff={run.dt - dt_ref:.3e})"
        )
    finally:
        run.close()


# ---------------------------------------------------------------------------
# 2a. Pin sentinel bit-wise over multiple steps (interior + ghost mask)
# ---------------------------------------------------------------------------


def test_pin_sentinel_bitwise_over_steps():
    """Pin sentinel over 3 steps: every masked cell (interior + ghost)
    keeps its five-slot sentinel value after each step.

    Config: 1D HD, reflect x1 (so ghost cells have defined values from
    fill_bc, then the pin overrides them). Mask covers some interior
    cells AND some ghost cells.
    """
    lib = load()
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [32, 1, 1]
    cfg["grid"]["ndim"] = 1
    cfg["time"] = {"cfl": 0.4, "tstop": 10.0, "dt_max": 0.0}
    cfg["ic"] = {"kind": "riemann", "x0": 5.5,
                 "left": {"n": 1.0e13, "v": 0.0, "T": 1.0e4},
                 "right": {"n": 1.25e12, "v": 0.0, "T": 8.0e3}}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        ncells = lib.xmf_state_ncells(h)
        nfluid = lib.xmf_state_nfluid(h)

        # Build mask: pin cells i=0..3 (two ghost + two interior) and
        # i=nxt-4..nxt-1 (two interior + two ghost). Plus one interior
        # cell in the middle.
        mask = np.zeros((nzt, nyt, nxt), dtype=np.int32)
        pin_indices = (list(range(0, 4)) + list(range(nxt - 4, nxt))
                       + [nxt // 2])
        for flat_idx in pin_indices:
            mask[0, 0, flat_idx] = 1

        # Sentinel values: different per slot so we can tell them apart.
        vals = np.empty((nfluid, 5, nzt, nyt, nxt), dtype=np.float64)
        for f in range(nfluid):
            for s in range(5):
                sentinel = 100.0 * (f * 5 + s + 1) + 7.0
                vals[f, s] = np.full((nzt, nyt, nxt), sentinel)

        run.set_pin(mask, vals)

        # Run 3 steps, check after each.
        for step_i in range(3):
            run.step()
            # Unpack and verify every masked cell.
            fld = _unpack_state(lib, h, nfluid, ncells)
            flat = fld.reshape(nfluid, 5, ncells)
            mask_flat = mask.ravel()
            for f in range(nfluid):
                for s in range(5):
                    sentinel = 100.0 * (f * 5 + s + 1) + 7.0
                    pinned = flat[f, s][mask_flat == 1]
                    assert np.all(pinned == sentinel), (
                        f"step {step_i}, fluid {f}, slot {s} "
                        f"({SLOT_NAMES[s]}): "
                        f"pinned cells differ from sentinel {sentinel!r} "
                        f"(max diff="
                        f"{np.max(np.abs(pinned - sentinel)):.3e})"
                    )
    finally:
        run.close()


# ---------------------------------------------------------------------------
# 2b. Pin does NOT touch P_e or B
# ---------------------------------------------------------------------------


def test_pin_does_not_touch_pe_or_b():
    """Pin must not alter P_e or B storage.

    Test strategy: pack P_e and B with known values, then apply the pin
    (via fill_bc which triggers pin_apply). Verify that the pin writes
    the sentinel to all five fluid slots of masked cells, but P_e and B
    retain their pre-pin values bit-wise. This tests the pin code path
    directly (xmf_pin_apply_{device,host} loops only over XMF_SLOT_NVAR
    = 5 fluid slots, never P_e or B blocks).

    The original control-vs-pinned approach is invalid: P_e and B
    evolve through hydro/CT which depend on the fluid state. The pin
    changes the fluid state, so P_e and B naturally diverge even though
    the pin code path never writes to them.

    Config: MHD gate open, single ion + pressure-only electrons.
    """
    lib = load()
    cfg = _cfl_mhd_cfg()
    cfg["electron"] = {"mode": "pressure-only"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        ncells = lib.xmf_state_ncells(h)

        # Pack known fluid state.
        _pack_state(lib, h, 0, RHO, np.full(ncells, 1.0))
        _pack_state(lib, h, 0, VX, np.full(ncells, 0.3))
        _pack_state(lib, h, 0, VY, np.full(ncells, -0.2))
        _pack_state(lib, h, 0, VZ, np.full(ncells, 0.1))
        _pack_state(lib, h, 0, P, np.full(ncells, 1.0))

        # Pack known P_e.
        pe_val = 0.5
        pe_buf = np.full(ncells, pe_val, dtype=np.float64)
        assert lib.xmf_state_write_pe(h, _ptr(pe_buf)) == ncells

        # Pack known B (uniform).
        b_vals = (0.4, 0.6, 0.8)
        for comp, bv in enumerate(b_vals):
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            nd = nxt if comp == 0 else nyt if comp == 1 else nzt
            n1 = nxt if d1 == 0 else nyt if d1 == 1 else nzt
            n2 = nxt if d2 == 0 else nyt if d2 == 1 else nzt
            nb = (nd + 1) * n1 * n2
            _pack_b(lib, h, comp, np.full(nb, bv))

        # Save P_e and B BEFORE pin apply.
        pe_before = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_copy_pe(h, _ptr(pe_before)) == ncells
        b_before = _unpack_b_all(lib, h, (nxt, nyt, nzt))

        # Set pin: mask ALL cells, sentinel fluid values.
        mask = np.ones((nzt, nyt, nxt), dtype=np.int32)
        sentinel = 999.0
        vals = np.full((1, 5, nzt, nyt, nxt), sentinel, dtype=np.float64)
        run.set_pin(mask, vals)

        # fill_bc triggers pin_apply. The pin writes sentinel to all
        # five fluid slots of every cell.
        assert lib.xmf_fill_bc(h) == 0

        # Check fluid slots: ALL cells must be sentinel.
        fld = _unpack_state(lib, h, 1, ncells)
        for s in range(5):
            assert np.all(fld[0, s] == sentinel), (
                f"slot {s} ({SLOT_NAMES[s]}): not all sentinel after "
                f"pin+fill_bc"
            )

        # Check P_e: must be UNCHANGED (bit-wise).
        pe_after = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_copy_pe(h, _ptr(pe_after)) == ncells
        assert np.all(pe_after == pe_before), (
            "P_e changed after pin+fill_bc (pin must not touch P_e)"
        )

        # Check B: must be UNCHANGED (bit-wise).
        b_after = _unpack_b_all(lib, h, (nxt, nyt, nzt))
        for comp in range(3):
            assert np.all(b_after[comp] == b_before[comp]), (
                f"B[{comp}] changed after pin+fill_bc "
                f"(pin must not touch B)"
            )
    finally:
        run.close()


# ---------------------------------------------------------------------------
# 3. Pin + user floor coexistence
# ---------------------------------------------------------------------------


def test_pin_and_user_floor_coexist():
    """Pin (interior) + user floor (x1 ghosts): pin cells keep sentinel,
    non-pin x1 ghosts keep the floor value, no cross-talk.

    Config: 1D, x1=user with boundary_floor. Pin covers interior cells
    only (i in [ng, ng+4)). Ghosts (i < ng, i >= ng+nx) are NOT
    pinned and must carry the floor value after fill_bc.
    """
    lib = load()
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [32, 1, 1]
    cfg["grid"]["ndim"] = 1
    cfg["time"] = {"cfl": 0.4, "tstop": 10.0, "dt_max": 0.0}
    cfg["boundary"] = {"x1": "user"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    ng = cfg["grid"]["nghost"]
    nytot = cfg["grid"]["n"][1] + 2 * ng
    nztot = cfg["grid"]["n"][2] + 2 * ng
    floor_val = 42.0
    nfluid = 1
    floor_buf = np.full(nfluid * 5 * nztot * nytot, floor_val,
                        dtype=np.float64)
    cfg["boundary_floor"] = floor_buf
    cfg["ic"] = {"kind": "riemann", "x0": 16.5,
                 "left": {"n": 1.0, "v": 0.0, "p": 1.0},
                 "right": {"n": 1.0, "v": 0.0, "p": 1.0}}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        ncells = lib.xmf_state_ncells(h)

        # Pack non-trivial interior state.
        _pack_state(lib, h, 0, RHO, np.full(ncells, 1.0))
        _pack_state(lib, h, 0, VX, np.full(ncells, 0.1))
        _pack_state(lib, h, 0, VY, np.full(ncells, 0.0))
        _pack_state(lib, h, 0, VZ, np.full(ncells, 0.0))
        _pack_state(lib, h, 0, P, np.full(ncells, 1.0))

        # Pin: interior cells i in [ng, ng+4). NOT the ghost cells.
        mask = np.zeros((nzt, nyt, nxt), dtype=np.int32)
        for ii in range(ng, ng + 4):
            mask[:, :, ii] = 1
        sentinel = 777.0
        vals = np.full((1, 5, nzt, nyt, nxt), sentinel, dtype=np.float64)
        run.set_pin(mask, vals)

        # Run 2 steps.
        for step_i in range(2):
            run.step()
            fld = _unpack_state(lib, h, 1, ncells)
            blk = fld.reshape(1, 5, nzt, nyt, nxt)

            # Pinned interior cells: must be sentinel.
            for s in range(5):
                pinned = blk[0, s, :, :, ng:ng + 4]
                assert np.all(pinned == sentinel), (
                    f"step {step_i} slot {s}: pinned cells not sentinel"
                )

            # Non-pin x1 ghost cells (i < ng): must be floor value.
            for s in range(5):
                ghost_lo = blk[0, s, :, :, :ng]
                assert np.all(ghost_lo == floor_val), (
                    f"step {step_i} slot {s}: x1beg ghost not floor "
                    f"(got {ghost_lo.ravel()[:4]!r}, expected "
                    f"{floor_val})"
                )
    finally:
        run.close()


# ---------------------------------------------------------------------------
# 4. Clear regression: never-loaded vs set-then-cleared
# ---------------------------------------------------------------------------


def test_pin_clear_matches_never_pinned():
    """1D empty table: never-pin and set-then-clear runs must produce
    bit-wise identical t, dt and state after 25 steps. Also compare
    against test_kokkos_serial's PIN_EMPTY_T_DUMP / PIN_EMPTY_DT_DUMP
    pins (read-only; this file does not touch that test)."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from test_kokkos_serial import (
        PIN_EMPTY_DT_DUMP,
        PIN_EMPTY_T_DUMP,
    )
    _sys.path.pop(0)

    lib = load()

    def _run_empty(pin_mode):
        """pin_mode: 'never', 'set_then_clear'."""
        cfg = xmf.shock_tube()
        run = xmf.setup(cfg)
        h = run._h
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        ncells = lib.xmf_state_ncells(h)
        nfluid = lib.xmf_state_nfluid(h)
        if pin_mode == "set_then_clear":
            mask = np.zeros((nzt, nyt, nxt), dtype=np.int32)
            mask[ng, ng, ng] = 1
            vals = np.full((nfluid, 5, nzt, nyt, nxt), 123.456)
            run.set_pin(mask, vals)
            run.set_pin(None, None)  # explicit clear
        for _ in range(25):
            run.step()
        t = run.t
        dt = run.dt
        fld = _unpack_state(lib, h, nfluid, ncells)
        return run, t, dt, fld

    run_a, t_a, dt_a, fld_a = _run_empty("never")
    run_b, t_b, dt_b, fld_b = _run_empty("set_then_clear")
    try:
        # Bit-wise identical between the two modes.
        assert t_a == t_b, f"t differs: never={t_a!r} vs clear={t_b!r}"
        assert dt_a == dt_b, (
            f"dt differs: never={dt_a!r} vs clear={dt_b!r}"
        )
        assert np.all(fld_a == fld_b), (
            "state differs between never/clear"
        )

        # Match the test_kokkos_serial pins (read-only check).
        assert t_a == PIN_EMPTY_T_DUMP, (
            f"never-pin t={t_a!r} != PIN_EMPTY_T_DUMP="
            f"{PIN_EMPTY_T_DUMP!r}"
        )
        assert dt_a == PIN_EMPTY_DT_DUMP, (
            f"never-pin dt={dt_a!r} != PIN_EMPTY_DT_DUMP="
            f"{PIN_EMPTY_DT_DUMP!r}"
        )
    finally:
        run_a.close()
        run_b.close()


# ---------------------------------------------------------------------------
# 5. Pit-1 regression: test_dt_reasonable_3d_hd in suite
# ---------------------------------------------------------------------------


def test_pit1_regression_in_suite():
    """Confirm test_dt_reasonable_3d_hd exists in test_mhd_3d and is
    green. Re-import and run it here so this file's pass/fail report
    carries the result explicitly."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from test_mhd_3d import test_dt_reasonable_3d_hd
    _sys.path.pop(0)
    lib = load()
    test_dt_reasonable_3d_hd(lib)


# ---------------------------------------------------------------------------
# 6. Run.set_pin Python shape validation
# ---------------------------------------------------------------------------


def test_set_pin_shape_validation():
    """Run.set_pin rejects wrong-shape mask/vals with ValueError, accepts
    correct shapes. None clears without error."""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [16, 1, 1]
    cfg["grid"]["ndim"] = 1
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        ncells = nxt * nyt * nzt
        nfluid = 1

        # Correct shape: should succeed silently.
        mask_ok = np.zeros((nzt, nyt, nxt), dtype=np.int32)
        mask_ok[ng, ng, ng] = 1
        vals_ok = np.zeros((nfluid, 5, nzt, nyt, nxt), dtype=np.float64)
        run.set_pin(mask_ok, vals_ok)

        # Clear via None: should succeed.
        run.set_pin(None, None)

        # Wrong mask size.
        bad_mask = np.zeros(ncells + 1, dtype=np.int32)
        with pytest.raises(ValueError, match="mask size"):
            run.set_pin(bad_mask, vals_ok)

        # Wrong vals size.
        bad_vals = np.zeros((nfluid, 5, nzt, nyt, nxt + 1),
                            dtype=np.float64)
        with pytest.raises(ValueError, match="vals size"):
            run.set_pin(mask_ok, bad_vals)

        # None mask, non-None vals: clear (no error).
        run.set_pin(None, vals_ok)
        # Non-None mask, None vals: clear (no error).
        run.set_pin(mask_ok, None)
    finally:
        run.close()
