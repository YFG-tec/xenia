"""Port B contract tests: device-side ghost skin fill.

Verifies the Port B contract (docs/SESSION.md item 6,
docs/cmd/PROMPT_PORT_B.md): the public xmf_fill_bc dispatches through
run->dev and the device fill is semantically equivalent to the host
fill_bc (bit-exact on Windows Kokkos OpenMP where HostSpace is the
device). Five classes of contract are pinned:

  1. Stale-definition class ("no-update-then-maintain"): xmf_fill_bc
     only writes ghost entries; interior cells (fluid, B faces, E
     edges) are bit-exact unchanged across the call, and across two
     consecutive calls.
  2. Periodic bit-exact: 3D periodic box ghost values match a
     Python-side wrap-around reference computed from the device
     interior. Includes a "pack -> step -> refill -> compare" segment
     on a non-uniform interior.
  3. Reflect bit-exact: ghost values match a Python mirror reference.
     Wall tier: B_n odd (flipped), B_t even (copied). Fluid: normal
     velocity slot odd, every other slot even.
  4. Degenerate axis: 2D box (nz=1) -- z ghost cells for fluid and
     B faces equal the active slice (transversely uniform copy).
     Caliber source: .tmp/port_b_degenerate_caliber.md (CT G template
     reads degenerate ghost cells, so the tree fills ALL directions).
  5. Dispatch entry: xmf_fill_bc returns 0 and ghost values are
     written (semantically not a no-op).

Ban self-check (docs/TODO.md section A):
  - no pluto.ini read (no file IO in this file);
  - no XUV_FACTOR==0 sentinel (no radiation);
  - fluid assertions cover all five slots (n, vx, vy, vz, P);
  - B assertions cover all three components;
  - E read via xmf_mag_unpack_e when needed.

Dimension coverage: 3D (8x6x4), 2D (8x6x1). All three velocities and
all three B components are exercised in the assertions.
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import load

_PTR = ctypes.POINTER(ctypes.c_double)

ION_FLUIDS = [{"name": "HII", "mass": 1.0, "charge": 1, "kind": "ion"}]


@pytest.fixture(scope="module")
def lib():
    """Shared library from the installed package."""
    return load()


# --- small helpers -----------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_PTR)


def _ntot_of(run) -> tuple[int, int, int]:
    n = run._cfg.n
    ng = run._cfg.ng
    return tuple(n[d] + 2 * ng for d in range(3))


def _len_b(ntot, comp: int) -> int:
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return (ntot[comp] + 1) * ntot[d1] * ntot[d2]


def _len_e(ntot, comp: int) -> int:
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return ntot[comp] * (ntot[d1] + 1) * (ntot[d2] + 1)


def _pack_b(lib, h, comp: int, blk: np.ndarray) -> None:
    got = lib.xmf_mag_pack(h, comp, _ptr(np.ascontiguousarray(blk.ravel())))
    assert got == blk.size, f"xmf_mag_pack B[{comp}] failed ({got})"


def _unpack_b(lib, h, comp: int, ntot) -> np.ndarray:
    n = _len_b(ntot, comp)
    buf = np.empty(n, dtype=np.float64)
    assert lib.xmf_mag_unpack(h, comp, _ptr(buf)) == n
    return buf


def _unpack_e(lib, h, comp: int, ntot) -> np.ndarray:
    n = _len_e(ntot, comp)
    buf = np.empty(n, dtype=np.float64)
    assert lib.xmf_mag_unpack_e(h, comp, _ptr(buf)) == n
    return buf


def _unpack_full(lib, h, f: int, var: int, ntot) -> np.ndarray:
    """Read the full (ghosts-included) fluid block shaped (nztot, nytot, nxtot)."""
    ncells = ntot[0] * ntot[1] * ntot[2]
    buf = np.empty(ncells, dtype=np.float64)
    got = lib.xmf_state_unpack(h, f, var, _ptr(buf))
    assert got == ncells, f"xmf_state_unpack failed ({got})"
    return buf.reshape(ntot[2], ntot[1], ntot[0])


def _pack_full(lib, h, f: int, var: int, arr: np.ndarray) -> None:
    """Write the full (ghosts-included) fluid block."""
    got = lib.xmf_state_pack(h, f, var, _ptr(np.ascontiguousarray(arr.ravel())))
    assert got == arr.size, f"xmf_state_pack failed ({got})"


# --- config factory ----------------------------------------------------------


def _box_cfg(n, bounds, bc, mhd=True, mhd_reflect="wall"):
    """3D box config. n=(nx,ny,nz), bounds=(x1max,x2max,x3max),
    bc='periodic'|'reflect'."""
    cfg = xmf.shock_tube()
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["grid"]["n"] = list(n)
    cfg["grid"]["ndim"] = 3
    cfg["grid"]["x1min"] = 0.0
    cfg["grid"]["x1max"] = bounds[0]
    cfg["grid"]["x2min"] = 0.0
    cfg["grid"]["x2max"] = bounds[1]
    cfg["grid"]["x3min"] = 0.0
    cfg["grid"]["x3max"] = bounds[2]
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "plm", "rk": "rk3"}
    cfg["time"] = {"cfl": 0.4, "tstop": 10.0, "dt_max": 0.0}
    if mhd:
        cfg["mhd"] = True
        cfg["mhd_reflect"] = mhd_reflect
        cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
        cfg["electron"] = {"mode": "off"}
    bc_dict = {}
    for ax in ("x1", "x2", "x3"):
        bc_dict[ax] = bc
    cfg["boundary"] = bc_dict
    cfg["ic"] = {
        "kind": "riemann",
        "x0": bounds[0] / 2.0,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    return cfg


def _fill_nonuniform_state(lib, h, ntot, ng):
    """Pack a non-uniform state into all five fluid slots + three B blocks.
    Returns the packed arrays for later comparison."""
    ncells = ntot[0] * ntot[1] * ntot[2]
    nint = tuple(ntot[d] - 2 * ng for d in range(3))
    dx = 1.0 / nint[0]
    dy = 1.0 / nint[1] if nint[1] > 1 else 1.0
    dz = 1.0 / nint[2] if nint[2] > 1 else 1.0

    rho = np.empty(ncells, dtype=np.float64)
    vx = np.empty(ncells, dtype=np.float64)
    vy = np.empty(ncells, dtype=np.float64)
    vz = np.empty(ncells, dtype=np.float64)
    P = np.empty(ncells, dtype=np.float64)

    for idx in range(ncells):
        i = idx % ntot[0]
        j = (idx // ntot[0]) % ntot[1]
        k = idx // (ntot[0] * ntot[1])
        xc = (i - ng + 0.5) * dx
        yc = (j - ng + 0.5) * dy if nint[1] > 1 else 0.5
        zc = (k - ng + 0.5) * dz if nint[2] > 1 else 0.5
        rho[idx] = 1.0 + 0.1 * math.sin(2.0 * math.pi * xc)
        vx[idx] = 0.2 + 0.05 * math.cos(2.0 * math.pi * yc)
        vy[idx] = 0.15 + 0.03 * math.sin(2.0 * math.pi * zc)
        vz[idx] = -0.1 + 0.02 * math.cos(2.0 * math.pi * xc)
        P[idx] = 1.0 + 0.08 * math.sin(2.0 * math.pi * (xc + yc))

    _pack_full(lib, h, 0, 0, rho)
    _pack_full(lib, h, 0, 1, vx)
    _pack_full(lib, h, 0, 2, vy)
    _pack_full(lib, h, 0, 3, vz)
    _pack_full(lib, h, 0, 4, P)

    return rho, vx, vy, vz, P


def _fill_nonuniform_b(lib, h, ntot, ng):
    """Pack non-uniform B into all three blocks. Returns the packed blocks."""
    nint = tuple(ntot[d] - 2 * ng for d in range(3))
    dx = 1.0 / nint[0]
    dy = 1.0 / nint[1] if nint[1] > 1 else 1.0
    dz = 1.0 / nint[2] if nint[2] > 1 else 1.0

    blocks = []
    for comp in range(3):
        d1, d2 = (comp + 1) % 3, (comp + 2) % 3
        n = _len_b(ntot, comp)
        blk = np.empty(n, dtype=np.float64)
        for idx in range(n):
            # storage layout: comp is fastest, then d1, then d2
            ic = idx % (ntot[comp] + 1)
            rem = idx // (ntot[comp] + 1)
            id1 = rem % ntot[d1]
            id2 = rem // ntot[d1]
            xc = (ic - ng + 0.5) * dx if comp == 0 else (ic - ng) * dx if ic >= ng else 0.0
            yc = (id1 - ng + 0.5) * dy if d1 == 1 else (id1 - ng) * dy if id1 >= ng else 0.0
            zc = (id2 - ng + 0.5) * dz if d2 == 2 else (id2 - ng) * dz if id2 >= ng else 0.0
            # use cell-center for face positions too for a smooth field
            if comp == 0:
                xf = (ic - ng) * dx
                blk[idx] = 0.4 + 0.1 * math.sin(2.0 * math.pi * yc) + 0.05 * math.cos(2.0 * math.pi * zc)
            elif comp == 1:
                yf = (id1 - ng) * dy
                blk[idx] = 0.6 + 0.08 * math.sin(2.0 * math.pi * zc) + 0.04 * math.cos(2.0 * math.pi * xc)
            else:
                zf = (id2 - ng) * dz
                blk[idx] = 0.8 + 0.06 * math.sin(2.0 * math.pi * xc) + 0.03 * math.cos(2.0 * math.pi * yc)
        _pack_b(lib, h, comp, blk)
        blocks.append(blk)
    return blocks


# ===========================================================================
# 1. Stale-definition class: interior cells unchanged by xmf_fill_bc
# ===========================================================================


def test_stale_definition_fluid(lib):
    """Stale-definition contract (fluid): xmf_fill_bc writes only ghost
    cells. Interior cells of all five slots are bit-exact unchanged
    across one fill and across two consecutive fills."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        # Pack non-uniform state.
        _fill_nonuniform_state(lib, h, ntot, ng)

        # Snapshot interior before fill.
        interior_before = {}
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot)
            inner = blk[ng:ng + nint[2], ng:ng + nint[1], ng:ng + nint[0]]
            interior_before[var] = inner.copy()

        # First fill.
        err = lib.xmf_fill_bc(h)
        assert err == 0, f"xmf_fill_bc returned {err}"

        # Interior must be bit-exact unchanged.
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot)
            inner = blk[ng:ng + nint[2], ng:ng + nint[1], ng:ng + nint[0]]
            assert np.all(inner == interior_before[var]), (
                f"fluid slot {var}: interior changed after first fill_bc")

        # Second fill: interior still bit-exact unchanged.
        err = lib.xmf_fill_bc(h)
        assert err == 0
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot)
            inner = blk[ng:ng + nint[2], ng:ng + nint[1], ng:ng + nint[0]]
            assert np.all(inner == interior_before[var]), (
                f"fluid slot {var}: interior changed after second fill_bc")
    finally:
        run.close()


def test_stale_definition_bface(lib):
    """Stale-definition contract (B faces): xmf_fill_bc writes only
    ghost entries. Interior face positions (ng..ng+nint along the
    component's own direction, interior cells transversely) are
    bit-exact unchanged across the fill."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        _fill_nonuniform_b(lib, h, ntot, ng)

        # Snapshot interior faces before fill. For each component,
        # interior faces are indices ng..ng+nint[comp] along comp,
        # and interior cells ng..ng+nint[d]-1 along each transverse d.
        interior_before = []
        for comp in range(3):
            blk = _unpack_b(lib, h, comp, ntot)
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            # storage: (ntot[d2], ntot[d1], ntot[comp]+1)
            shaped = blk.reshape(ntot[d2], ntot[d1], ntot[comp] + 1)
            inner = shaped[
                ng:ng + nint[d2],
                ng:ng + nint[d1],
                ng:ng + nint[comp] + 1,
            ]
            interior_before.append(inner.copy())

        err = lib.xmf_fill_bc(h)
        assert err == 0

        for comp in range(3):
            blk = _unpack_b(lib, h, comp, ntot)
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            shaped = blk.reshape(ntot[d2], ntot[d1], ntot[comp] + 1)
            inner = shaped[
                ng:ng + nint[d2],
                ng:ng + nint[d1],
                ng:ng + nint[comp] + 1,
            ]
            assert np.all(inner == interior_before[comp]), (
                f"B[{comp}]: interior faces changed after fill_bc")
    finally:
        run.close()


def test_stale_definition_edge(lib):
    """Stale-definition contract (E edges): xmf_fill_bc does not write
    E edges (E is filled by mag_induce during step, not by fill_bc).
    Interior edges are bit-exact unchanged across fill_bc.

    Note: E edges are not filled by xmf_fill_bc at all; the device E
    ghost fill (xmf_bc_fill_e_ghosts_device) is called from mag_induce.
    This test pins that fill_bc itself does not touch E."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        _fill_nonuniform_state(lib, h, ntot, ng)
        _fill_nonuniform_b(lib, h, ntot, ng)

        # Step once to populate E (mag_induce fills interior E).
        run.step()

        # Snapshot interior edges.
        interior_before = []
        for comp in range(3):
            e = _unpack_e(lib, h, comp, ntot)
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            # E storage: (ntot[d2]+1, ntot[d1]+1, ntot[comp])
            shaped = e.reshape(ntot[d2] + 1, ntot[d1] + 1, ntot[comp])
            inner = shaped[
                ng:ng + nint[d2] + 1,
                ng:ng + nint[d1] + 1,
                ng:ng + nint[comp],
            ]
            interior_before.append(inner.copy())

        # fill_bc should not touch E.
        err = lib.xmf_fill_bc(h)
        assert err == 0

        for comp in range(3):
            e = _unpack_e(lib, h, comp, ntot)
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            shaped = e.reshape(ntot[d2] + 1, ntot[d1] + 1, ntot[comp])
            inner = shaped[
                ng:ng + nint[d2] + 1,
                ng:ng + nint[d1] + 1,
                ng:ng + nint[comp],
            ]
            assert np.all(inner == interior_before[comp]), (
                f"E[{comp}]: interior edges changed after fill_bc")
    finally:
        run.close()


# ===========================================================================
# 2. Periodic bit-exact: ghost == Python wrap-around reference
# ===========================================================================


def _python_periodic_fluid_ghost(blk_3d, ntot, nint, ng, face):
    """Compute expected ghost values for one fluid face using Python
    wrap-around. blk_3d is (nztot, nytot, nxtot). Returns list of
    (ghost_indices, expected_values) tuples."""
    d = face // 2
    side = face % 2
    results = []
    for m in range(ng):
        if side == 0:
            g_idx = ng - 1 - m
            src_idx = ng + nint[d] - 1 - m
        else:
            g_idx = ng + nint[d] + m
            src_idx = ng + m
        # Build slice tuples for ghost and source.
        g_sl = [slice(None)] * 3
        s_sl = [slice(None)] * 3
        # axis order is (k, j, i) = (z, y, x)
        ax = 2 - d  # d=0->ax=2(i), d=1->ax=1(j), d=2->ax=0(k)
        g_sl[ax] = g_idx
        s_sl[ax] = src_idx
        ghost_vals = blk_3d[tuple(g_sl)]
        src_vals = blk_3d[tuple(s_sl)]
        results.append((ghost_vals, src_vals))
    return results


def test_periodic_fluid_bitexact(lib):
    """Periodic fill bit-exact (fluid 5 slots): ghost values equal the
    Python wrap-around reference computed from the device interior.
    8x6x4 full periodic box with non-uniform state."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        _fill_nonuniform_state(lib, h, ntot, ng)
        _fill_nonuniform_b(lib, h, ntot, ng)

        err = lib.xmf_fill_bc(h)
        assert err == 0

        # Check all six faces, all five slots.
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot)
            for face in range(6):
                pairs = _python_periodic_fluid_ghost(blk, ntot, nint, ng, face)
                for ghost_vals, src_vals in pairs:
                    assert np.all(ghost_vals == src_vals), (
                        f"periodic fluid slot {var} face {face}: "
                        f"ghost != wrap source (bit-exact)")
    finally:
        run.close()


def _python_periodic_b_ghost(blk_flat, comp, ntot, ng, nint, face):
    """Compute expected periodic B ghost values. blk_flat is the flat
    B block. Storage: (ntot[d2], ntot[d1], ntot[comp]+1)."""
    d = face // 2
    side = face % 2
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    nint_d = nint[d]
    shaped = blk_flat.reshape(ntot[d2], ntot[d1], ntot[comp] + 1)

    results = []
    if comp == d:
        # B_d: face-centred along d. Ghost face indices.
        for m in range(ng):
            delta = m + 1
            if side == 0:
                g_f = ng - delta
                src_f = g_f + nint_d
            else:
                g_f = ng + nint_d + delta
                src_f = g_f - nint_d
            # Transverse walk: all cells in d1, d2.
            g_sl = [slice(None)] * 3
            s_sl = [slice(None)] * 3
            ax_d = 2  # comp is fastest axis in storage
            # storage axes: (d2=0, d1=1, comp=2)
            g_sl[ax_d] = g_f
            s_sl[ax_d] = src_f
            results.append((shaped[tuple(g_sl)], shaped[tuple(s_sl)]))
    else:
        # B_t: cell-centred along d. Ghost cell indices.
        for m in range(ng):
            delta = m + 1
            if side == 0:
                g_c = ng - delta
                src_c = g_c + nint_d
            else:
                g_c = ng + nint_d + delta - 1
                src_c = g_c - nint_d
            g_sl = [slice(None)] * 3
            s_sl = [slice(None)] * 3
            g_sl[2] = g_c
            s_sl[2] = src_c
            results.append((shaped[tuple(g_sl)], shaped[tuple(s_sl)]))
    return results


def test_periodic_b_bitexact(lib):
    """Periodic fill bit-exact (B faces, all three components): ghost
    face values equal the Python wrap-around reference. 8x6x4 box with
    non-uniform B."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        _fill_nonuniform_state(lib, h, ntot, ng)
        _fill_nonuniform_b(lib, h, ntot, ng)

        err = lib.xmf_fill_bc(h)
        assert err == 0

        for comp in range(3):
            blk = _unpack_b(lib, h, comp, ntot)
            # Check faces in the comp's own direction (d == comp).
            for face in (2 * comp, 2 * comp + 1):
                pairs = _python_periodic_b_ghost(
                    blk, comp, ntot, ng, nint, face)
                for ghost_vals, src_vals in pairs:
                    assert np.all(ghost_vals == src_vals), (
                        f"periodic B[{comp}] face {face}: "
                        f"ghost != wrap source (bit-exact)")
    finally:
        run.close()


def test_periodic_after_step_bitexact(lib):
    """Periodic fill bit-exact after one step (non-uniform interior).
    pack -> step -> fill_bc -> compare ghosts against Python wrap.

    This is the critical non-trivial test: on a uniform field any fill
    is trivially correct; only a non-uniform evolved interior catches
    wrap-indexing errors."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        _fill_nonuniform_state(lib, h, ntot, ng)
        _fill_nonuniform_b(lib, h, ntot, ng)

        # Step once to evolve the interior (non-uniform after step).
        run.step()

        # Re-fill after step.
        err = lib.xmf_fill_bc(h)
        assert err == 0

        # Fluid: all six faces, all five slots.
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot)
            for face in range(6):
                pairs = _python_periodic_fluid_ghost(blk, ntot, nint, ng, face)
                for ghost_vals, src_vals in pairs:
                    assert np.all(ghost_vals == src_vals), (
                        f"post-step periodic fluid slot {var} face {face}: "
                        f"ghost != wrap source (bit-exact)")

        # B: all three components, own-direction faces.
        for comp in range(3):
            blk = _unpack_b(lib, h, comp, ntot)
            for face in (2 * comp, 2 * comp + 1):
                pairs = _python_periodic_b_ghost(
                    blk, comp, ntot, ng, nint, face)
                for ghost_vals, src_vals in pairs:
                    assert np.all(ghost_vals == src_vals), (
                        f"post-step periodic B[{comp}] face {face}: "
                        f"ghost != wrap source (bit-exact)")
    finally:
        run.close()


# ===========================================================================
# 3. Reflect bit-exact: ghost == Python mirror reference
# ===========================================================================


def _python_reflect_fluid_ghost(blk_3d, ntot, nint, ng, face, flip):
    """Compute expected reflect ghost values. blk_3d is (nztot, nytot, nxtot).
    flip=True for the normal velocity slot, False for others."""
    d = face // 2
    side = face % 2
    results = []
    for m in range(ng):
        if side == 0:
            g_idx = ng - 1 - m
            src_idx = ng + m
        else:
            g_idx = ng + nint[d] + m
            src_idx = ng + nint[d] - 1 - m
        ax = 2 - d
        g_sl = [slice(None)] * 3
        s_sl = [slice(None)] * 3
        g_sl[ax] = g_idx
        s_sl[ax] = src_idx
        ghost_vals = blk_3d[tuple(g_sl)]
        src_vals = blk_3d[tuple(s_sl)]
        if flip:
            src_vals = -src_vals
        results.append((ghost_vals, src_vals))
    return results


def _python_reflect_b_normal_ghost(blk_flat, comp, ntot, ng, nint, face, sgn_n):
    """Compute expected reflect ghost values for the NORMAL B component
    (comp == d). Wall tier: sgn_n = -1 (B_n odd). Symmetric: sgn_n = +1."""
    d = face // 2
    side = face % 2
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    shaped = blk_flat.reshape(ntot[d2], ntot[d1], ntot[comp] + 1)

    results = []
    # Wall tier: boundary face pinned to 0.
    if sgn_n < 0:
        w = ng if side == 0 else ng + nint[d]
        w_sl = [slice(None)] * 3
        w_sl[2] = w
        wall_vals = shaped[tuple(w_sl)]
        results.append(("wall", wall_vals, 0.0))

    for m in range(ng):
        delta = m + 1
        if side == 0:
            g_f = ng - delta
            src_f = ng + delta
        else:
            g_f = ng + nint[d] + delta
            src_f = ng + nint[d] - delta
        g_sl = [slice(None)] * 3
        s_sl = [slice(None)] * 3
        g_sl[2] = g_f
        s_sl[2] = src_f
        results.append(("ghost", shaped[tuple(g_sl)], sgn_n * shaped[tuple(s_sl)]))
    return results


def _python_reflect_b_tangent_ghost(blk_flat, t, d, ntot, ng, nint, face, sgn_t):
    """Compute expected reflect ghost values for a TANGENTIAL B component
    (t != d). Wall tier: sgn_t = +1 (B_t even). Symmetric: sgn_t = -1.

    Storage layout for component t: (ntot[d2_t], ntot[d1_t], ntot[t]+1)
    where d1_t=(t+1)%3, d2_t=(t+2)%3. The ghost walk is along the
    FACE's normal direction d, which is cell-indexed for tangential
    components. In storage, d occupies one of the first two axes
    depending on how d maps to d1_t / d2_t."""
    side = face % 2
    d1_t, d2_t = (t + 1) % 3, (t + 2) % 3
    shaped = blk_flat.reshape(ntot[d2_t], ntot[d1_t], ntot[t] + 1)

    # Figure out which storage axis corresponds to the face direction d.
    # Storage axes: 0 = d2_t, 1 = d1_t, 2 = t.
    if d == d2_t:
        d_ax = 0
    elif d == d1_t:
        d_ax = 1
    else:
        raise ValueError(f"d={d} not transverse to t={t}")

    results = []
    for m in range(ng):
        delta = m + 1
        if side == 0:
            g_c = ng - delta
            src_c = ng + delta - 1
        else:
            g_c = ng + nint[d] - 1 + delta
            src_c = ng + nint[d] - delta
        g_sl = [slice(None)] * 3
        s_sl = [slice(None)] * 3
        g_sl[d_ax] = g_c
        s_sl[d_ax] = src_c
        results.append((shaped[tuple(g_sl)], sgn_t * shaped[tuple(s_sl)]))
    return results


def test_reflect_fluid_bitexact(lib):
    """Reflect fill bit-exact (fluid 5 slots): ghost values match the
    Python mirror reference. Wall tier: normal velocity slot is odd
    (sign-flipped), every other slot is even (copied). 8x6x4 box."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "reflect", mhd=True,
                    mhd_reflect="wall")
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        _fill_nonuniform_state(lib, h, ntot, ng)
        # B = 0 for reflect (avoid wall-BC divB complications).
        for comp in range(3):
            blk = np.zeros(_len_b(ntot, comp), dtype=np.float64)
            _pack_b(lib, h, comp, blk)

        err = lib.xmf_fill_bc(h)
        assert err == 0

        # normvar[d] = d (0->vx, 1->vy, 2->vz)
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot)
            for face in range(6):
                d = face // 2
                is_normal_vel = (var == d + 1)  # vx=1, vy=2, vz=3
                pairs = _python_reflect_fluid_ghost(
                    blk, ntot, nint, ng, face, flip=is_normal_vel)
                for ghost_vals, src_vals in pairs:
                    assert np.all(ghost_vals == src_vals), (
                        f"reflect fluid slot {var} face {face}: "
                        f"ghost != mirror reference (bit-exact)")
    finally:
        run.close()


def test_reflect_b_bitexact_wall(lib):
    """Reflect fill bit-exact (B faces, wall tier): B_n is odd (ghost
    faces sign-flipped, boundary face pinned to 0); B_t is even (ghost
    cells copied without sign change). 8x6x4 box with non-uniform B."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "reflect", mhd=True,
                    mhd_reflect="wall")
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        _fill_nonuniform_state(lib, h, ntot, ng)
        _fill_nonuniform_b(lib, h, ntot, ng)

        err = lib.xmf_fill_bc(h)
        assert err == 0

        sgn_n = -1.0  # wall: B_n odd
        sgn_t = 1.0   # wall: B_t even

        for face in range(6):
            d = face // 2
            side = face % 2

            # Normal component: B_d
            blk_d = _unpack_b(lib, h, d, ntot)
            pairs = _python_reflect_b_normal_ghost(
                blk_d, d, ntot, ng, nint, face, sgn_n)
            for entry in pairs:
                if entry[0] == "wall":
                    wall_vals = entry[1]
                    assert np.all(wall_vals == 0.0), (
                        f"reflect wall B[{d}] face {face}: "
                        f"boundary face not pinned to 0")
                else:
                    ghost_vals, expect = entry[1], entry[2]
                    assert np.all(ghost_vals == expect), (
                        f"reflect B[{d}] face {face}: "
                        f"normal ghost != mirror reference (bit-exact)")

            # Tangential components: B_t for t != d
            for t in range(3):
                if t == d:
                    continue
                blk_t = _unpack_b(lib, h, t, ntot)
                pairs = _python_reflect_b_tangent_ghost(
                    blk_t, t, d, ntot, ng, nint, face, sgn_t)
                for ghost_vals, expect in pairs:
                    assert np.all(ghost_vals == expect), (
                        f"reflect B[{t}] face {face}: "
                        f"tangential ghost != mirror reference (bit-exact)")
    finally:
        run.close()


def test_reflect_b_bitexact_symmetric(lib):
    """Reflect fill bit-exact (B faces, symmetric tier): B_n is even
    (copied), B_t is odd (sign-flipped). 8x6x4 box with non-uniform B.
    Contrasts the wall tier to pin both reflect branches."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "reflect", mhd=True,
                    mhd_reflect="symmetric")
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        _fill_nonuniform_state(lib, h, ntot, ng)
        _fill_nonuniform_b(lib, h, ntot, ng)

        err = lib.xmf_fill_bc(h)
        assert err == 0

        sgn_n = 1.0   # symmetric: B_n even
        sgn_t = -1.0  # symmetric: B_t odd

        for face in range(6):
            d = face // 2

            # Normal component: B_d (no wall pinning in symmetric tier).
            blk_d = _unpack_b(lib, h, d, ntot)
            pairs = _python_reflect_b_normal_ghost(
                blk_d, d, ntot, ng, nint, face, sgn_n)
            for entry in pairs:
                if entry[0] == "wall":
                    continue  # symmetric tier: no wall pinning
                ghost_vals, expect = entry[1], entry[2]
                assert np.all(ghost_vals == expect), (
                    f"symmetric B[{d}] face {face}: "
                    f"normal ghost != mirror reference (bit-exact)")

            # Tangential components.
            for t in range(3):
                if t == d:
                    continue
                blk_t = _unpack_b(lib, h, t, ntot)
                pairs = _python_reflect_b_tangent_ghost(
                    blk_t, t, d, ntot, ng, nint, face, sgn_t)
                for ghost_vals, expect in pairs:
                    assert np.all(ghost_vals == expect), (
                        f"symmetric B[{t}] face {face}: "
                        f"tangential ghost != mirror reference (bit-exact)")
    finally:
        run.close()


# ===========================================================================
# 4. Degenerate axis: z ghost == active slice copy (nz=1)
# ===========================================================================


def test_degenerate_axis_fluid_and_b(lib):
    """Degenerate axis contract (nz=1): z-direction ghost cells for
    fluid and B faces equal the active slice (transversely uniform
    copy). The tree fills ALL directions including degenerate ones
    (caliber source: .tmp/port_b_degenerate_caliber.md -- CT G
    template reads degenerate ghost cells, so all directions are
    filled; the old card text said "degenerate axis not filled" but
    the caliber ruling overrides it).

    2D box: 8x6x1, x1/x2 periodic, x3 copy (config forbids periodic
    on nz<2). After fill_bc, z ghost cells must equal the active
    slice (k=ng) bit-exact for fluid; B1/B2 z-ghost cells copy the
    active cell; B3 z-ghost faces copy the boundary face."""
    cfg = _box_cfg((8, 6, 1), (1.0, 1.0, 1.0), "periodic", mhd=True)
    # Override x3 to copy (periodic requires nz >= 2).
    cfg["boundary"]["x3beg"] = "copy"
    cfg["boundary"]["x3end"] = "copy"
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        assert nint[2] == 1, f"expected nz=1, got {nint[2]}"
        assert ntot[2] == 1 + 2 * ng, (
            f"expected nztot={1 + 2 * ng}, got {ntot[2]}")

        _fill_nonuniform_state(lib, h, ntot, ng)
        _fill_nonuniform_b(lib, h, ntot, ng)

        err = lib.xmf_fill_bc(h)
        assert err == 0

        # Fluid: z ghost slices must equal the active slice (k=ng).
        # With nz=1 and copy BC, every ghost copies the outermost
        # active layer, which IS the only active layer k=ng.
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot)
            active = blk[ng, :, :]  # active slice at k=ng
            for m in range(ng):
                ghost_beg = blk[ng - 1 - m, :, :]
                assert np.all(ghost_beg == active), (
                    f"degenerate fluid slot {var} z-beg ghost layer {m}: "
                    f"not equal to active slice")
                ghost_end = blk[ng + 1 + m, :, :]
                assert np.all(ghost_end == active), (
                    f"degenerate fluid slot {var} z-end ghost layer {m}: "
                    f"not equal to active slice")

        # B: for each component, z-direction ghost entries must equal
        # the active z value.
        for comp in range(3):
            blk = _unpack_b(lib, h, comp, ntot)
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            shaped = blk.reshape(ntot[d2], ntot[d1], ntot[comp] + 1)
            if comp == 0:
                # shaped = (ntot[2], ntot[1], ntot[0]+1), z is axis 0
                # B1 cell-centred along z: active cell at k=ng.
                active_z = shaped[ng, :, :]
                for m in range(ng):
                    assert np.all(shaped[ng - 1 - m, :, :] == active_z), (
                        f"degenerate B[0] z-beg ghost layer {m}")
                    assert np.all(shaped[ng + 1 + m, :, :] == active_z), (
                        f"degenerate B[0] z-end ghost layer {m}")
            elif comp == 1:
                # shaped = (ntot[0], ntot[2], ntot[1]+1), z is axis 1
                # B2 cell-centred along z: active cell at k=ng.
                active_z = shaped[:, ng, :]
                for m in range(ng):
                    assert np.all(shaped[:, ng - 1 - m, :] == active_z), (
                        f"degenerate B[1] z-beg ghost layer {m}")
                    assert np.all(shaped[:, ng + 1 + m, :] == active_z), (
                        f"degenerate B[1] z-end ghost layer {m}")
            else:  # comp == 2
                # shaped = (ntot[1], ntot[0], ntot[2]+1), z is axis 2
                # B3 face-centred along z: active faces ng and ng+1.
                # Copy BC: ghost faces replicate the boundary face.
                # beg ghost face ng-1 copies boundary face ng.
                # end ghost face ng+2 copies boundary face ng+1.
                for m in range(ng):
                    beg = shaped[:, :, ng - 1 - m]
                    src_beg = shaped[:, :, ng]
                    assert np.all(beg == src_beg), (
                        f"degenerate B[2] z-beg ghost face {m}")
                    end = shaped[:, :, ng + 1 + 1 + m]
                    src_end = shaped[:, :, ng + 1]
                    assert np.all(end == src_end), (
                        f"degenerate B[2] z-end ghost face {m}")
    finally:
        run.close()


# ===========================================================================
# 5. Dispatch entry: xmf_fill_bc returns 0 and ghosts are written
# ===========================================================================


def test_dispatch_entry(lib):
    """Dispatch entry contract: xmf_fill_bc returns 0 (success) and
    ghost cells are actually written (not left as zero from calloc).
    On Windows OpenMP the device path IS the host path (HostSpace),
    so this pins "the dispatch went through run->dev and the semantic
    result is not a no-op"."""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        nint = run._cfg.n

        # Pack non-uniform interior; ghosts start as whatever pack left.
        _fill_nonuniform_state(lib, h, ntot, ng)
        _fill_nonuniform_b(lib, h, ntot, ng)

        # Zero out all ghost cells explicitly so we can detect a fill.
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot).copy()
            # Zero beg/end ghost slabs in each direction.
            for d in range(3):
                ax = 2 - d
                for m in range(ng):
                    sl = [slice(None)] * 3
                    sl[ax] = m
                    blk[tuple(sl)] = 0.0
                    sl2 = [slice(None)] * 3
                    sl2[ax] = ntot[d] - 1 - m
                    blk[tuple(sl2)] = 0.0
            _pack_full(lib, h, 0, var, blk)

        # fill_bc should return 0 and write non-zero ghosts.
        err = lib.xmf_fill_bc(h)
        assert err == 0, f"xmf_fill_bc returned {err}"

        # At least some ghost cells must be non-zero (the interior is
        # non-uniform with values near 1.0, so periodic wrap will copy
        # non-zero values into the ghosts).
        any_nonzero = False
        for var in range(5):
            blk = _unpack_full(lib, h, 0, var, ntot)
            for d in range(3):
                ax = 2 - d
                for m in range(ng):
                    sl = [slice(None)] * 3
                    sl[ax] = m
                    if np.any(blk[tuple(sl)] != 0.0):
                        any_nonzero = True
                        break
                    sl2 = [slice(None)] * 3
                    sl2[ax] = ntot[d] - 1 - m
                    if np.any(blk[tuple(sl2)] != 0.0):
                        any_nonzero = True
                        break
                if any_nonzero:
                    break
            if any_nonzero:
                break
        assert any_nonzero, (
            "xmf_fill_bc returned 0 but no ghost cell was written "
            "(dispatch may have been a no-op)")

        # B ghosts also non-zero.
        any_b_nonzero = False
        for comp in range(3):
            blk = _unpack_b(lib, h, comp, ntot)
            if np.any(blk != 0.0):
                any_b_nonzero = True
                break
        assert any_b_nonzero, (
            "xmf_fill_bc returned 0 but B ghosts are all zero "
            "(magnetic dispatch may have been a no-op)")
    finally:
        run.close()
