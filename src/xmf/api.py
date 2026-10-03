from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from . import config as cfg
from ._native import bc_tapes_to_c, load, problem_to_c, tapes_to_c

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        _LIB = load()
    return _LIB


@dataclass
class FluidFields:
    x: np.ndarray
    n: np.ndarray
    v: np.ndarray
    p: np.ndarray


class Run:
    def __init__(self, handle: int, packed: cfg.KernelConfig):
        self._h = handle
        self._cfg = packed
        self.tstop = packed.tstop

    @property
    def t(self) -> float:
        return float(_lib().xmf_time(self._h))

    @property
    def dt(self) -> float:
        return float(_lib().xmf_dt(self._h))

    @property
    def num_tapes(self) -> int:
        """Number of source tapes loaded into the kernel (A4.6). 0 for an
        empty sources table."""
        return int(_lib().xmf_ir_ntape(self._h))

    @property
    def num_states(self) -> int:
        """N, the number of independent variables of the implicit
        Jacobian (A4.7): five stored slots (n, vx, vy, vz, P) per fluid,
        indexed f*5 + slot. Grid coordinates r/x1/x2/x3, F_b and
        constants{} are never independent variables. B2: with
        pressure-only electrons on, P_e is ONE extra unknown, last
        (N = 5*nfluid + 1); it never enters the fluid five slots."""
        return int(_lib().xmf_ir_nstate(self._h))

    @property
    def nbeam(self) -> int:
        """A5: number of radiation beams loaded through the beam-table
        port (0 when no radiation state is loaded)."""
        return int(_lib().xmf_rad_nbeam(self._h))

    @property
    def source_resid(self) -> float:
        """A5: largest relative one-step residual of the linearised
        implicit source solve over the last completed step (0 when no
        source step ran). Report-only -- the solve is a single linearised
        backward-Euler step and never iterates on it."""
        if not self._h:
            raise RuntimeError("run is closed")
        return float(_lib().xmf_source_resid(self._h))

    @property
    def divb_max(self) -> float:
        """M5: maximum normalised discrete divB over the interior cells.
        C-IDF-4 area-weighted form (dimensionless; 0 for a
        constructively divergence-free field). Report-only -- the
        threshold 1e-6 is enforced in the test suite, never by the
        kernel. Returns 0 when the magnetic gate is closed."""
        if not self._h:
            raise RuntimeError("run is closed")
        return float(_lib().xmf_divb_max(self._h))

    def _interior_shape(self) -> tuple[tuple[int, int, int], int, tuple[int, int, int]]:
        lib = _lib()
        ntot = tuple(int(lib.xmf_mesh_ncell(self._h, d)) for d in range(3))
        ncells = int(lib.xmf_state_ncells(self._h))
        nint = tuple(self._cfg.n)  # (nx, ny, nz)
        return ntot, ncells, nint

    def fields_full(self) -> dict[str, Any]:
        """A5 read port: every fluid's five slots over the interior grid.

        Returns {fluid_name: {"n","vx","vy","vz","P"}} with each array
        shaped (nz, ny, nx) -- the (k, j, i) cell order of the kernel.
        Slot 0 stores the mass density; "n" is converted back to the
        number density rho/mass, and every slot is returned in physical
        cgs-style units through the problem's unit triplet (the legacy
        single-fluid ``fields`` property keeps its exact convention).

        B1: with pressure-only electrons on, the electron pressure block
        is additionally exposed as out["P_e"] -- one (nz, ny, nx) array
        in the same pressure units, active cells only. Electrons off:
        the kernel has no P_e block and the key simply does not exist.
        """
        if not self._h:
            raise RuntimeError("run is closed")
        lib = _lib()
        ntot, ncells, nint = self._interior_shape()
        ng = self._cfg.ng
        nfluid = int(lib.xmf_state_nfluid(self._h))
        ptr = ctypes.POINTER(ctypes.c_double)
        buf = np.empty(ncells, dtype=np.float64)
        out: dict[str, dict[str, np.ndarray]] = {}
        for f in range(nfluid):
            slots: dict[str, np.ndarray] = {}
            for var, name in enumerate(("n", "vx", "vy", "vz", "P")):
                got = lib.xmf_state_unpack(self._h, f, var, buf.ctypes.data_as(ptr))
                if got != ncells:
                    raise RuntimeError(f"xmf_state_unpack failed ({got})")
                blk = buf.reshape(ntot[2], ntot[1], ntot[0])
                inner = blk[ng : ng + nint[2], ng : ng + nint[1], ng : ng + nint[0]]
                slots[name] = inner.copy()
            mass_amu = self._cfg.fluids[f].mass
            u = self._cfg.units
            slots["n"] = slots["n"] * u.density / (mass_amu * cfg.AMU)
            for key in ("vx", "vy", "vz"):
                slots[key] = slots[key] * u.velocity
            slots["P"] = slots["P"] * u.density * u.velocity * u.velocity
            out[self._cfg.fluids[f].name] = slots
        # B1 electron pressure read port: present exactly when the kernel
        # carries the optional P_e block (pressure-only electrons on);
        # xmf_state_copy_pe refuses with -1 when electrons are off, so no
        # key appears then. Same pressure units as the fluid "P" slot.
        got = lib.xmf_state_copy_pe(self._h, buf.ctypes.data_as(ptr))
        if got == ncells:
            blk = buf.reshape(ntot[2], ntot[1], ntot[0])
            inner = blk[ng : ng + nint[2], ng : ng + nint[1], ng : ng + nint[0]]
            u = self._cfg.units
            out["P_e"] = inner.copy() * u.density * u.velocity * u.velocity
        # M2 staggered-B read port (the P_e precedent): the B1/B2/B3 keys
        # appear exactly when the magnetic gate is open (xmf_mag_load
        # landed) -- xmf_mag_unpack refuses with -1 otherwise, so the keys
        # simply do not exist with magnetics off. Block comp is laid out
        # (ntot[comp]+1) faces along its own direction, cells across the
        # other two, i fastest; the active read takes every cell's LOWER
        # face in the block's own direction (face index == cell index),
        # shaped (nz, ny, nx) in the kernel's (k, j, i) order. Physical
        # units are sqrt(unit_density) * unit_velocity.
        u = self._cfg.units
        unit_b = math.sqrt(u.density) * u.velocity
        for comp, key in enumerate(("B1", "B2", "B3")):
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            nb_len = (ntot[comp] + 1) * ntot[d1] * ntot[d2]
            bbuf = np.empty(nb_len, dtype=np.float64)
            got = lib.xmf_mag_unpack(self._h, comp, bbuf.ctypes.data_as(ptr))
            if got != nb_len:
                if comp == 0:
                    break  # gate closed: no B keys at all
                raise RuntimeError(f"xmf_mag_unpack failed ({got})")
            blk = bbuf.reshape(ntot[d2], ntot[d1], ntot[comp] + 1)
            # blk axes are (direction d2, direction d1, direction comp);
            # reorder to (k, j, i) = (x3, x2, x1).
            axis_of = {d2: 0, d1: 1, comp: 2}
            perm = (axis_of[2], axis_of[1], axis_of[0])
            sl = [slice(None)] * 3
            for q, ax in axis_of.items():
                sl[ax] = slice(ng, ng + nint[q])
            out[key] = blk[tuple(sl)].transpose(perm).copy() * unit_b
        return out

    def flux(self) -> np.ndarray:
        """A5 read port: the scanned transmitted beam flux F_b on the
        interior grid, shape (nbeam, nz, ny, nx), in the table's own
        units (erg/cm^2/s for the shipped SED tables). With the
        radiation disabled every entry reads the loaded boundary value
        (the vacuum limit). Raises when no beam table was loaded."""
        if not self._h:
            raise RuntimeError("run is closed")
        lib = _lib()
        nb = int(lib.xmf_rad_nbeam(self._h))
        if nb <= 0:
            raise RuntimeError("no radiation state loaded")
        ntot, ncells, nint = self._interior_shape()
        ng = self._cfg.ng
        ptr = ctypes.POINTER(ctypes.c_double)
        buf = np.empty(ncells, dtype=np.float64)
        out = np.empty((nb, nint[2], nint[1], nint[0]), dtype=np.float64)
        for b in range(nb):
            got = lib.xmf_rad_flux_copy(self._h, b, buf.ctypes.data_as(ptr))
            if got != ncells:
                raise RuntimeError(f"xmf_rad_flux_copy failed ({got})")
            blk = buf.reshape(ntot[2], ntot[1], ntot[0])
            out[b] = blk[ng : ng + nint[2], ng : ng + nint[1], ng : ng + nint[0]]
        return out

    def jacobian(self, k: int, j: int, i: int) -> np.ndarray:
        """A4.7 helper: the dense local Jacobian J = dL_c/du at one
        interior cell.

        k, j, i are full-grid indices (ghost offset included), each in
        [ng, ng + n_d). Returns a row-major float64 array of shape
        (N, N) with N = num_states: rows are the write-target components
        in the same f*5 + slot order as the state, columns the seeded
        state slots. Each column seeds one slot with udot = 1 and runs
        the very tapes already loaded; a bare dvx/dvy/dvz tape adds its
        derivative to every fluid's vx/vy/vz row (apply's ADDS
        convention). All zeros for an empty sources table. Never called
        by step; ghosts are never evaluated.
        """
        if not self._h:
            raise RuntimeError("run is closed")
        n = int(_lib().xmf_ir_nstate(self._h))
        mat = np.empty(n * n, dtype=np.float64)
        err = _lib().xmf_ir_jac(
            self._h,
            int(k),
            int(j),
            int(i),
            mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if err != 0:
            raise RuntimeError(f"xmf_ir_jac failed ({err})")
        return mat.reshape((n, n))

    def step(self) -> float:
        err = _lib().xmf_step(self._h)
        if err != 0:
            raise RuntimeError(f"xmf_step failed ({err})")
        return self.t

    def set_pin(
        self,
        mask: np.ndarray | None,
        vals: np.ndarray | None,
    ) -> None:
        """Port C: load or clear the frozen-cell pin mask.

        mask is an int32 array of shape (nztot, nytot, nxtot) -- ghosts
        included -- with 0/1 entries. vals is a float64 array of shape
        (nfluid, 5, nztot, nytot, nxtot) in the same flat slot order as
        xmf_state_pack (rho, vx, vy, vz, P per fluid). Every step and
        every ghost fill automatically overwrites masked cells with their
        pinned values. P_e is never touched; B is never touched (the CT
        path's B is not pinned by this port).

        Passing None for either mask or vals clears the pin (both are
        treated as None regardless of the other argument). The same call
        replaces a previously loaded pin. Wrong-sized arrays raise
        ValueError without touching the C loader, so any previously
        loaded pin stays intact.
        """
        if not self._h:
            raise RuntimeError("run is closed")
        lib = _lib()
        iptr = ctypes.POINTER(ctypes.c_int)
        dptr = ctypes.POINTER(ctypes.c_double)
        if mask is None or vals is None:
            err = lib.xmf_bc_set_pin(self._h, None, None)
            if err != 0:
                raise RuntimeError(f"xmf_bc_set_pin failed ({err})")
            return
        ncells = int(lib.xmf_state_ncells(self._h))
        nfluid = int(lib.xmf_state_nfluid(self._h))
        m = np.ascontiguousarray(mask, dtype=np.int32)
        v = np.ascontiguousarray(vals, dtype=np.float64)
        expected_val_size = nfluid * 5 * ncells
        if m.size != ncells:
            raise ValueError(
                f"set_pin: mask size {m.size} does not match "
                f"ncells {ncells}"
            )
        if v.size != expected_val_size:
            raise ValueError(
                f"set_pin: vals size {v.size} does not match "
                f"nfluid*5*ncells {expected_val_size}"
            )
        err = lib.xmf_bc_set_pin(
            self._h,
            m.ctypes.data_as(iptr),
            v.ctypes.data_as(dptr),
        )
        if err != 0:
            raise RuntimeError(f"xmf_bc_set_pin failed ({err})")

    def apply_sources(self) -> None:
        """A4.6 helper: one xmf_ir_apply over the interior cells.

        Locked convention: every loaded tape's instantaneous value is
        added to its write target (dvx -> vx += expr, dn -> n += expr,
        dP -> P += expr). No dt factor, no hydro step, ghosts untouched.
        A no-op when no tapes were loaded (empty sources table). Time
        semantics are A4.7/A5's business.
        """
        if not self._h:
            raise RuntimeError("run is closed")
        err = _lib().xmf_ir_apply(self._h)
        if err != 0:
            raise RuntimeError(f"xmf_ir_apply failed ({err})")

    def advance(self, tstop: float | None = None) -> float:
        limit = self.tstop if tstop is None else tstop
        while self.t < limit:
            if self.t + self.dt > limit and self.dt > 0:
                # last CFL step may overshoot slightly; keep the kernel simple
                pass
            self.step()
            if self.dt <= 0:
                break
        return self.t

    @property
    def fields(self) -> dict[str, FluidFields]:
        n = int(_lib().xmf_ncell(self._h))  # interior cells along x1
        x = np.empty(n, dtype=np.float64)
        rho = np.empty(n, dtype=np.float64)
        v = np.empty(n, dtype=np.float64)
        p = np.empty(n, dtype=np.float64)
        ptr = ctypes.POINTER(ctypes.c_double)
        _lib().xmf_fields(
            self._h,
            x.ctypes.data_as(ptr),
            rho.ctypes.data_as(ptr),
            v.ctypes.data_as(ptr),
            p.ctypes.data_as(ptr),
        )
        out = {
            self._cfg.fluid: FluidFields(
                x=x,
                n=rho * self._cfg.units.density / (self._cfg.mass * cfg.AMU),
                v=v * self._cfg.units.velocity,
                p=p * self._cfg.units.density * self._cfg.units.velocity**2,
            )
        }
        # B1: expose P_e along the same interior x1 row when the kernel
        # carries the electron pressure block (electrons off: no key).
        # The row sits in the k=ng, j=ng transverse plane of the block.
        lib = _lib()
        ntot0 = int(lib.xmf_mesh_ncell(self._h, 0))
        ntot1 = int(lib.xmf_mesh_ncell(self._h, 1))
        ntot2 = int(lib.xmf_mesh_ncell(self._h, 2))
        ncells = ntot0 * ntot1 * ntot2
        pe_buf = np.empty(ncells, dtype=np.float64)
        got = lib.xmf_state_copy_pe(self._h, pe_buf.ctypes.data_as(ptr))
        if got == ncells:
            ng = self._cfg.ng
            blk = pe_buf.reshape(ntot2, ntot1, ntot0)
            u = self._cfg.units
            out["P_e"] = (
                blk[ng, ng, ng : ng + n].copy() * u.density * u.velocity**2
            )
        return out

    def close(self) -> None:
        if self._h:
            _lib().xmf_finalize(self._h)
            self._h = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def setup(user_cfg: Mapping[str, Any]) -> Run:
    packed = cfg.pack(user_cfg)
    blob = problem_to_c(packed)
    dptr = ctypes.POINTER(ctypes.c_double)
    # A1 stretch supplement: the three per-direction ACTIVE face lattices
    # ride beside the flat blob (variable-length arrays never enter the
    # fixed-size XmfConfig). normalize() built them (user array or the
    # uniform/log factory); the kernel validates strict monotonicity,
    # pads ghosts with the adjacent active width and derives x/dx/A/dV.
    xf = tuple(
        np.ascontiguousarray(getattr(packed, f"xf{d}"), dtype=np.float64)
        for d in (1, 2, 3)
    )
    handle = _lib().xmf_setup(
        ctypes.byref(blob),
        xf[0].ctypes.data_as(dptr),
        xf[1].ctypes.data_as(dptr),
        xf[2].ctypes.data_as(dptr),
    )
    if not handle:
        raise RuntimeError("xmf_setup failed")
    try:
        # A5 physics port (opt-in): per-fluid masses and the unit
        # triplet. Must land before the beam table (the scan converts the
        # absorber density through mass_code) and before the tapes (they
        # are validated against the loadable slots).
        phys = packed.extra.get("phys")
        if phys:
            mass_amu = np.asarray(phys["mass_amu"], dtype=np.float64)
            err = _lib().xmf_phys_load(
                handle,
                mass_amu.ctypes.data_as(dptr),
                float(phys["unit_density"]),
                float(phys["unit_velocity"]),
                float(phys["unit_length"]),
            )
            if err != 0:
                raise RuntimeError(f"xmf_phys_load failed ({err})")
        # B1 charge port (opt-in): per-fluid charges q_s in elementary-
        # charge multiples. normalize() builds the table exactly when
        # electrons are on; the kernel's ion role is q_s != 0 and its
        # closure gate opens only with this table loaded (n_e divides by
        # mass_code, hence after xmf_phys_load; before the first step).
        # Electrons off -> no table pushed -> pre-B1 hydro bit for bit.
        charges = packed.extra.get("charge")
        if charges is not None:
            charge = np.asarray(charges, dtype=np.float64)
            err = _lib().xmf_charge_load(handle, charge.ctypes.data_as(dptr))
            if err != 0:
                raise RuntimeError(f"xmf_charge_load failed ({err})")
        # M block 1 magnetic port (opt-in): normalize() records the
        # problem-level mhd switch in extra["mhd"] exactly when it is on.
        # The face-centred B blocks arrive zero-initialised through
        # xmf_mag_load (block 1 = HD evolution + frozen B bystander;
        # initial values are packed afterwards through xmf_mag_pack when a
        # problem needs them). Magnetics off -> no port pushed -> every
        # mag helper refuses (-1) and the path stays bit for bit pre-M.
        mag = packed.extra.get("mhd")
        if mag:
            ntot = tuple(packed.n[d] + 2 * packed.ng for d in range(3))
            b_blocks = []
            for dd in range(3):
                d1, d2 = (dd + 1) % 3, (dd + 2) % 3
                b_blocks.append(
                    np.zeros((ntot[dd] + 1) * ntot[d1] * ntot[d2], dtype=np.float64)
                )
            err = _lib().xmf_mag_load(
                handle,
                int(mag["reflect"]),
                b_blocks[0].ctypes.data_as(dptr),
                b_blocks[1].ctypes.data_as(dptr),
                b_blocks[2].ctypes.data_as(dptr),
            )
            if err != 0:
                raise RuntimeError(f"xmf_mag_load failed ({err})")
        # A5 beam table: must land before the tapes so FLUX slots are
        # loadable when the compiled tapes are validated. With the
        # radiation disabled normalize() has zeroed the boundary fluxes,
        # so FLUX reads return the vacuum value 0.
        rad = packed.extra.get("rad_load")
        if rad:
            sigma = np.asarray(rad["sigma"], dtype=np.float64)
            hnu = np.asarray(rad["hnu"], dtype=np.float64)
            finf = np.asarray(rad["finf"], dtype=np.float64)
            got = _lib().xmf_rad_load(
                handle,
                int(rad["enabled"]),
                int(rad["mode"]),
                int(rad["dilution"]),
                int(rad["sense"]),
                int(rad["absorber"]),
                len(sigma),
                sigma.ctypes.data_as(dptr),
                hnu.ctypes.data_as(dptr),
                finf.ctypes.data_as(dptr),
            )
            if got != len(sigma):
                raise RuntimeError(f"xmf_rad_load failed ({got})")
        # B3 multi-absorber port (opt-in, APPENDED): present exactly when
        # normalize() saw more than one absorber name. Pushed right after
        # xmf_rad_load; while loaded, the scan's per-beam optical depth is
        # tau_b = (sum_s sigma_{s,b} n_s) ds over the table's fluids. The
        # single-absorber path never reaches this port.
        radm = packed.extra.get("rad_load_multi")
        if radm:
            absorbers = np.asarray(radm["absorbers"], dtype=np.int32)
            sigma_m = np.asarray(radm["sigma"], dtype=np.float64)
            got = _lib().xmf_rad_load_absorbers(
                handle,
                int(len(radm["absorbers"])),
                absorbers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                sigma_m.ctypes.data_as(dptr),
            )
            if got != len(radm["absorbers"]):
                raise RuntimeError(f"xmf_rad_load_absorbers failed ({got})")
        # A5 user boundary floor (pinned base state of the x1 user face).
        floor = packed.extra.get("user_bc")
        if floor is not None:
            data = np.asarray(floor, dtype=np.float64)
            err = _lib().xmf_bc_set_user(handle, data.ctypes.data_as(dptr))
            if err != 0:
                raise RuntimeError(f"xmf_bc_set_user failed ({err})")
        # A3 supplement: analytic user boundary tapes. Per-face sets with
        # assignment semantics, loaded through xmf_bc_load_tapes -- never
        # merged into the source tape set below (normalize keeps them
        # mutually exclusive with the floor on x1 faces). They evaluate
        # at every ghost fill: the setup fill already ran before this
        # port, so the first step (or an explicit xmf_fill_bc) applies
        # them -- exactly the floor's activation convention.
        for face, tapes in packed.extra.get("bc_tapes") or ():
            header, ops, consts = bc_tapes_to_c(tapes)
            got = _lib().xmf_bc_load_tapes(
                handle,
                int(face),
                len(tapes),
                header.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ops.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                consts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            if got != len(tapes):
                raise RuntimeError(f"xmf_bc_load_tapes failed ({got})")
        # A4.6: a non-empty sources table was compiled into integer tapes by
        # normalize() (Problem.extra["source_tapes"]); push them into the
        # kernel after the A5 ports. An empty table loads nothing, keeping
        # the pure-hydro path (shock tube) bit-for-bit untouched. The
        # kernel only ever sees opcodes and slots -- never a formula
        # string.
        tapes = packed.extra.get("source_tapes") or ()
        if tapes:
            header, ops, consts = tapes_to_c(tapes)
            got = _lib().xmf_ir_load(
                handle,
                len(tapes),
                header.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ops.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                consts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            if got != len(tapes):
                raise RuntimeError(f"xmf_ir_load failed ({got})")
    except Exception:
        _lib().xmf_finalize(handle)
        raise
    return Run(handle, packed)
