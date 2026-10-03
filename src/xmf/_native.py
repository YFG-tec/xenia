"""Load the prebuilt C kernel shipped inside the installed xmf package.

The shared library is compiled at install time by the build backend
(scikit-build-core + CMake), never at import time. This module only
locates the artifact next to this file and binds its C signatures.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
_LIB = _DIR / ("xmf_core.dll" if sys.platform == "win32" else "libxmf_core.so")


class XmfConfigC(ctypes.Structure):
    """Must match src/xmf/core/xmf_config.h."""

    _fields_ = [
        ("schema_version", ctypes.c_int32),
        ("ndim", ctypes.c_int32),
        ("n", ctypes.c_int32 * 3),
        ("ng", ctypes.c_int32),
        ("geometry", ctypes.c_int32),
        ("nfluid", ctypes.c_int32),
        ("sources_enabled", ctypes.c_int32),
        ("electron_mode", ctypes.c_int32),
        ("backend", ctypes.c_int32),
        ("riemann", ctypes.c_int32),
        ("recon", ctypes.c_int32),
        ("rk", ctypes.c_int32),
        ("stretch", ctypes.c_int32),
        # B2 shock-heating share tier (enum XmfShockHeat). This field used
        # to be the unused reserved0 slot; the byte position is FROZEN (no
        # field may be inserted after electron_mode and shift the blob),
        # only the name changed. Matches core/xmf_config.h.
        ("shock_heat", ctypes.c_int32),
        ("bc", ctypes.c_int32 * 6),
        ("x1min", ctypes.c_double),
        ("x1max", ctypes.c_double),
        ("x2min", ctypes.c_double),
        ("x2max", ctypes.c_double),
        ("x3min", ctypes.c_double),
        ("x3max", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("cfl", ctypes.c_double),
        ("dt_max", ctypes.c_double),
        ("tstop", ctypes.c_double),
        ("x_disc", ctypes.c_double),
        ("rho_l", ctypes.c_double),
        ("v_l", ctypes.c_double),
        ("p_l", ctypes.c_double),
        ("rho_r", ctypes.c_double),
        ("v_r", ctypes.c_double),
        ("p_r", ctypes.c_double),
    ]


def problem_to_c(p) -> XmfConfigC:
    from . import config as cfg

    c = XmfConfigC()
    c.schema_version = p.schema
    c.ndim = p.ndim
    c.n[0], c.n[1], c.n[2] = p.n
    c.ng = p.ng
    c.geometry = cfg.GEOM[p.geometry]
    c.nfluid = len(p.fluids)
    c.sources_enabled = 1 if p.sources_enabled else 0
    c.electron_mode = cfg.ELECTRON[p.electron.mode]
    c.backend = cfg.BACKEND[p.backend]
    c.riemann = cfg.RIEMANN[p.numerics.riemann]
    c.recon = cfg.RECON[p.numerics.reconstruction]
    c.rk = cfg.RK[p.numerics.rk]
    c.stretch = cfg.STRETCH[p.stretch]
    # B2: the former reserved0 slot now carries the shock-heating share
    # tier (validated by normalize via cfg.SHOCK_HEAT; unknown names never
    # reach this point).
    c.shock_heat = cfg.SHOCK_HEAT[p.electron.shock_heat]
    # Per-face boundary enums: face order matches cfg.FACES (0=x1beg ..
    # 5=x3end); values come from cfg.BC_ENUM, which is derived from
    # cfg.BC_TYPES and matches enum XmfBcType in core/xmf_config.h.
    for i, face in enumerate(cfg.FACES):
        c.bc[i] = cfg.BC_ENUM[p.boundary[face]]
    c.x1min, c.x1max = p.x1min, p.x1max
    c.x2min, c.x2max = p.x2min, p.x2max
    c.x3min, c.x3max = p.x3min, p.x3max
    c.gamma = p.gamma
    c.cfl = p.numerics.cfl
    c.dt_max = p.numerics.dt_max
    c.tstop = p.numerics.tstop
    c.x_disc = p.x_disc
    c.rho_l, c.v_l, c.p_l = p.rho_l, p.v_l, p.p_l
    c.rho_r, c.v_r, c.p_r = p.rho_r, p.v_r, p.p_r
    return c


def tapes_to_c(tapes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten compiled SourceTape objects for xmf_ir_load.

    Returns (header, ops, consts): header holds four int32 per tape --
    dest_kind, dest_fluid (-1 = bare dvx/dvy/dvz = every fluid), nops,
    nconst; ops holds nops (opcode, argument) pairs per tape as int32;
    consts holds nconst float64 per tape. Concatenated in tape order,
    exactly the layout xmf_ir_load expects. The kernel copies everything
    into its own memory, so these arrays need not outlive the call.
    """
    header: list[int] = []
    ops: list[int] = []
    consts: list[float] = []
    for tape in tapes:
        header.extend(
            [int(tape.dest.kind), int(tape.dest.dest_fluid), len(tape.ops), len(tape.consts)]
        )
        for op, arg in tape.ops:
            ops.extend([int(op), int(arg)])
        consts.extend(float(v) for v in tape.consts)
    return (
        np.asarray(header, dtype=np.int32),
        np.asarray(ops, dtype=np.int32),
        np.asarray(consts, dtype=np.float64),
    )


def bc_tapes_to_c(tapes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten compiled BcTape objects for xmf_bc_load_tapes.

    Returns (header, ops, consts): header holds four int32 per tape --
    dest_var (the assigned field slot 0..4 = n,vx,vy,vz,P), dest_fluid,
    nops, nconst; ops holds nops (opcode, argument) pairs per tape as
    int32; consts holds nconst float64 per tape. Concatenated in tape
    order, exactly the layout xmf_bc_load_tapes expects. The kernel
    copies everything into its own memory, so these arrays need not
    outlive the call.
    """
    header: list[int] = []
    ops: list[int] = []
    consts: list[float] = []
    for tape in tapes:
        header.extend([int(tape.dest_var), int(tape.dest_fluid), len(tape.ops), len(tape.consts)])
        for op, arg in tape.ops:
            ops.extend([int(op), int(arg)])
        consts.extend(float(v) for v in tape.consts)
    return (
        np.asarray(header, dtype=np.int32),
        np.asarray(ops, dtype=np.int32),
        np.asarray(consts, dtype=np.float64),
    )


def load() -> ctypes.CDLL:
    """Load the prebuilt shared library from the installed package directory."""
    if not _LIB.exists():
        raise FileNotFoundError(
            f"prebuilt xmf C kernel not found: {_LIB}\n"
            "The kernel is compiled at install time. "
            "Run `pip install .` in the repository root to build and "
            "install the package."
        )
    lib = ctypes.CDLL(str(_LIB))
    lib.xmf_setup.restype = ctypes.c_void_p
    # A1 stretch supplement: beside the flat blob, setup takes the three
    # per-direction ACTIVE face lattices (n[d]+1 absolute interface
    # positions each, ghosts excluded). Variable-length arrays never enter
    # the fixed-size XmfConfig; the kernel validates strict monotonicity,
    # pads ghosts with the adjacent active width and derives x/dx/A/dV.
    lib.xmf_setup.argtypes = [
        ctypes.POINTER(XmfConfigC),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_step.restype = ctypes.c_int
    lib.xmf_step.argtypes = [ctypes.c_void_p]
    lib.xmf_time.restype = ctypes.c_double
    lib.xmf_time.argtypes = [ctypes.c_void_p]
    lib.xmf_dt.restype = ctypes.c_double
    lib.xmf_dt.argtypes = [ctypes.c_void_p]
    lib.xmf_ncell.restype = ctypes.c_int
    lib.xmf_ncell.argtypes = [ctypes.c_void_p]
    lib.xmf_fields.restype = None
    lib.xmf_fields.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_finalize.restype = None
    lib.xmf_finalize.argtypes = [ctypes.c_void_p]
    # Read-only mesh queries (A1), helper style like xmf_ncell/xmf_time:
    # d is the direction (0=x1, 1=x2, 2=x3). ncell counts cells including
    # ghosts (n_d + 2*ng); nface is ncell + 1. The copy_* functions write
    # into caller-owned buffers (x/dx/dV: ncell doubles; xf/A: nface
    # doubles) and return the number of doubles written, or -1 on invalid
    # arguments.
    lib.xmf_mesh_ncell.restype = ctypes.c_int
    lib.xmf_mesh_ncell.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.xmf_mesh_nface.restype = ctypes.c_int
    lib.xmf_mesh_nface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.xmf_mesh_copy_x.restype = ctypes.c_int
    lib.xmf_mesh_copy_x.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_mesh_copy_xf.restype = ctypes.c_int
    lib.xmf_mesh_copy_xf.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_mesh_copy_dx.restype = ctypes.c_int
    lib.xmf_mesh_copy_dx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_mesh_copy_A.restype = ctypes.c_int
    lib.xmf_mesh_copy_A.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_mesh_copy_dV.restype = ctypes.c_int
    lib.xmf_mesh_copy_dV.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    # Multi-fluid state queries (A2), helper style like xmf_mesh_copy_*:
    # fields are laid out [fluid][slot][cell], slot order (n, vx, vy, vz,
    # P) == offsets 0..4, fluids addressed by integer index f in
    # [0, xmf_state_nfluid(h)). Every block is xmf_state_ncells(h)
    # doubles long: nxtot*nytot*nztot INCLUDING ghosts -- not the interior
    # x1 count returned by xmf_ncell. pack copies a caller buffer INTO the
    # kernel, unpack copies OUT into a caller buffer; both move a whole
    # block and return the number of doubles moved, or -1 (moving nothing)
    # on invalid f / slot / buffer. The optional electron pressure P_e
    # (electron_mode == pressure-only) is one separate block of the same
    # length: copy_pe writes it out, write_pe copies into it, and both
    # return -1 without touching anything when electrons are off.
    lib.xmf_state_ncells.restype = ctypes.c_int
    lib.xmf_state_ncells.argtypes = [ctypes.c_void_p]
    lib.xmf_state_nfluid.restype = ctypes.c_int
    lib.xmf_state_nfluid.argtypes = [ctypes.c_void_p]
    lib.xmf_state_pack.restype = ctypes.c_int
    lib.xmf_state_pack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_state_unpack.restype = ctypes.c_int
    lib.xmf_state_unpack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_state_copy_pe.restype = ctypes.c_int
    lib.xmf_state_copy_pe.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_state_write_pe.restype = ctypes.c_int
    lib.xmf_state_write_pe.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    # Per-face boundary helpers (A3): face order is fixed, 0=x1beg ..
    # 5=x3end, and the type integers match enum XmfBcType in
    # core/xmf_config.h (reflect=0, copy=1, outflow=2, user=3, none=4).
    # xmf_bc_type returns the type stored for one face, or -1 on invalid
    # arguments. xmf_fill_bc is a helper on the same level as
    # xmf_state_pack (not a fifth public verb): it immediately re-pads
    # every face's ghosts per the run's current six face types and
    # returns 0, or -1 on invalid arguments.
    lib.xmf_bc_type.restype = ctypes.c_int
    lib.xmf_bc_type.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.xmf_fill_bc.restype = ctypes.c_int
    lib.xmf_fill_bc.argtypes = [ctypes.c_void_p]
    # Source-tape helpers (A4.6), same helper level as xmf_state_pack --
    # not a fifth public verb. xmf_ir_load copies a whole compiled tape set
    # (see tapes_to_c for the header/ops/consts layout) into the run after
    # setup and returns the tape count, or -1 on error (previous set kept).
    # xmf_ir_apply adds every tape's instantaneous value to its write
    # target on the interior cells: no dt factor, no hydro -- the locked
    # convention lives in xmf_core.h / xmf_ir.c. xmf_ir_ntape returns the
    # loaded count (0 for an empty sources table).
    lib.xmf_ir_load.restype = ctypes.c_int
    lib.xmf_ir_load.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_ir_apply.restype = ctypes.c_int
    lib.xmf_ir_apply.argtypes = [ctypes.c_void_p]
    lib.xmf_ir_ntape.restype = ctypes.c_int
    lib.xmf_ir_ntape.argtypes = [ctypes.c_void_p]
    # Forward-AD helpers (A4.7), same helper level as xmf_ir_apply -- not
    # a fifth public verb. xmf_ir_nstate returns N = 5*nfluid: the
    # independent variables of the implicit Jacobian are the stored
    # per-fluid slots (n, vx, vy, vz, P), indexed f*5 + slot; r/x1/x2/x3,
    # F_b and constants{} are never seeded. xmf_ir_jac writes the dense
    # row-major N*N Jacobian dL_c/du of one interior cell (k, j, i are
    # full grid indices, ghosts included); an empty tape set writes all
    # zeros; bare dvx/dvy/dvz tapes add into every fluid's velocity row,
    # matching apply's ADDS convention. xmf_ir_eval_tape evaluates one
    # loaded tape (index t) at one interior cell with a single seeded
    # slot direction (seed_slot = -1 or a loadable packed slot code,
    # seeded with derivative 1): a self-check seam -- seeding r gives
    # d(-G*M/r^2)/dr = 2*G*M/r^3 without ever becoming a jac column.
    # Full conventions live in core/xmf_core.h.
    lib.xmf_ir_nstate.restype = ctypes.c_int
    lib.xmf_ir_nstate.argtypes = [ctypes.c_void_p]
    lib.xmf_ir_jac.restype = ctypes.c_int
    lib.xmf_ir_jac.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_ir_eval_tape.restype = ctypes.c_int
    lib.xmf_ir_eval_tape.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    # A5 auxiliary load/read ports, same helper level as xmf_ir_load --
    # never a fifth public verb. xmf_phys_load converts amu masses to
    # code mass densities and records the unit triplet (returns 0/-1);
    # xmf_charge_load (B1) copies the nfluid charges in elementary-charge
    # multiples q_s into the run (ion role q_s != 0, names never
    # consulted; call after xmf_phys_load, before the first step; returns
    # 0/-1, charges default to 0.0 = pre-B1 hydro bit for bit);
    # xmf_rad_load copies the beam table (sigma/hnu/finf, nbeam each)
    # and allocates the transmitted-flux state (returns nbeam, or -1 on
    # invalid arguments / unsupported mode / dilution / sense);
    # xmf_rad_nbeam returns the loaded beam count; xmf_rad_flux_copy
    # copies one beam's transmitted flux (ghosts included);
    # xmf_bc_set_user loads the x1 user-face floor buffer;
    # xmf_source_resid reads the last step's relative source residual.
    lib.xmf_phys_load.restype = ctypes.c_int
    lib.xmf_phys_load.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.xmf_charge_load.restype = ctypes.c_int
    lib.xmf_charge_load.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_rad_load.restype = ctypes.c_int
    lib.xmf_rad_load.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    # B3 appended multi-absorber port: requires a beam table already
    # loaded by xmf_rad_load; absorbers is nabs fluid indices (int32),
    # sigma the nabs*nbeam cross sections row-major. Returns nabs, or -1.
    lib.xmf_rad_load_absorbers.restype = ctypes.c_int
    lib.xmf_rad_load_absorbers.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_rad_nbeam.restype = ctypes.c_int
    lib.xmf_rad_nbeam.argtypes = [ctypes.c_void_p]
    lib.xmf_rad_flux_copy.restype = ctypes.c_int
    lib.xmf_rad_flux_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_bc_set_user.restype = ctypes.c_int
    lib.xmf_bc_set_user.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    # Port C: frozen-cell pin mask. Auxiliary load port at the same helper
    # level as xmf_bc_set_user -- not a fifth public verb. mask/vals
    # lengths are validated in Python (api.py Run.set_pin); the C side
    # takes raw pointers, signature verbatim the hd one.
    lib.xmf_bc_set_pin.restype = ctypes.c_int
    lib.xmf_bc_set_pin.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
    ]
    # M block 1 magnetic ports (M2 staggered storage + M3 boundary
    # helpers), same helper level as xmf_state_pack / xmf_bc_set_user --
    # not a fifth public verb. xmf_mag_load opens the gate (allocates the
    # face-centred B and edge-centred E blocks, copies b1/b2/b3 in,
    # records the reflect tier enum XmfMagReflect) and returns 0/-1;
    # pack/unpack move one whole B block (ghosts included) in/out and
    # return the doubles moved or -1 on a closed gate; unpack_e reads one
    # edge-centred E block (the M2 read seam); xmf_mag_set_user loads the
    # x1 user-face tangential-B floor [tcomp][k*nytot+j].
    lib.xmf_mag_load.restype = ctypes.c_int
    lib.xmf_mag_load.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_mag_pack.restype = ctypes.c_int
    lib.xmf_mag_pack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_mag_unpack.restype = ctypes.c_int
    lib.xmf_mag_unpack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_mag_unpack_e.restype = ctypes.c_int
    lib.xmf_mag_unpack_e.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_mag_set_user.restype = ctypes.c_int
    lib.xmf_mag_set_user.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
    ]
    # A3 supplement: analytic user boundary tapes. Auxiliary load port at
    # the same helper level as xmf_bc_set_user (not a fifth public verb):
    # per-face assignment tapes compiled by xmf.bc_ir; see bc_tapes_to_c
    # for the header/ops/consts layout and xmf_core.h for the contract
    # (face must already be user; slots without a tape copy @1; reads are
    # the ghost's own grid centre, constants and @1/@2 neighbour layers;
    # never merged into the source tape set / Newton solve). Returns the
    # tape count, or -1 on invalid arguments (previous face set kept);
    # ntape == 0 clears the face's set.
    lib.xmf_bc_load_tapes.restype = ctypes.c_int
    lib.xmf_bc_load_tapes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.xmf_source_resid.restype = ctypes.c_double
    lib.xmf_source_resid.argtypes = [ctypes.c_void_p]
    # M5: maximum normalised discrete divB over the interior cells.
    # Report-only, returns 0 when the gate is closed. Helper level like
    # xmf_source_resid (not a fifth public verb).
    lib.xmf_divb_max.restype = ctypes.c_double
    lib.xmf_divb_max.argtypes = [ctypes.c_void_p]
    return lib
