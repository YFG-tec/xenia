"""Python-side problem config. Names and tables live here; C only sees XmfConfig.

User dict (names) → normalize() → Problem → XmfConfig (ints/doubles).

``gamma`` is the adiabatic index, not PLUTO ``g_inputParam[GAMMA]``.

Grid (A1 stretch supplement): a NON-uniform lattice is the default. Every
direction's grid is handed to the C kernel as the ABSOLUTE positions of the
n_d+1 active interfaces (ghosts not included) -- ``grid.x1`` / ``grid.x2``
/ ``grid.x3`` (aliases ``x`` / ``y`` / ``z``). When a direction carries no
array, the factory below builds uniform faces from ``xdmin``/``xdmax``.
``grid.stretch`` is a PYTHON-ONLY factory ('none' = uniform, 'log'/'x1'
stretches x1 geometrically with ``grid.stretch_factor``); the kernel never
sees a recipe or a cell-width array, only faces. User arrays are checked
here for length/finiteness and their endpoints must equal xdmin/xdmax;
monotonicity is validated authoritatively by the C kernel.

Source terms that depend on the grid or fluid fields go in as symbolic
expressions (later). The C kernel only evaluates the packed IR; it must
not contain specialized chemistry or body-force formulas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA = 1
KB = 1.3806505e-16
AMU = 1.66053886e-24
UNIT_DENSITY = 1.0e-12
UNIT_VELOCITY = 1.0e6
UNIT_LENGTH = 1.0
UNIT_PRS = UNIT_DENSITY * UNIT_VELOCITY * UNIT_VELOCITY

GEOM = {"cartesian": 0, "spherical": 1, "cylindrical": 2}
BACKEND = {"serial": 0, "host": 1, "cuda": 2}
ELECTRON = {"off": 0, "pressure-only": 1}
# Shock-heating share tiers (B2): how the Godunov sweep's non-adiabatic
# (entropy-producing) pressure increment is split between the ion pressure
# P_s and the electron pressure P_e. Tóth's warning is the contract:
# T_e/T_i is NOT unique at a shock, so none of the three tiers is physical
# truth -- they are switchable model options the tube tests contrast.
# ions-only keeps every increment in P_s (the B1 two-pass path, bit for
# bit); by-charge hands each ion's charge share w_s of its increment to
# P_e; electron hands the whole increment to P_e. The integer lands in the
# flat blob's former reserved0 slot (byte position frozen, name moved);
# values must match enum XmfShockHeat in core/xmf_config.h.
SHOCK_HEAT = {"ions-only": 0, "by-charge": 1, "electron": 2}
# Riemann solvers supported by the C kernel (A4): hll, hllc and tvdlf are
# implemented. Must match enum XmfRiemann in core/xmf_config.h (hll=0,
# hllc=1, tvdlf=2); roe/exact do not exist any more. M block 1 (M1): hlld
# is enumerated so configs can name the MHD solver, but the kernel does not
# implement it in this slice -- xmf_setup refuses it (there is no silent
# fallback), exactly like rk4.
RIEMANN = {"hll": 0, "hllc": 1, "tvdlf": 2, "hlld": 3}
# Magnetic reflect tier (M block 1, M3): the mirror rule a reflect face
# applies to the face-centred B block. wall (PLUTO REFLECTIVE / AthenaK):
# the normal component flips and the wall face is pinned to 0 -- a
# conducting wall; symmetric (PLUTO EQTSYMMETRIC / Idefix): the tangential
# components flip (axial-vector mirroring), the normal one copies -- a
# symmetry plane. The fluid five slots use the same reflect fill either
# way. Values must match enum XmfMagReflect in core/xmf_config.h. The tier
# rides the auxiliary xmf_mag_load port -- never the flat XmfConfig blob.
MHD_REFLECT = {"wall": 0, "symmetric": 1}
RECON = {"pcm": 0, "plm": 1, "weno3": 2}
# Time integrators (A4 supplement): rk1/rk2/rk3 are implemented by the C
# kernel (forward Euler / Heun / Shu-Osher, all wrapping the whole-step
# L_f). rk4 is enumerated so normalize()/problem_to_c accept the name, but
# the kernel does not implement it -- xmf_setup refuses it (there is no
# silent fallback to rk1). Values must match enum XmfRK in
# core/xmf_config.h.
RK = {"rk1": 0, "rk2": 1, "rk3": 2, "rk4": 3}
# Stretch names are a PYTHON-side face-array factory only (A1 stretch
# supplement): 'none' builds uniform faces from xdmin/xdmax; 'log'
# stretches x1 geometrically (widths grow by grid.stretch_factor per cell);
# 'x1' is the legacy alias of 'log'. The integers still land in the flat
# blob for schema stability, but the C kernel ignores them -- it receives
# the resulting face arrays and knows no recipe.
STRETCH = {"none": 0, "log": 1, "x1": 1}
FACES = ("x1beg", "x1end", "x2beg", "x2end", "x3beg", "x3end")
BC_TYPES = ("reflect", "copy", "outflow", "user", "none", "periodic")
# Kernel integer codes for the per-face boundary types (XmfConfig.bc).
# Derived from BC_TYPES so there is exactly one string table; the values
# must match enum XmfBcType in core/xmf_config.h (reflect=0, copy=1,
# outflow=2, user=3, none=4, periodic=5).
BC_ENUM = {name: i for i, name in enumerate(BC_TYPES)}


@dataclass(frozen=True)
class FluidSpec:
    name: str
    mass: float
    charge: float
    # B1: the species ROLE -- 'neutral' or 'ion' -- never a solver label.
    # normalize() rewrites the legacy solver label 'hd' into the role from
    # the charge (q=0 -> neutral, q!=0 -> ion) and validates explicit roles
    # against the charge; 'mhd' stays refused (MHD is the problem-level
    # switch cfg['mhd'], never a per-fluid label). The kernel's ion test is
    # q_s != 0.
    kind: str = "neutral"


@dataclass(frozen=True)
class ElectronSpec:
    mode: str = "off"
    share: str = "by-charge"
    shock_heat: str = "ions-only"


@dataclass(frozen=True)
class Numerics:
    riemann: str
    reconstruction: str
    rk: str
    cfl: float
    dt_max: float
    tstop: float


@dataclass(frozen=True)
class Units:
    density: float
    velocity: float
    length: float


@dataclass
class Problem:
    """Validated frontend. Extra physics is stored even if the kernel ignores it."""

    schema: int
    gamma: float
    backend: str
    ndim: int
    n: tuple[int, int, int]
    ng: int
    geometry: str
    stretch: str
    x1min: float
    x1max: float
    x2min: float
    x2max: float
    x3min: float
    x3max: float
    # Absolute positions of the ACTIVE interfaces per direction (A1 stretch
    # supplement): len == n_d + 1, ghosts not included, xf[0] == xdmin and
    # xf[-1] == xdmax. Built by normalize() from a user array or the
    # uniform/log factory; handed to xmf_setup beside the flat blob. The
    # authoritative monotonicity check lives in C (strictly increasing).
    xf1: tuple[float, ...]
    xf2: tuple[float, ...]
    xf3: tuple[float, ...]
    numerics: Numerics
    units: Units
    fluids: list[FluidSpec]
    electron: ElectronSpec
    sources: list[Mapping[str, Any]]
    sources_enabled: bool
    radiation: Mapping[str, Any]
    constants: Mapping[str, Any]
    body_force: Mapping[str, Any] | str
    boundary: dict[str, str]
    ic: Mapping[str, Any]
    ic_kind: str
    mass: float
    fluid: str
    x_disc: float
    rho_l: float
    v_l: float
    p_l: float
    rho_r: float
    v_r: float
    p_r: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cfl(self) -> float:
        return self.numerics.cfl

    @property
    def tstop(self) -> float:
        return self.numerics.tstop


def _req(cfg: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = cfg
    path = []
    for k in keys:
        path.append(k)
        if not isinstance(cur, Mapping) or k not in cur:
            raise KeyError("missing config field: " + ".".join(path))
        cur = cur[k]
    return cur


def _pick(name: str, table: Mapping[str, int], raw: Any) -> str:
    key = str(raw).lower()
    if key not in table:
        raise ValueError(f"unknown {name} {raw!r}; want {sorted(table)}")
    return key


def n_to_rho(n: float, mass: float) -> float:
    return n * mass * AMU / UNIT_DENSITY


def rho_to_n(rho: float, mass: float) -> float:
    return rho * UNIT_DENSITY / (mass * AMU)


def _as_n(grid: Mapping[str, Any]) -> tuple[int, int, int]:
    if "n" in grid:
        raw = list(grid["n"])
        while len(raw) < 3:
            raw.append(1)
        return int(raw[0]), int(raw[1]), int(raw[2])
    nx = int(grid["nx"])
    ny = int(grid.get("ny", 1))
    nz = int(grid.get("nz", 1))
    return nx, ny, nz


def _faces(raw: Mapping[str, Any] | None) -> dict[str, str]:
    out = {f: "none" for f in FACES}
    if not raw:
        out["x1beg"] = out["x1end"] = "reflect"
        return out
    for axis, both in (("x1", "reflect"), ("x2", "none"), ("x3", "none")):
        if axis in raw and isinstance(raw[axis], str):
            out[f"{axis}beg"] = out[f"{axis}end"] = str(raw[axis]).lower()
    for key in FACES:
        if key in raw:
            out[key] = str(raw[key]).lower()
    for key, val in out.items():
        if val not in BC_TYPES:
            raise ValueError(f"unknown boundary {key}={val}")
    # Periodic validation: must be paired (xbeg ⟺ xend); x2/x3 periodic
    # requires ny>1/nz>1 (checked later in _grid_faces after grid is known).
    for axis in ("x1", "x2", "x3"):
        beg, end = out[f"{axis}beg"], out[f"{axis}end"]
        if (beg == "periodic") != (end == "periodic"):
            raise ValueError(
                f"periodic boundary on {axis} must be paired "
                f"({axis}beg={beg}, {axis}end={end})"
            )
    return out


# ---- Per-direction active face arrays (A1 stretch supplement) ------------
# The kernel's grid authority is the n_d+1 absolute interface positions of
# each direction's ACTIVE domain (ghosts excluded). Python builds them --
# from a user array (grid.x1/x2/x3, aliases x/y/z) or a factory -- and C
# validates strict monotonicity, pads ghosts with the adjacent active
# width and derives x/dx/A/dV. Cell widths never cross into C.

_GRID_ALIASES = {"x": "x1", "y": "x2", "z": "x3"}


def _grid_faces_raw(grid: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """Per-direction user face arrays with the x/y/z aliases resolved.
    Giving both spellings of one direction (x1 AND x) is refused."""
    raw: list[Any] = [None, None, None]
    for key in grid:
        if key in ("x1", "x2", "x3"):
            d = int(key[1]) - 1
        elif key in _GRID_ALIASES:
            d = int(_GRID_ALIASES[key][1]) - 1
        else:
            continue
        if raw[d] is not None:
            raise ValueError(
                f"grid.{key} duplicates the face array of direction {d + 1} "
                f"(keys x1/x2/x3 and their aliases x/y/z are the same slot)"
            )
        raw[d] = grid[key]
    return raw[0], raw[1], raw[2]


def _axis_faces(name: str, raw: Any, n: int, xmin: float, xmax: float) -> list[float]:
    """Validate one user face array: a 1-D sequence of n+1 finite reals
    whose endpoints equal xdmin/xdmax exactly (the array is never silently
    rewritten). Monotonicity is NOT checked here -- the C kernel is the
    authoritative validator (strictly increasing, else setup fails)."""
    try:
        vals = list(raw)
    except TypeError:
        raise ValueError(f"grid.{name} must be a 1-D sequence of reals") from None
    if len(vals) != n + 1:
        raise ValueError(
            f"grid.{name} must carry n+1={n + 1} active interface "
            f"positions, got {len(vals)}"
        )
    out: list[float] = []
    for idx, v in enumerate(vals):
        if isinstance(v, (str, bytes)):
            raise ValueError(f"grid.{name}[{idx}] is not a real number: {v!r}")
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"grid.{name}[{idx}] is not a real number: {v!r}") from None
        if not math.isfinite(fv):
            raise ValueError(f"grid.{name}[{idx}] must be finite, got {fv!r}")
        out.append(fv)
    if out[0] != xmin or out[-1] != xmax:
        raise ValueError(
            f"grid.{name} endpoints ({out[0]!r}, {out[-1]!r}) must equal "
            f"xdmin/xdmax ({xmin!r}, {xmax!r})"
        )
    return out


def _uniform_faces(xmin: float, xmax: float, n: int) -> list[float]:
    """Equidistant interfaces over [xmin, xmax] (the stretch='none' and
    default factory). xf[k] = xmin + k*dx with the last face pinned to
    xmax, so the generated array also satisfies the endpoint contract."""
    dx = (xmax - xmin) / n
    faces = [xmin + k * dx for k in range(n)]
    faces.append(xmax)
    return faces


def _log_faces(xmin: float, xmax: float, n: int, factor: float) -> list[float]:
    """Geometric (log-stretched) x1 factory: widths grow as
    dx_j = dx0 * factor**j, so
        dx0 = (xmax-xmin) * (factor-1) / (factor**n - 1)
    and the faces span [xmin, xmax] exactly. factor > 1 crowds cells at
    xmin (inner-dense), factor < 1 at xmax. PYTHON-only: the kernel
    receives the resulting faces, never this recipe."""
    if isinstance(factor, (str, bytes)) or not math.isfinite(float(factor)):
        raise ValueError(f"grid.stretch_factor must be a finite number, got {factor!r}")
    factor = float(factor)
    if factor <= 0.0:
        raise ValueError(f"grid.stretch_factor must be positive, got {factor!r}")
    if factor == 1.0:
        return _uniform_faces(xmin, xmax, n)
    if n * abs(math.log(factor)) > 700.0:
        raise ValueError(
            f"grid.stretch_factor {factor!r} overflows across n={n} cells"
        )
    total = xmax - xmin
    dx0 = total * (factor - 1.0) / (factor**n - 1.0)
    faces = [xmin + dx0 * (factor**k - 1.0) / (factor - 1.0) for k in range(n + 1)]
    faces[0] = xmin
    faces[-1] = xmax
    return faces


def _species_kind(name: str, kind_in: str, charge: float) -> str:
    """B1: fluids[].kind is the species ROLE ('neutral' | 'ion'), never a
    solver label. The legacy solver label 'hd' is rewritten from the charge
    (q == 0 -> neutral, q != 0 -> ion) so pre-B1 configs (e.g.
    examples/rad_hd_1d) keep working unchanged. 'mhd' stays refused: MHD is
    the problem-level switch cfg['mhd'] (all ions at once, face-centred B
    + constrained transport) -- never a per-fluid label and never "hd run
    as MHD with B=0". Explicit roles are validated against the charge:
    ions carry q_s != 0, neutrals q_s == 0; mismatches are refused here so
    xmf.setup() fails instead of silently running."""
    if kind_in == "hd":  # legacy solver label -> species role by charge
        return "neutral" if charge == 0.0 else "ion"
    if kind_in == "mhd":
        raise ValueError(
            f"fluid {name!r}: kind='mhd' is a solver label and stays "
            "refused; fluids[].kind is the species role 'neutral' or "
            "'ion' (the legacy kind='hd' is rewritten from charge). MHD "
            "is the problem-level switch cfg['mhd'] (all ions at once, "
            "face-centred B + constrained transport) -- never a "
            "per-fluid label and never run hd as MHD with B=0."
        )
    if kind_in not in ("neutral", "ion"):
        raise ValueError(
            f"fluid {name!r}: kind={kind_in!r} is not a species role; "
            "want 'neutral' or 'ion' (the legacy 'hd' is rewritten from "
            "the charge)"
        )
    if kind_in == "ion" and charge == 0.0:
        raise ValueError(
            f"fluid {name!r}: kind='ion' needs a non-zero charge q_s "
            "(ions are recognised by kind=ion and q_s != 0)"
        )
    if kind_in == "neutral" and charge != 0.0:
        raise ValueError(
            f"fluid {name!r}: kind='neutral' needs charge 0; a charged "
            "species is an ion (kind='ion')"
        )
    return kind_in


def _state(side: Mapping[str, Any], mass: float, units: Units) -> tuple[float, float, float]:
    n = float(side["n"])
    v = float(side.get("v", 0.0))
    if "p" in side:
        p = float(side["p"])
    else:
        p = n * KB * float(side["T"])
    unit_p = units.density * units.velocity * units.velocity
    return n * mass * AMU / units.density, v / units.velocity, p / unit_p


def normalize(cfg: Mapping[str, Any]) -> Problem:
    schema = int(cfg.get("schema", SCHEMA))
    if schema != SCHEMA:
        raise ValueError(f"unsupported config schema {schema} (want {SCHEMA})")
    if "GAMMA" in cfg:
        raise ValueError(
            "use numeric field 'gamma' (adiabatic index), not PLUTO label 'GAMMA'"
        )

    grid = _req(cfg, "grid")
    n = _as_n(grid)
    if n[0] < 1:
        raise ValueError("grid.n[0] must be >= 1")
    ndim = int(cfg.get("ndim", grid.get("ndim", 1 if n[1] == 1 and n[2] == 1 else 2 if n[2] == 1 else 3)))
    geom = _pick("geometry", GEOM, grid.get("geometry", cfg.get("geometry", "cartesian")))
    stretch = _pick("grid.stretch", STRETCH, grid.get("stretch", "none"))
    backend = _pick("backend", BACKEND, cfg.get("backend", "serial"))

    # Active-domain endpoints (needed below to build/validate face arrays).
    x1min = float(_req(cfg, "grid", "x1min"))
    x1max = float(_req(cfg, "grid", "x1max"))
    x2min = float(grid.get("x2min", 0.0))
    x2max = float(grid.get("x2max", 1.0))
    x3min = float(grid.get("x3min", 0.0))
    x3max = float(grid.get("x3max", 1.0))

    # Per-direction ACTIVE interface arrays (A1 stretch supplement). The
    # kernel's grid authority is these n_d+1 absolute positions each; C
    # validates strict monotonicity and pads ghosts. Python supplies them
    # from one of three sources per direction:
    #   1. an explicit user array grid.x1/x2/x3 (aliases x/y/z) --
    #      length n_d+1, finite, endpoints exactly xdmin/xdmax;
    #   2. the log factory for x1 when stretch is 'log' (legacy alias
    #      'x1') -- geometric widths via grid.stretch_factor;
    #   3. the uniform factory everywhere else (also the stretch='none'
    #      case and the degenerate factor==1.0).
    # An explicit x1 array and the log factory are mutually exclusive.
    faces_raw = _grid_faces_raw(grid)
    if stretch in ("log", "x1") and faces_raw[0] is not None:
        raise ValueError(
            "grid.x1 (explicit interface array) and grid.stretch='log' "
            "are mutually exclusive: the stretch factory builds x1's "
            "faces from stretch_factor; give one or the other"
        )
    faces: list[list[float]] = []
    for d, (raw_d, nd, lo, hi) in enumerate(
        (
            (faces_raw[0], n[0], x1min, x1max),
            (faces_raw[1], n[1], x2min, x2max),
            (faces_raw[2], n[2], x3min, x3max),
        )
    ):
        name = f"x{d + 1}"
        if raw_d is not None:
            faces.append(_axis_faces(name, raw_d, nd, lo, hi))
        elif d == 0 and stretch in ("log", "x1"):
            faces.append(
                _log_faces(lo, hi, nd, grid.get("stretch_factor", 1.0))
            )
        else:
            faces.append(_uniform_faces(lo, hi, nd))

    time = _req(cfg, "time")
    num_in = cfg.get("numerics") or {}
    numerics = Numerics(
        riemann=_pick("numerics.riemann", RIEMANN, num_in.get("riemann", "hll")),
        reconstruction=_pick("numerics.reconstruction", RECON, num_in.get("reconstruction", "pcm")),
        rk=_pick("numerics.rk", RK, num_in.get("rk", time.get("rk", "rk1"))),
        cfl=float(time["cfl"]),
        dt_max=float(time.get("dt_max", 0.0)),
        tstop=float(time["tstop"]),
    )
    u_in = cfg.get("units") or {}
    units = Units(
        density=float(u_in.get("density", UNIT_DENSITY)),
        velocity=float(u_in.get("velocity", UNIT_VELOCITY)),
        length=float(u_in.get("length", UNIT_LENGTH)),
    )

    fluids_raw = _req(cfg, "fluids")
    if not fluids_raw:
        raise ValueError("fluids must contain at least one entry")
    fluids = [
        FluidSpec(
            name=str(f.get("name", f"f{i}")),
            mass=float(f.get("mass", 1.00794)),
            charge=float(f.get("charge", 0.0)),
            kind=_species_kind(
                str(f.get("name", f"f{i}")),
                str(f.get("kind", "hd")).lower(),
                float(f.get("charge", 0.0)),
            ),
        )
        for i, f in enumerate(fluids_raw)
    ]
    ele_raw = cfg.get("electron") or {}
    electron = ElectronSpec(
        mode=_pick("electron.mode", ELECTRON, ele_raw.get("mode", "off")),
        share=str(ele_raw.get("share", "by-charge")),
        # B2: the tier is a kernel enum now (flat blob's former
        # reserved0 slot); unknown names are refused here, never silently
        # run as ions-only.
        shock_heat=_pick(
            "electron.shock_heat", SHOCK_HEAT, ele_raw.get("shock_heat", "ions-only")
        ),
    )

    # M block 1 (M1): the problem-level MHD switch. A boolean (mhd: true)
    # turns magnetics on for the WHOLE problem -- all ions at once,
    # face-centred B + constrained transport -- never a per-fluid solver
    # label (kind='mhd' stays refused by _species_kind above). The switch
    # rides the auxiliary xmf_mag_load port (extra["mhd"] below, pushed by
    # api.setup): it never enters the flat XmfConfig blob, whose byte
    # layout stays frozen. mhd_reflect picks the reflect-face mirror tier
    # for the B block (wall / symmetric, MHD_REFLECT); it means nothing
    # with magnetics off, so giving it without the switch is refused.
    mhd_raw = cfg.get("mhd", False)
    if isinstance(mhd_raw, (str, bytes)) or not isinstance(mhd_raw, (bool, int)):
        raise ValueError(f"mhd must be a boolean switch, got {mhd_raw!r}")
    mhd_on = bool(mhd_raw)
    mhd_reflect_in = cfg.get("mhd_reflect")
    if mhd_reflect_in is not None and not mhd_on:
        raise ValueError(
            "mhd_reflect requires the problem-level mhd switch: the "
            "reflect tier is the magnetic block's mirror rule and means "
            "nothing with magnetics off"
        )
    mhd_reflect = _pick(
        "mhd_reflect", MHD_REFLECT,
        "wall" if mhd_reflect_in is None else mhd_reflect_in,
    )

    sources = list(cfg.get("sources") or [])
    radiation = dict(cfg.get("radiation") or {"enabled": False, "absorbers": []})
    constants = dict(cfg.get("constants") or {})
    body = cfg.get("body_force", "none")
    if body not in ("none", None, "") and not isinstance(body, Mapping):
        raise ValueError(
            "body_force is not a C enum; write a sources[] expression or omit it"
        )
    boundary = _faces(cfg.get("boundary"))
    # Periodic dimension validation: x2/x3 periodic requires ny>1/nz>1
    # (the ghost wrap needs a non-degenerate transverse extent).
    if boundary.get("x2beg") == "periodic" and n[1] < 2:
        raise ValueError("x2 periodic requires grid.ny >= 2")
    if boundary.get("x3beg") == "periodic" and n[2] < 2:
        raise ValueError("x3 periodic requires grid.nz >= 2")

    ic = dict(_req(cfg, "ic"))
    ic_kind = str(ic.get("kind", "riemann" if "left" in ic else "none"))
    mass = fluids[0].mass
    rho_l = v_l = p_l = rho_r = v_r = p_r = 0.0
    x_disc = float(ic.get("x0", 0.0))
    if ic_kind == "riemann":
        rho_l, v_l, p_l = _state(ic["left"], mass, units)
        rho_r, v_r, p_r = _state(ic["right"], mass, units)
    elif ic_kind not in ("file", "array", "none"):
        raise ValueError(f"unknown ic.kind {ic_kind}")

    gamma = float(cfg.get("gamma", 5.0 / 3.0))
    if gamma <= 1.0:
        raise ValueError("gamma must be the adiabatic index > 1, not a table slot")

    extra: dict[str, Any] = {}
    if sources:
        from .source_ir import compile_sources, env_from

        n_beam = 1
        rad = radiation if isinstance(radiation, Mapping) else {}
        raw_bins = rad.get("bins") if isinstance(rad.get("bins"), Mapping) else {}
        if isinstance(raw_bins, Mapping) and "n" in raw_bins:
            n_beam = int(raw_bins["n"])
        ir_env = env_from(
            fluids,
            constants,
            electron=electron.mode != "off",
            n_beam=n_beam,
        )
        extra["source_tapes"] = compile_sources(sources, ir_env, constants)

    # A3 supplement: analytic user boundaries. The user dict key
    # ``user_bc`` attaches closed assignment expressions per face:
    #     "user_bc": {"x1beg": {"n_HI": "n_HI@1", "P_HI": "P_HI@1"}}
    # Every named face must already carry boundary type ``user``. The
    # expressions are compiled by xmf.bc_ir into boundary tapes and go to
    # the kernel through xmf_bc_load_tapes (per-face sets, assignment
    # semantics) -- NEVER through the source tape set. The floor channel
    # (boundary_floor -> extra["user_bc"] -> xmf_bc_set_user) is a
    # separate key and stays the pinned-base-state path: the two are
    # mutually exclusive on the x1 faces the floor pins, checked here
    # before either reaches extra.
    user_bc_in = cfg.get("user_bc")
    if user_bc_in is not None:
        if not isinstance(user_bc_in, Mapping):
            raise ValueError(
                "user_bc must be a mapping face -> {slot: expression}"
            )
        if cfg.get("boundary_floor") is not None and any(
            name in ("x1beg", "x1end") for name in user_bc_in
        ):
            raise ValueError(
                "user_bc and boundary_floor both act on the same x1 user "
                "face and are mutually exclusive per face: the floor pins "
                "ghosts AND the innermost active cell, the tapes assign "
                "ghosts only -- pick one per x1 face"
            )
        # Periodic faces cannot carry user_bc: the ghost values come from
        # the opposite active domain, not from expressions.
        for face_name in user_bc_in:
            if boundary.get(face_name) == "periodic":
                raise ValueError(
                    f"user_bc on {face_name} conflicts with periodic boundary: "
                    "periodic ghosts wrap from the opposite active domain"
                )
        from .bc_ir import compile_user_bc
        from .source_ir import env_from

        # Boundary expressions never read beams or electron quantities;
        # the environment is built with both closed regardless of the
        # problem's own settings (bc_ir refuses the names itself too).
        bc_env = env_from(fluids, constants)
        extra["bc_tapes"] = compile_user_bc(user_bc_in, boundary, bc_env, constants)

    # A5 radiation beam table (auxiliary LOAD PORT -- never the flat
    # XmfConfig). Present exactly when radiation.bins carries a table;
    # api.setup pushes it through xmf_rad_load right after setup. The
    # scan enums are named here and mapped to the kernel integers
    # (mode grid_line=1, dilution none=0, sense inward=0); anything else
    # is refused at the config layer. With radiation.enabled false the
    # boundary fluxes are zeroed so FLUX slots read the vacuum value 0
    # while the source tapes stay loaded (off-radiation comparison).
    rad_bins = radiation.get("bins") if isinstance(radiation, Mapping) else None
    if isinstance(rad_bins, Mapping) and "n" in rad_bins:
        scan = radiation.get("scan") if isinstance(radiation.get("scan"), Mapping) else {}
        mode = str(scan.get("mode", "grid_line"))
        dilution = str(scan.get("dilution", "none"))
        sense = str(scan.get("sense", "inward"))
        if mode != "grid_line":
            raise ValueError(f"radiation.scan.mode {mode!r} is not implemented (want 'grid_line')")
        if dilution != "none":
            raise ValueError(f"radiation.scan.dilution {dilution!r} is not implemented (want 'none')")
        if sense != "inward":
            raise ValueError(f"radiation.scan.sense {sense!r} is not implemented (want 'inward')")
        absorbers = list(radiation.get("absorbers") or [])
        fluid_names = [f.name for f in fluids]
        if not absorbers or any(str(a) not in fluid_names for a in absorbers):
            raise ValueError("radiation.bins needs radiation.absorbers naming a fluid")
        nbeam = int(rad_bins["n"])
        hnu = [float(v) for v in rad_bins["hnu"]]
        finf = [float(v) for v in rad_bins["flux_inf"]]
        if len(hnu) != nbeam or len(finf) != nbeam:
            raise ValueError("radiation.bins sigma/hnu/flux_inf must each carry bins.n entries")
        # B3 sigma shape contract: one absorber keeps the A5 flat table
        # (sigma of length bins.n, bit for bit the old rad_load payload);
        # several absorbers need ONE sigma row per absorber (each of
        # length bins.n) -- absorbers[1:] are never silently dropped.
        sigma_raw = rad_bins["sigma"]
        if len(absorbers) == 1:
            sigma_rows = [[float(v) for v in sigma_raw]]
            if len(sigma_rows[0]) != nbeam:
                raise ValueError("radiation.bins sigma/hnu/flux_inf must each carry bins.n entries")
        else:
            rows = list(sigma_raw)
            if rows and not isinstance(rows[0], (list, tuple)):
                raise ValueError(
                    "radiation.bins sigma must carry one bins.n row per absorber "
                    f"({len(absorbers)} absorbers, got a flat table); "
                    "absorbers beyond the first are never silently dropped"
                )
            sigma_rows = [list(row) for row in rows]
            if len(sigma_rows) != len(absorbers):
                raise ValueError(
                    "radiation.bins sigma must carry one bins.n row per absorber "
                    f"({len(absorbers)} absorbers, got {len(sigma_rows)} rows); "
                    "absorbers beyond the first are never silently dropped"
                )
            sigma_rows = [[float(v) for v in row] for row in sigma_rows]
            if any(len(row) != nbeam for row in sigma_rows):
                raise ValueError(
                    "radiation.bins sigma rows must each carry bins.n entries"
                )
        # Vacuum contract: with enabled false the scan never runs and the
        # IR FLUX slots read the loaded boundary value finf. Off-radiation
        # comparison runs therefore pass finf == 0 in the table itself
        # (the factory's choice), never by rewriting the table here.
        extra["rad_load"] = {
            "enabled": 1 if bool(radiation.get("enabled")) else 0,
            "mode": 1,
            "dilution": 0,
            "sense": 0,
            "absorber": fluid_names.index(str(absorbers[0])),
            "sigma": sigma_rows[0],
            "hnu": hnu,
            "finf": finf,
        }
        if len(absorbers) > 1:
            # B3 appended multi-absorber port: index table + sigma row-major
            # [nabs x nbeam]; api.setup pushes it through
            # xmf_rad_load_absorbers right after xmf_rad_load.
            extra["rad_load_multi"] = {
                "absorbers": [fluid_names.index(str(a)) for a in absorbers],
                "sigma": [v for row in sigma_rows for v in row],
            }

    # A5 physics port (auxiliary LOAD PORT, opt-in): real per-fluid
    # masses and units are loaded exactly when a compiled tape reads a
    # derived T or a beam flux F_b, or when a beam table is loaded (the
    # scan needs the absorber number density). Plain configs keep every
    # mass_code and unit at 1.0 -- the pre-A5 numerics, bit for bit
    # (the empty-table pcm Sod pins stay untouched). B1 electron-on
    # configs deliberately do NOT force this port: the quasi-neutral
    # closure divides rho by the SAME mass_code the IR boundary uses
    # (n = rho/mass_code), so with no phys load n_e = sum q_s rho stays
    # the code-unit convention bit for bit, exactly like LOAD n -- and
    # the pre-B1 electron-on fixtures (bc-tape ghost pins) keep their
    # mass_code == 1 numerics.
    need_phys = "rad_load" in extra
    if not need_phys:
        for tape in extra.get("source_tapes") or ():
            for op, arg in tape.ops:
                if op == 1:  # LOAD_SLOT
                    kind = (arg >> 16) & 0xFF
                    b = arg & 0xFF
                    if kind == 3 or (kind == 1 and b == 5):  # FLUX or fluid T
                        need_phys = True
                        break
            if need_phys:
                break
    if not need_phys:
        # Boundary tapes reading a neighbour temperature T@1/T@2 need the
        # same conversion (T = P*unit_P/(n*kB)) as the source IR's T.
        for _face, tapes in extra.get("bc_tapes") or ():
            for tape in tapes:
                for op, arg in tape.ops:
                    if op == 1:  # LOAD_SLOT
                        kind = (arg >> 16) & 0xFF
                        if kind == 4 and (arg & 0x0F) == 5:  # NB T
                            need_phys = True
                            break
                if need_phys:
                    break
            if need_phys:
                break
    if need_phys:
        extra["phys"] = {
            "mass_amu": [f.mass for f in fluids],
            "unit_density": units.density,
            "unit_velocity": units.velocity,
            "unit_length": units.length,
        }

    # B1 charge port (auxiliary LOAD PORT): the per-fluid charges q_s in
    # elementary-charge multiples ride beside the masses -- NEVER the flat
    # XmfConfig (a variable-length charge table never enters the blob).
    # Loaded exactly when electrons are on; the kernel's ion role is
    # q_s != 0 (the kernel never recognises a species name). With
    # electrons off no charge table is pushed and the kernel's closure
    # gate stays off, keeping the pre-B1 hydro path bit for bit.
    # M4: the magnetic gate recognises ions through the SAME table
    # (xmf_mag_load refuses a gate open without any charge > 0 fluid),
    # so magnetics on pushes the charges even with electrons off -- the
    # kernel's electron closure gate still requires the P_e block and
    # stays off, so the hydro path without magnetics is untouched.
    if electron.mode != "off" or mhd_on:
        extra["charge"] = [f.charge for f in fluids]

    # M block 1 magnetic port (auxiliary LOAD PORT, opt-in): present
    # exactly when the problem-level mhd switch is on. api.setup pushes it
    # through xmf_mag_load right after the charge port: the face-centred B
    # blocks start zero (block 1 = HD evolution + frozen B bystander) and
    # the reflect tier is the chosen mirror rule. With the switch off no
    # port is pushed and every magnetic helper refuses (-1), keeping the
    # pre-M path bit for bit.
    if mhd_on:
        extra["mhd"] = {"reflect": MHD_REFLECT[mhd_reflect]}

    # A5 user boundary floor: the pinned base state of an x1 user face
    # arrives as a plain buffer [fluid][slot][k*nytot+j]; api.setup
    # pushes it through xmf_bc_set_user. Without it a user face keeps
    # the pre-A5 copy padding.
    floor = cfg.get("boundary_floor")
    if floor is not None:
        # Periodic x1 faces conflict with boundary_floor: the floor pins
        # ghost values, but periodic ghosts wrap from the opposite domain.
        if boundary.get("x1beg") == "periodic":
            raise ValueError(
                "boundary_floor conflicts with periodic x1 boundary: "
                "periodic ghosts wrap from the opposite active domain"
            )
        extra["user_bc"] = floor

    return Problem(
        schema=schema,
        gamma=gamma,
        backend=backend,
        ndim=ndim,
        n=n,
        ng=int(grid.get("nghost", 2)),
        geometry=geom,
        stretch=stretch,
        x1min=x1min,
        x1max=x1max,
        x2min=x2min,
        x2max=x2max,
        x3min=x3min,
        x3max=x3max,
        xf1=tuple(faces[0]),
        xf2=tuple(faces[1]),
        xf3=tuple(faces[2]),
        numerics=numerics,
        units=units,
        fluids=fluids,
        electron=electron,
        sources=sources,
        sources_enabled=bool(sources),
        radiation=radiation,
        constants=constants,
        body_force=body,
        boundary=boundary,
        ic=ic,
        ic_kind=ic_kind,
        mass=mass,
        fluid=fluids[0].name,
        x_disc=x_disc,
        rho_l=rho_l,
        v_l=v_l,
        p_l=p_l,
        rho_r=rho_r,
        v_r=v_r,
        p_r=p_r,
        extra=extra,
    )


KernelConfig = Problem


def pack(cfg: Mapping[str, Any]) -> Problem:
    return normalize(cfg)


def shock_tube() -> dict[str, Any]:
    """Uniform 1D tube. Demo kernel is first-order HLL + RK1."""
    return {
        "schema": SCHEMA,
        "backend": "serial",
        "gamma": 5.0 / 3.0,
        "units": {
            "density": UNIT_DENSITY,
            "velocity": UNIT_VELOCITY,
            "length": UNIT_LENGTH,
        },
        "numerics": {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"},
        "grid": {
            "geometry": "cartesian",
            "ndim": 1,
            "n": [256, 1, 1],
            "stretch": "none",
            "x1min": 1.0,
            "x1max": 10.0,
            "nghost": 2,
        },
        "time": {"cfl": 0.4, "tstop": 0.15, "dt_max": 0.0},
        # B1: kind is the species role; the legacy 'hd' label is still
        # accepted and rewritten from the charge (q=0 -> neutral).
        "fluids": [{"name": "HI", "mass": 1.00794, "charge": 0, "kind": "neutral"}],
        "electron": {"mode": "off"},
        "sources": [],
        "constants": {},
        "radiation": {"enabled": False, "absorbers": []},
        "boundary": {"x1": "reflect"},
        "ic": {
            "kind": "riemann",
            "x0": 5.5,
            "left": {"n": 1.0e13, "v": 0.0, "T": 1.0e4},
            "right": {"n": 1.25e12, "v": 0.0, "T": 8.0e3},
        },
    }


def full_schema_example() -> dict[str, Any]:
    """All frontend keys. Kernel may still ignore sources; reconstruction
    (weno3 here) is real -- the kernel never silently downgrades it."""
    cfg = shock_tube()
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "weno3", "rk": "rk3"}
    cfg["grid"]["geometry"] = "spherical"
    cfg["grid"]["stretch"] = "x1"
    cfg["grid"]["x2min"] = 0.0
    cfg["grid"]["x2max"] = 3.141592653589793
    # B1: kind is the species role ('neutral' | 'ion'), consistent with
    # the charge; the legacy solver label 'hd' is still accepted and
    # rewritten from the charge by normalize().
    cfg["fluids"] = [
        {"name": "HI", "mass": 1.00794, "charge": 0, "kind": "neutral"},
        {"name": "HEI", "mass": 4.0026, "charge": 0, "kind": "neutral"},
        {"name": "HII", "mass": 1.00794, "charge": 1, "kind": "ion"},
        {"name": "HEII", "mass": 4.0026, "charge": 1, "kind": "ion"},
    ]
    cfg["electron"] = {
        "mode": "pressure-only",
        "share": "by-charge",
        "shock_heat": "ions-only",
    }
    cfg["constants"] = {
        "G": 6.6743e-8,
        "M_p": 1.0,
        "M_star": 1.0,
        "a": 1.0,
        "k_cx": 1.0e-9,
    }
    cfg["sources"] = [
        {
            "name": "charge_exchange",
            "expr": {"dn_HI": "-k_cx * n_HI * n_HEII", "dn_HEII": "k_cx * n_HI * n_HEII"},
        },
        {
            "name": "gravity_tide",
            "expr": {"dvx": "-G * M_p / r**2 + 3 * G * M_star * r / a**3"},
        },
    ]
    cfg["radiation"] = {"enabled": True, "absorbers": ["HI", "HEI"]}
    cfg["boundary"] = {"x1beg": "user", "x1end": "outflow"}
    return cfg
