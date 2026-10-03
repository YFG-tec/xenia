"""Source-term identifier contract and a small AST → opcode tape.

Python owns names. The C kernel will later see only opcodes and slot
indices (see src/xmf/core/xmf_ir.h). This module does not run inside
xmf_step.

Read names:  n_<fluid> vx_ vy_ vz_ P_ T_ ; n_e P_e T_e ; r x1 x2 x3 ; F F_b ;
             Bx By Bz (M1 magnetic name space -- parsed here, the C kernel
             refuses the kind at load until a later M block supplies B)
Write names: dn_ dvx_ dvy_ dvz_ dP_ + fluid ; bare dvx/dvy/dvz = every fluid ;
             dP_e (B2) writes the electron pressure block directly -- only
             when electron.mode != off, never a fluid slot ;
             dBx dBy dBz (M1 appended write kinds 6..8 -- numbers assigned
             now, the C kernel refuses them at load this slice)
Constants:   keys of constants{}. T is derived from P; there is no dT_.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# Keep numeric values in lockstep with src/xmf/core/xmf_ir.h.
OP_LOAD_CONST = 0
OP_LOAD_SLOT = 1
OP_ADD = 2
OP_SUB = 3
OP_MUL = 4
OP_DIV = 5
OP_POW = 6
OP_NEG = 7
OP_EXP = 8
OP_LOG = 9

KIND_GRID = 0
KIND_FLUID = 1
KIND_ELEC = 2
KIND_FLUX = 3
# A3 supplement: neighbour reads of boundary assignment tapes (xmf.bc_ir).
# Appended kind only -- opcodes 0..9 stay frozen, source tapes never carry
# it (xmf_ir_load refuses it) and it never enters the Newton solve.
KIND_NB = 4
# M block 1 (M1): magnetic-field reads, packed slot a = 0/1/2 = B1/B2/B3
# component. Appended kind only -- opcodes 0..9 stay frozen and the fluid
# five slots never carry it. Python compiles the names; the C kernel
# refuses to LOAD the kind this slice (no magnetic tape-slot supply yet),
# exactly the NB precedent -- appended, enumerated, never silently
# loadable. The magnetic state itself never enters xmf_ir_nstate.
KIND_MAG = 5

GRID_X1 = 0
GRID_X2 = 1
GRID_X3 = 2
# r is an alias of x1 (radial / first axis), not a fourth coordinate.

FLUID_N = 0
FLUID_VX = 1
FLUID_VY = 2
FLUID_VZ = 3
FLUID_P = 4
FLUID_T = 5  # derived; never a stored slot

ELEC_N = 0
ELEC_P = 1
ELEC_T = 2

# M1 magnetic read components (KIND_MAG slot a): B1/B2/B3.
MAG_BX = 0
MAG_BY = 1
MAG_BZ = 2

WRITE_DN = 0
WRITE_DVX = 1
WRITE_DVY = 2
WRITE_DVZ = 3
WRITE_DP = 4
# B2 appended write name: the increment enters the optional electron
# pressure block P_e DIRECTLY (never a fluid slot -- no "write into ion P
# then project"). dest_fluid is -1 and the C kernel refuses the tape at
# load when electrons are off. Appended name only: opcodes 0..9 frozen.
WRITE_DPE = 5
# M block 1 (M1) appended write names: dBx/dBy/dBz increments for the
# magnetic block. The numbers are assigned now so the IR stays append-only
# (never a renumbering of the frozen DN..DPE), but this slice wires no
# magnetic increment target -- the C xmf_ir_load refuses these kinds at
# load, exactly the rk4 precedent (enumerated, never silently runnable).
# dest_fluid is always -1 (the magnetic block is not a fluid).
WRITE_DBX = 6
WRITE_DBY = 7
WRITE_DBZ = 8

FIELD_READ = {
    "n": FLUID_N,
    "vx": FLUID_VX,
    "vy": FLUID_VY,
    "vz": FLUID_VZ,
    "P": FLUID_P,
    "T": FLUID_T,
}
FIELD_WRITE = {
    "dn": WRITE_DN,
    "dvx": WRITE_DVX,
    "dvy": WRITE_DVY,
    "dvz": WRITE_DVZ,
    "dP": WRITE_DP,
}

RESERVED = frozenset(
    {
        "r",
        "x1",
        "x2",
        "x3",
        "F",
        "e",
        "n",
        "vx",
        "vy",
        "vz",
        "P",
        "T",
        "dn",
        "dvx",
        "dvy",
        "dvz",
        "dP",
        "dP_e",
        # M1 magnetic name space (read Bx/By/Bz + write dBx/dBy/dBz):
        # reserved so a fluid or constant can never shadow a B slot.
        "Bx",
        "By",
        "Bz",
        "dBx",
        "dBy",
        "dBz",
        "exp",
        "log",
        "VX1",
        "RHO",
        "IEND",
        "IBEG",
    }
)
_BANNED_BARE = frozenset(
    {"VX1", "VX2", "VX3", "RHO", "IEND", "IBEG", "NDT", "PRS", "TMP"}
)


def _is_flux_alias(name: str) -> bool:
    """True for the beam identifiers F_0..F_b (F itself handled separately)."""
    return name.startswith("F_") and name[2:].isdigit()


def pack_slot(kind: int, a: int, b: int = 0) -> int:
    """kind in 0..3, a in 0..255, b in 0..255."""
    return (kind << 16) | ((a & 0xFF) << 8) | (b & 0xFF)


def unpack_slot(code: int) -> tuple[int, int, int]:
    return (code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF


@dataclass(frozen=True)
class WriteTarget:
    """dest_fluid is -1 when a bare dvx/dvy/dvz applies to every fluid."""

    kind: int
    dest_fluid: int


@dataclass(frozen=True)
class SourceTape:
    dest: WriteTarget
    ops: tuple[tuple[int, int], ...]
    consts: tuple[float, ...]
    expr: str


@dataclass(frozen=True)
class NameEnv:
    fluids: tuple[str, ...]
    constants: tuple[str, ...]
    electron: bool
    n_beam: int

    @property
    def fluid_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.fluids)}


def check_fluid_name(name: str) -> str:
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"fluid name {name!r} is not a legal identifier")
    if name in RESERVED:
        raise ValueError(f"fluid name {name!r} collides with a reserved source-IR word")
    if _is_flux_alias(name):
        raise ValueError(f"fluid name {name!r} collides with the flux identifier F_b")
    return name


def env_from(
    fluids: Sequence[Any],
    constants: Mapping[str, Any],
    *,
    electron: bool = False,
    n_beam: int = 1,
) -> NameEnv:
    names = []
    for spec in fluids:
        name = spec if isinstance(spec, str) else spec.name
        names.append(check_fluid_name(str(name)))
    if len(set(names)) != len(names):
        raise ValueError("fluid names must be unique")
    # Constants are folded into the tape at compile time (LOAD_CONST), so
    # a constant key must never shadow a read name: reserved words, the
    # electron aliases, fluid names, or any <field>_<fluid> slot name.
    taken = set(RESERVED)
    taken.update(("n_e", "P_e", "T_e", "dP_e"))
    taken.update(names)
    for fname in names:
        for prefix in FIELD_READ:
            taken.add(prefix + "_" + fname)
        for prefix in FIELD_WRITE:
            taken.add(prefix + "_" + fname)
    const_names = []
    for key in constants:
        key = str(key)
        if not key.isidentifier():
            raise ValueError(f"constant {key!r} is not a legal identifier")
        if key in taken:
            raise ValueError(f"constant {key!r} shadows a reserved source-IR name")
        if _is_flux_alias(key):
            raise ValueError(f"constant {key!r} collides with the flux identifier F_b")
        const_names.append(key)
    if n_beam < 0:
        raise ValueError("n_beam must be >= 0")
    return NameEnv(
        fluids=tuple(names),
        constants=tuple(const_names),
        electron=bool(electron),
        n_beam=int(n_beam),
    )


def parse_write_name(name: str, env: NameEnv) -> WriteTarget:
    if name in FIELD_WRITE:
        if name == "dn" or name == "dP":
            raise ValueError(
                f"{name!r} must be qualified with a fluid (e.g. {name}_H)"
            )
        return WriteTarget(kind=FIELD_WRITE[name], dest_fluid=-1)
    if name == "dP_e":
        # B2: exact intercept BEFORE the dP_ prefix loop below -- "e" is
        # not a fluid (RESERVED), the electron pressure block is. The
        # write targets the block directly; compiling it with electrons
        # off is refused here (the C xmf_ir_load repeats the refusal).
        if not env.electron:
            raise ValueError(
                "write name 'dP_e' requires electron.mode != off: the "
                "electron pressure block does not exist when electrons "
                "are off"
            )
        return WriteTarget(kind=WRITE_DPE, dest_fluid=-1)
    if name in ("dBx", "dBy", "dBz"):
        # M1 appended magnetic write names: compiled into the appended
        # XmfIrWrite numbers 6..8 with dest_fluid -1 (the magnetic block
        # is not a fluid). The C kernel refuses them at load this slice
        # (no magnetic increment target yet) -- exactly the rk4
        # precedent. Intercepted before the dP_/dn_ prefix loop below.
        return WriteTarget(
            kind={"dBx": WRITE_DBX, "dBy": WRITE_DBY, "dBz": WRITE_DBZ}[name],
            dest_fluid=-1,
        )
    for prefix, kind in FIELD_WRITE.items():
        tag = prefix + "_"
        if name.startswith(tag):
            fluid = name[len(tag) :]
            if fluid not in env.fluid_index:
                raise ValueError(f"unknown fluid {fluid!r} in write name {name!r}")
            return WriteTarget(kind=kind, dest_fluid=env.fluid_index[fluid])
    if name.startswith("dT"):
        raise ValueError("dT_ is not a source slot; write dP_ and derive T from P")
    raise ValueError(f"illegal write name {name!r}")


def _resolve_read(name: str, env: NameEnv) -> int:
    if name in _BANNED_BARE or name in env.fluids:
        raise ValueError(
            f"bare name {name!r} is not a field; use n_{name} / P_{name} / …"
            if name in env.fluids
            else f"banned identifier {name!r} (PLUTO-era slot or loop index)"
        )
    if name == "r" or name == "x1":
        return pack_slot(KIND_GRID, GRID_X1)
    if name == "x2":
        return pack_slot(KIND_GRID, GRID_X2)
    if name == "x3":
        return pack_slot(KIND_GRID, GRID_X3)
    if name == "F":
        if env.n_beam < 1:
            raise ValueError("F is illegal when n_beam == 0")
        return pack_slot(KIND_FLUX, 0)
    if _is_flux_alias(name):
        b = int(name[2:])
        if b < 0 or b >= env.n_beam:
            raise ValueError(f"{name!r} is outside n_beam={env.n_beam}")
        return pack_slot(KIND_FLUX, b)
    if name in ("n_e", "P_e", "T_e"):
        if not env.electron:
            raise ValueError(f"{name!r} requires electron.mode != off")
        return pack_slot(KIND_ELEC, {"n_e": ELEC_N, "P_e": ELEC_P, "T_e": ELEC_T}[name])
    if name in ("Bx", "By", "Bz"):
        # M1 magnetic read names. Python parses them into the appended
        # KIND_MAG packed slot; the C kernel refuses to load the kind in
        # this slice (the rk4/NB precedent -- enumerated, never silently
        # loadable), so a tape using them fails at setup, not mid-step.
        return pack_slot(KIND_MAG, {"Bx": MAG_BX, "By": MAG_BY, "Bz": MAG_BZ}[name])
    for prefix, var in FIELD_READ.items():
        tag = prefix + "_"
        if name.startswith(tag):
            fluid = name[len(tag) :]
            if fluid not in env.fluid_index:
                raise ValueError(f"unknown fluid {fluid!r} in {name!r}")
            return pack_slot(KIND_FLUID, env.fluid_index[fluid], var)
    raise ValueError(f"unknown identifier {name!r}")


class _Compiler(ast.NodeVisitor):
    def __init__(self, env: NameEnv, const_values: Mapping[str, float]):
        self.env = env
        self.const_values = {k: float(v) for k, v in const_values.items()}
        self.ops: list[tuple[int, int]] = []
        self.consts: list[float] = []

    def _const(self, value: float) -> None:
        self.ops.append((OP_LOAD_CONST, len(self.consts)))
        self.consts.append(float(value))

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"illegal literal {node.value!r}")
        self._const(float(node.value))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, ast.USub):
            raise ValueError("only unary minus is allowed")
        self.visit(node.operand)
        self.ops.append((OP_NEG, 0))

    def visit_BinOp(self, node: ast.BinOp) -> None:
        table = {
            ast.Add: OP_ADD,
            ast.Sub: OP_SUB,
            ast.Mult: OP_MUL,
            ast.Div: OP_DIV,
            ast.Pow: OP_POW,
        }
        op = table.get(type(node.op))
        if op is None:
            raise ValueError(f"illegal operator {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)
        self.ops.append((op, 0))

    def visit_Call(self, node: ast.Call) -> None:
        if node.keywords or len(node.args) != 1 or not isinstance(node.func, ast.Name):
            raise ValueError("only exp(x) and log(x) calls are allowed")
        fn = node.func.id
        if fn not in ("exp", "log"):
            raise ValueError(f"illegal function {fn!r}")
        self.visit(node.args[0])
        self.ops.append((OP_EXP if fn == "exp" else OP_LOG, 0))

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("assignment inside an expression is illegal")
        if node.id in self.const_values:
            self._const(self.const_values[node.id])
            return
        self.ops.append((OP_LOAD_SLOT, _resolve_read(node.id, self.env)))

    def generic_visit(self, node: ast.AST) -> None:
        raise ValueError(f"illegal syntax {type(node).__name__}")


def compile_expr(
    expr: str,
    env: NameEnv,
    constants: Mapping[str, Any],
    dest_name: str,
) -> SourceTape:
    dest = parse_write_name(dest_name, env)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {expr!r}: {exc}") from exc
    comp = _Compiler(env, {k: float(v) for k, v in constants.items()})
    comp.visit(tree)
    if not comp.ops:
        raise ValueError(f"empty expression {expr!r}")
    return SourceTape(
        dest=dest,
        ops=tuple(comp.ops),
        consts=tuple(comp.consts),
        expr=expr,
    )


def compile_sources(
    sources: Iterable[Mapping[str, Any]],
    env: NameEnv,
    constants: Mapping[str, Any],
) -> tuple[SourceTape, ...]:
    tapes: list[SourceTape] = []
    for item in sources:
        exprs = item.get("expr")
        if not isinstance(exprs, Mapping) or not exprs:
            raise ValueError("each sources[] entry needs a non-empty expr mapping")
        for dest_name, raw in exprs.items():
            tapes.append(compile_expr(str(raw), env, constants, str(dest_name)))
    return tuple(tapes)


def eval_tape(
    tape: SourceTape,
    *,
    slots: Mapping[int, float],
) -> float:
    """Host-side interpreter for tests. Same ops as the future C stack machine."""
    stack: list[float] = []
    for op, arg in tape.ops:
        if op == OP_LOAD_CONST:
            stack.append(tape.consts[arg])
        elif op == OP_LOAD_SLOT:
            if arg not in slots:
                raise KeyError(f"missing slot {arg}")
            stack.append(float(slots[arg]))
        elif op == OP_NEG:
            stack.append(-stack.pop())
        elif op == OP_EXP:
            stack.append(math.exp(stack.pop()))
        elif op == OP_LOG:
            stack.append(math.log(stack.pop()))
        else:
            b = stack.pop()
            a = stack.pop()
            if op == OP_ADD:
                stack.append(a + b)
            elif op == OP_SUB:
                stack.append(a - b)
            elif op == OP_MUL:
                stack.append(a * b)
            elif op == OP_DIV:
                stack.append(a / b)
            elif op == OP_POW:
                stack.append(a**b)
            else:
                raise ValueError(f"unknown opcode {op}")
    if len(stack) != 1:
        raise ValueError("source tape did not leave a single value")
    return stack[0]
