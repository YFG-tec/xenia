"""Boundary assignment tapes for analytic ``user`` faces (A3 supplement).

Python owns the names; the C kernel (xmf_bc.c) only evaluates the packed
integer tapes on ghost cells. This module compiles the closed per-face
expressions of the ``user_bc`` config key -- it never runs inside
xmf_step.

Semantics are ASSIGNMENT, not the increment semantics of the source IR
(xmf.source_ir): every tape writes one fluid slot of every ghost cell of
its face. There are no dn_/dvx_/dP_/dT_ write names here, and boundary
tapes never enter run->tapes, xmf_ir_apply or the source Newton solve.

Write names (the ghost cell being filled):
    n_<fluid>  vx_<fluid>  vy_<fluid>  vz_<fluid>  P_<fluid>
Read names, neighbour layers (face-relative, beg/end and axis agnostic):
    <field>_<fluid>@1 / @2   with field in n vx vy vz P T
    @1 = the outermost ACTIVE cell of the face, @2 = the next one inward.
    Deeper layers (@3, @i-1, ...) are refused.
Read names, the ghost's own cell centre:
    r x1 x2 x3   (r aliases x1; the coordinates of the ghost being
                  filled, so a hydrostatic P(r) profile can be written)
Constants:
    keys of constants{}. Beam fluxes F_b and electron quantities
    (n_e/P_e/T_e) are refused in boundary expressions.

Neighbour slots carry the appended IR kind KIND_NB = 4 (opcodes 0..9 are
frozen, only the kind table grows): packed slot
    (KIND_NB << 16) | (fluid << 8) | ((layer << 4) | var)
with layer in {1, 2} and var the usual fluid-slot numbers
(n=0, vx=1, vy=2, vz=3, P=4, T=5 derived).

Parsing note: Python's own grammar treats ``@`` as the matmul operator
(same precedence as ``*``, so ``2*n@1`` would mis-group). The compiler
therefore rewrites every ``<name>@<digits>`` token into the single
identifier ``<name>__AT<digits>`` BEFORE ast.parse; ``@`` never reaches
the AST and the layer binds exactly like part of the name. Any ``@``
left over after the rewrite (e.g. ``@i-1``) is refused.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .source_ir import (
    FIELD_READ,
    KIND_GRID,
    KIND_NB,
    OP_LOAD_SLOT,
    GRID_X1,
    GRID_X2,
    GRID_X3,
    NameEnv,
    _BANNED_BARE,
    _Compiler,
    _is_flux_alias,
    pack_slot,
)

# Boundary write slots reuse the field-slot numbers (n, vx, vy, vz, P).
BC_WRITE_FIELDS = {
    "n": 0,
    "vx": 1,
    "vy": 2,
    "vz": 3,
    "P": 4,
}
NB_LAYERS = ("1", "2")

# ``@`` is Python's matmul operator (precedence of ``*``), so a raw
# ``n_HI@1`` would mis-parse inside products like ``2*n_HI@1``. Before
# ast.parse we therefore fold every ``<name>@<digits>`` token into the
# single identifier ``<name>__AT__<digits>``; the layer then binds like
# part of the name and grouping is exactly the caller's. Any ``@`` left
# after the fold (e.g. ``@i-1``) is refused in compile_bc_expr.
_NB_ENCODE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*@\s*([0-9]+)")
_NB_DECODE = re.compile(r"^(?P<base>.+)__AT__(?P<layer>[0-9]+)$")


def _encode_layers(expr: str) -> str:
    out = _NB_ENCODE.sub(lambda m: m.group(1) + "__AT__" + m.group(2), expr)
    if "@" in out:
        raise ValueError(
            f"illegal neighbour layer reference in {expr!r}: neighbour reads "
            "are <field>_<fluid>@1 or @2 only"
        )
    return out


@dataclass(frozen=True)
class BcTape:
    """One compiled boundary assignment: value -> slot (dest_var) of fluid
    dest_fluid on every ghost cell of the face it is loaded onto."""

    dest_fluid: int
    dest_var: int  # 0..4 = n, vx, vy, vz, P (field slot order)
    ops: tuple[tuple[int, int], ...]
    consts: tuple[float, ...]
    expr: str


def pack_nb_slot(fluid: int, layer: int, var: int) -> int:
    """Packed KIND_NB slot; layer in {1, 2}, var in 0..5."""
    return pack_slot(KIND_NB, fluid, (layer << 4) | var)


def unpack_nb_slot(code: int) -> tuple[int, int, int]:
    """(fluid, layer, var) of a packed KIND_NB slot."""
    fluid = (code >> 8) & 0xFF
    lb = code & 0xFF
    return fluid, (lb >> 4) & 0x0F, lb & 0x0F


def parse_bc_write_name(name: str, env: NameEnv) -> tuple[int, int]:
    """(dest_fluid, dest_var) of a boundary write name.

    Boundary tapes ASSIGN the ghost slot, so the source-IR increment
    prefixes (dn_/dvx_/dP_/... and dT_) are refused with a pointed
    message; write names carry no @layer (they name the ghost itself).
    """
    if "@" in name:
        raise ValueError(
            f"write name {name!r} names the ghost cell itself; @ layers are "
            "read-only in user_bc"
        )
    if name.startswith("dT"):
        raise ValueError(
            "dT_ is not a boundary slot; T is derived from P -- write P_<fluid>"
        )
    for prefix in ("dn", "dvx", "dvy", "dvz", "dP"):
        if name.startswith(prefix + "_") or name == prefix:
            raise ValueError(
                f"user_bc assigns ghost values; increment name {name!r} is a "
                f"source-IR write name -- write {prefix[1:]}_<fluid> instead"
            )
    for prefix, var in BC_WRITE_FIELDS.items():
        tag = prefix + "_"
        if name.startswith(tag):
            fluid = name[len(tag):]
            if fluid not in env.fluid_index:
                raise ValueError(f"unknown fluid {fluid!r} in write name {name!r}")
            return env.fluid_index[fluid], var
    raise ValueError(
        f"illegal boundary write name {name!r}; want n_/vx_/vy_/vz_/P_<fluid>"
    )


def _resolve_bc_read(name: str, env: NameEnv) -> int:
    if name in _BANNED_BARE:
        raise ValueError(f"banned identifier {name!r} (PLUTO-era slot or loop index)")
    if name in env.fluids:
        raise ValueError(f"bare name {name!r} is not a field; use n_{name} / P_{name} / …")
    if name == "F" or _is_flux_alias(name):
        raise ValueError(f"beam flux {name!r} is not readable in user_bc expressions")
    if name in ("n_e", "P_e", "T_e"):
        raise ValueError(f"electron quantity {name!r} is not readable in user_bc expressions")
    if name == "r" or name == "x1":
        return pack_slot(KIND_GRID, GRID_X1)
    if name == "x2":
        return pack_slot(KIND_GRID, GRID_X2)
    if name == "x3":
        return pack_slot(KIND_GRID, GRID_X3)
    nb = _NB_DECODE.match(name)
    if nb is not None:
        # Encoded neighbour read <field>_<fluid>@<layer> (see _encode_layers).
        base, layer_str = nb.group("base"), nb.group("layer")
        if layer_str not in NB_LAYERS:
            raise ValueError(
                f"illegal neighbour layer in {base}@{layer_str}: only @1 and @2 "
                "exist (the two innermost active cells of the face)"
            )
        layer = int(layer_str)
        for prefix, var in FIELD_READ.items():
            tag = prefix + "_"
            if base.startswith(tag):
                fluid = base[len(tag):]
                if fluid not in env.fluid_index:
                    raise ValueError(f"unknown fluid {fluid!r} in {base}@{layer_str}")
                return pack_nb_slot(env.fluid_index[fluid], layer, var)
        if base.startswith("d"):
            raise ValueError(
                f"increment name {base}@{layer_str} has no boundary meaning; "
                "read n_/vx_/vy_/vz_/P_/T_<fluid>@1 or @2"
            )
        raise ValueError(
            f"unknown neighbour read {base}@{layer_str}; want "
            "<field>_<fluid>@1 or @2 with field in n/vx/vy/vz/P/T"
        )
    # No layer suffix, not a grid coordinate: the ghost's own fluid state
    # is never readable (it is what gets overwritten).
    for prefix in FIELD_READ:
        if name.startswith(prefix + "_"):
            raise ValueError(
                f"{name!r} reads the ghost's own state; boundary reads need a "
                f"neighbour layer (@1/@2) -- e.g. {name}@1"
            )
    raise ValueError(f"unknown identifier {name!r}")


class _BcCompiler(_Compiler):
    """Same frozen AST subset as the source IR (+ - * / **, unary minus,
    exp/log -- nothing else); only the name resolution differs."""

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise ValueError("assignment inside an expression is illegal")
        if node.id in self.const_values:
            self._const(self.const_values[node.id])
            return
        self.ops.append((OP_LOAD_SLOT, _resolve_bc_read(node.id, self.env)))


def compile_bc_expr(
    expr: str,
    env: NameEnv,
    constants: Mapping[str, Any],
    dest_name: str,
) -> BcTape:
    dest_fluid, dest_var = parse_bc_write_name(dest_name, env)
    encoded = _encode_layers(expr)
    try:
        tree = ast.parse(encoded, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {expr!r}: {exc}") from exc
    comp = _BcCompiler(env, {k: float(v) for k, v in constants.items()})
    comp.visit(tree)
    if not comp.ops:
        raise ValueError(f"empty expression {expr!r}")
    return BcTape(
        dest_fluid=dest_fluid,
        dest_var=dest_var,
        ops=tuple(comp.ops),
        consts=tuple(comp.consts),
        expr=expr,
    )


def compile_user_bc(
    user_bc: Mapping[str, Any],
    boundary: Mapping[str, str],
    env: NameEnv,
    constants: Mapping[str, Any],
) -> tuple[tuple[int, tuple[BcTape, ...]], ...]:
    """Compile the whole ``user_bc`` config key.

    user_bc maps a face name (x1beg..x3end) to a mapping
    {slot write name: expression}. The face must already carry the
    ``user`` boundary type. Slots without an expression fall back to
    copy @1 in the kernel (they simply get no tape). Returns
    ((face_index, tapes), ...) sorted by face index; face order is
    0=x1beg .. 5=x3end as in config.FACES.
    """
    from .config import FACES  # local import: no module cycle at import time

    out: list[tuple[int, tuple[BcTape, ...]]] = []
    for face_name, exprs in user_bc.items():
        if face_name not in FACES:
            raise ValueError(
                f"unknown user_bc face {face_name!r}; want one of {list(FACES)}"
            )
        if boundary.get(face_name) != "user":
            raise ValueError(
                f"user_bc expressions require boundary[{face_name!r}] == 'user', "
                f"got {boundary.get(face_name)!r}"
            )
        if not isinstance(exprs, Mapping) or not exprs:
            raise ValueError(
                f"user_bc[{face_name!r}] must be a non-empty mapping "
                "{slot: expression}"
            )
        tapes: list[BcTape] = []
        seen: set[tuple[int, int]] = set()
        for dest_name, raw in exprs.items():
            tape = compile_bc_expr(str(raw), env, constants, str(dest_name))
            key = (tape.dest_fluid, tape.dest_var)
            if key in seen:
                raise ValueError(
                    f"user_bc[{face_name!r}]: duplicate expression for slot "
                    f"{dest_name!r}"
                )
            seen.add(key)
            tapes.append(tape)
        out.append((FACES.index(face_name), tuple(tapes)))
    out.sort(key=lambda item: item[0])
    return tuple(out)


def eval_bc_tape(
    tape: BcTape,
    *,
    slots: Mapping[int, float],
) -> float:
    """Host-side interpreter for tests: same frozen opcodes as the C fill.

    slots maps packed slot codes (KIND_GRID / KIND_NB encodings) to
    values; constants are baked into the tape.
    """
    import math

    stack: list[float] = []
    for op, arg in tape.ops:
        if op == 0:  # LOAD_CONST
            stack.append(tape.consts[arg])
        elif op == 1:  # LOAD_SLOT
            if arg not in slots:
                raise KeyError(f"missing slot {arg}")
            stack.append(float(slots[arg]))
        elif op == 7:  # NEG
            stack.append(-stack.pop())
        elif op == 8:  # EXP
            stack.append(math.exp(stack.pop()))
        elif op == 9:  # LOG
            stack.append(math.log(stack.pop()))
        else:
            b = stack.pop()
            a = stack.pop()
            if op == 2:
                stack.append(a + b)
            elif op == 3:
                stack.append(a - b)
            elif op == 4:
                stack.append(a * b)
            elif op == 5:
                stack.append(a / b)
            elif op == 6:
                stack.append(a**b)
            else:
                raise ValueError(f"unknown opcode {op}")
    if len(stack) != 1:
        raise ValueError("boundary tape did not leave a single value")
    return stack[0]
