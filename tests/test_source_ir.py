"""A4.6 source-IR identifier contract (names → slots; no C step)."""
from __future__ import annotations

import pytest

import xmf
from xmf.config import full_schema_example, normalize, shock_tube
from xmf.source_ir import (
    KIND_FLUID,
    KIND_GRID,
    WRITE_DN,
    WRITE_DVX,
    compile_expr,
    env_from,
    eval_tape,
    pack_slot,
    parse_write_name,
)


def test_shock_tube_has_no_tapes():
    p = normalize(shock_tube())
    assert p.sources_enabled is False
    assert "source_tapes" not in p.extra


def test_full_schema_compiles_systematic_names():
    p = normalize(full_schema_example())
    tapes = p.extra["source_tapes"]
    assert len(tapes) == 3
    dests = [(t.dest.kind, t.dest.dest_fluid) for t in tapes]
    assert dests[0] == (WRITE_DN, 0)  # dn_HI
    assert dests[1][0] == WRITE_DN and dests[1][1] == 3  # dn_HEII
    assert dests[2] == (WRITE_DVX, -1)  # bare dvx: every fluid


def test_gravity_tape_matches_algebra():
    env = env_from(["H"], {"G": 1.0, "M": 2.0})
    tape = compile_expr("-G * M / r**2", env, {"G": 1.0, "M": 2.0}, "dvx")
    assert tape.dest.dest_fluid == -1
    assert tape.dest.kind == WRITE_DVX
    r = pack_slot(KIND_GRID, 0)
    got = eval_tape(tape, slots={r: 4.0})
    assert got == pytest.approx(-1.0 * 2.0 / 16.0)


def test_product_of_two_fluids():
    env = env_from(["HI", "HEII"], {"k_cx": 1.0e-9})
    tape = compile_expr(
        "-k_cx * n_HI * n_HEII", env, {"k_cx": 1.0e-9}, "dn_HI"
    )
    n_hi = pack_slot(KIND_FLUID, 0, 0)
    n_he = pack_slot(KIND_FLUID, 1, 0)
    got = eval_tape(tape, slots={n_hi: 2.0, n_he: 3.0})
    assert got == pytest.approx(-1.0e-9 * 2.0 * 3.0)
    assert parse_write_name("dn_HI", env).dest_fluid == 0


def test_reject_bare_fluid_name():
    env = env_from(["HI"], {})
    with pytest.raises(ValueError, match="bare name"):
        compile_expr("HI", env, {}, "dn_HI")


def test_reject_pluto_slot_names():
    env = env_from(["H"], {})
    with pytest.raises(ValueError, match="banned"):
        compile_expr("VX1", env, {}, "dvx_H")
    with pytest.raises(ValueError, match="banned"):
        compile_expr("RHO", env, {}, "dn_H")


def test_reject_reserved_fluid_name():
    with pytest.raises(ValueError, match="reserved"):
        env_from(["r"], {})


def test_reject_illegal_call_and_dT():
    env = env_from(["H"], {})
    with pytest.raises(ValueError, match="illegal function"):
        compile_expr("sin(r)", env, {}, "dvx_H")
    with pytest.raises(ValueError, match="dT_"):
        parse_write_name("dT_H", env)


def test_normalize_rejects_bad_source_name():
    cfg = shock_tube()
    cfg["fluids"] = [{"name": "H", "mass": 1.0, "charge": 0, "kind": "hd"}]
    cfg["constants"] = {"G": 1.0, "M": 1.0}
    cfg["sources"] = [{"name": "g", "expr": {"dv": "-G*M/r**2"}}]
    with pytest.raises(ValueError, match="illegal write name"):
        normalize(cfg)


def test_r_is_alias_of_x1_same_tape():
    """卡片读名单「r = x1」：两者编出的带子（操作码 + 常数）逐条相同。"""
    env = env_from(["H"], {"G": 2.5, "M": 3.75})
    const = {"G": 2.5, "M": 3.75}
    tape_r = compile_expr("-G*M/r**2", env, const, "dvx")
    tape_x = compile_expr("-G*M/x1**2", env, const, "dvx")
    assert tape_r.ops == tape_x.ops
    assert tape_r.consts == tape_x.consts
    assert tape_r.dest == tape_x.dest


def test_reject_attribute_subscript_and_iend():
    """拒属性/下标（ast 子集之外）与 PLUTO 时代循环索引名。"""
    env = env_from(["H"], {})
    with pytest.raises(ValueError, match="illegal syntax"):
        compile_expr("n_H[0]", env, {}, "dn_H")
    with pytest.raises(ValueError, match="illegal syntax"):
        compile_expr("n_H.real", env, {}, "dn_H")
    with pytest.raises(ValueError, match="banned"):
        compile_expr("IEND", env, {}, "dn_H")
