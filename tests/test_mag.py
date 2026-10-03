"""M 块 1（M1+M2+M3）：磁场 IR 名 / 面心 B + 边心 E 存储 / B 边界单测。

规格来源：docs/SESSION.md（本刀卡片）「交什么」1..7。逐条对应：

  1. M1 配置层：问题级开关 ``mhd``（布尔）与 ``mhd_reflect``
     （wall / symmetric，默认 wall）；关磁默认不产生 extra["mhd"]；
     mhd_reflect 不允许在关磁时单独出现；非布尔开关拒收。
  2. M1 枚举：RIEMANN 表含 ``hlld = 3``（块 1 只登记；M4 已实现，
     磁开 + 离子时 setup 通过，见
     test_hlld_passes_config_and_setup_after_m4）；
     MHD_REFLECT = {"wall": 0, "symmetric": 1}；kind='mhd' 拒收且
     错误消息带 'hd'。blob（XmfConfigC）不开新字段——磁开关走 extra
     端口，22×int32 钉死布局不漂移。
  3. M1 IR 名空间：Bx/By/Bz（读，KIND_MAG=5）与 dBx/dBy/dBz
     （写，kind 6/7/8，dest_fluid=-1）只追加不改操作码；RESERVED
     防撞；Python 编译得过，C xmf_ir_load 一律拒收（NB/rk4 前例），
     经 setup() 变 RuntimeError；xmf_ir_nstate 不因磁改变（N=5*nfluid）。
  4. M2 存储：门开（xmf_mag_load 落地）才有 B1/B2/B3 键与六个块；
     bf[d] 长 (ntot[d]+1)*ntot[d1]*ntot[d2]、ef[d] 长
     ntot[d]*(ntot[d1]+1)*(ntot[d2]+1)；pack/unpack 含鬼区逐位往返；
     E 块恒零且与 B 不共存储；关门四个磁口全 -1、fields_full 无 B 键。
  5. M2 读口：fields_full 的 B1/B2/B3 = 活动格在本块方向的下表面
     （面序 == 格序），形状 (nz, ny, nx)，单位 sqrt(rho_u)*v_u。
  6. M3 边界：reflect 面按 mag_reflect 分层——wall 层（PLUTO
     REFLECTIVE）法向翻号且壁面钉 0、切向复制；symmetric 层（PLUTO
     EQTSYMMETRIC）切向翻号、法向复制、壁面不动。copy / outflow 面
     零阶垫。user 面切向取 xmf_mag_set_user 地板（x1 面）否则
     copy @1，法向鬼面由离散 divB=0 反解（逐鬼格散度归零、边界面
     是活动值不被覆盖、换板重填无陈旧值）。
  7. M4 后的步语义（本刀收编）：块 1「磁开 = HD 演化 + 冻结 B 旁观」
     已废——磁开时 B 随一次有限体积感应演化。本文件保留的是存储 /
     边界合同（块长、pack/unpack、reflect 两档、copy/outflow 垫、
     user 切向地板 + divB 反解）与「E 块 M4 不写、恒零」。M4 闸门
     要求磁开必须有 charge>0 流体（xmf_mag_load 拒中性表），本文件
     所有开门算例统一换单支 HII 离子；开门算例不做步级演化断言。
     零 B + hll 与关磁流体五槽逐位相等（B=0 ⇒ MHD HLL 退回 HD HLL）
     与磁开四步链在 tests/test_mhd.py，不在此重复。

禁令自查（卡片 + docs/TODO.md §A）：
  - 不读 pluto.ini：本文件无任何文件 IO；
  - 不用 XUV_FACTOR==0 哨兵：本文件不涉辐射；
  - 不只测 VX1：流体对照覆盖五槽 (n, vx, vy, vz, P)，B 覆盖三分量、
    鬼层与活动区分别断言。

维度覆盖：1D（256×1×1，横方向退化仍整面写）与薄 2D（8×4×1）。
断言锚定中央横排 (k, j) = (ng, ng)：1D 下 x2/x3 面垫边会把中央活动
排广播到横方向鬼排（流体五槽向来如此），多面角区按面序后写覆盖，
中央排是各面写入的唯一无歧义交集（与 test_bc_user 同口径）。

契约测试审查补强（第 7..9 节）：卡片 §7 逐层对账后收紧——
  - 两档 reflect 的鬼层离散 ∇·B 噪声量级断言（卡片点名，1D 下横
    向项被垫边抹平、无法构造非零容母，故落在薄 2D）；
  - 两档 reflect 配对合同从单条中央排扩到全内区横排（含切向分量
    的整面行走）；
  - 重复 fill_bc 的逐位幂等；无逐流体 hd|mhd 字段的正面断言；
    IR 旧编号（kind 0..4 / 写名 0..5 / 操作码）未漂移的正面钉。
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import XmfConfigC, load, tapes_to_c
from xmf.config import BC_ENUM, MHD_REFLECT, RIEMANN
from xmf.source_ir import (
    KIND_MAG,
    MAG_BX,
    MAG_BY,
    MAG_BZ,
    RESERVED,
    WRITE_DBX,
    WRITE_DBY,
    WRITE_DBZ,
    compile_expr,
    env_from,
    pack_slot,
    unpack_slot,
)

# --- 用例参数（取自 shock_tube() 默认值，防硬编码漂移） ----------------------
_BASE = xmf.shock_tube()
_BASE_GRID = _BASE["grid"]
_NG = _BASE_GRID["nghost"]       # 2
_NYTOT = 1 + 2 * _NG             # 5（1D 横方向退化，仍整面写）
_NZTOT = 1 + 2 * _NG             # 5

NX1_2D, NX2_2D = 8, 4            # 薄 2D；核 sanity 下限 n[0]>=8，同 test_bc
_NTOT2D = (NX1_2D + 2 * _NG, NX2_2D + 2 * _NG, 1 + 2 * _NG)

_PTR = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（mag / state / mesh / bc 查询符号已绑定）。"""
    return load()


# --- 只读小助手 ---------------------------------------------------------------


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


def _bidx(ntot, comp: int, c) -> int:
    """Flat index into bf[comp]: c[comp] is the FACE index, the other two
    entries are cell indices (xmf_run.h layout)."""
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return (c[d2] * ntot[d1] + c[d1]) * (ntot[comp] + 1) + c[comp]


def _pval(c, salt: float) -> float:
    """Everywhere-nonzero, coordinate-dependent pattern value; the three
    slopes are mutually distinct so swapped axes or wrong pairing never
    collide by accident."""
    return salt + 0.11 * c[0] + 0.31 * c[1] + 0.17 * c[2]


def _pat_b(ntot, comp: int, salt: float) -> np.ndarray:
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    arr = np.empty(_len_b(ntot, comp), dtype=np.float64)
    c = [0, 0, 0]
    for k in range(ntot[d2]):
        for j in range(ntot[d1]):
            for f in range(ntot[comp] + 1):
                c[comp], c[d1], c[d2] = f, j, k
                arr[_bidx(ntot, comp, c)] = _pval(c, salt)
    return arr


def _pack_b(lib, h, comp: int, pat: np.ndarray) -> None:
    assert lib.xmf_mag_pack(h, comp, _ptr(pat)) == len(pat)


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


# M4 gate: xmf_mag_load only opens with a charge > 0 fluid present.
# shock_tube()'s default neutral HI is therefore replaced by a single
# HII ion in every gate-open case below -- the B storage / boundary
# contracts ride the magnetic port and do not depend on which ion
# carries the gate, so assertion strength is unchanged.
ION_FLUIDS = [{"name": "HII", "mass": 1.0, "charge": 1, "kind": "ion"}]


def _setup_mhd(cfg_overrides: dict | None = None):
    """shock_tube + problem-level mhd switch + optional top-level overrides
    (boundary / mhd_reflect / ...; whole-key replacement)."""
    cfg = xmf.shock_tube()
    cfg["mhd"] = True
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    if cfg_overrides:
        for key, val in cfg_overrides.items():
            cfg[key] = val
    return xmf.setup(cfg)


def _mesh_arrays(lib, h):
    """Per-direction face area A (nface doubles) and cell volume dV
    (ncell doubles) -- the same arrays the kernel's divB back-solve uses."""
    A, dV = [], []
    for d in range(3):
        nc = lib.xmf_mesh_ncell(h, d)
        nf = lib.xmf_mesh_nface(h, d)
        a = np.empty(nf, dtype=np.float64)
        v = np.empty(nc, dtype=np.float64)
        assert lib.xmf_mesh_copy_A(h, d, _ptr(a)) == nf
        assert lib.xmf_mesh_copy_dV(h, d, _ptr(v)) == nc
        A.append(a)
        dV.append(v)
    return A, dV


def _div_at(blks, ntot, A, dV, c):
    """Discrete div of one cell: sum_e (A_e[c_e+1]B_out - A_e[c_e]B_in)/dV.
    Returns (div, largest single term) for a relative tolerance."""
    s, scale = 0.0, 0.0
    for e in range(3):
        c_out = list(c)
        c_out[e] += 1
        bout = blks[e][_bidx(ntot, e, c_out)]
        bin_ = blks[e][_bidx(ntot, e, c)]
        term = (A[e][c[e] + 1] * bout - A[e][c[e]] * bin_) / dV[e][c[e]]
        s += term
        scale = max(scale, abs(term))
    return s, scale


# --- 1. M1 配置层 -------------------------------------------------------------


def test_mhd_gate_default_off():
    """关磁默认：extra 无 'mhd'，blob 之外没有任何磁端口痕迹。"""
    packed = xmf.normalize(xmf.shock_tube())
    assert "mhd" not in packed.extra


def test_mhd_on_extra_and_reflect_tiers():
    packed = xmf.normalize({**xmf.shock_tube(), "mhd": True})
    assert packed.extra["mhd"] == {"reflect": MHD_REFLECT["wall"]}
    packed_s = xmf.normalize(
        {**xmf.shock_tube(), "mhd": True, "mhd_reflect": "symmetric"}
    )
    assert packed_s.extra["mhd"] == {"reflect": MHD_REFLECT["symmetric"]}
    assert MHD_REFLECT == {"wall": 0, "symmetric": 1}


def test_mhd_reflect_requires_switch():
    with pytest.raises(ValueError, match="mhd"):
        xmf.normalize({**xmf.shock_tube(), "mhd_reflect": "wall"})


@pytest.mark.parametrize("bad", ["yes", 2.5, [True], None])
def test_mhd_switch_type_checked(bad):
    with pytest.raises(ValueError):
        xmf.normalize({**xmf.shock_tube(), "mhd": bad})


def test_bad_reflect_name_refused():
    with pytest.raises(ValueError):
        xmf.normalize({**xmf.shock_tube(), "mhd": True, "mhd_reflect": "mirror"})


def test_riemann_table_enumerates_hlld():
    """RIEMANN 四档：块 1 登记 hlld，M4 起内核实现它（setup 行为见下条）。"""
    assert RIEMANN == {"hll": 0, "hllc": 1, "tvdlf": 2, "hlld": 3}


def test_hlld_passes_config_and_setup_after_m4():
    """hlld 从块 1 的「只登记、setup 拒」变成 M4 实现：配置层通过不变；
    xmf_setup 上界放到 hlld——关磁（HD 派发）过，磁开 + 离子过且一步
    能跑。磁开 + hllc 仍拒（见 tests/test_mhd.py）。"""
    cfg = xmf.shock_tube()
    cfg["numerics"]["riemann"] = "hlld"
    packed = xmf.normalize(cfg)
    assert packed.numerics.riemann == "hlld"
    run = xmf.setup(cfg)  # 关磁：HD 派发，setup 过
    try:
        assert run.dt > 0.0
    finally:
        run.close()
    cfg_on = xmf.shock_tube()
    cfg_on["numerics"]["riemann"] = "hlld"
    cfg_on["mhd"] = True
    cfg_on["fluids"] = [dict(f) for f in ION_FLUIDS]
    run_on = xmf.setup(cfg_on)  # 磁开：MHD HLLD，setup 过
    try:
        run_on.step()
        assert run_on.dt > 0.0
        fld = run_on.fields_full()
        assert {"B1", "B2", "B3"} <= set(fld.keys())
    finally:
        run_on.close()


def test_kind_mhd_refused_message_carries_hd():
    """kind 槽位是组分角色，'mhd' 不是角色；消息必须点出 hd 问题级开关。"""
    cfg = xmf.shock_tube()
    cfg["fluids"] = [{"name": "HI", "mass": 1.00794, "charge": 0, "kind": "mhd"}]
    with pytest.raises(ValueError, match="hd"):
        xmf.normalize(cfg)


def test_blob_layout_has_no_mhd_field():
    """磁开关走 extra 端口，XmfConfig blob 布局不开新字段。"""
    names = [name for name, _ in XmfConfigC._fields_]
    assert not any("mhd" in name or "mag" in name or "reflect" in name
                   for name in names)


# --- 2. M1 IR 名空间 ----------------------------------------------------------


def test_mag_names_reserved():
    for word in ("Bx", "By", "Bz", "dBx", "dBy", "dBz"):
        assert word in RESERVED


def test_mag_name_collisions_refused():
    """流体名与常数名都撞不得磁名。"""
    with pytest.raises(ValueError):
        env_from(["Bx"], {})
    with pytest.raises(ValueError):
        env_from(["HI"], {"dBx": 1.0})


def test_bx_read_compiles_to_kind_mag():
    env = env_from(["HI"], {})
    for name, slot in (("Bx", MAG_BX), ("By", MAG_BY), ("Bz", MAG_BZ)):
        tape = compile_expr(f"{name} * 2.0", env, {}, "dn_HI")
        loads = [(op, arg) for op, arg in tape.ops if op == 1]
        assert loads == [(1, pack_slot(KIND_MAG, slot))]
        kind, a, b = unpack_slot(loads[0][1])
        assert (kind, a, b) == (KIND_MAG, slot, 0)
    assert (KIND_MAG, MAG_BX, MAG_BY, MAG_BZ) == (5, 0, 1, 2)


def test_dbx_write_names_parse():
    env = env_from(["HI"], {})
    for name, kind in (("dBx", WRITE_DBX), ("dBy", WRITE_DBY), ("dBz", WRITE_DBZ)):
        tape = compile_expr("1.0", env, {}, name)
        assert tape.dest.kind == kind
        assert tape.dest.dest_fluid == -1  # 磁块不是流体
    assert (WRITE_DBX, WRITE_DBY, WRITE_DBZ) == (6, 7, 8)


def test_nstate_unchanged_with_gate_open(lib):
    """磁态永不进隐式 Jacobian 的独立变量：N = 5*nfluid 不变。"""
    run = _setup_mhd()
    try:
        assert run.num_states == 5 * len(run._cfg.fluids)
    finally:
        run.close()


def _load_raw_tape(lib, h, tape) -> int:
    header, ops, consts = tapes_to_c([tape])
    return lib.xmf_ir_load(
        h,
        1,
        header.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ops.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        consts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )


def test_mag_read_kind_refused_at_ir_load(lib):
    """Bx 读带：Python 编译得过，C loader 拒收 KIND_MAG（门开也拒）。"""
    env = env_from(["HII"], {})
    tape = compile_expr("Bx * 0.0", env, {}, "dn_HII")
    run = _setup_mhd()
    try:
        assert _load_raw_tape(lib, run._h, tape) == -1
        assert run.num_tapes == 0  # 拒收后旧表保留（此处为空表）
    finally:
        run.close()


def test_mag_write_kind_refused_at_ir_load(lib):
    """dBx 写带：追加写名 6..8 在内核侧拒收（rk4 前例）。"""
    env = env_from(["HII"], {})
    for name in ("dBx", "dBy", "dBz"):
        tape = compile_expr("1.0", env, {}, name)
        run = _setup_mhd()
        try:
            assert _load_raw_tape(lib, run._h, tape) == -1
            assert run.num_tapes == 0
        finally:
            run.close()


@pytest.mark.parametrize(
    "expr_map",
    [{"dn_HI": "Bx * 0.0"}, {"dBx": "1.0"}],
    ids=["read_Bx", "write_dBx"],
)
def test_setup_refuses_mag_source_tapes(expr_map):
    """端到端：含磁名的源表在 setup 就炸，不是跑到半路才炸。"""
    cfg = xmf.shock_tube()
    cfg["sources"] = [{"expr": expr_map}]
    xmf.normalize(cfg)  # Python 侧编译得过
    with pytest.raises(RuntimeError):
        xmf.setup(cfg)


# --- 3. M2 存储：关门 / 开门 --------------------------------------------------


def test_gate_closed_all_ports_refuse(lib):
    run = xmf.setup(xmf.shock_tube())
    try:
        h = run._h
        buf = np.zeros(8, dtype=np.float64)
        assert lib.xmf_mag_pack(h, 0, _ptr(buf)) == -1
        assert lib.xmf_mag_unpack(h, 0, _ptr(buf)) == -1
        assert lib.xmf_mag_unpack_e(h, 0, _ptr(buf)) == -1
        assert lib.xmf_mag_set_user(h, _ptr(buf)) == -1
        fld = run.fields_full()
        assert not any(key.startswith("B") for key in fld)
        assert lib.xmf_fill_bc(h) == 0  # 关门补边本身照常（磁派发 no-op）
    finally:
        run.close()


def test_mag_load_refuses_bad_reflect_mode(lib):
    """reflect 模式只认 0/1；拒收后磁门仍关。"""
    run = xmf.setup(xmf.shock_tube())
    try:
        h = run._h
        ntot = _ntot_of(run)
        blocks = [np.zeros(_len_b(ntot, d), dtype=np.float64) for d in range(3)]
        for mode in (2, -1):
            assert (
                lib.xmf_mag_load(
                    h, mode, _ptr(blocks[0]), _ptr(blocks[1]), _ptr(blocks[2])
                )
                == -1
            )
        buf = np.zeros(4, dtype=np.float64)
        assert lib.xmf_mag_unpack(h, 0, _ptr(buf)) == -1  # 仍关门
    finally:
        run.close()


def test_gate_open_block_lengths(lib):
    run = _setup_mhd()
    try:
        ntot = _ntot_of(run)
        for comp in range(3):
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            nb = _len_b(ntot, comp)
            ne = _len_e(ntot, comp)
            assert nb == (ntot[comp] + 1) * ntot[d1] * ntot[d2]
            assert ne == ntot[comp] * (ntot[d1] + 1) * (ntot[d2] + 1)
            assert lib.xmf_mag_unpack(run._h, comp, _ptr(np.empty(nb))) == nb
            assert lib.xmf_mag_unpack_e(run._h, comp, _ptr(np.empty(ne))) == ne
    finally:
        run.close()


def test_e_blocks_zero_and_separate(lib):
    """E 块 calloc 零；塞满 B 之后 E 仍零——两块不共存储。"""
    run = _setup_mhd()
    try:
        ntot = _ntot_of(run)
        for comp in range(3):
            assert np.all(_unpack_e(lib, run._h, comp, ntot) == 0.0)
            _pack_b(lib, run._h, comp, _pat_b(ntot, comp, 3.0 * comp + 1.0))
        for comp in range(3):
            assert np.all(_unpack_e(lib, run._h, comp, ntot) == 0.0)
            assert np.any(_unpack_b(lib, run._h, comp, ntot) != 0.0)
    finally:
        run.close()


def test_pack_unpack_roundtrip_bitwise(lib):
    run = _setup_mhd()
    try:
        ntot = _ntot_of(run)
        for comp in range(3):
            pat = _pat_b(ntot, comp, 0.25 * (comp + 1))
            _pack_b(lib, run._h, comp, pat)
            assert np.array_equal(_unpack_b(lib, run._h, comp, ntot), pat)
    finally:
        run.close()


def test_pack_bad_comp_refused(lib):
    run = _setup_mhd()
    try:
        buf = np.zeros(4, dtype=np.float64)
        assert lib.xmf_mag_pack(run._h, 3, _ptr(buf)) == -1
        assert lib.xmf_mag_unpack(run._h, -1, _ptr(buf)) == -1
        assert lib.xmf_mag_unpack_e(run._h, 3, _ptr(buf)) == -1
    finally:
        run.close()


def test_fields_full_reads_lower_faces(lib):
    """B1/B2/B3 键 = 活动格在本块方向的下表面 × sqrt(rho_u)*v_u，
    三分量各验一遍（comp=1/2 同时钉死轴置换路径）。"""
    run = _setup_mhd()
    try:
        ntot = _ntot_of(run)
        nint = run._cfg.n
        ng = run._cfg.ng
        pats = {}
        for comp in range(3):
            pats[comp] = _pat_b(ntot, comp, 1.0 + comp)
            _pack_b(lib, run._h, comp, pats[comp])
        fld = run.fields_full()
        assert set(k for k in fld if k.startswith("B")) == {"B1", "B2", "B3"}
        u = run._cfg.units
        unit_b = math.sqrt(u.density) * u.velocity
        for comp, key in enumerate(("B1", "B2", "B3")):
            d1, d2 = (comp + 1) % 3, (comp + 2) % 3
            blk = pats[comp].reshape(ntot[d2], ntot[d1], ntot[comp] + 1)
            axis_of = {d2: 0, d1: 1, comp: 2}
            perm = (axis_of[2], axis_of[1], axis_of[0])
            sl = [slice(None)] * 3
            for q, ax in axis_of.items():
                sl[ax] = slice(ng, ng + nint[q])
            expect = blk[tuple(sl)].transpose(perm) * unit_b
            assert fld[key].shape == (nint[2], nint[1], nint[0])
            assert np.array_equal(fld[key], expect), key
    finally:
        run.close()


def test_thin_2d_block_shapes(lib):
    """薄 2D（8×4×1）：形状与块长公式不变，横方向鬼区仍在。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["mhd"] = True
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]  # M4 闸门：离子才开门
    run = xmf.setup(cfg)
    try:
        ntot = _ntot_of(run)
        assert ntot == _NTOT2D
        fld = run.fields_full()
        for key in ("B1", "B2", "B3"):
            assert fld[key].shape == (1, NX2_2D, NX1_2D)
        for comp in range(3):
            assert (
                lib.xmf_mag_unpack(
                    run._h, comp, _ptr(np.empty(_len_b(ntot, comp)))
                )
                == _len_b(ntot, comp)
            )
    finally:
        run.close()


def test_lifecycle_reopen(lib):
    """开→关→再开：finalize 清块，再次 setup 一切照旧。"""
    run = _setup_mhd()
    ntot = _ntot_of(run)
    _pack_b(lib, run._h, 0, _pat_b(ntot, 0, 9.0))
    run.close()
    run2 = _setup_mhd()
    try:
        ntot2 = _ntot_of(run2)
        assert np.all(_unpack_b(lib, run2._h, 0, ntot2) == 0.0)  # 新块归零
    finally:
        run2.close()


# --- 4. M3 边界：reflect 两层 / copy / outflow --------------------------------


def test_reflect_wall_tier_pairing(lib):
    """wall 层（默认）：法向鬼面翻号、壁面钉 0、切向鬼格复制；
    活动区（含壁面以外的法向活动面）逐位不动。断言锚中央横排。"""
    run = _setup_mhd()  # shock_tube 默认 x1 两面 reflect
    try:
        h = run._h
        ntot = _ntot_of(run)
        assert lib.xmf_bc_type(h, 0) == BC_ENUM["reflect"]
        nx = run._cfg.n[0]
        ng = run._cfg.ng
        pats = {}
        for comp in range(3):
            pats[comp] = _pat_b(ntot, comp, 2.0 * comp + 0.5)
            _pack_b(lib, h, comp, pats[comp])
        assert lib.xmf_fill_bc(h) == 0
        b = [ _unpack_b(lib, h, comp, ntot) for comp in range(3) ]

        def n0(i):
            return b[0][_bidx(ntot, 0, [i, ng, ng])]

        def p0(i):
            return pats[0][_bidx(ntot, 0, [i, ng, ng])]

        def nt(comp, i):
            c = [ng, ng, ng]
            c[0] = i  # comp 1/2 cell-indexed along x1
            return b[comp][_bidx(ntot, comp, c)]

        def pt(comp, i):
            c = [ng, ng, ng]
            c[0] = i
            return pats[comp][_bidx(ntot, comp, c)]

        # 壁面钉 0（两面）
        assert n0(ng) == 0.0
        assert n0(ng + nx) == 0.0
        # 法向鬼面 = -镜像面
        for m in range(1, ng + 1):
            assert n0(ng - m) == -p0(ng + m)
            assert n0(ng + nx + m) == -p0(ng + nx - m)
        # 切向鬼格 = +镜像格
        for comp in (1, 2):
            for m in range(1, ng + 1):
                assert nt(comp, ng - m) == pt(comp, ng + m - 1)
                assert nt(comp, ng + nx - 1 + m) == pt(comp, ng + nx - m)
        # 活动区逐位不动：法向活动面除两壁面，切向全体活动格
        for i in range(ng + 1, ng + nx):
            assert n0(i) == p0(i)
        for comp in (1, 2):
            for i in range(ng, ng + nx):
                assert nt(comp, i) == pt(comp, i)
    finally:
        run.close()


def test_reflect_symmetric_tier_pairing(lib):
    """symmetric 层：切向翻号、法向复制、壁面（活动值）不钉零。"""
    run = _setup_mhd({"mhd_reflect": "symmetric"})
    try:
        h = run._h
        ntot = _ntot_of(run)
        nx = run._cfg.n[0]
        ng = run._cfg.ng
        pats = {}
        for comp in range(3):
            pats[comp] = _pat_b(ntot, comp, 4.0 * comp + 0.75)
            _pack_b(lib, h, comp, pats[comp])
        assert lib.xmf_fill_bc(h) == 0
        b = [_unpack_b(lib, h, comp, ntot) for comp in range(3)]

        def n0(i):
            return b[0][_bidx(ntot, 0, [i, ng, ng])]

        def p0(i):
            return pats[0][_bidx(ntot, 0, [i, ng, ng])]

        def nt(comp, i):
            c = [ng, ng, ng]
            c[0] = i
            return b[comp][_bidx(ntot, comp, c)]

        def pt(comp, i):
            c = [ng, ng, ng]
            c[0] = i
            return pats[comp][_bidx(ntot, comp, c)]

        # 壁面不动（仍是活动值）
        assert n0(ng) == p0(ng)
        assert n0(ng + nx) == p0(ng + nx)
        # 法向鬼面 = +镜像面（偶）
        for m in range(1, ng + 1):
            assert n0(ng - m) == p0(ng + m)
            assert n0(ng + nx + m) == p0(ng + nx - m)
        # 切向鬼格 = -镜像格（奇）
        for comp in (1, 2):
            for m in range(1, ng + 1):
                assert nt(comp, ng - m) == -pt(comp, ng + m - 1)
                assert nt(comp, ng + nx - 1 + m) == -pt(comp, ng + nx - m)
    finally:
        run.close()


@pytest.mark.parametrize("bc_name", ["copy", "outflow"])
def test_copy_and_outflow_zero_order_pad(lib, bc_name):
    """copy / outflow 面：法向鬼面取边界面值，切向鬼格取最外活动格。"""
    run = _setup_mhd({"boundary": {"x1": bc_name}})
    try:
        h = run._h
        ntot = _ntot_of(run)
        assert lib.xmf_bc_type(h, 0) == BC_ENUM[bc_name]
        nx = run._cfg.n[0]
        ng = run._cfg.ng
        pats = {}
        for comp in range(3):
            pats[comp] = _pat_b(ntot, comp, 6.0 * comp + 1.5)
            _pack_b(lib, h, comp, pats[comp])
        assert lib.xmf_fill_bc(h) == 0
        b = [_unpack_b(lib, h, comp, ntot) for comp in range(3)]

        def n0(i):
            return b[0][_bidx(ntot, 0, [i, ng, ng])]

        def p0(i):
            return pats[0][_bidx(ntot, 0, [i, ng, ng])]

        def nt(comp, i):
            c = [ng, ng, ng]
            c[0] = i
            return b[comp][_bidx(ntot, comp, c)]

        def pt(comp, i):
            c = [ng, ng, ng]
            c[0] = i
            return pats[comp][_bidx(ntot, comp, c)]

        for m in range(1, ng + 1):
            assert n0(ng - m) == p0(ng)            # beg：边界面值
            assert n0(ng + nx + m) == p0(ng + nx)  # end：边界面值
        for comp in (1, 2):
            for m in range(1, ng + 1):
                assert nt(comp, ng - m) == pt(comp, ng)
                assert nt(comp, ng + nx - 1 + m) == pt(comp, ng + nx - 1)
    finally:
        run.close()


# --- 5. M3 边界：user 面（地板 + divB 反解）----------------------------------


def _floor_2d() -> np.ndarray:
    """x1-face tangential floor, layout [tcomp][k*nytot+j] (tcomp 0=B2,
    1=B3); j/k-dependent so the fill is provably not a constant pad."""
    ntot = _NTOT2D
    nt = ntot[2] * ntot[1]
    floor = np.empty(2 * nt, dtype=np.float64)
    for k in range(ntot[2]):
        for j in range(ntot[1]):
            floor[0 * nt + k * ntot[1] + j] = 5.0 + 0.5 * j + 2.0 * k
            floor[1 * nt + k * ntot[1] + j] = -1.5 + 0.1 * j
    return floor


def _setup_user_2d(with_floor: bool):
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["mhd"] = True
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]  # M4 闸门：离子才开门
    cfg["boundary"] = {"x1beg": "user", "x1end": "copy"}
    run = xmf.setup(cfg)
    if with_floor:
        assert load().xmf_mag_set_user(run._h, _ptr(_floor_2d())) == 0
    return run


def _fill_user_patterns(lib, run):
    ntot = _NTOT2D
    pats = {}
    for comp in range(3):
        pats[comp] = _pat_b(ntot, comp, 0.5 * comp + 0.125)
        _pack_b(lib, run._h, comp, pats[comp])
    assert lib.xmf_fill_bc(run._h) == 0
    return [_unpack_b(lib, run._h, comp, ntot) for comp in range(3)], pats


def _assert_divb_zero_ghosts(lib, run, blks, ghost_i, jrows):
    """Ghost cells at x1 indices `ghost_i`, transverse rows jrows x {k=ng}:
    per-cell discrete divB vanishes (back-solve contract), with a relative
    tolerance against the largest single flux term."""
    ntot = _NTOT2D
    ng = _NG
    A, dV = _mesh_arrays(lib, run._h)
    max_div, max_term = 0.0, 0.0
    for i in ghost_i:
        for j in jrows:
            s, scale = _div_at(blks, ntot, A, dV, [i, j, ng])
            max_div = max(max_div, abs(s))
            max_term = max(max_term, scale)
    assert max_term > 0.0  # 图案有真横向结构，容母不是 0/0
    assert max_div <= 1e-9 * max_term, (max_div, max_term)


def test_user_face_floor_tangential_and_divb(lib):
    """x1beg user 面 + 地板：切向鬼格在边界平面内区取地板值；法向鬼面
    由离散 divB=0 反解——逐鬼格散度归零，边界面（活动值）逐位不动，且
    反解结果确实不等于零阶复制（横向通量项真被用了）。

    内区 = 横向坐标全在活动范围内的条目：B2 取 j 活动面（ng..ng+ny）
    × k 活动格，B3 取 j 活动格 × k 活动面。边界平面的横向鬼排（角区）
    由 x2/x3 面按面序后写覆盖，不是本面契约（同 test_bc_user 口径）。"""
    run = _setup_user_2d(with_floor=True)
    try:
        h = run._h
        ntot = _NTOT2D
        ng = _NG
        assert lib.xmf_bc_type(h, 0) == BC_ENUM["user"]
        blks, pats = _fill_user_patterns(lib, run)
        nt = ntot[2] * ntot[1]
        floor = _floor_2d()
        jfaces = range(ng, ng + NX2_2D + 1)  # B2 自己的方向 = 面
        jcells = range(ng, ng + NX2_2D)      # B3 横跨方向 = 格
        kfaces = (ng, ng + 1)                # nz=1：活动 k 面 ng..ng+1
        kcells = (ng,)                       # nz=1：活动 k 格

        # B2 鬼格（两层，本面方向 i）= 地板（内区）
        for m in range(1, ng + 1):
            for j in jfaces:
                for k in kcells:
                    got = blks[1][_bidx(ntot, 1, [ng - m, j, k])]
                    assert got == floor[0 * nt + k * ntot[1] + j]
        # B3 鬼格 = 地板（内区）
        for m in range(1, ng + 1):
            for j in jcells:
                for k in kfaces:
                    got = blks[2][_bidx(ntot, 2, [ng - m, j, k])]
                    assert got == floor[1 * nt + k * ntot[1] + j]
        # 法向边界面（活动值）逐位不动（内区）
        for j in jcells:
            for k in kcells:
                got = blks[0][_bidx(ntot, 0, [ng, j, k])]
                want = pats[0][_bidx(ntot, 0, [ng, j, k])]
                assert got == want
        # 逐鬼格 divB = 0（横向全活动排；角区归 x2/x3 面，不参与本面契约）
        jrows = range(ng, ng + NX2_2D)
        _assert_divb_zero_ghosts(lib, run, blks, (ng - 1, ng - 2), jrows)
        # 反解 ≠ 零阶复制：至少一个鬼面与边界面值不同（S 项非零）
        diff = False
        for j in jrows:
            g1 = blks[0][_bidx(ntot, 0, [ng - 1, j, ng])]
            bw = blks[0][_bidx(ntot, 0, [ng, j, ng])]
            if g1 != bw:
                diff = True
        assert diff
    finally:
        run.close()


def test_user_face_no_floor_copy_at_1_and_divb(lib):
    """无地板的 user 面：切向回退 copy @1（内区，随横排变），法向仍反解
    归零。"""
    run = _setup_user_2d(with_floor=False)
    try:
        ntot = _NTOT2D
        ng = _NG
        blks, pats = _fill_user_patterns(lib, run)
        jfaces = range(ng, ng + NX2_2D + 1)
        jcells = range(ng, ng + NX2_2D)
        kfaces = (ng, ng + 1)
        kcells = (ng,)
        for m in range(1, ng + 1):
            for j in jfaces:
                for k in kcells:
                    got = blks[1][_bidx(ntot, 1, [ng - m, j, k])]
                    want = pats[1][_bidx(ntot, 1, [ng, j, k])]
                    assert got == want
            for j in jcells:
                for k in kfaces:
                    got = blks[2][_bidx(ntot, 2, [ng - m, j, k])]
                    want = pats[2][_bidx(ntot, 2, [ng, j, k])]
                    assert got == want
        # copy @1 真随横排变（图案横向有结构），证明不是常值垫
        rows = {
            blks[1][_bidx(ntot, 1, [ng - 1, j, ng])] for j in jcells
        }
        assert len(rows) > 1
        jrows = range(ng, ng + NX2_2D)
        _assert_divb_zero_ghosts(lib, run, blks, (ng - 1, ng - 2), jrows)
    finally:
        run.close()


def test_user_floor_requires_gate_open(lib):
    run = xmf.setup(xmf.shock_tube())
    try:
        buf = np.zeros(2 * _NYTOT * _NZTOT, dtype=np.float64)
        assert lib.xmf_mag_set_user(run._h, _ptr(buf)) == -1
    finally:
        run.close()


def test_refill_after_repack_no_stale(lib):
    """换图案 + 换地板后重填：鬼格跟新值走，无陈旧残留。"""
    run = _setup_user_2d(with_floor=True)
    try:
        ntot = _NTOT2D
        ng = _NG
        _fill_user_patterns(lib, run)  # 第一轮
        pats2 = {}
        for comp in range(3):
            pats2[comp] = _pat_b(ntot, comp, 50.0 + 0.5 * comp)
            _pack_b(lib, run._h, comp, pats2[comp])
        nt = ntot[2] * ntot[1]
        # 新地板随 j 变（42+0.25j）：既与旧地板 5+0.5j+2k 处处不相等，
        # 又给反解提供非零横向通量（证明反解按新地板重算）。
        floor2 = np.empty(2 * nt, dtype=np.float64)
        for k in range(ntot[2]):
            for j in range(ntot[1]):
                floor2[0 * nt + k * ntot[1] + j] = 42.0 + 0.25 * j
                floor2[1 * nt + k * ntot[1] + j] = 42.0 + 0.25 * j
        assert lib.xmf_mag_set_user(run._h, _ptr(floor2)) == 0
        assert lib.xmf_fill_bc(run._h) == 0
        b2 = _unpack_b(lib, run._h, 1, ntot)
        # 新地板生效（内区）
        for j in range(ng, ng + NX2_2D + 1):
            assert b2[_bidx(ntot, 1, [ng - 1, j, ng])] == 42.0 + 0.25 * j
        # 法向边界面跟新图案的活动值（内区）
        blks = [_unpack_b(lib, run._h, comp, ntot) for comp in range(3)]
        assert blks[0][_bidx(ntot, 0, [ng, ng, ng])] == pats2[0][
            _bidx(ntot, 0, [ng, ng, ng])
        ]
        jrows = range(ng, ng + NX2_2D)
        _assert_divb_zero_ghosts(lib, run, blks, (ng - 1, ng - 2), jrows)
    finally:
        run.close()


# --- 6. M4 后步语义：E 块恒零（冻结 B 已废） ---------------------------------
# 块 1 的「磁开五槽逐位同关磁」与「非零 B 逐步冻结旁观」断言已随 M4 废除：
# 磁开时 B 随一次感应演化。继任合同——
#   * 零 B + hll 与关磁流体五槽逐位（B=0 ⇒ MHD HLL 退回 HD HLL，感应为 0）
#     在 tests/test_mhd.py 的零 B 用例（≥10 步），此处不重复；
#   * B 的演化行为（Brio–Wu 管、两支共动还原）同样在 tests/test_mhd.py。


def test_e_blocks_written_by_ct(lib):
    """M5 CT writes edge-centred E = -u_+ x B: shock_tube() 的默认初值
    rho_l != rho_r 产生非零 u_+，一步之后至少一个 E 分量非零（E 是
    M5 的物理量，不再是 M4 的全零占位）。B 读口仍通；divb_max 可
    读。均匀 B + 零速度初值仍给零 E（E = -0 x B_const = 0）。"""
    run = _setup_mhd({"boundary": {"x1": "copy"}})
    try:
        h = run._h
        ntot = _ntot_of(run)
        for comp in range(3):
            pat = np.full(_len_b(ntot, comp), 0.1 + 0.1 * comp, dtype=np.float64)
            _pack_b(lib, h, comp, pat)
        # 多步让 Riemann 问题产生非零 u_+
        for _ in range(3):
            run.step()
        any_nonzero = False
        for comp in range(3):
            e = _unpack_e(lib, h, comp, ntot)
            assert np.all(np.isfinite(e)), f"E[{comp}] has non-finite values"
            if np.any(e != 0.0):
                any_nonzero = True
        assert any_nonzero, "all E blocks still zero after steps (CT should write E)"
        # divb_max 可读
        d = run.divb_max
        assert isinstance(d, float) and d >= 0.0
    finally:
        run.close()


def test_four_step_chain_with_gate_open(lib):
    """setup → step → fields_full/fields → close 全链；重复开关不炸。"""
    run = _setup_mhd()
    try:
        assert run.dt > 0.0
        run.step()
        fld = run.fields_full()
        assert {"B1", "B2", "B3"} <= set(fld.keys())
        assert "HII" in fld
        legacy = run.fields
        assert "HII" in legacy
    finally:
        run.close()
    run2 = _setup_mhd()
    run2.step()
    run2.close()


# --- 7. 契约审查补强：两档 reflect 鬼层离散 ∇·B（卡片 §7 点名） ------------


def _setup_2d_mhd(cfg_overrides: dict | None = None):
    """薄 2D（8×4×1）磁开配置；x1 两面保持 shock_tube 默认 reflect，
    x2/x3 走默认 none。顶层整键覆盖（如 mhd_reflect）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["mhd"] = True
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]  # M4 闸门：离子才开门
    if cfg_overrides:
        for key, val in cfg_overrides.items():
            cfg[key] = val
    return xmf.setup(cfg)


def _face_coords(lib, h):
    """Kernel-authority face coordinates (xf) of all three directions,
    ghost extension included -- the uniform grid makes every direction
    equidistant, so these double as the staggered pattern's axis."""
    out = []
    for d in range(3):
        nf = lib.xmf_mesh_nface(h, d)
        buf = np.empty(nf, dtype=np.float64)
        assert lib.xmf_mesh_copy_xf(h, d, _ptr(buf)) == nf
        out.append(buf)
    return out


def _pack_b_fn(lib, h, ntot, comp, fn):
    """Pack one whole B block from fn(c): c[comp] is the face index along
    the component direction, the other two entries are cell indices."""
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    arr = np.empty(_len_b(ntot, comp), dtype=np.float64)
    c = [0, 0, 0]
    for k in range(ntot[d2]):
        for j in range(ntot[d1]):
            for f in range(ntot[comp] + 1):
                c[comp], c[d1], c[d2] = f, j, k
                arr[_bidx(ntot, comp, c)] = fn(c)
    _pack_b(lib, h, comp, arr)


def _unpack_all_b(lib, h, ntot):
    return [_unpack_b(lib, h, comp, ntot) for comp in range(3)]


@pytest.mark.parametrize("tier", ["wall", "symmetric"])
def test_reflect_tiers_ghost_divb_at_noise_level(lib, tier):
    """卡片 §7 边界层：两档 reflect 下鬼段离散 ∇·B 在表示噪声量级。

    初值取真离散无散场（笛卡尔均匀网格，逐项通量非零、和恰为零）：
      壁型：导体壁相容要求壁面 B_n = 0，取 B = (0, y面坐标, -z面坐标)，
            y 项与 z 项互抵（A_y·dy == A_z·dz 是笛卡尔恒等式）；
      对称型：法向可以非零，取 B = (x面坐标, -y面坐标, 0)，x 项与 y 项
            互抵；壁面不钉零（活动值保留）。
    前提钉（活动区自身无散）先于 fill 断言，防止图案构造悄悄漂移；
    divB 全部由测试用 xmf_mag_unpack + xmf_mesh_copy_A/dV 在 Python 侧
    自算（卡片点名做法；本刀无 C 的 divB 助手）。1D 下横向项被垫边
    广播抹平、容母归零无法设防，故本断言落在薄 2D（横排真结构）。
    """
    overrides = {} if tier == "wall" else {"mhd_reflect": "symmetric"}
    run = _setup_2d_mhd(overrides)
    try:
        h = run._h
        ntot = _NTOT2D
        ng = _NG
        nx, ny = NX1_2D, NX2_2D
        xf, yf, zf = _face_coords(lib, h)
        assert lib.xmf_bc_type(h, 0) == BC_ENUM["reflect"]
        if tier == "wall":
            fns = {
                0: lambda c: 0.0,
                1: lambda c: yf[c[1]],
                2: lambda c: -zf[c[2]],
            }
        else:
            fns = {
                0: lambda c: xf[c[0]],
                1: lambda c: -yf[c[1]],
                2: lambda c: 0.0,
            }
        for comp in range(3):
            _pack_b_fn(lib, h, ntot, comp, fns[comp])
        jrows = range(ng, ng + ny)
        # 前提：活动区逐格离散散度本就在噪声量级（图案自洽钉）
        _assert_divb_zero_ghosts(
            lib, run, _unpack_all_b(lib, h, ntot), range(ng, ng + nx), jrows
        )
        assert lib.xmf_fill_bc(h) == 0
        blks = _unpack_all_b(lib, h, ntot)
        # x1 鬼层两端两层逐鬼格归零（内区横排；角区归 x2/x3 面）
        _assert_divb_zero_ghosts(lib, run, blks, (ng - 1, ng - 2), jrows)
        _assert_divb_zero_ghosts(lib, run, blks, (ng + nx, ng + nx + 1), jrows)
    finally:
        run.close()


# --- 8. 契约审查补强：reflect 配对合同的全内区横排口径 ----------------------


def _blk3(flat, ntot, comp):
    """Reshape one unpacked B block to (dir d2, dir d1, own-dir faces);
    the same axis convention fields_full uses."""
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return flat.reshape(ntot[d2], ntot[d1], ntot[comp] + 1)


@pytest.mark.parametrize("tier", ["wall", "symmetric"])
def test_reflect_pairing_all_interior_rows_2d(lib, tier):
    """两档 reflect 的翻号/复制合同扩到全部内区横排（不只中央排）：
    法向鬼面逐对内区 (j, k) 行、切向鬼格逐自身方向的面行（含最上活动
    面行）断言；活动区（壁型扣除被钉零的壁面）逐位不动。图案三分量
    斜率互异，轴错位 / 错层配对无法巧合相等。"""
    overrides = {} if tier == "wall" else {"mhd_reflect": "symmetric"}
    run = _setup_2d_mhd(overrides)
    try:
        h = run._h
        ntot = _NTOT2D
        ng = _NG
        nx, ny, nz = NX1_2D, NX2_2D, 1
        pats = {}
        for comp in range(3):
            pats[comp] = _pat_b(ntot, comp, 7.0 * comp + 2.25)
            _pack_b(lib, h, comp, pats[comp])
        assert lib.xmf_fill_bc(h) == 0
        b = [_blk3(_unpack_b(lib, h, comp, ntot), ntot, comp) for comp in range(3)]
        p = [_blk3(pats[comp], ntot, comp) for comp in range(3)]
        wall = tier == "wall"
        sgn_n = -1.0 if wall else 1.0
        sgn_t = 1.0 if wall else -1.0
        js, ks = slice(ng, ng + ny), slice(ng, ng + nz)
        jf, kf = slice(ng, ng + ny + 1), slice(ng, ng + nz + 1)

        # B1（法向，轴 (k, j, i面)）
        if wall:
            assert np.all(b[0][ks, js, ng] == 0.0)          # 壁面钉 0
            assert np.all(b[0][ks, js, ng + nx] == 0.0)
        else:
            assert np.array_equal(b[0][ks, js, ng], p[0][ks, js, ng])
            assert np.array_equal(b[0][ks, js, ng + nx], p[0][ks, js, ng + nx])
        for m in range(1, ng + 1):
            assert np.array_equal(b[0][ks, js, ng - m], sgn_n * p[0][ks, js, ng + m])
            assert np.array_equal(
                b[0][ks, js, ng + nx + m], sgn_n * p[0][ks, js, ng + nx - m]
            )
        # B2（切向，轴 (i, k, j面)）/ B3（切向，轴 (j, i, k面)）
        for m in range(1, ng + 1):
            assert np.array_equal(
                b[1][ng - m, ks, jf], sgn_t * p[1][ng + m - 1, ks, jf]
            )
            assert np.array_equal(
                b[1][ng + nx - 1 + m, ks, jf], sgn_t * p[1][ng + nx - m, ks, jf]
            )
            assert np.array_equal(
                b[2][js, ng - m, kf], sgn_t * p[2][js, ng + m - 1, kf]
            )
            assert np.array_equal(
                b[2][js, ng + nx - 1 + m, kf], sgn_t * p[2][js, ng + nx - m, kf]
            )
        # 活动区逐位不动（壁型扣除两端被钉零的壁面）
        act_n = slice(ng + 1, ng + nx) if wall else slice(ng, ng + nx + 1)
        assert np.array_equal(b[0][ks, js, act_n], p[0][ks, js, act_n])
        assert np.array_equal(b[1][ng : ng + nx, ks, jf],
                              p[1][ng : ng + nx, ks, jf])
        assert np.array_equal(b[2][js, ng : ng + nx, kf],
                              p[2][js, ng : ng + nx, kf])
    finally:
        run.close()


def test_fill_bc_repeat_is_bitwise_idempotent(lib):
    """重复 xmf_fill_bc（输入不变）逐位幂等：user 面（地板 + 法向反解）
    这条最复杂的路径上，鬼格重填不产生陈旧漂移。"""
    run = _setup_user_2d(with_floor=True)
    try:
        ntot = _NTOT2D
        _fill_user_patterns(lib, run)  # 打包图案 + 第一次 fill
        first = _unpack_all_b(lib, run._h, ntot)
        assert lib.xmf_fill_bc(run._h) == 0
        for comp in range(3):
            assert np.array_equal(
                _unpack_b(lib, run._h, comp, ntot), first[comp]
            ), comp
    finally:
        run.close()


# --- 9. 契约审查补强：config 正面钉 + IR 旧编号不漂移 -----------------------


def test_no_per_fluid_solver_choice_field():
    """卡片 §7 config 层：无逐流体 hd|mhd 选择字段。FluidSpec 只携带
    组分角色（name/mass/charge/kind），任何 solver 级字段都不存在；
    磁开时 normalize 出的每支流体 kind 仍只是组分角色。"""
    import dataclasses

    from xmf.config import FluidSpec

    names = {f.name for f in dataclasses.fields(FluidSpec)}
    assert names == {"name", "mass", "charge", "kind"}
    assert names.isdisjoint({"solver", "hd", "mhd", "riemann", "mhd_reflect"})
    packed = xmf.normalize({**xmf.shock_tube(), "mhd": True})
    assert all(f.kind in ("neutral", "ion") for f in packed.fluids)


def test_ir_frozen_numbers_below_mag_appends():
    """追加不重排（卡片禁令「操作码数字不改」的正面钉）：kind 0..4 与
    写名 0..5 的旧编号逐字不动；编译产物里实际出现的操作码数字
    （0/1/2/9）也不漂移。"""
    from xmf.source_ir import (
        KIND_ELEC,
        KIND_FLUID,
        KIND_FLUX,
        KIND_GRID,
        KIND_NB,
        WRITE_DN,
        WRITE_DPE,
        WRITE_DP,
        WRITE_DVX,
        WRITE_DVY,
        WRITE_DVZ,
    )

    assert (KIND_GRID, KIND_FLUID, KIND_ELEC, KIND_FLUX, KIND_NB) == (0, 1, 2, 3, 4)
    assert (WRITE_DN, WRITE_DVX, WRITE_DVY, WRITE_DVZ, WRITE_DP, WRITE_DPE) == (
        0, 1, 2, 3, 4, 5,
    )
    env = env_from(["HI"], {})
    tape = compile_expr("2.0 + n_HI", env, {}, "dn_HI")
    # LOAD_CONST=0 / LOAD_SLOT=1 / ADD=2（操作数发射顺序不钉，钉数字）
    assert sorted(op for op, _ in tape.ops) == [0, 1, 2]
    tape_log = compile_expr("log(3.0)", env, {}, "dn_HI")
    assert [op for op, _ in tape_log.ops] == [0, 9]  # LOAD_CONST, LOG
