"""A3 补：解析 ``user`` 边界赋值带子契约单测（test_bc_user）。

规格来源：docs/SESSION.md（本刀卡片）§交什么 4 与 docs/cmd/PROMPT_A3_USERBC.md
§4「单测」。逐条对应（一条不少）：

  1. 无带子 ``user`` 面仍按 copy 垫（旧测
     test_bc::test_user_face_queryable_and_ghost_defined 不动、保持绿；
     本文件再钉一条紧凑版防线）。
  2. 常值赋值 ``n_HI = 7.0``：该面**全体** ghost（两层、整张横平面）为 7，
     内区活动格不动。
  3. ``@1``：``n_HI@1`` 与无带子 copy 逐位同值（两条 run 直接比）。
  4. ``@2``：``2*n_HI@1 - n_HI@2`` 线性外推，内区构造真斜率，对到相对
     容差 1e-12，且与纯 copy 撞不上值（证明 @2 真被读了）；beg/end 两面。
  5. 用本 ghost 的 r：``n0*exp(-(r-R)/H)`` 对解析值，常数走 constants{}，
     逐 ghost 用其自身格心坐标（两层 ghost 值必须不同）。
  6. 薄 2D：只垫 x2beg，不影响 i 方向内区；同 run 的 x1 copy、x2end
     reflect 行为不变。
  7. 拒绝路径（setup/normalize 抛 ValueError）：@3、max(/min(、增量写名
     dn_/dvx_/dP_、给非 user 面挂式子、同一 x1 面既 boundary_floor 又
     user_bc。另附 builder 交接提到的 C loader 拒绝（薄面读 @2）经
     setup() 变 RuntimeError（可选项，顺手钉上）。
  8. 回归钉：sources=[] 空表 Sod L1 三数逐位钉死；user_bc 带子在场时
     源项口 xmf_ir_ntape 不增加（带子挂面上，不进 run->tapes）。
  9. B1 补（开门 user 面 P_e 鬼格补边，§12）：开门（q≠0 离子 +
     pressure-only）后，带子面 / floor 面（配置键 boundary_floor）/
     无带子无 floor 的 user 面 / 薄 2D x2beg 带子面：P_e 鬼格全层 ==
     同横排最外活动层（零阶复制），活动区含最内活动格逐位不变，且鬼格
     确被重写（旧核早退跳过补边、留旧值必红）；关门现钉（§9 的
     test_pe_block_untouched_by_tape_fill）不放宽。来源：B1 补卡片
     （docs/SESSION.md §交什么 4 与 docs/cmd/PROMPT_B1SUP_PEBC.md §4）。

卡片另两条：
  - 禁止只测 VX1 / 单槽：本文件写名覆盖 n_/vx_/P_ 三个槽，并显式断言
    未写槽（vy_/vz_ 等）回退 copy @1。
  - 电子 P_e 不碰：pressure-only 配置下 fill 前后 Pe 块逐位不动。

约定（docs/TODO.md §A）：一块一文件；断言只打 bc 带子模块契约，不靠整条
激波管「看起来像」；维数覆盖 1D（nx2=nx3=1）与薄 2D（8×4）。不读
pluto.ini、不用 XUV_FACTOR==0 哨兵、除回归钉外不跑 advance 物理。

带口语义（builder 交接 + xmf_bc.c 注释）：
  - 配置：``"boundary": {"x1beg": "user"}, "user_bc": {"x1beg": {<写名>: <式子>}}``；
    未写槽自动 copy @1；@1 = 本面内侧第 1 活动格，@2 = 第 2，面相对。
  - 带口**只赋值 ghost**，不钉最内活动格（那是 A5 地板的事，不混）。
  - setup 的初始 fill 先于带口加载，故显式 ``xmf_fill_bc`` 激活带子
    （与 A5 地板同口径）；step 开头的 fill 也会跑带子。
  - 默认配置未走 xmf_phys_load，mass_code == 1.0，密度槽读写不换算：
    带子写 ``n_HI = v`` 存的就是 v，``n_HI@k`` 读的就是存的值。
  - B1 补口径：开门（q≠0 离子 + pressure-only）时 user 面 P_e 鬼格由
    通用辅助块口（xmf_bc.c bc_fill_aux_blocks）零阶复制补齐——floor 缓冲
    无 P_e 槽、带子槽位检查拒收电子槽，P_e 无规定值来源，「从内复制」
    是唯一语义，最内活动格永不覆盖。1D 横方向退化：面 2..5 的垫边会把
    唯一活动横排 (k=j=ng) 广播到所有横排（流体五槽向来如此），断言锚定
    中央行；多面角区按面序 0→5 后写者覆盖，判「鬼格 == copy@1」时对比
    终态同行而非原始图案同 (k, i)。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import BC_ENUM, KB

# --- 用例参数（取自 shock_tube() 默认值，防硬编码漂移） ----------------------
_BASE_GRID = xmf.shock_tube()["grid"]
_NX1 = _BASE_GRID["n"][0]        # 256
_NG = _BASE_GRID["nghost"]       # 2
_NXTOT = _NX1 + 2 * _NG          # 260
_NYTOT = 1 + 2 * _NG             # 5（1D 横方向退化，仍整面写）
_NZTOT = 1 + 2 * _NG             # 5

NX1_2D, NX2_2D = 8, 4            # 薄 2D；核 sanity 下限 n[0]>=8，同 test_bc
_NXTOT2 = NX1_2D + 2 * _NG       # 12
_NYTOT2 = NX2_2D + 2 * _NG       # 8
_NZTOT2 = 1 + 2 * _NG            # 5
NCELLS_2D = _NXTOT2 * _NYTOT2 * _NZTOT2

# 槽序 (n, vx, vy, vz, P) == 0..4（A2 契约）
SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4
FACE_X1BEG, FACE_X1END, FACE_X2BEG, FACE_X2END = 0, 1, 2, 3

# 空表 Sod 回归钉（与 tests/test_step_empty.py 逐位同值；口径：
# scale = (n_l, max|v_ex|, max p_ex)，精确解取 advance 实际到达的 t）
PIN_T = 0.15224840316529667
PIN_L1_N = 0.0035152402455663314   # 3.515e-3
PIN_L1_V = 0.0097162878019579242   # 9.716e-3
PIN_L1_P = 0.0040293309314733058   # 4.029e-3

_PTR = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（bc / state / mesh 查询符号已绑定）。"""
    return load()


# --- 只读小助手（与 test_bc 同口径） -----------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_PTR)


def _pattern(ncells: int, var: int, salt: float = 0.0) -> np.ndarray:
    """处处非零、逐位可复现、随 var/salt 互不相同；沿 i 斜率恒为 1。"""
    return np.arange(ncells, dtype=np.float64) + 0.25 * (var + 1) + salt


def _pack_all(lib, h, ncells: int, salt: float = 0.0) -> dict:
    pats = {}
    for var in range(5):
        pat = _pattern(ncells, var, salt)
        assert lib.xmf_state_pack(h, 0, var, _ptr(pat)) == ncells, (
            f"槽 {var}：pack 应返回 ncells={ncells}"
        )
        pats[var] = pat
    return pats


def _unpack_all(lib, h, ncells: int) -> dict:
    out = {}
    for var in range(5):
        buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(h, 0, var, _ptr(buf)) == ncells, (
            f"槽 {var}：unpack 应返回 ncells={ncells}"
        )
        out[var] = buf
    return out


def _user_cfg(boundary=None, user_bc=None, **extra_top):
    """shock_tube() + user 面 + 可选带子；额外顶层键（如 constants）并入。"""
    cfg = xmf.shock_tube()
    cfg["boundary"] = boundary if boundary is not None else {
        "x1beg": "user", "x1end": "reflect",
    }
    if user_bc is not None:
        cfg["user_bc"] = user_bc
    for key, val in extra_top.items():
        cfg[key] = val
    return cfg


def _shape1():
    return (_NZTOT, _NYTOT, _NXTOT)


# --- 1. 无带子 user 面：仍按 copy 垫（旧测的紧凑防线） ------------------------


def test_user_face_without_tape_still_copy(lib):
    """无 user_bc 带子的 user 面：C 查询仍为 3，ghost 按 copy 垫。

    与旧测 test_bc::test_user_face_queryable_and_ghost_defined 同口径：
    每一 (k,j) 行 ghost == 该行最外层内区；对照面 x1end（reflect）仍翻号。
    """
    run = xmf.setup(_user_cfg())
    try:
        h = run._h
        assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["user"] == 3
        assert lib.xmf_bc_type(h, FACE_X1END) == BC_ENUM["reflect"] == 0
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = _shape1()
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            assert np.all(np.isfinite(got)), f"槽 {var}：fill 后不得有 NaN/Inf"
            for k in range(_NZTOT):
                for j in range(_NYTOT):
                    for m in range(_NG):
                        assert got[k, j, _NG - 1 - m] == got[k, j, _NG], (
                            f"槽 {var} 无带子 user 面 (k={k},j={j})：应按 copy 垫"
                        )
            # 中心行锚定 pack 原值
            k0 = j0 = _NG
            np.testing.assert_array_equal(
                got[k0, j0, _NG : _NG + _NX1], pat[k0, j0, _NG : _NG + _NX1],
                err_msg=f"槽 {var}：fill 不得触碰内区",
            )
        # 对照面仍 reflect：中心行 vx 翻号
        vx = out[SLOT_VX].reshape(shape)
        vx_pat = pats[SLOT_VX].reshape(shape)
        k0 = j0 = _NG
        for m in range(_NG):
            assert vx[k0, j0, _NG + _NX1 + m] == -vx_pat[k0, j0, _NG + _NX1 - 1 - m]
    finally:
        run.close()


# --- 2. 常值赋值：该面全体 ghost 为 7，内区活动格不动 --------------------------


def test_constant_assigns_every_ghost_of_face(lib):
    """``n_HI = 7.0``：x1beg 面**全体** ghost（两层 × 整张横平面）为 7。

    卡片第 2 条。内区活动格（1D 即中心行）五槽逐位不动；未写槽回退
    copy @1。写名用当前组分 HI（shock_tube 唯一组分）。
    """
    run = xmf.setup(_user_cfg(user_bc={"x1beg": {"n_HI": "7.0"}}))
    try:
        h = run._h
        assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["user"]
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = _shape1()
        n_got = out[SLOT_N].reshape(shape)
        # 全体 ghost：两层、整张横平面（含横方向 ghost 行）逐位 7.0
        assert np.all(n_got[:, :, :_NG] == 7.0), (
            "常值带子应把该面全体 ghost 的 n 槽赋成 7.0"
            f"（实际范围 [{n_got[:, :, :_NG].min()}, {n_got[:, :, :_NG].max()}]）"
        )
        # 内区活动格（中心行）五槽逐位不动
        k0 = j0 = _NG
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            np.testing.assert_array_equal(
                got[k0, j0, _NG : _NG + _NX1], pat[k0, j0, _NG : _NG + _NX1],
                err_msg=f"槽 {var}：带子只写 ghost，不得钉内区活动格",
            )
        # 未写槽（vx..P）回退 copy @1：每行 ghost == 该行最外层内区
        for var in (SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P):
            got = out[var].reshape(shape)
            for m in range(_NG):
                np.testing.assert_array_equal(
                    got[:, :, m], got[:, :, _NG],
                    err_msg=f"槽 {var}：未写槽应回退 copy @1（零梯度）",
                )
    finally:
        run.close()


# --- 3. 多槽写名 + 未写槽回退（禁只测单槽） ----------------------------------


def test_multi_slot_writes_and_unwritten_fallback(lib):
    """同面写 n_/vx_/P_ 三个槽，未写的 vy_/vz_ 回退 copy @1。

    卡片：禁止只测单一槽，要钉写名对应的槽并断言未写槽走 @1。
    n_HI=7.0、vx_HI=-3.5（与内区正模式撞不上值）、P_HI=P_HI@1；
    模式内区恒正，故写后的负 vx 直接证明是「赋值」而非 copy。
    """
    run = xmf.setup(_user_cfg(user_bc={"x1beg": {
        "n_HI": "7.0",
        "vx_HI": "-3.5",
        "P_HI": "P_HI@1",
    }}))
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert np.all(pats[SLOT_VX] > 0.0), "内区 vx 模式应恒正（赋负值才撞不出）"
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = _shape1()
        n_got = out[SLOT_N].reshape(shape)
        vx_got = out[SLOT_VX].reshape(shape)
        p_got = out[SLOT_P].reshape(shape)

        # 写名槽：全体 ghost 逐位赋值
        assert np.all(n_got[:, :, :_NG] == 7.0), "n_HI 应被赋为 7.0"
        assert np.all(vx_got[:, :, :_NG] == -3.5), "vx_HI 应被赋为 -3.5"
        # 赋值 != copy：模式恒正，负速度不可能是 copy 来的
        assert np.all(vx_got[:, :, :_NG] != vx_got[:, :, _NG : _NG + 1]), (
            "vx ghost 与 @1 不同值才证明是带子赋值而非 copy 兜底"
        )
        # P_HI@1：每行 ghost == 该行最外层内区 P
        for m in range(_NG):
            np.testing.assert_array_equal(
                p_got[:, :, m], p_got[:, :, _NG],
                err_msg="P_HI@1 应等于该行 @1 的 P",
            )
        # 未写槽 vy_/vz_：回退 copy @1
        for var in (SLOT_VY, SLOT_VZ):
            got = out[var].reshape(shape)
            for m in range(_NG):
                np.testing.assert_array_equal(
                    got[:, :, m], got[:, :, _NG],
                    err_msg=f"槽 {var}：未写槽应回退 copy @1",
                )
        # 内区（中心行）五槽逐位不动
        k0 = j0 = _NG
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            np.testing.assert_array_equal(
                got[k0, j0, _NG : _NG + _NX1], pat[k0, j0, _NG : _NG + _NX1],
                err_msg=f"槽 {var}：fill 不得触碰内区",
            )
    finally:
        run.close()


# --- 4. @1 与无带子 copy 逐位同值 ---------------------------------------------


def test_at1_tape_matches_no_tape_copy(lib):
    """``n_HI@1``：两条同模式 run（无带子 / 有带子）fill 后五槽逐位相同。

    卡片第 3 条。同一确定性模式分别喂给无带子 user 面（copy 垫）与
    ``n_HI = n_HI@1`` 带子面，unpack 全量比对：@1 语义就是 copy。
    """
    outs = {}
    for tag, ubc in (("plain", None), ("tape", {"x1beg": {"n_HI": "n_HI@1"}})):
        run = xmf.setup(_user_cfg(user_bc=ubc))
        try:
            h = run._h
            ncells = lib.xmf_state_ncells(h)
            _pack_all(lib, h, ncells)  # 确定性模式，两条 run 完全一致
            assert lib.xmf_fill_bc(h) == 0
            outs[tag] = _unpack_all(lib, h, ncells)
        finally:
            run.close()
    for var in range(5):
        np.testing.assert_array_equal(
            outs["tape"][var], outs["plain"][var],
            err_msg=f"槽 {var}：n_HI@1 带子应与无带子 copy 逐位同值",
        )


# --- 5. @2 线性外推（真斜率，beg/end 两面） -----------------------------------


def test_at2_linear_extrapolation_both_ends(lib):
    """``2*n_HI@1 - n_HI@2``：线性外推对到相对 1e-12，且不与纯 copy 撞值。

    卡片第 4 条。内区模式沿 i 斜率恒为 1（真斜率），外推值 = @1 - 1，
    与纯 copy 的 @1 必不同 ⇒ @2 真被读了。beg/end 两面同一条式子，
    钉面相对语义（@1/@2 与 beg/end、轴向无关）。
    """
    run = xmf.setup(_user_cfg(
        boundary={"x1beg": "user", "x1end": "user"},
        user_bc={
            "x1beg": {"n_HI": "2*n_HI@1 - n_HI@2"},
            "x1end": {"n_HI": "2*n_HI@1 - n_HI@2"},
        },
    ))
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = _shape1()
        k0 = j0 = _NG
        n_got = out[SLOT_N].reshape(shape)
        n_pat = pats[SLOT_N].reshape(shape)

        # 真斜率哨兵：@2 与 @1 差 1（否则外推与 copy 无法区分）
        assert n_pat[k0, j0, _NG + 1] - n_pat[k0, j0, _NG] == 1.0

        # x1beg：两层 ghost 都是 2*@1 - @2
        want_beg = 2.0 * n_pat[k0, j0, _NG] - n_pat[k0, j0, _NG + 1]
        np.testing.assert_allclose(
            n_got[k0, j0, :_NG], want_beg, rtol=1e-12,
            err_msg="x1beg ghost 应为 2*n@1 - n@2（线性外推）",
        )
        # x1end：@1/@2 面相对（内侧第 1、第 2 活动格）
        i1, i2 = _NG + _NX1 - 1, _NG + _NX1 - 2
        want_end = 2.0 * n_pat[k0, j0, i1] - n_pat[k0, j0, i2]
        np.testing.assert_allclose(
            n_got[k0, j0, _NG + _NX1 :], want_end, rtol=1e-12,
            err_msg="x1end ghost 应为 2*n@1 - n@2（@1/@2 面相对）",
        )
        # 证明 @2 真被读：外推值与纯 copy（@1）撞不上
        assert np.all(n_got[k0, j0, :_NG] != n_pat[k0, j0, _NG]), (
            "x1beg 外推值不得等于纯 copy 值（否则 @2 没被读）"
        )
        assert np.all(n_got[k0, j0, _NG + _NX1 :] != n_pat[k0, j0, i1]), (
            "x1end 外推值不得等于纯 copy 值（否则 @2 没被读）"
        )
        # 内区（中心行）逐位不动；未写槽回退 copy @1
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            np.testing.assert_array_equal(
                got[k0, j0, _NG : _NG + _NX1], pat[k0, j0, _NG : _NG + _NX1],
                err_msg=f"槽 {var}：fill 不得触碰内区",
            )
            if var != SLOT_N:
                for m in range(_NG):
                    assert got[k0, j0, _NG - 1 - m] == got[k0, j0, _NG]
                    assert got[k0, j0, _NG + _NX1 + m] == got[k0, j0, _NG + _NX1 - 1]
    finally:
        run.close()


# --- 6. 用本 ghost 的 r：n0*exp(-(r-R)/H) 对解析 ------------------------------


def test_ghost_r_analytic_profile(lib):
    """``n_HI = n0*exp(-(r-R)/H)``：逐 ghost 用其自身格心坐标。

    卡片第 5 条。常数走 constants{}；r 是正在填的那颗 ghost 的格心，
    故两层 ghost 值必须不同（同一常值或纯 @1 都做不到）。x 坐标经
    xmf_mesh_copy_x 取回（含 ghost），解析值逐 ghost 比对。
    """
    n0, r0, h_scale = 2.5, 1.0, 0.3
    run = xmf.setup(_user_cfg(
        user_bc={"x1beg": {"n_HI": "n0 * exp(-(r - R)/H)"}},
        constants={"n0": n0, "R": r0, "H": h_scale},
    ))
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)

        # ghost 格心坐标（含 ghost 段，长 _NXTOT）
        x = np.empty(_NXTOT, dtype=np.float64)
        assert lib.xmf_mesh_copy_x(h, 0, _ptr(x)) == _NXTOT

        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)
        n_got = out[SLOT_N].reshape(_shape1())

        for m in range(_NG):
            want = n0 * np.exp(-(x[m] - r0) / h_scale)
            np.testing.assert_allclose(
                n_got[:, :, m], want, rtol=1e-12,
                err_msg=f"ghost 层 m={m}：应等于 n0*exp(-(r-R)/H) 在格心 x[{m}] 的值",
            )
        # 逐 ghost 的 r：两层格心不同 → 值必须不同（排除常值/@1 冒充）
        assert np.all(n_got[:, :, 0] != n_got[:, :, 1]), (
            "两层 ghost 的解析值应不同（各用自己的格心 r）"
        )
        # 内区（中心行）五槽逐位不动
        k0 = j0 = _NG
        for var in range(5):
            got = out[var].reshape(_shape1())
            pat = pats[var].reshape(_shape1())
            np.testing.assert_array_equal(
                got[k0, j0, _NG : _NG + _NX1], pat[k0, j0, _NG : _NG + _NX1],
                err_msg=f"槽 {var}：fill 不得触碰内区",
            )
    finally:
        run.close()


# --- 7. 薄 2D：只垫 x2beg，不影响 i 方向内区 ----------------------------------


def test_2d_x2beg_tape_does_not_touch_i_interior(lib):
    """薄 2D（8×4）：x2beg 挂带子，只垫 j 面 ghost，i 内区逐位不动。

    卡片第 6 条。x2beg 写 n_HI=7.0、vx_HI=-2.0（全 i 行，含 i ghost 列）；
    未写的法向槽 vy 回退 copy @1（不翻号，与 x2end reflect 对照）；
    同 run 的 x1 两端仍是 copy、x2end 仍是 reflect —— 证明带子按面挂、
    不串面。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "copy", "x2beg": "user", "x2end": "reflect"}
    cfg["user_bc"] = {"x2beg": {"n_HI": "7.0", "vx_HI": "-2.0"}}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        assert ncells == NCELLS_2D
        assert lib.xmf_bc_type(h, FACE_X2BEG) == BC_ENUM["user"]

        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT2, _NYTOT2, _NXTOT2)
        k0 = _NG
        jin = slice(_NG, _NG + NX2_2D)   # j 内区
        iin = slice(_NG, _NG + NX1_2D)   # i 内区

        # (a) 内区（i 与 j 方向）五槽逐位不动 —— 「不影响 i 方向内区」
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            np.testing.assert_array_equal(
                got[k0, jin, iin], pat[k0, jin, iin],
                err_msg=f"槽 {var}：带子只写 x2beg ghost，内区不得改动",
            )

        # (b) x2beg ghost 行（全 i 行，两层）：写名槽逐位赋值
        n_got = out[SLOT_N].reshape(shape)
        vx_got = out[SLOT_VX].reshape(shape)
        assert np.all(n_got[k0, :_NG, :] == 7.0), "x2beg 全行 n 应为 7.0"
        assert np.all(vx_got[k0, :_NG, :] == -2.0), "x2beg 全行 vx 应为 -2.0"

        # (c) 未写槽（含法向 vy）回退 copy @1：整行 == @1 行（j=ng）
        for var in (SLOT_VY, SLOT_VZ, SLOT_P):
            got = out[var].reshape(shape)
            for m in range(_NG):
                np.testing.assert_array_equal(
                    got[k0, _NG - 1 - m, :], got[k0, _NG, :],
                    err_msg=f"槽 {var} x2beg m={m}：未写槽应回退 copy @1",
                )
            for m in range(_NG):
                np.testing.assert_array_equal(
                    got[k0, _NG - 1 - m, iin], pats[var].reshape(shape)[k0, _NG, iin],
                    err_msg=f"槽 {var} x2beg m={m}：回退值应锚定 pack 的 @1 行",
                )
        # 法向槽 vy 未写 → 不翻号（与 reflect 区分；模式恒正）
        vy_got = out[SLOT_VY].reshape(shape)
        assert np.all(vy_got[k0, :_NG, iin] > 0.0), (
            "x2beg 未写的法向槽走 copy @1，不得翻号"
        )

        # (d) 同 run 的 x1 面仍是 copy（内区 j 行上断言，锚定 pack）
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            for m in range(_NG):
                np.testing.assert_array_equal(
                    got[k0, jin, _NG - 1 - m], pat[k0, jin, _NG],
                    err_msg=f"槽 {var} x1beg m={m}：仍应按 copy 垫",
                )
                np.testing.assert_array_equal(
                    got[k0, jin, _NG + NX1_2D + m], pat[k0, jin, _NG + NX1_2D - 1],
                    err_msg=f"槽 {var} x1end m={m}：仍应按 copy 垫",
                )

        # (e) 同 run 的 x2end 仍是 reflect：法向槽 vy 奇、其余偶
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            flip = var == SLOT_VY
            sign = -1.0 if flip else 1.0
            for m in range(_NG):
                je = _NG + NX2_2D + m
                jse = _NG + NX2_2D - 1 - m
                np.testing.assert_array_equal(
                    got[k0, je, iin], sign * pat[k0, jse, iin],
                    err_msg=f"槽 {var} x2end m={m}：仍应镜像"
                            f"{'翻号（奇）' if flip else '同号（偶）'}",
                )
        assert np.all(vy_got[k0, _NG + NX2_2D :, iin] < 0.0), (
            "x2end reflect 的 vy ghost 应翻号为负（与 x2beg 带子面对照）"
        )
    finally:
        run.close()


# --- 8. 拒绝路径：setup/normalize 抛 ValueError --------------------------------


def test_reject_layer_3():
    """卡片第 7 条：更深层 @3 拒绝（只许 @1/@2）。"""
    cfg = _user_cfg(user_bc={"x1beg": {"n_HI": "n_HI@3"}})
    with pytest.raises(ValueError, match="neighbour layer"):
        xmf.setup(cfg)


def test_reject_max_min_calls():
    """卡片第 7 条：不为边界开 max/min 操作码，出现即拒。

    两参调用命中「只许 exp(x)/log(x) 调用」分支，单参命中
    「illegal function」分支——两条消息都接受，总之必须 ValueError。
    """
    for bad in ("max(n_HI@1, 1.0)", "min(n_HI@1, 1.0)", "max(n_HI@1)"):
        cfg = _user_cfg(user_bc={"x1beg": {"n_HI": bad}})
        with pytest.raises(ValueError, match="(illegal function|calls are allowed)"):
            xmf.setup(cfg)


@pytest.mark.parametrize("bad_name", ["dn_HI", "dvx_HI", "dP_HI"])
def test_reject_increment_write_names(bad_name):
    """卡片第 7 条：增量写名（源项语义）不得出现在边界赋值带子。"""
    cfg = _user_cfg(user_bc={"x1beg": {bad_name: "n_HI@1"}})
    with pytest.raises(ValueError, match="increment name"):
        xmf.setup(cfg)


@pytest.mark.parametrize("bc_kind", ["copy", "reflect"])
def test_reject_tape_on_non_user_face(bc_kind):
    """卡片第 7 条：带子只能挂在已是 user 的面上。"""
    cfg = _user_cfg(
        boundary={"x1beg": bc_kind, "x1end": "reflect"},
        user_bc={"x1beg": {"n_HI": "n_HI@1"}},
    )
    with pytest.raises(ValueError, match="require boundary"):
        xmf.setup(cfg)


def test_reject_floor_and_tape_conflict():
    """卡片第 7 条：同一 x1 面既 boundary_floor 又 user_bc 带子 → 拒。

    地板钉底层（ghost + 最内活动格），带子只赋值 ghost，语义互斥。
    """
    cfg = _user_cfg(user_bc={"x1beg": {"n_HI": "n_HI@1"}})
    # floor 形状 [nfluid][5][nztot*nytot]；拒绝发生在用之前，内容不重要
    cfg["boundary_floor"] = [[[0.0] * (_NZTOT * _NYTOT) for _ in range(5)]]
    with pytest.raises(ValueError, match="mutually exclusive"):
        xmf.setup(cfg)


def test_c_loader_rejects_at2_on_thin_face():
    """builder 交接可选项：C loader 拒绝法向内区不足 2 格却读 @2。

    Python 侧 @2 合法；1D 的 x2 面法向内区只有 1 格，
    xmf_bc_load_tapes 验证失败返回 -1，经 setup() 变 RuntimeError。
    """
    cfg = _user_cfg(
        boundary={"x1beg": "user", "x2beg": "user"},
        user_bc={"x2beg": {"n_HI": "n_HI@2"}},
    )
    with pytest.raises(RuntimeError, match="bc_load_tapes"):
        xmf.setup(cfg)


# --- 9. 电子 P_e 不碰 ----------------------------------------------------------


def test_pe_block_untouched_by_tape_fill(lib):
    """pressure-only 电子：带子 fill 前后 Pe 块逐位不动（卡片：不碰 P_e）。

    顺带断言流体 ghost 确实被带子赋值（证明 fill 真跑了，Pe 的「不动」
    不是「fill 没跑」）。
    """
    cfg = _user_cfg(user_bc={"x1beg": {"n_HI": "7.0"}})
    cfg["electron"] = {"mode": "pressure-only"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        _pack_all(lib, h, ncells)

        pe_in = np.arange(ncells, dtype=np.float64) + 1234.5
        assert lib.xmf_state_write_pe(h, _ptr(pe_in)) == ncells

        assert lib.xmf_fill_bc(h) == 0

        pe_out = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_copy_pe(h, _ptr(pe_out)) == ncells
        np.testing.assert_array_equal(
            pe_out, pe_in, err_msg="带子 fill 不得触碰 Pe 块（含 ghost）"
        )
        # fill 真跑了：流体 n 槽 ghost 已被赋成 7.0
        n_buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(h, 0, SLOT_N, _ptr(n_buf)) == ncells
        assert np.all(n_buf.reshape(_shape1())[:, :, :_NG] == 7.0)
    finally:
        run.close()


# --- 10. 回归钉：空表 Sod L1 与源项口 -----------------------------------------


def test_empty_sod_l1_pin_unchanged():
    """卡片第 8 条：sources=[] 空表 Sod 的 pcm L1 三数逐位钉不动。

    口径与 tests/test_step_empty.py 逐字相同（scale = (n_l, max|v_ex|,
    max p_ex)，精确解取 advance 实际到达的 t，全精度 ==）。边界带子这
    一刀不得动纯 hydro 空路径。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.riemann import exact_hi  # noqa: E402

    cfg = xmf.shock_tube()
    assert cfg["sources"] == [], "本钉只对空 sources 表有效"
    run = xmf.setup(cfg)
    try:
        assert run.num_tapes == 0
        run.advance()
        assert run.t == PIN_T, (
            f"空表最终 t 与基线不一致：实测 {run.t!r}，钉值 {PIN_T!r}"
        )
        hi = run.fields["HI"]
        ic = cfg["ic"]
        p_l = ic["left"]["n"] * KB * ic["left"]["T"]
        p_r = ic["right"]["n"] * KB * ic["right"]["T"]
        n_ex, v_ex, p_ex = exact_hi(
            hi.x, run.t, ic["x0"], ic["left"]["n"], ic["right"]["n"],
            0.0, 0.0, p_l, p_r, gamma=run._cfg.gamma,
        )
        scale_n = ic["left"]["n"]
        scale_v = max(float(np.max(np.abs(v_ex))), 1.0)
        scale_p = float(np.max(p_ex))
        err_n = float(np.mean(np.abs(hi.n - n_ex)) / scale_n)
        err_v = float(np.mean(np.abs(hi.v - v_ex)) / scale_v)
        err_p = float(np.mean(np.abs(hi.p - p_ex)) / scale_p)
        assert err_n == PIN_L1_N, (
            f"空表 L1(n)/scale 逐位钉失效：实测 {err_n!r}，钉值 {PIN_L1_N!r}"
        )
        assert err_v == PIN_L1_V, (
            f"空表 L1(v)/scale 逐位钉失效：实测 {err_v!r}，钉值 {PIN_L1_V!r}"
        )
        assert err_p == PIN_L1_P, (
            f"空表 L1(P)/scale 逐位钉失效：实测 {err_p!r}，钉值 {PIN_L1_P!r}"
        )
    finally:
        run.close()


def test_bc_tapes_do_not_enter_source_tape_set(lib):
    """卡片第 8 条：user_bc 带子在场时 xmf_ir_ntape 不增加。

    带子挂在面上（run->bc_tapes），不进 run->tapes / Newton。为证明
    「带子确实在场」（否则 ntape==0 平凡成立），先验证 ghost 已被带子
    赋值，再断言源项口计数仍为 0。
    """
    run = xmf.setup(_user_cfg(user_bc={"x1beg": {"n_HI": "7.0", "P_HI": "P_HI@1"}}))
    try:
        h = run._h
        # 带子在场：fill 后 ghost 被赋值（不是 copy 的内区值）
        ncells = lib.xmf_state_ncells(h)
        _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        n_buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(h, 0, SLOT_N, _ptr(n_buf)) == ncells
        assert np.all(n_buf.reshape(_shape1())[:, :, :_NG] == 7.0), (
            "前置：带子应已生效（否则本测的 ntape==0 没有证明力）"
        )
        # 源项口不因边界带子增加
        assert run.num_tapes == 0, (
            "user_bc 带子不得进源项口：num_tapes 应为 0"
        )
        assert lib.xmf_ir_ntape(h) == 0, (
            "xmf_ir_ntape 应为 0（边界带子挂面上，不进 run->tapes）"
        )
    finally:
        run.close()


# --- 11. 四步冒烟：带子在场时 setup → step → fields 仍通 ----------------------


def test_four_step_smoke_with_user_tape():
    """带子挂在 user 面上时四步 API 仍通：step 开头的 fill 会跑带子。

    用 @1 语义带子（等价 copy，数值上最稳）：三步 step 后 fields 有限、
    密度为正、Sod 台阶仍在。
    """
    run = xmf.setup(_user_cfg(user_bc={"x1beg": {
        "n_HI": "n_HI@1",
        "vx_HI": "vx_HI@1",
        "P_HI": "P_HI@1",
    }}))
    try:
        for _ in range(3):
            run.step()
        assert run.t > 0.0
        hi = run.fields["HI"]
        assert hi.n.shape == (_NX1,)
        assert np.all(np.isfinite(hi.n)) and np.all(hi.n > 0.0)
        assert np.all(np.isfinite(hi.v)) and np.all(np.isfinite(hi.p))
        assert float(np.max(hi.n)) / float(np.min(hi.n)) > 2.0, (
            "三步后 Sod 台阶应仍在（带子未破坏初值/推进）"
        )
    finally:
        run.close()


# --- 12. B1 补：开门 user 面 P_e 鬼格补边（通用辅助块口） ---------------------
#
# 合同（docs/SESSION.md B1 补 §交什么 2/4 与 PROMPT_B1SUP_PEBC.md §4）：
#   - 开门 = 至少一支 q≠0 离子 + electron: pressure-only（核内闭合门
#     elec_closure_active 为真）；关门现钉在 §9，逐位不放宽。
#   - user 面（带子 / floor / 无带子无 floor 三条早退或通用路径）P_e 鬼格
#     语义钉死：全部鬼层 == 同横排最外活动层的值（零阶复制，等效
#     fill_face_block(run, run->pe, face, 0, 0)）；最内活动格永不覆盖。
#   - 每条都带「鬼格确被重写」哨兵：旧核（两条 user 早退跳过 P_e 补边）
#     鬼格留 setup 旧值，必红。
#   - 口径见模块 docstring 末条：1D 横排广播锚定中央行；多面角区按面序
#     后写者覆盖，判等对比终态同行。


def _gate_on_cfg(boundary=None, user_bc=None, floor=None):
    """开门配置：HII（q=1 离子）+ HI（中性）+ pressure-only 电子。

    带子走 ``user_bc``（config.normalize 编译成面带，config.py:522-544）；
    地板走 ``boundary_floor``（进核内 floor 缓冲，config.py:652-654）。
    两者在同一 x1 面互斥（normalize 拒），故一次只给一条。
    """
    cfg = _user_cfg(boundary=boundary, user_bc=user_bc)
    cfg["fluids"] = [
        {"name": "HII", "mass": 1.00794, "charge": 1, "kind": "ion"},
        {"name": "HI", "mass": 1.00794, "charge": 0, "kind": "neutral"},
    ]
    cfg["electron"] = {"mode": "pressure-only"}
    if floor is not None:
        cfg["boundary_floor"] = floor
    return cfg


def _pack_two_fluids(lib, h, ncells: int) -> dict:
    """两支流体五槽全打包：确定性模式，流体间盐距 100，逐槽可辨。"""
    pats = {}
    for f in range(2):
        pats[f] = {}
        for var in range(5):
            pat = _pattern(ncells, var, salt=100.0 * f)
            assert lib.xmf_state_pack(h, f, var, _ptr(pat)) == ncells, (
                f"流体 {f} 槽 {var}：pack 应返回 ncells={ncells}"
            )
            pats[f][var] = pat
    return pats


def _read_pe_block(lib, h, ncells: int) -> np.ndarray:
    """开门 run 读回整块 P_e（含 ghost）；读不回本身即违约。"""
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_copy_pe(h, _ptr(buf)) == ncells, (
        "开门（pressure-only）run 必须能读回 P_e 块"
    )
    return buf


def test_pe_ghost_copy_at1_on_tape_face_gate_open(lib):
    """开门 + 带子 user 面（x1beg）：P_e 鬼格 == copy@1，活动区逐位不变。

    卡片 §交什么 4①。写名 ``n_HII`` 是该配置真实存在的流体槽（流体 0，
    q=1 离子）。断言链：
      - P_e 中央行鬼格全层 == 最外活动层原值，且确被重写（堵洞哨兵）；
      - 中央行活动区（含最内活动格）逐位不变（P_e 是演化量，带子拒收
        电子槽，无规定值可给）；
      - 流体鬼格确被带子赋值（证明 fill 真跑了，P_e 的变化不是「fill
        没跑」）；
      - 对照面 x1end（reflect，非 user）仍镜面（偶量不翻号）。
    """
    cfg = _gate_on_cfg(user_bc={"x1beg": {"n_HII": "7.0"}})
    run = xmf.setup(cfg)
    try:
        h = run._h
        assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["user"]
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_two_fluids(lib, h, ncells)
        pe0 = np.arange(ncells, dtype=np.float64) + 7777.25
        assert lib.xmf_state_write_pe(h, _ptr(pe0)) == ncells

        assert lib.xmf_fill_bc(h) == 0

        shape = _shape1()
        pe1 = _read_pe_block(lib, h, ncells).reshape(shape)
        pe0b = pe0.reshape(shape)
        k0 = j0 = _NG
        # C2补3 口径收窄：退化轴鬼格不再填皮（1D 的 k!=ng/j!=ng 行无读者），
        # 只断言活动方向（x1）中央行上的 P_e 鬼格。
        mid, mid0 = pe1[k0, j0, :], pe0b[k0, j0, :]
        # x1beg 鬼格 == copy@1（全部鬼层），且真被重写（堵洞哨兵）
        np.testing.assert_array_equal(
            mid[:_NG], np.full(_NG, mid0[_NG]),
            err_msg="开门带子面 P_e 鬼格应全层 == 同横排最外活动层（零阶复制）",
        )
        assert np.all(mid[:_NG] != mid0[:_NG]), (
            "P_e 鬼格必须真被重写（旧核早退跳过补边会留旧值）"
        )
        # 活动区（含最内活动格）逐位不变
        np.testing.assert_array_equal(
            mid[_NG : _NG + _NX1], mid0[_NG : _NG + _NX1],
            err_msg="P_e 是演化量：user 面补边不得覆盖活动区（含最内活动格）",
        )
        # 对照面 x1end（reflect，非 user）：P_e 镜面，偶量不翻号
        for m in range(_NG):
            assert mid[_NG + _NX1 + m] == mid0[_NG + _NX1 - 1 - m], (
                f"x1end reflect 的 P_e 鬼格 m={m} 仍应镜面（不翻号）"
            )
        # fill 真跑了：n_HII 鬼格被带子赋成 7.0（中央行 x1beg 鬼格）；
        # 中央行内区不动
        n_buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(h, 0, SLOT_N, _ptr(n_buf)) == ncells
        n1 = n_buf.reshape(shape)
        assert np.all(n1[k0, j0, :_NG] == 7.0), (
            "前置：带子应已赋值中央行流体鬼格（否则 P_e 的变化不能归因于补边）"
        )
        np.testing.assert_array_equal(
            n1[k0, j0, _NG : _NG + _NX1],
            pats[0][SLOT_N].reshape(shape)[k0, j0, _NG : _NG + _NX1],
            err_msg="带子只写 ghost：HII 内区不得改动",
        )
    finally:
        run.close()


def test_pe_ghost_copy_on_floor_face_gate_open(lib):
    """开门 + floor user 面（boundary_floor）：P_e 鬼格 == 最内活动层复制。

    卡片 §交什么 4②。配置键是 ``boundary_floor``（进核内 floor 缓冲），
    不是带子的 ``user_bc``：floor 只钉流体五槽（鬼格 + 最内活动格），
    缓冲布局没有 P_e 的位置。断言链：
      - P_e 中央行鬼格全层 == 最内活动层原值，且确被重写；
      - 中央行活动区逐位不变（floor 无 P_e 槽，最内活动格也不覆盖）；
      - 流体五槽「鬼格 + 最内活动格钉住」行为不变，其后内区不动；
      - 对照面 x1end（reflect）P_e 仍镜面。
    """
    nt = _NZTOT * _NYTOT
    # 基态逐流体逐槽可辨，且远离模式值（模式 ghost 段 >= 3000）
    floor = [
        [[2.0 + 10.0 * f + 0.25 * var] * nt for var in range(5)]
        for f in range(2)
    ]
    run = xmf.setup(_gate_on_cfg(floor=floor))
    try:
        h = run._h
        assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["user"]
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_two_fluids(lib, h, ncells)
        pe0 = np.arange(ncells, dtype=np.float64) + 8888.75
        assert lib.xmf_state_write_pe(h, _ptr(pe0)) == ncells

        assert lib.xmf_fill_bc(h) == 0

        shape = _shape1()
        pe1 = _read_pe_block(lib, h, ncells).reshape(shape)
        pe0b = pe0.reshape(shape)
        k0 = j0 = _NG
        # C2补3 口径收窄：退化轴鬼格不再填皮，只断言中央行。
        mid, mid0 = pe1[k0, j0, :], pe0b[k0, j0, :]
        # 鬼格 == 最内活动层复制，且真被重写（堵洞哨兵）
        np.testing.assert_array_equal(
            mid[:_NG], np.full(_NG, mid0[_NG]),
            err_msg="开门 floor 面 P_e 鬼格应全层 == 最内活动层的值",
        )
        assert np.all(mid[:_NG] != mid0[:_NG]), (
            "P_e 鬼格必须真被重写（旧核早退跳过补边会留旧值）"
        )
        # 活动区（含最内活动格）逐位不变
        np.testing.assert_array_equal(
            mid[_NG : _NG + _NX1], mid0[_NG : _NG + _NX1],
            err_msg="floor 无 P_e 槽：P_e 活动区（含最内活动格）不得改动",
        )
        # 对照面 x1end（reflect）：P_e 镜面，偶量不翻号
        for m in range(_NG):
            assert mid[_NG + _NX1 + m] == mid0[_NG + _NX1 - 1 - m], (
                f"x1end reflect 的 P_e 鬼格 m={m} 仍应镜面（不翻号）"
            )
        # 流体五槽钉住行为不变：全横截面鬼格 + 最内活动格 == floor 基态；
        # 其后内区逐位不动（中央行锚定）
        for f in range(2):
            for var in range(5):
                want = 2.0 + 10.0 * f + 0.25 * var
                buf = np.empty(ncells, dtype=np.float64)
                assert lib.xmf_state_unpack(h, f, var, _ptr(buf)) == ncells
                got = buf.reshape(shape)
                assert np.all(got[:, :, :_NG] == want), (
                    f"流体 {f} 槽 {var}：floor 鬼格钉住丢了"
                )
                assert got[k0, j0, _NG] == want, (
                    f"流体 {f} 槽 {var}：floor 最内活动格钉住丢了"
                )
                np.testing.assert_array_equal(
                    got[k0, j0, _NG + 1 : _NG + _NX1],
                    pats[f][var].reshape(shape)[k0, j0, _NG + 1 : _NG + _NX1],
                    err_msg=f"流体 {f} 槽 {var}：floor 只钉最内活动格，其后内区不动",
                )
    finally:
        run.close()


def test_pe_ghost_copy_on_bare_user_face_gate_open(lib):
    """开门 + 无带子无 floor 的 user 面：通用 copy 垫路径 P_e 同步补齐。

    冒烟第 4 例落常备：该面流体五槽向来按 copy 垫（§1 关门钉），开门后
    P_e 同俗——鬼格 == copy@1、活动区逐位不变、鬼格确被重写。钉的是
    bc_fill_face 通用路径并入 bc_fill_aux_blocks 后逐位语义不变。
    """
    run = xmf.setup(_gate_on_cfg())
    try:
        h = run._h
        assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["user"]
        ncells = lib.xmf_state_ncells(h)
        _pack_two_fluids(lib, h, ncells)
        pe0 = np.arange(ncells, dtype=np.float64) + 5555.5
        assert lib.xmf_state_write_pe(h, _ptr(pe0)) == ncells

        assert lib.xmf_fill_bc(h) == 0

        shape = _shape1()
        pe1 = _read_pe_block(lib, h, ncells).reshape(shape)
        pe0b = pe0.reshape(shape)
        k0 = j0 = _NG
        mid, mid0 = pe1[k0, j0, :], pe0b[k0, j0, :]
        np.testing.assert_array_equal(
            mid[:_NG], np.full(_NG, mid0[_NG]),
            err_msg="开门无带子 user 面 P_e 鬼格应按 copy@1 垫",
        )
        assert np.all(mid[:_NG] != mid0[:_NG]), "P_e 鬼格必须真被重写"
        np.testing.assert_array_equal(
            mid[_NG : _NG + _NX1], mid0[_NG : _NG + _NX1],
            err_msg="P_e 活动区（含最内活动格）不得改动",
        )
    finally:
        run.close()


def test_pe_ghost_copy_on_2d_x2beg_tape_face_gate_open(lib):
    """开门 + 薄 2D x2beg 带子面（卡片 §交什么 4④）：逐行 copy@1。

    与 test_2d_x2beg_tape_does_not_touch_i_interior 同几何（8×4、x1 两端
    copy、x2beg user、x2end reflect），开门后加同样 P_e 断言。口径：
      - x3 退化（nz=1）：面 4/5 垫边把 k=ng 行广播到所有 k 行；
      - 多面角区按面序 0→5 后写者覆盖：面 0/1（x1 copy）先把 i 鬼格垫好，
        面 2（x2beg）再整行复制——判「鬼格 == copy@1」对比终态同行
        （j=ng 行），不对比原始图案同 (k, i)；
      - 内区盒子（ng<=j<ng+ny、ng<=i<ng+nx）含最内活动层 j=ng 逐位不变。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["fluids"] = [
        {"name": "HII", "mass": 1.00794, "charge": 1, "kind": "ion"},
        {"name": "HI", "mass": 1.00794, "charge": 0, "kind": "neutral"},
    ]
    cfg["electron"] = {"mode": "pressure-only"}
    cfg["boundary"] = {"x1": "copy", "x2beg": "user", "x2end": "reflect"}
    cfg["user_bc"] = {"x2beg": {"n_HII": "9.0"}}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        assert ncells == NCELLS_2D
        assert lib.xmf_bc_type(h, FACE_X2BEG) == BC_ENUM["user"]
        pats = _pack_two_fluids(lib, h, ncells)
        pe0 = np.arange(ncells, dtype=np.float64) * 0.5 + 3333.25
        assert lib.xmf_state_write_pe(h, _ptr(pe0)) == ncells

        assert lib.xmf_fill_bc(h) == 0

        shape = (_NZTOT2, _NYTOT2, _NXTOT2)
        pe1 = _read_pe_block(lib, h, ncells).reshape(shape)
        pe0b = pe0.reshape(shape)
        k0 = _NG
        jin = slice(_NG, _NG + NX2_2D)   # j 内区
        iin = slice(_NG, _NG + NX1_2D)   # i 内区
        # C2补3 口径收窄：x3 退化（nz=1），k!=ng 行不再填皮，只断言 k=ng 平面。
        # x2beg 鬼格：j<ng 的行整行等于终态 j=ng 行（角区归后写面）
        for m in range(_NG):
            np.testing.assert_array_equal(
                pe1[k0, _NG - 1 - m, :], pe1[k0, _NG, :],
                err_msg=f"x2beg 鬼格行 m={m} 应整行 == 终态 j=ng 行（copy@1）",
            )
        assert np.any(pe1[k0, :_NG, :] != pe0b[k0, :_NG, :]), (
            "P_e 鬼格必须真被重写（旧核早退跳过补边会留旧值）"
        )
        # 内区盒子（含最内活动层 j=ng）逐位不变
        np.testing.assert_array_equal(
            pe1[k0, jin, iin], pe0b[k0, jin, iin],
            err_msg="P_e 内区盒子（含最内活动层）不得改动",
        )
        # 带子真跑了：n_HII 在 x2beg 鬼格 == 9.0（k0 行全 i）；内区盒子不动
        n_buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(h, 0, SLOT_N, _ptr(n_buf)) == ncells
        n1 = n_buf.reshape(shape)
        assert np.all(n1[k0, :_NG, :] == 9.0), (
            "前置：带子应已赋值 x2beg 鬼格（否则 P_e 的变化不能归因于补边）"
        )
        np.testing.assert_array_equal(
            n1[k0, jin, iin], pats[0][SLOT_N].reshape(shape)[k0, jin, iin],
            err_msg="带子只写 ghost：HII 内区盒子不得改动",
        )
    finally:
        run.close()
