"""C2补4 冻格口（内部边界 / 流体障碍）契约单测（test_bc_pin）。

规格来源：docs/SESSION.md（本刀卡片）§交什么 4 与 docs/cmd/PROMPT_C2SUP_IBC.md。
逐条对应（一条不少）：

  1. 钉住不动：薄 2D，掩膜格钉 v=0，若干步后掩膜格速度仍精确为 0（逐位
     == 0.0）；掩膜外必须可演化（max|vx| > 0）。
  2. 换钉值生效且不整场回滚：同一口换成非零钉值（vy=1），步进后掩膜格
     == 新钉值；掩膜外的场没被这次加载重置（加载瞬间掩膜外逐位不变）。
  3. 无掩膜 Sod 三钉逐位：走 shock_tube() Sod，在「从不加载钉」和「显式
     set_pin(None,None) 清除后」两种情形下三钉逐位不放宽。回归保护。
  4. 鬼格掩膜测：掩膜含 x1beg 端一列鬼格，xmf_fill_bc 后这些鬼格 == 钉值。
     钉值与外边界自然值不同（-7 这种一眼可辨的值），否则测不出。
  5. 不只测 VX1：第 1/2 条已含 vy；本文件显式断言 vy 和第二方向。
  6. 同一 run 先后两口：set_pin 与 user 皮（boundary_floor / user_bc）
     可在同一 run 先后加载，各自生效互不清除。
  7. 清除语义：显式 set_pin(None, None) 或全 0 掩膜后，原掩膜格恢复自由
     演化（下一步后该格值改变）。
  8. 坏输入不破坏上一拍：错误尺寸的 mask/vals 抛异常，且抛异常前已生效
     的钉仍然生效。

约定（docs/TODO.md §A）：一块一文件；断言只打冻格模块契约，不靠整条
激波管「看起来像」；维数覆盖薄 2D（16×8）与 1D（nx2=nx3=1）。不读
pluto.ini、不用 XUV_FACTOR==0 哨兵、禁止只测 VX1。

带口语义（builder 交接 + xmf_bc.c 注释）：
  - 加载口 ``run.set_pin(mask, vals)``：mask 是 (nztot,nytot,nxtot) 的
    int32 数组（0/1，含鬼格），vals 是 (nfluid,5,nztot,nytot,nxtot) 的
    float64 数组。同一口可换钉值 / 掩膜；None 或全 0 清除。
  - 施加时机：``lf_all_dirs`` 每活动方向扫后 + ``xmf_fill_bc`` 填皮后。
  - P_e 不碰。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import KB

# --- 用例参数（取自 shock_tube() 默认值） -------------------------------------
_BASE_GRID = xmf.shock_tube()["grid"]
_NX1 = _BASE_GRID["n"][0]        # 256
_NG = _BASE_GRID["nghost"]       # 2
_NXTOT = _NX1 + 2 * _NG          # 260
_NYTOT = 1 + 2 * _NG             # 5（1D 横方向退化）
_NZTOT = 1 + 2 * _NG             # 5

NX1_2D, NX2_2D = 16, 8           # 薄 2D
_NXTOT2 = NX1_2D + 2 * _NG       # 20
_NYTOT2 = NX2_2D + 2 * _NG       # 12
_NZTOT2 = 1 + 2 * _NG            # 5

# 槽序 (n, vx, vy, vz, P) == 0..4（A2 契约）
SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

_PTR = ctypes.POINTER(ctypes.c_double)

# 空表 Sod 回归钉（与 tests/test_step_empty.py 逐位同值）
PIN_T = 0.15224840316529667
PIN_L1_N = 0.0035152402455663314
PIN_L1_V = 0.0097162878019579242
PIN_L1_P = 0.0040293309314733058


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库。"""
    return load()


# --- 只读小助手 --------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_PTR)


def _unpack(lib, h, var, ncells):
    """unpack 流体 0 槽 var 的整块（含 ghost），返回 flat 缓冲。"""
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(h, 0, var, _ptr(buf)) == ncells
    return buf


def _make_2d_run(cfg_extra=None):
    """创建薄 2D (16×8) run，默认边界 x1=reflect, x2=copy。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["ndim"] = 2
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["x2min"] = 0.0
    cfg["grid"]["x2max"] = 1.0
    cfg["time"]["tstop"] = 1.0
    cfg["boundary"] = {"x1": "reflect", "x2": "copy"}
    if cfg_extra:
        for k, v in cfg_extra.items():
            cfg[k] = v
    return xmf.setup(cfg)


def _build_pin_vals(lib, h, ncells, shape, nfluid=1):
    """从当前场读全量并构造 (nfluid, 5, *shape) 的钉值缓冲。"""
    vals = np.zeros((nfluid, 5) + shape, dtype=np.float64)
    for var in range(5):
        buf = _unpack(lib, h, var, ncells)
        vals[0, var] = buf.reshape(shape)
    return vals


# ============================================================================
# 1. 钉住不动：掩膜格钉 v=0，若干步后速度仍精确为 0；掩膜外可演化
# ============================================================================


def test_pin_zero_velocity_holds(lib):
    """卡片 §交什么 4①：薄 2D 掩膜钉 v=0，扫若干步后掩膜格速度仍精确为 0。

    掩膜外必须可演化（max|vx| > 0）。同时断言 vy 也被钉住（不只测 VX1）。
    """
    run = _make_2d_run()
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        # 掩膜：内区中央一块 3×4 的区域
        mask = np.zeros(shape, dtype=np.int32)
        mask[_NG, _NG + 2 : _NG + 5, _NG + 4 : _NG + 8] = 1
        assert np.sum(mask) > 0, "掩膜应至少有一个格"

        # 钉值：从当前场读 rho/n/P，把三个速度分量钉为 0
        vals = _build_pin_vals(lib, h, ncells, shape)
        vals[0, SLOT_VX][mask == 1] = 0.0
        vals[0, SLOT_VY][mask == 1] = 0.0
        vals[0, SLOT_VZ][mask == 1] = 0.0

        run.set_pin(mask, vals)

        # 步进若干步
        for _ in range(4):
            run.step()

        # 掩膜格速度仍精确为 0
        vx = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        vy = _unpack(lib, h, SLOT_VY, ncells).reshape(shape)
        vz = _unpack(lib, h, SLOT_VZ, ncells).reshape(shape)

        assert np.all(vx[mask == 1] == 0.0), (
            f"掩膜格 vx 应精确为 0，实测 max|vx|="
            f"{float(np.max(np.abs(vx[mask == 1]))):.3e}"
        )
        assert np.all(vy[mask == 1] == 0.0), (
            f"掩膜格 vy 应精确为 0，实测 max|vy|="
            f"{float(np.max(np.abs(vy[mask == 1]))):.3e}"
        )
        assert np.all(vz[mask == 1] == 0.0), (
            f"掩膜格 vz 应精确为 0，实测 max|vz|="
            f"{float(np.max(np.abs(vz[mask == 1]))):.3e}"
        )

        # 掩膜外可演化（vx 不全为 0）
        assert float(np.max(np.abs(vx[mask == 0]))) > 0.0, (
            "掩膜外应可演化：max|vx| > 0"
        )
    finally:
        run.close()


# ============================================================================
# 2. 换钉值生效且不整场回滚
# ============================================================================


def test_retarget_pin_value_no_field_rollback(lib):
    """卡片 §交什么 4②：同一口换成非零钉值（vy=1），掩膜外不被重置。

    加载瞬间掩膜外逐位不变（对比加载前后掩膜外某格值）。
    """
    run = _make_2d_run()
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        # 掩膜：同上
        mask = np.zeros(shape, dtype=np.int32)
        mask[_NG, _NG + 2 : _NG + 5, _NG + 4 : _NG + 8] = 1

        # 第一拍：v=0
        vals = _build_pin_vals(lib, h, ncells, shape)
        vals[0, SLOT_VX][mask == 1] = 0.0
        vals[0, SLOT_VY][mask == 1] = 0.0
        vals[0, SLOT_VZ][mask == 1] = 0.0
        run.set_pin(mask, vals)

        # 步进几步让场演化
        for _ in range(3):
            run.step()

        # 记录换钉前掩膜外的场
        rho_before = _unpack(lib, h, SLOT_N, ncells).reshape(shape).copy()
        vy_before = _unpack(lib, h, SLOT_VY, ncells).reshape(shape).copy()

        # 换钉值：vy=1.0（其他保持当前场值）
        vals2 = _build_pin_vals(lib, h, ncells, shape)
        vals2[0, SLOT_VY][mask == 1] = 1.0
        vals2[0, SLOT_VX][mask == 1] = 0.0
        vals2[0, SLOT_VZ][mask == 1] = 0.0
        run.set_pin(mask, vals2)

        # 加载瞬间掩膜外逐位不变
        rho_after_load = _unpack(lib, h, SLOT_N, ncells).reshape(shape)
        np.testing.assert_array_equal(
            rho_after_load[mask == 0], rho_before[mask == 0],
            err_msg="换钉加载瞬间：掩膜外 n 应逐位不变",
        )

        # 步进一步
        run.step()

        # 掩膜格 vy == 新钉值 1.0
        vy = _unpack(lib, h, SLOT_VY, ncells).reshape(shape)
        assert np.all(vy[mask == 1] == 1.0), (
            f"换钉后掩膜格 vy 应 == 1.0，实测唯一值 {np.unique(vy[mask == 1])}"
        )
    finally:
        run.close()


# ============================================================================
# 3. 无掩膜 Sod 三钉逐位（回归保护）
# ============================================================================


def _run_sod_three_pin(lib, clear_first=False):
    """跑空表 Sod advance，返回 (t, err_n, err_v, err_p)。

    clear_first=True 时在 advance 前显式 set_pin(None, None)。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.riemann import exact_hi

    cfg = xmf.shock_tube()
    assert cfg["sources"] == [], "本钉只对空 sources 表有效"
    run = xmf.setup(cfg)
    try:
        if clear_first:
            run.set_pin(None, None)
        run.advance()
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
        return run.t, err_n, err_v, err_p
    finally:
        run.close()


def test_no_pin_sod_regression_never_loaded():
    """卡片 §交什么 4③（情形一）：从不加载钉的空表 Sod 三钉逐位。

    回归保护：冻格口没碰空表路径。
    """
    t, err_n, err_v, err_p = _run_sod_three_pin(None, clear_first=False)
    assert t == PIN_T, f"空表 t={t!r}，钉 {PIN_T!r}"
    assert err_n == PIN_L1_N, f"L1(n)={err_n!r}，钉 {PIN_L1_N!r}"
    assert err_v == PIN_L1_V, f"L1(v)={err_v!r}，钉 {PIN_L1_V!r}"
    assert err_p == PIN_L1_P, f"L1(P)={err_p!r}，钉 {PIN_L1_P!r}"


def test_no_pin_sod_regression_after_explicit_clear():
    """卡片 §交什么 4③（情形二）：显式 set_pin(None,None) 清除后三钉逐位。

    清除后应与从未加载钉的结果逐位一致。
    """
    t, err_n, err_v, err_p = _run_sod_three_pin(None, clear_first=True)
    assert t == PIN_T, f"清除后空表 t={t!r}，钉 {PIN_T!r}"
    assert err_n == PIN_L1_N, f"清除后 L1(n)={err_n!r}，钉 {PIN_L1_N!r}"
    assert err_v == PIN_L1_V, f"清除后 L1(v)={err_v!r}，钉 {PIN_L1_V!r}"
    assert err_p == PIN_L1_P, f"清除后 L1(P)={err_p!r}，钉 {PIN_L1_P!r}"


# ============================================================================
# 4. 鬼格掩膜测（fill_bc 后再钉分支）
# ============================================================================


def test_ghost_cell_pin_after_fill_bc(lib):
    """卡片 §交什么 4④：掩膜含 x1beg 端一列鬼格，fill_bc 后鬼格 == 钉值。

    钉值与外边界自然值不同（-7 这种一眼可辨），确保测出的是「钉上去的」
    而非「边界本来就这样」。这是唯一打到「填皮后再钉」分支的测试。
    """
    run = _make_2d_run()
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        # 掩膜：x1beg 端鬼格列（i=0..ng-1），j 范围取中间几行
        mask = np.zeros(shape, dtype=np.int32)
        mask[:, _NG + 1 : _NG + 5, :_NG] = 1
        assert np.sum(mask) > 0

        # 钉值：用一眼可辨的值（-7 系列），与自然边界 copy/reflect 不同
        PIN_VAL_N = -7.0
        PIN_VAL_VX = -7.5
        PIN_VAL_VY = -7.25
        vals = np.zeros((1, 5) + shape, dtype=np.float64)
        vals[0, SLOT_N][mask == 1] = PIN_VAL_N
        vals[0, SLOT_VX][mask == 1] = PIN_VAL_VX
        vals[0, SLOT_VY][mask == 1] = PIN_VAL_VY
        vals[0, SLOT_VZ][mask == 1] = 0.0
        vals[0, SLOT_P][mask == 1] = -7.75  # P 也用可辨值

        run.set_pin(mask, vals)

        # fill_bc 应先填皮（copy/reflect）再钉
        assert lib.xmf_fill_bc(h) == 0

        # 鬼格 == 钉值
        n_got = _unpack(lib, h, SLOT_N, ncells).reshape(shape)
        vx_got = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        vy_got = _unpack(lib, h, SLOT_VY, ncells).reshape(shape)
        p_got = _unpack(lib, h, SLOT_P, ncells).reshape(shape)

        assert np.all(n_got[mask == 1] == PIN_VAL_N), (
            f"鬼格 n 应 == {PIN_VAL_N}，实测唯一值 {np.unique(n_got[mask == 1])}"
        )
        assert np.all(vx_got[mask == 1] == PIN_VAL_VX), (
            f"鬼格 vx 应 == {PIN_VAL_VX}，实测唯一值 {np.unique(vx_got[mask == 1])}"
        )
        assert np.all(vy_got[mask == 1] == PIN_VAL_VY), (
            f"鬼格 vy 应 == {PIN_VAL_VY}，实测唯一值 {np.unique(vy_got[mask == 1])}"
        )
        assert np.all(p_got[mask == 1] == -7.75), (
            f"鬼格 P 应 == -7.75，实测唯一值 {np.unique(p_got[mask == 1])}"
        )
    finally:
        run.close()


# ============================================================================
# 5. 不只测 VX1（已覆盖在第 1/2 条中，此处加一条显式断言第二方向）
# ============================================================================


def test_pin_covers_vy_and_second_direction(lib):
    """卡片 §交什么 4⑤：至少碰到 vy 或第二方向。

    掩膜覆盖 x2 方向的一行（j 固定），钉 vy=0.5，步进后掩膜格 vy == 0.5。
    """
    run = _make_2d_run()
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        # 掩膜：x2 方向中间一行（j=ng+3），整条 i 轴内区
        mask = np.zeros(shape, dtype=np.int32)
        j_pin = _NG + 3
        mask[_NG, j_pin, _NG : _NG + NX1_2D] = 1
        assert np.sum(mask) > 0

        vals = _build_pin_vals(lib, h, ncells, shape)
        PIN_VY = 0.5
        vals[0, SLOT_VY][mask == 1] = PIN_VY
        vals[0, SLOT_VX][mask == 1] = 0.0
        vals[0, SLOT_VZ][mask == 1] = 0.0

        run.set_pin(mask, vals)
        for _ in range(3):
            run.step()

        vy = _unpack(lib, h, SLOT_VY, ncells).reshape(shape)
        assert np.all(vy[mask == 1] == PIN_VY), (
            f"掩膜格 vy 应 == {PIN_VY}，实测唯一值 {np.unique(vy[mask == 1])}"
        )
    finally:
        run.close()


# ============================================================================
# 6. 同一 run 先后两口：set_pin 与 user 皮各自生效互不清除
# ============================================================================


def test_pin_and_user_coexist_on_same_run(lib):
    """卡片 §交什么 4⑥：set_pin 与 user 皮（boundary_floor）同一 run 共存。

    配置给 x1beg user 面 + floor；setup 后加载冻格掩膜（内区中央一块）。
    两者各自生效：user 面的 ghost 被 floor 赋值，掩膜格被钉住。
    """
    nt = _NZTOT2 * _NYTOT2
    floor_val_n = 5.5
    floor_val_vx = -1.5
    floor = [
        [
            [floor_val_n] * nt,
            [floor_val_vx] * nt,
            [0.0] * nt,
            [0.0] * nt,
            [3.0] * nt,
        ]
    ]
    cfg = xmf.shock_tube()
    cfg["grid"]["ndim"] = 2
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["boundary"] = {"x1beg": "user", "x1end": "copy", "x2": "copy"}
    cfg["boundary_floor"] = floor
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        # 加载冻格掩膜：内区中央一块
        mask = np.zeros(shape, dtype=np.int32)
        mask[_NG, _NG + 2 : _NG + 5, _NG + 4 : _NG + 8] = 1
        PIN_VY = 2.0
        vals = _build_pin_vals(lib, h, ncells, shape)
        vals[0, SLOT_VY][mask == 1] = PIN_VY
        vals[0, SLOT_VX][mask == 1] = 0.0
        vals[0, SLOT_VZ][mask == 1] = 0.0
        run.set_pin(mask, vals)

        # fill_bc 应同时跑 floor 和 pin
        assert lib.xmf_fill_bc(h) == 0

        # user 面（x1beg）的 ghost 被 floor 赋值
        n_got = _unpack(lib, h, SLOT_N, ncells).reshape(shape)
        vx_got = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        k0 = _NG
        jin = slice(_NG, _NG + NX2_2D)
        for m in range(_NG):
            np.testing.assert_array_equal(
                n_got[k0, jin, m], floor_val_n,
                err_msg=f"x1beg floor ghost n m={m} 应 == {floor_val_n}",
            )
            np.testing.assert_array_equal(
                vx_got[k0, jin, m], floor_val_vx,
                err_msg=f"x1beg floor ghost vx m={m} 应 == {floor_val_vx}",
            )

        # 掩膜格被钉住（vy == PIN_VY）
        vy_got = _unpack(lib, h, SLOT_VY, ncells).reshape(shape)
        assert np.all(vy_got[mask == 1] == PIN_VY), (
            f"掩膜格 vy 应 == {PIN_VY}（与 floor 共存）"
        )
    finally:
        run.close()


# ============================================================================
# 7. 清除语义：清除后原掩膜格恢复自由演化
# ============================================================================


def test_clear_pin_restores_free_evolution(lib):
    """卡片 §交什么 4⑦：显式 set_pin(None, None) 后掩膜格恢复自由演化。

    步骤：加载钉 → 步进确认钉住 → 清除 → 步进 → 原掩膜格值改变。
    """
    run = _make_2d_run()
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        # 掩膜 + 钉 v=0
        mask = np.zeros(shape, dtype=np.int32)
        mask[_NG, _NG + 2 : _NG + 5, _NG + 4 : _NG + 8] = 1
        vals = _build_pin_vals(lib, h, ncells, shape)
        vals[0, SLOT_VX][mask == 1] = 0.0
        vals[0, SLOT_VY][mask == 1] = 0.0
        vals[0, SLOT_VZ][mask == 1] = 0.0
        run.set_pin(mask, vals)

        # 步进确认钉住
        for _ in range(3):
            run.step()
        vx_pinned = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        assert np.all(vx_pinned[mask == 1] == 0.0), "前置：钉住应生效"

        # 清除（None 方式）
        run.set_pin(None, None)

        # 步进：原掩膜格应可演化
        run.step()
        vx_after = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        # 清除后掩膜格不再被钉住：值应与 0 不同（场在演化）
        assert not np.all(vx_after[mask == 1] == 0.0), (
            "清除后掩膜格应恢复自由演化（vx 不再全为 0）"
        )
    finally:
        run.close()


def test_clear_pin_via_all_zero_mask(lib):
    """卡片 §交什么 4⑦（补充）：全 0 掩膜也清除钉。"""
    run = _make_2d_run()
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        # 加载钉
        mask = np.zeros(shape, dtype=np.int32)
        mask[_NG, _NG + 2 : _NG + 5, _NG + 4 : _NG + 8] = 1
        vals = _build_pin_vals(lib, h, ncells, shape)
        vals[0, SLOT_VX][mask == 1] = 0.0
        vals[0, SLOT_VY][mask == 1] = 0.0
        vals[0, SLOT_VZ][mask == 1] = 0.0
        run.set_pin(mask, vals)

        # 步进确认钉住
        for _ in range(2):
            run.step()
        vx_pinned = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        assert np.all(vx_pinned[mask == 1] == 0.0), "前置：钉住应生效"

        # 全 0 掩膜清除
        zero_mask = np.zeros(shape, dtype=np.int32)
        run.set_pin(zero_mask, vals)  # 全 0 掩膜 = 清除

        # 步进：原掩膜格应可演化
        run.step()
        vx_after = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        assert not np.all(vx_after[mask == 1] == 0.0), (
            "全 0 掩膜清除后原掩膜格应恢复自由演化"
        )
    finally:
        run.close()


# ============================================================================
# 8. 坏输入不破坏上一拍：错误尺寸抛异常，已生效的钉仍然生效
# ============================================================================


def test_bad_input_preserves_previous_pin(lib):
    """卡片 §交什么 4⑧：错误尺寸的 mask/vals 抛异常，上一拍钉仍生效。

    卡片规定 ValueError（或 RuntimeError），且抛异常前已生效的钉仍然生效。
    """
    run = _make_2d_run()
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        # 先加载一个合法的钉
        mask = np.zeros(shape, dtype=np.int32)
        mask[_NG, _NG + 2 : _NG + 5, _NG + 4 : _NG + 8] = 1
        vals = _build_pin_vals(lib, h, ncells, shape)
        vals[0, SLOT_VX][mask == 1] = 0.0
        vals[0, SLOT_VY][mask == 1] = 0.0
        vals[0, SLOT_VZ][mask == 1] = 0.0
        run.set_pin(mask, vals)

        # 确认钉生效
        assert lib.xmf_fill_bc(h) == 0
        vx_before = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        assert np.all(vx_before[mask == 1] == 0.0), "前置：合法钉应生效"

        # 尝试加载错误尺寸的 mask/vals
        bad_mask = np.zeros((3,), dtype=np.int32)
        bad_vals = np.zeros((1, 5, 3), dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            run.set_pin(bad_mask, bad_vals)

        # 上一拍钉仍然生效
        assert lib.xmf_fill_bc(h) == 0
        vx_after = _unpack(lib, h, SLOT_VX, ncells).reshape(shape)
        assert np.all(vx_after[mask == 1] == 0.0), (
            "坏输入抛异常后上一拍钉应仍然生效"
        )
    finally:
        run.close()


# ============================================================================
# 补充：四步冒烟（setup → step → fields → finalize 仍通）
# ============================================================================


def test_four_step_smoke_no_pin():
    """无钉时四步 API 仍通：step 后 fields 有限、密度为正。"""
    run = _make_2d_run()
    try:
        for _ in range(3):
            run.step()
        assert run.t > 0.0
        hi = run.fields["HI"]
        assert np.all(np.isfinite(hi.n)) and np.all(hi.n > 0.0)
        assert np.all(np.isfinite(hi.v)) and np.all(np.isfinite(hi.p))
    finally:
        run.close()


def test_four_step_smoke_with_pin():
    """有钉时四步 API 仍通：step 后 fields 有限、密度为正。"""
    run = _make_2d_run()
    try:
        lib = load()
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        shape = (_NZTOT2, _NYTOT2, _NXTOT2)

        mask = np.zeros(shape, dtype=np.int32)
        mask[_NG, _NG + 2 : _NG + 5, _NG + 4 : _NG + 8] = 1
        vals = _build_pin_vals(lib, h, ncells, shape)
        vals[0, SLOT_VX][mask == 1] = 0.0
        vals[0, SLOT_VY][mask == 1] = 0.0
        vals[0, SLOT_VZ][mask == 1] = 0.0
        run.set_pin(mask, vals)

        for _ in range(3):
            run.step()
        assert run.t > 0.0
        hi = run.fields["HI"]
        assert np.all(np.isfinite(hi.n)) and np.all(hi.n > 0.0)
    finally:
        run.close()
