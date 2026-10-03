"""B1 准中性电子压强闭合合同单测（test_pe）：P_eff 三向进 hydro。

合同来自本刀卡片（docs/SESSION.md B1）与 docs/cmd/PROMPT_B1_PE.md：
  1. kind 语义：'hd' 遗留标签按 charge 改写（q=0→neutral、q≠0→ion）；
     显式角色与 charge 不一致、未知角色、'mhd' 一律在 normalize 拒掉
     （mhd 的报错必须含 "hd" 子串）。核认离子靠 q_s != 0，不认名字。
  2. 电荷走辅助加载口 xmf_charge_load：NULL / NaN 拒（-1），合法 0。
     不进扁平 XmfConfig。
  3. 一步逐位等价钉（P_eff 真的进了 Riemann）：开电子跑 A（离子+中性，
     P_s 与 P_e 分槽）对关电子跑 B（同两支流体，P 槽打包 P_s+P_e）。
     单离子时 w = (q rho/m)/(q rho/m) = 1.0 逐位，P_eff = P_s + P_e 与
     numpy 的 P_s+pe 同一加法逐位相同；CFL 又只读储存的 rho、P → dt
     逐位相同 → 一步之后离子 n 与各方向速度逐位相同，中性支五槽逐位
     相同（w=0 不吃 P_e），而两支的 P 槽必然不同（A 只留 P_s 增量）。
     模式只沿被测方向变化、v=0 → 其余方向的 sweep 是逐位 no-op，等价
     关系对三个方向各自成立（1D=x1、薄 2D=x2、薄 3D=x3 各一例）。
  4. n_e <= 0 的格：w=0、u_e=0、P_eff 退回 P_s；离子密度全 0 时两步
     与关电子单流体逐位相同，且 P_e 块逐位不动（电子伪扫只给地板态）。
  5. CFL 离子声速读 P_eff：均匀离子+均匀 P_e 的 dt == cfl/(s/dx)，
     s = sqrt(gamma*(P_s + w P_e)/rho)（w=1），相对差 <= 1e-12。
  6. IR：开电子放行派生读 LOAD n_e / T_e / P_e（操作码 0..9 不动、牛顿
     未知量仍 5*nfluid）；apply 值与解析的 n_e = sum q_s rho_s/mass_code、
     T_e = P_e*unit_P/(n_e kB)（1e-30 地板）对 1e-12：a) 单离子均匀态
     钉值 + jac；b) 双离子（q=1/q=2、质量不同）+ 中性支、非均匀剖面的
     逐格钉——Σ 是真和、逐格是真逐格，中性支不进 n_e；jac 的 d/drho
     列每支离子各一列、中性列为 0；LOAD P_e 读储存块原值，且 P_e 不进
     牛顿向量（该带子 jac 全 0）；关电子时 n_e / T_e / P_e 三名编译即拒
     （错误含 "electron"）。
  7. fields_full / fields：开电子给 P_e（活动区、同压强口径）；关电子
     两支读口都不出现该键。

口径（同 test_fields_shape / test_ir_ad）：unpack/write_pe 块长 =
xmf_state_ncells 含 ghost，reshape (nztot, nytot, nxtot)，(k,j,i) i 最快；
fields_full 只吐内区 (nz,ny,nx)。默认单位下 unit_P = 1e-12*1e6² = 1.0。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；不只测 VX1——
n / vx / vy / vz / P 五槽与 P_e 都钉；等价钉对三个方向分别给例。
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf import config as cfgmod
from xmf._native import load
from xmf.config import AMU, KB

_PTR = ctypes.POINTER(ctypes.c_double)

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

# 等价钉初值（码单位，O(1)）：左右台阶 + 中性支均匀冷背景。
# P_e 与 P_s 同量级，保证两支 P 槽的分裂肉眼可见。
RHO_L, RHO_R = 1.0, 0.3
PS_L, PS_R = 1.0, 0.2
PE_L, PE_R = 0.6, 0.1
RHO_N, PN = 0.5, 0.1  # 中性支均匀态：w=0，应保持逐位等价于关电子跑
MASS_ION, MASS_NEUTRAL = 1.00794, 4.0026
MASS_HEII = 4.0026  # 逐格钉里的第二离子：与 HII 不同质量、不同电荷

GAMMA = 5.0 / 3.0


@pytest.fixture(scope="module")
def lib():
    return load()


# --- 打包小助手（同 test_fields_shape 的口径） ---------------------------------


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _pack_block(lib, run, f: int, var: int, block: np.ndarray) -> None:
    ncells = lib.xmf_state_ncells(run._h)
    data = np.ascontiguousarray(block, dtype=np.float64).ravel()
    assert data.size == ncells
    assert lib.xmf_state_pack(run._h, f, var, data.ctypes.data_as(_PTR)) == ncells


def _write_pe(lib, run, block: np.ndarray) -> None:
    ncells = lib.xmf_state_ncells(run._h)
    data = np.ascontiguousarray(block, dtype=np.float64).ravel()
    assert data.size == ncells
    assert lib.xmf_state_write_pe(run._h, data.ctypes.data_as(_PTR)) == ncells


def _unpack_block(lib, run, f: int, var: int) -> np.ndarray:
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, f, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _copy_pe(lib, run) -> np.ndarray:
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_copy_pe(run._h, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _step_along(axis: int, shape: np.ndarray, left: float, right: float) -> np.ndarray:
    """沿 axis（0=i、1=j、2=k，对 shape (nztot,nytot,nxtot)）取左/右台阶值，
    分界在块中线。只用于构造初值。"""
    blk = np.empty(shape, dtype=np.float64)
    n = shape[2 - axis]  # axis 0->nxtot(idx2), 1->nytot(idx1), 2->nztot(idx0)
    for idx in range(n):
        val = left if idx < n // 2 else right
        if axis == 0:
            blk[:, :, idx] = val
        elif axis == 1:
            blk[:, idx, :] = val
        else:
            blk[idx, :, :] = val
    return blk


def _pair_cfg(n, ndim, axis):
    """等价钉的 A / B 公共配置：两支流体（离子 + 中性）、pcm/hll/rk1、
    六面 reflect。A 开电子（normalize 自动推电荷表），B 关电子。"""

    def base(electron_mode):
        c = xmf.shock_tube()
        c["grid"]["n"] = list(n)
        c["grid"]["ndim"] = ndim
        c["fluids"] = [
            {"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"},
            {"name": "HI", "mass": MASS_NEUTRAL, "charge": 0, "kind": "neutral"},
        ]
        c["electron"] = {"mode": electron_mode}
        c["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
        c["ic"] = {"kind": "none"}
        return c

    return base("pressure-only"), base("off")


def _init_pair(lib, run_a, run_b, axis):
    """同一初值打进 A / B：HII 带台阶 (rho, P_s)，A 的 P_e 单独一块；
    B 的 P 槽直接打包 numpy 的 P_s + pe（与核内 q + w*pe 同一加法）。
    HI 均匀冷背景。返回打包用的 pe 全块。"""
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run_a)
    shape = (nzt, nyt, nxt)
    assert _geom(run_b) == (ng, nx, ny, nz, nxt, nyt, nzt)

    rho = _step_along(axis, shape, RHO_L, RHO_R)
    p_s = _step_along(axis, shape, PS_L, PS_R)
    pe = _step_along(axis, shape, PE_L, PE_R)

    for run in (run_a, run_b):
        for var, blk in ((SLOT_N, rho), (SLOT_P, p_s)):
            _pack_block(lib, run, 0, var, blk)
        # 横/纵向速度保持 0（calloc 初值），无需打包
    # B 的 HII 压强槽 = P_s + P_e（numpy 逐元素一次加法，与核内逐位同）
    _pack_block(lib, run_b, 0, SLOT_P, p_s + pe)
    # A 的 P_e 块（ghost 一起）；B 无 pe 块
    _write_pe(lib, run_a, pe)

    for run in (run_a, run_b):
        for var, val in ((SLOT_N, RHO_N), (SLOT_P, PN)):
            _pack_block(lib, run, 1, var, np.full(shape, val))
    return pe


# --- 1. kind 语义（config 层） ------------------------------------------------


def test_kind_legacy_hd_rewritten_by_charge():
    c = xmf.shock_tube()
    c["fluids"] = [
        {"name": "HI", "mass": 1.00794, "charge": 0, "kind": "hd"},
        {"name": "HII", "mass": 1.00794, "charge": 1, "kind": "hd"},
    ]
    p = cfgmod.normalize(c)
    assert p.fluids[0].kind == "neutral", "q=0 的遗留 hd 应改写为 neutral"
    assert p.fluids[1].kind == "ion", "q!=0 的遗留 hd 应改写为 ion"


def test_kind_role_charge_mismatch_refused():
    for kind, charge, frag in (
        ("ion", 0.0, "non-zero charge"),
        ("neutral", 1.0, "charge 0"),
    ):
        c = xmf.shock_tube()
        c["fluids"] = [{"name": "X", "mass": 1.0, "charge": charge, "kind": kind}]
        with pytest.raises(ValueError, match=frag):
            cfgmod.normalize(c)


def test_kind_unknown_and_mhd_refused():
    c = xmf.shock_tube()
    c["fluids"] = [{"name": "X", "mass": 1.0, "charge": 0, "kind": "banana"}]
    with pytest.raises(ValueError, match="species role"):
        cfgmod.normalize(c)
    c["fluids"] = [{"name": "X", "mass": 1.0, "charge": 0, "kind": "mhd"}]
    with pytest.raises(ValueError, match="hd"):
        cfgmod.normalize(c)


# --- 2. 电荷辅助加载口（C 层） -------------------------------------------------


def test_charge_load_port_validates(lib):
    run = xmf.setup(xmf.shock_tube())
    try:
        h = run._h
        assert lib.xmf_charge_load(h, None) == -1, "NULL 表必须拒"
        bad = np.array([float("nan")], dtype=np.float64)
        assert lib.xmf_charge_load(h, bad.ctypes.data_as(_PTR)) == -1, "NaN 必须拒"
        ok = np.array([0.0], dtype=np.float64)
        assert lib.xmf_charge_load(h, ok.ctypes.data_as(_PTR)) == 0
        assert lib.xmf_charge_load(None, ok.ctypes.data_as(_PTR)) == -1
    finally:
        run.close()


# --- 3. 一步逐位等价钉：P_eff 三向进 Riemann -----------------------------------


def _assert_equivalence_one_step(lib, n, ndim, axis):
    cfg_a, cfg_b = _pair_cfg(n, ndim, axis)
    run_a = xmf.setup(cfg_a)
    run_b = xmf.setup(cfg_b)
    try:
        pe0 = _init_pair(lib, run_a, run_b, axis)

        run_a.step()
        run_b.step()
        # 本步实际使用的 dt 逐位相同（CFL 读的是同一份 rho 与
        # P_eff = P_s + P_e），时间因此也逐位相同
        assert run_a.dt == run_b.dt, (
            f"dt 应逐位相同：A={run_a.dt!r} B={run_b.dt!r}"
        )
        assert run_a.t == run_b.t

        # 离子支：n 与三个速度槽逐位相同（pass A 与 B 的整次扫同一算子）
        for var in (SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ):
            a = _unpack_block(lib, run_a, 0, var)
            b = _unpack_block(lib, run_b, 0, var)
            assert np.array_equal(a, b), (
                f"HII 槽 {var} 一步后应逐位相同（含 ghost）；"
                f"max|diff|={float(np.max(np.abs(a - b))):.3e}"
            )
        # 两支的 P 槽必须分裂：A 只留 P_s 增量，B 是总压演化
        pa = _unpack_block(lib, run_a, 0, SLOT_P)
        pb = _unpack_block(lib, run_b, 0, SLOT_P)
        assert np.any(pa != pb), "A 的 P_s 与 B 的总压演化应已分裂"
        # 中性支五槽逐位相同（w=0，不吃 P_e）
        for var in range(5):
            a = _unpack_block(lib, run_a, 1, var)
            b = _unpack_block(lib, run_b, 1, var)
            assert np.array_equal(a, b), f"HI 槽 {var} 应逐位相同（w=0）"
        # P_e 确实被推进了（闭合门真的开了）
        pe1 = _copy_pe(lib, run_a)
        assert np.any(pe1 != pe0.ravel()), "一步后 P_e 应变化（伪流体扫的压力增量）"
        # 读口：A 有 P_e 键，B 没有
        fa = run_a.fields_full()
        fb = run_b.fields_full()
        assert "P_e" in fa and "P_e" not in fb
        assert fa["P_e"].shape == (run_a._cfg.n[2], run_a._cfg.n[1], run_a._cfg.n[0])
        assert "P_e" in run_a.fields and "P_e" not in run_b.fields
    finally:
        run_a.close()
        run_b.close()


def test_pe_equivalence_1d_x1(lib):
    """1D：模式沿 x1，唯一活动方向就是被测方向。"""
    _assert_equivalence_one_step(lib, (16, 1, 1), 1, axis=0)


def test_pe_equivalence_2d_x2(lib):
    """薄 2D：模式只沿 x2；x1 方向初值均匀且 v=0，x1 扫是逐位 no-op，
    等价关系因此落在 x2 扫上。"""
    _assert_equivalence_one_step(lib, (8, 4, 1), 2, axis=1)


def test_pe_equivalence_3d_x3(lib):
    """薄 3D：模式只沿 x3；x1/x2 方向均匀且 v=0，等价落在 x3 扫上。"""
    _assert_equivalence_one_step(lib, (8, 4, 2), 3, axis=2)


# --- 4. n_e <= 0：w=0、u_e=0、P_eff 退回 P_s、P_e 冻结 --------------------------


def test_pe_zero_ion_cells_fall_back_to_p(lib):
    """离子密度全 0（n_e=0）：两步之后与关电子单流体逐位相同，P_e 块
    逐位不动（伪扫只给地板态、地板格永不回写）。"""
    c_a = xmf.shock_tube()
    c_a["grid"]["n"] = [16, 1, 1]
    c_a["fluids"] = [{"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"}]
    c_a["electron"] = {"mode": "pressure-only"}
    c_a["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    c_a["ic"] = {"kind": "none"}
    import copy as _copy

    c_b = _copy.deepcopy(c_a)
    c_b["electron"] = {"mode": "off"}

    run_a = xmf.setup(c_a)
    run_b = xmf.setup(c_b)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run_a)
        shape = (nzt, nyt, nxt)
        rho = _step_along(0, shape, RHO_L, RHO_R) * 0.0  # 全 0：n_e == 0
        p_s = _step_along(0, shape, PS_L, PS_R)
        pe = _step_along(0, shape, PE_L, PE_R)
        for run in (run_a, run_b):
            _pack_block(lib, run, 0, SLOT_N, rho)
            _pack_block(lib, run, 0, SLOT_P, p_s)
        _write_pe(lib, run_a, pe)

        for _ in range(2):
            run_a.step()
            run_b.step()
        assert run_a.dt == run_b.dt and run_a.t == run_b.t
        for var in range(5):
            a = _unpack_block(lib, run_a, 0, var)
            b = _unpack_block(lib, run_b, 0, var)
            assert np.array_equal(a, b), (
                f"n_e=0 时槽 {var} 应与关电子逐位相同（P_eff 退回 P_s）"
            )
        assert np.array_equal(_copy_pe(lib, run_a), pe.ravel()), (
            "n_e=0 的格 P_e 必须冻结：伪扫地板格不回写"
        )
    finally:
        run_a.close()
        run_b.close()


# --- 5. CFL 离子声速读 P_eff ---------------------------------------------------


def test_cfl_reads_peff(lib):
    """均匀离子 + 均匀 P_e（单离子 w=1）：dt == cfl / (s/dx)，
    s = sqrt(gamma*(P_s + P_e)/rho)，相对差 <= 1e-12。"""
    c = xmf.shock_tube()
    c["grid"]["n"] = [16, 1, 1]
    c["fluids"] = [{"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"}]
    c["electron"] = {"mode": "pressure-only"}
    c["ic"] = {"kind": "none"}
    run = xmf.setup(c)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        rho_c, ps_c, pe_c = 1.4, 0.9, 0.7
        _pack_block(lib, run, 0, SLOT_N, np.full(shape, rho_c))
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, ps_c))
        _write_pe(lib, run, np.full(shape, pe_c))
        # 重新填充 ghost 并取 dt：直接一步前的 dt 即 compute_dt 的结果
        dx_buf = np.empty(nxt, dtype=np.float64)
        assert lib.xmf_mesh_copy_dx(run._h, 0, dx_buf.ctypes.data_as(_PTR)) == nxt
        dx = dx_buf[ng]
        # 一步：步首 compute_dt 用打包后的态重算 dt（均匀态，扫本身 no-op）
        run.step()
        peff = ps_c + 1.0 * pe_c  # 与核内 fp + w*pe（w=1）同一加法
        s = math.sqrt(GAMMA * peff / rho_c)
        expect = run._cfg.numerics.cfl / (s / dx)
        assert run.dt > 0.0
        assert abs(run.dt - expect) <= 1.0e-12 * expect, (
            f"dt={run.dt!r} 应为 cfl/(s/dx)={expect!r}（CFL 读 P_eff）"
        )
    finally:
        run.close()


def test_cfl_reads_peff_2d(lib):
    """薄 2D + 开电子 + P_eff（C2补5 卡片 §交什么 4 的正交缺口）：
    均匀单离子（w=1）+ 均匀 P_e，nx2>1 使 x2 进求和。
    step() 后 run.dt == cfl/(s1/dx1 + s2/dx2)，
    s1 = |vx| + sqrt(gamma*(P_s+P_e)/rho)，s2 = sqrt(gamma*(P_s+P_e)/rho)
    （vy=0），相对差 <= 1e-12。x2 项必须真的进了求和（dt < 0.5 * dt_x1_only）。

    本测同时打到：设备归约（step 内 compute_dt 走 parallel_reduce）+
    P_eff 公式 + 第二方向进 CFL 求和。
    """
    c = xmf.shock_tube()
    c["grid"]["n"] = [8, 4, 1]
    c["grid"]["ndim"] = 2
    c["grid"]["x2min"] = 0.0
    c["grid"]["x2max"] = 2.0
    c["fluids"] = [{"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"}]
    c["electron"] = {"mode": "pressure-only"}
    c["ic"] = {"kind": "none"}
    run = xmf.setup(c)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        rho_c, ps_c, pe_c, vx_c = 1.4, 0.9, 0.7, 0.3
        _pack_block(lib, run, 0, SLOT_N, np.full(shape, rho_c))
        _pack_block(lib, run, 0, SLOT_VX, np.full(shape, vx_c))
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, ps_c))
        _write_pe(lib, run, np.full(shape, pe_c))

        # 读两方向的实际格宽（与核内 mesh[d].dx 同一份）
        dx_buf0 = np.empty(nxt, dtype=np.float64)
        dx_buf1 = np.empty(nyt, dtype=np.float64)
        assert (
            lib.xmf_mesh_copy_dx(run._h, 0, dx_buf0.ctypes.data_as(_PTR)) == nxt
        )
        assert (
            lib.xmf_mesh_copy_dx(run._h, 1, dx_buf1.ctypes.data_as(_PTR)) == nyt
        )
        dx1 = dx_buf0[ng]
        dx2 = dx_buf1[ng]

        # 一步：步首 compute_dt 用打包后的态重算 dt（均匀态，扫本身 no-op）
        run.step()

        peff = ps_c + 1.0 * pe_c  # 与核内 fp + w*pe（w=1）同一加法
        cs = math.sqrt(GAMMA * peff / rho_c)
        s1 = abs(vx_c) + cs
        s2 = cs  # vy = 0
        expect = run._cfg.numerics.cfl / (s1 / dx1 + s2 / dx2)
        assert run.dt > 0.0
        assert abs(run.dt - expect) <= 1.0e-12 * expect, (
            f"2D dt={run.dt!r} 应为 cfl/(s1/dx1+s2/dx2)={expect!r}"
            f"（CFL 读 P_eff，x2 进求和）"
        )
        # 判别性：x2 项确实压低了 dt
        dt_x1_only = run._cfg.numerics.cfl / (s1 / dx1)
        assert run.dt < 0.5 * dt_x1_only, (
            "x2 方向应真实进入 CFL 求和（n_d>1 的方向不得漏）"
        )
    finally:
        run.close()


# --- 6. IR 派生读 n_e / T_e ----------------------------------------------------


def _ir_cfg(electron_mode, sources):
    c = xmf.shock_tube()
    c["grid"]["n"] = [16, 1, 1]
    c["fluids"] = [
        {"name": "HI", "mass": MASS_NEUTRAL, "charge": 0, "kind": "neutral"},
        {"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"},
    ]
    c["electron"] = {"mode": electron_mode}
    c["sources"] = sources
    # 辐射表（enabled=false）：只用来触发 phys 口（真实质量 + 单位），
    # 扫描不跑、FLUX 读 0，对 apply/jac 无副作用。
    c["radiation"] = {
        "enabled": False,
        "absorbers": ["HI"],
        "bins": {"n": 1, "sigma": [1.0e-18], "hnu": [2.0e-11], "flux_inf": [0.0]},
    }
    c["ic"] = {"kind": "none"}
    return c


def test_ir_load_ne_te_values_and_jacobian(lib):
    """开电子：LOAD n_e / T_e 放行；apply 值与解析式 1e-12；jac 的
    d/drho_HII 列同式。B2：开门时牛顿未知量 N = 5*nfluid + 1（P_e
    殿后）；n_e 不带 P_e 列，T_e 带 d/dP_e = unit_P/(n_e kB)。"""
    sources = [
        {"name": "ne_read", "expr": {"dP_HI": "n_e"}},
        {"name": "te_read", "expr": {"dP_HII": "T_e"}},
    ]
    run = xmf.setup(_ir_cfg("pressure-only", sources))
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        assert run.num_states == 11, (
            "B2 开门：N = 5*nfluid + 1（P_e 殿后）；n_e/T_e 仍不入未知量"
        )

        rho_n, rho_i = 0.8, 1.2
        pe_c = 0.35
        mc_n = MASS_NEUTRAL * AMU / run._cfg.units.density
        mc_i = MASS_ION * AMU / run._cfg.units.density
        _pack_block(lib, run, 0, SLOT_N, np.full(shape, rho_n))
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, 0.5))
        _pack_block(lib, run, 1, SLOT_N, np.full(shape, rho_i))
        _pack_block(lib, run, 1, SLOT_P, np.full(shape, 0.6))
        _write_pe(lib, run, np.full(shape, pe_c))

        # 解析值：n_e = sum q_s rho_s / mass_code（中性支 q=0 不进）；
        # T_e = P_e*unit_P/(n_e kB)（n_e 地板 1e-30，同流体 T 闭合）。
        ne = 1.0 * rho_i / mc_i
        unit_p = run._cfg.units.density * run._cfg.units.velocity**2
        te = pe_c * unit_p / (ne * KB)

        p_hi0 = _unpack_block(lib, run, 0, SLOT_P).copy()
        p_hii0 = _unpack_block(lib, run, 1, SLOT_P).copy()
        run.apply_sources()
        d_hi = (_unpack_block(lib, run, 0, SLOT_P) - p_hi0).reshape(shape)
        d_hii = (_unpack_block(lib, run, 1, SLOT_P) - p_hii0).reshape(shape)
        # xmf_ir_apply 的锁定约定：只写内区、ghost 永不求值（同
        # test_ir_eval 的 pin）。解析值只对内区比，ghost 必须逐位不动。
        d_hi_int = d_hi[ng : ng + nz, ng : ng + ny, ng : ng + nx]
        d_hii_int = d_hii[ng : ng + nz, ng : ng + ny, ng : ng + nx]
        assert float(np.max(np.abs(d_hi_int - ne))) <= 1.0e-12 * ne, (
            "apply 的 n_e 增量应对 sum q_s rho_s/mass_code 1e-12"
        )
        assert float(np.max(np.abs(d_hii_int - te))) <= 1.0e-12 * te, (
            "apply 的 T_e 增量应对 P_e*unit_P/(n_e kB) 1e-12"
        )
        ghost = np.ones(shape, dtype=bool)
        ghost[ng : ng + nz, ng : ng + ny, ng : ng + nx] = False
        assert np.all(d_hi[ghost] == 0.0), "apply 不得触碰 ghost 区"
        assert np.all(d_hii[ghost] == 0.0), "apply 不得触碰 ghost 区"

        # jac：行 4 = HI 的 P（n_e 带），行 9 = HII 的 P（T_e 带）；
        # 列 0 = rho_HI（中性，两式导数均 0），列 5 = rho_HII，
        # 列 10 = B2 开门新增的 P_e 未知量（最后一列）。
        J = run.jacobian(ng, ng, ng + nx // 2)
        assert J.shape == (11, 11)
        assert abs(J[4, 5] - 1.0 / mc_i) <= 1.0e-12 * abs(1.0 / mc_i), (
            "d(n_e)/d(rho_HII) 应为 q/mass_code"
        )
        assert J[4, 0] == 0.0, "中性支不进 n_e"
        assert J[4, 10] == 0.0, "n_e = Σ q_s n_s 不含 P_e"
        d_te_drho = -pe_c * unit_p / (ne * ne * KB) * (1.0 / mc_i)
        assert abs(J[9, 5] - d_te_drho) <= 1.0e-12 * abs(d_te_drho), (
            "d(T_e)/d(rho_HII) 应为 -P_e unit_P/(n_e^2 kB) * q/mass_code"
        )
        assert J[9, 0] == 0.0
        d_te_dpe = unit_p / (ne * KB)
        assert abs(J[9, 10] - d_te_dpe) <= 1.0e-12 * abs(d_te_dpe), (
            "B2：T_e 带子在 P_e 列应为 unit_P/(n_e kB)"
        )
    finally:
        run.close()


def _ir_cfg_multi(sources):
    """双离子（q=1 质量 1.00794 / q=2 质量 4.0026）+ 一中性的逐格钉配置。
    空跑束表只为触发 phys 口，让 mass_code 是真实质量——Σ 里的
    1/mass_code 因子才有判别力（同 _ir_cfg 的注释）。"""
    c = xmf.shock_tube()
    c["grid"]["n"] = [16, 1, 1]
    c["fluids"] = [
        {"name": "HI", "mass": MASS_ION, "charge": 0, "kind": "neutral"},
        {"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"},
        {"name": "HEII", "mass": MASS_HEII, "charge": 2, "kind": "ion"},
    ]
    c["electron"] = {"mode": "pressure-only"}
    c["sources"] = sources
    c["radiation"] = {
        "enabled": False,
        "absorbers": ["HI"],
        "bins": {"n": 1, "sigma": [1.0e-18], "hnu": [2.0e-11], "flux_inf": [0.0]},
    }
    c["ic"] = {"kind": "none"}
    return c


def test_ir_ne_te_multi_ion_cellwise(lib):
    """双离子 + 中性、非均匀剖面：n_e = Σ q_s rho_s/mass_code 与
    T_e = P_e*unit_P/(n_e kB) 逐格对解析 1e-12；jac 每支离子 rho 各一列、
    中性列为 0；B2 开门 num_states == 16（5*3 + P_e 殿后），T_e 带子
    在 P_e 列多一个 unit_P/(n_e kB) 非零。

    收紧点（相对单离子均匀态钉）：Σ 是真和（两项、不同 q、不同
    mass_code），逐格是真逐格（沿 i 的斜坡剖面），中性支密度非零但
    不得进 n_e。核按流体下标求和、不认组分名：解析式用同样的下标序
    与同样的结合序 (q*rho)/mass_code 逐项累加。
    """
    sources = [
        {"name": "ne_read", "expr": {"dP_HI": "n_e"}},
        {"name": "te_read", "expr": {"dP_HEII": "T_e"}},
    ]
    run = xmf.setup(_ir_cfg_multi(sources))
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        assert run.num_states == 16, (
            "B2 开门：3 流体 × 5 槽 + P_e 殿后；n_e/T_e 仍不入未知量"
        )

        idx = np.arange(nzt * nyt * nxt, dtype=np.float64).reshape(shape)
        rho_hi = 0.5 + 0.001 * idx
        rho_hii = 1.2 - 0.0015 * idx
        rho_heii = 0.7 + 0.002 * idx
        pe_blk = 0.2 + 0.0008 * idx
        assert np.all(rho_hii > 0.0) and np.all(rho_heii > 0.0)

        _pack_block(lib, run, 0, SLOT_N, rho_hi)
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, 0.4))
        _pack_block(lib, run, 1, SLOT_N, rho_hii)
        _pack_block(lib, run, 1, SLOT_P, np.full(shape, 0.5))
        _pack_block(lib, run, 2, SLOT_N, rho_heii)
        _pack_block(lib, run, 2, SLOT_P, np.full(shape, 0.6))
        _write_pe(lib, run, pe_blk)

        # 解析值：rho 用读回值（与核同一份数），Σ 按流体下标序累加
        # （中性支 q=0 项恰为 0，不进和）。unit_P 默认 1.0。
        mc_hii = MASS_ION * AMU / run._cfg.units.density
        mc_heii = MASS_HEII * AMU / run._cfg.units.density
        unit_p = run._cfg.units.density * run._cfg.units.velocity**2
        r_hii = _unpack_block(lib, run, 1, SLOT_N).reshape(shape)
        r_heii = _unpack_block(lib, run, 2, SLOT_N).reshape(shape)
        ne = (1.0 * r_hii) / mc_hii + (2.0 * r_heii) / mc_heii
        ne_safe = np.where(ne > 1.0e-30, ne, 1.0e-30)
        te = pe_blk * unit_p / (ne_safe * KB)

        p_hi0 = _unpack_block(lib, run, 0, SLOT_P).copy()
        p_he0 = _unpack_block(lib, run, 2, SLOT_P).copy()
        run.apply_sources()
        d_hi = (_unpack_block(lib, run, 0, SLOT_P) - p_hi0).reshape(shape)
        d_he = (_unpack_block(lib, run, 2, SLOT_P) - p_he0).reshape(shape)
        inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
        rel_ne = float(np.max(np.abs((d_hi[inner] - ne[inner]) / ne[inner])))
        assert rel_ne <= 1.0e-12, (
            f"apply 的 n_e 增量应对 Σ q_s rho_s/mass_code 逐格 1e-12，"
            f"实测最大相对差 {rel_ne:.3e}"
        )
        rel_te = float(np.max(np.abs((d_he[inner] - te[inner]) / te[inner])))
        assert rel_te <= 1.0e-12, (
            f"apply 的 T_e 增量应对 P_e*unit_P/(n_e kB) 逐格 1e-12，"
            f"实测最大相对差 {rel_te:.3e}"
        )
        # ghost 不求值；无人写的 HII P 槽逐位不动
        ghost = np.ones(shape, dtype=bool)
        ghost[inner] = False
        assert np.all(d_hi[ghost] == 0.0), "apply 不得触碰 ghost 区"
        assert np.all(d_he[ghost] == 0.0), "apply 不得触碰 ghost 区"
        assert np.all(_unpack_block(lib, run, 1, SLOT_P) == 0.5), (
            "两条带子都不得写 HII 的 P 槽"
        )

        # jac：行 4 = HI 的 P（n_e 带），行 14 = HEII 的 P（T_e 带）；
        # 列 0/5/10 = HI/HII/HEII 的 rho，列 15 = B2 开门的 P_e 未知量。
        # 逐格钉两个不同位置的内区格。
        for iq in (2, nx - 3):
            cell = (ng, ng, ng + iq)
            J = run.jacobian(*cell)
            assert J.shape == (16, 16)
            ne_c = float(ne[cell])
            pe_c = float(pe_blk[cell])
            assert abs(J[4, 5] - 1.0 / mc_hii) <= 1.0e-12 / mc_hii, (
                "d(n_e)/d(rho_HII) 应为 q_HII/mass_code_HII"
            )
            assert abs(J[4, 10] - 2.0 / mc_heii) <= 1.0e-12 * (2.0 / mc_heii), (
                "d(n_e)/d(rho_HEII) 应为 q_HEII/mass_code_HEII"
            )
            assert J[4, 0] == 0.0, "中性支不进 n_e"
            k_te = -pe_c * unit_p / (ne_c * ne_c * KB)
            assert abs(J[14, 5] - k_te / mc_hii) <= 1.0e-12 * abs(k_te / mc_hii), (
                "d(T_e)/d(rho_HII) 应为 -P_e unit_P/(n_e² kB) · q/mass_code"
            )
            assert abs(J[14, 10] - 2.0 * k_te / mc_heii) <= 1.0e-12 * abs(
                2.0 * k_te / mc_heii
            ), "d(T_e)/d(rho_HEII) 应为 -P_e unit_P/(n_e² kB) · q/mass_code"
            assert J[14, 0] == 0.0, "中性支不进 T_e"
            assert J[4, 15] == 0.0, "n_e = Σ q_s n_s 不含 P_e"
            d_te_dpe = unit_p / (ne_c * KB)
            assert abs(J[14, 15] - d_te_dpe) <= 1.0e-12 * abs(d_te_dpe), (
                "B2：T_e 带子在 P_e 列应为 unit_P/(n_e kB)"
            )
            assert np.count_nonzero(J) == 5, (
                "两带子的 J 应恰有五个非零（行 4/14 的两支离子 rho 列 + "
                "行 14 的 P_e 列），"
                f"实测 {np.argwhere(J != 0.0).tolist()}"
            )
    finally:
        run.close()


def test_ir_load_pe_value_and_newton_column(lib):
    """LOAD P_e 在 pe!=NULL 时合法：apply 增量与储存的 P_e 块逐格一致
    （读原值、无单位换算）。B2 合同翻转（相对 B1）：开门时 P_e 是最后
    一个牛顿未知量——N = 5*nfluid + 1，LOAD P_e 带子的 jac 在
    (行 = dP_HII 的 P 槽, 列 = P_e) 上恰为 1.0，其余逐位为 0。"""
    c = xmf.shock_tube()
    c["grid"]["n"] = [16, 1, 1]
    c["fluids"] = [{"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"}]
    c["electron"] = {"mode": "pressure-only"}
    c["sources"] = [{"name": "pe_read", "expr": {"dP_HII": "P_e"}}]
    c["ic"] = {"kind": "none"}
    run = xmf.setup(c)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        assert run.num_states == 6, "B2 开门：单流体 5 槽 + P_e 殿后"

        pe_blk = (
            np.arange(nzt * nyt * nxt, dtype=np.float64).reshape(shape) + 0.25
        ) * 0.5
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, 0.75))
        _write_pe(lib, run, pe_blk)

        p0 = _unpack_block(lib, run, 0, SLOT_P).copy()
        run.apply_sources()
        d = (_unpack_block(lib, run, 0, SLOT_P) - p0).reshape(shape)
        inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
        rel = float(np.max(np.abs((d[inner] - pe_blk[inner]) / pe_blk[inner])))
        assert rel <= 1.0e-12, (
            f"LOAD P_e 的 apply 增量应与储存块逐格一致，实测最大相对差 {rel:.3e}"
        )
        ghost = np.ones(shape, dtype=bool)
        ghost[inner] = False
        assert np.all(d[ghost] == 0.0), "apply 不得触碰 ghost 区"

        J = run.jacobian(ng, ng, ng + nx // 2)
        assert J.shape == (6, 6)
        assert J[4, 5] == 1.0, (
            "B2 开门：d(dP_HII)/d(P_e) = 1（P_e 是最后一个牛顿未知量）"
        )
        Jc = J.copy()
        Jc[4, 5] = 0.0
        assert np.all(Jc == 0.0), (
            "LOAD P_e 带子的 jac 除 (P_HII 行, P_e 列) 外应逐位全 0"
        )
    finally:
        run.close()


@pytest.mark.parametrize("read_name", ["n_e", "T_e", "P_e"])
def test_ir_elec_reads_refused_when_electrons_off(read_name):
    """关电子：n_e / T_e / P_e 三个 ELEC 读名一律编译期拒绝（含 "electron"）。"""
    c = _ir_cfg("off", [{"name": "bad", "expr": {"dP_HI": read_name}}])
    with pytest.raises(ValueError, match="electron"):
        xmf.setup(c)


# --- 7. fields / fields_full 的 P_e 键 -----------------------------------------


def test_fields_pe_key_presence_and_values(lib):
    c_on = xmf.shock_tube()
    c_on["grid"]["n"] = [16, 1, 1]
    c_on["fluids"] = [{"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"}]
    c_on["electron"] = {"mode": "pressure-only"}
    c_on["ic"] = {"kind": "none"}
    run = xmf.setup(c_on)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        pe = np.arange(nzt * nyt * nxt, dtype=np.float64).reshape(nzt, nyt, nxt) + 0.25
        _write_pe(lib, run, pe)
        u = run._cfg.units

        ff = run.fields_full()
        expect = pe[ng : ng + nz, ng : ng + ny, ng : ng + nx]
        got = expect * u.density * u.velocity * u.velocity
        assert np.array_equal(ff["P_e"], got), "fields_full 的 P_e 应与块内区逐位一致"

        f1d = run.fields
        assert "P_e" in f1d
        row = pe[ng, ng, ng : ng + nx] * u.density * u.velocity**2
        assert np.array_equal(f1d["P_e"], row), "legacy fields 的 P_e 取 (ng,ng) 行"
    finally:
        run.close()

    run_off = xmf.setup(xmf.shock_tube())
    try:
        assert "P_e" not in run_off.fields_full()
        assert "P_e" not in run_off.fields
    finally:
        run_off.close()
