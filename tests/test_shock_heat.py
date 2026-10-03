"""B2 激波加热份额（shock_heat）管测单测：三档份额按公式走，关电子回退现钉。

合同来自本刀卡片（docs/SESSION.md B2 §交什么 3/5 与 docs/cmd/PROMPT_B2_H_PE.md）：
  1. 笛卡尔 1D、至少一支离子 + pressure-only：份额换档时，离子 / P_e 的
     加热增量按公式走。核内实现（xmf_hydro.c sweep_line_pe 的 keep 环）：
       H      = max(P_B - P_s^n * (rho'_1/rho'_0)^gamma, 0)   （pass B 自测）
       f_s    = 0 (ions-only) | w_s (by-charge) | 1 (electron)
       P_s'   = max(P_B - f_s*H, 1e-30)
       P_e   += P_B - P_s'        （真被拿走的那份，floor 未激活时 == f_s*H）
     其中 P_B 是 pass B 扫出的总压增量，rho'_0 用 pass B 装置原式重算、
     rho'_1 是扫后的伪密度。deposit 在同一方向的伪流体扫之后落账，
     伪扫的可逆 PdV 部分不被档位触碰。
  2. 1D 单步逐位关系（唯一活动方向里，离子扫期间 P_e 停在 U^n，各档的
     pass A/B 输入逐位相同，dt 也逐位相同）：
       d          = P_s^ions-only - P_s^tier >= 0
       P_e^tier   == P_e^ions-only + d                 （逐位）
       单离子时 by-charge 的 w = q n / (q n) = 1.0 逐位 ⇒ by-charge 与
       electron 全槽逐位相同；中性支 w=0、走普通单遍扫，五槽各档逐位不变；
       多离子时 by-charge 的 P_e 增量 == 各离子 d 之和（按流体下标序相加）。
  3. ions-only 是默认档且就是 B1 路径：显式 "ions-only" 与省略键的默认跑
     一步全槽逐位相同（B1 的 P_eff 等价钉在 test_pe 里，空表 Sod 钉在
     test_step_empty 里——两者的默认档现在都是 ions-only）。
  4. 关电子回退：三档名字分别挂在 electron.mode=off 上 advance() 到底，
     复现 test_step_empty 的空表 pcm Sod 三钉与 PIN_T（逐位 ==）。
  5. 拒绝面：normalize 对未知档名 raise（含 "shock_heat"）；C 级
     xmf_setup 对枚举越界（改 blob 绕过前端）返回 NULL。

口径（同 test_pe）：unpack/write_pe 块长 = xmf_state_ncells 含 ghost，
reshape (nztot, nytot, nxtot)，(k,j,i) i 最快；默认单位下 unit_P = 1。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；钉的是五槽 + P_e 的
逐位关系与整条 advance 链，不只测 VX1。
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf import config as cfgmod
from xmf._native import load, problem_to_c
from xmf.config import KB, normalize

_PTR = ctypes.POINTER(ctypes.c_double)

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

# 管测初值（码单位，O(1)）：沿 x1 的台阶制造激波/耗散；P_e 与 P_s 同量级。
RHO_L, RHO_R = 1.0, 0.3
PS_L, PS_R = 1.0, 0.2
PE_L, PE_R = 0.6, 0.1
RHO_N, PN = 0.5, 0.1  # 中性支均匀冷背景（w=0）
MASS_ION, MASS_NEUTRAL = 1.00794, 4.0026
MASS_HEII = 4.0026  # 多离子钉的第二支：q=2、质量不同
TIERS = ("ions-only", "by-charge", "electron")


@pytest.fixture(scope="module")
def lib():
    return load()


# --- 打包小助手（同 test_pe 的口径） ------------------------------------------


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


def _step_along_x1(shape, left: float, right: float) -> np.ndarray:
    """沿 i（nxtot 轴）取左/右台阶值，分界在块中线（只用于构造初值）。"""
    blk = np.empty(shape, dtype=np.float64)
    n = shape[2]
    for idx in range(n):
        blk[:, :, idx] = left if idx < n // 2 else right
    return blk


def _tube_cfg(shock_heat=None, fluids=None, n=64):
    """1D 笛卡尔管：pcm/hll/rk1、六面 reflect、ic none（手工打包）。
    shock_heat=None 时省略该键，测默认档。"""
    c = xmf.shock_tube()
    c["grid"]["n"] = [n, 1, 1]
    c["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    c["fluids"] = fluids if fluids is not None else [
        {"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"},
        {"name": "HI", "mass": MASS_NEUTRAL, "charge": 0, "kind": "neutral"},
    ]
    ele = {"mode": "pressure-only"}
    if shock_heat is not None:
        ele["shock_heat"] = shock_heat
    c["electron"] = ele
    c["ic"] = {"kind": "none"}
    return c


def _init_tube(lib, run):
    """同一份台阶初值打进一支离子 + 一支中性 + P_e 块。"""
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    shape = (nzt, nyt, nxt)
    rho = _step_along_x1(shape, RHO_L, RHO_R)
    p_s = _step_along_x1(shape, PS_L, PS_R)
    pe = _step_along_x1(shape, PE_L, PE_R)
    _pack_block(lib, run, 0, SLOT_N, rho)
    _pack_block(lib, run, 0, SLOT_P, p_s)
    _pack_block(lib, run, 1, SLOT_N, np.full(shape, RHO_N))
    _pack_block(lib, run, 1, SLOT_P, np.full(shape, PN))
    _write_pe(lib, run, pe)
    return shape


# --- 1. 默认档 = ions-only，且与 B1 路径逐位相同 --------------------------------


def test_default_tier_is_ions_only_bitwise(lib):
    """省略 shock_heat 键的默认档 == 显式 "ions-only"：一步之后两支流体
    五槽与 P_e 块逐位相同。默认档逐位就是 B1 的两遍扫路径（heat 缓冲
    不分配，keep 环原样写 P_B）。"""
    p_def = normalize(_tube_cfg(shock_heat=None))
    p_exp = normalize(_tube_cfg(shock_heat="ions-only"))
    assert p_def.electron.shock_heat == "ions-only", "缺省档必须是 ions-only"
    assert p_exp.electron.shock_heat == "ions-only"

    run_def = xmf.setup(_tube_cfg(shock_heat=None))
    run_exp = xmf.setup(_tube_cfg(shock_heat="ions-only"))
    try:
        _init_tube(lib, run_def)
        _init_tube(lib, run_exp)
        run_def.step()
        run_exp.step()
        assert run_def.dt == run_exp.dt
        for f in range(2):
            for var in range(5):
                a = _unpack_block(lib, run_def, f, var)
                b = _unpack_block(lib, run_exp, f, var)
                assert np.array_equal(a, b), (
                    f"流体 {f} 槽 {var}：默认档与显式 ions-only 一步后应逐位相同"
                )
        assert np.array_equal(_copy_pe(lib, run_def), _copy_pe(lib, run_exp)), (
            "P_e 块：默认档与显式 ions-only 一步后应逐位相同"
        )
    finally:
        run_def.close()
        run_exp.close()


# --- 2. 单离子三档增量关系（1D 一步逐位钉） -------------------------------------


def _run_tiers(lib, shock_heats, fluids=None, n=64):
    """同一初值、各档各跑一步，返回 {tier: run}（调用方负责 close）。"""
    runs = {}
    for tier in shock_heats:
        run = xmf.setup(_tube_cfg(shock_heat=tier, fluids=fluids, n=n))
        _init_tube(lib, run)
        run.step()
        runs[tier] = run
    return runs


def test_tier_increment_relations_single_ion(lib):
    """单离子（HII + 中性 HI）：d = P_s^io - P_s^tier >= 0、激波处 d > 0，
    P_e^tier == P_e^io + d 逐位；单离子 by-charge 的 w = qn/qn = 1.0 逐位
    ⇒ by-charge 与 electron 全槽逐位相同；中性支五槽各档逐位不变。

    逐位关系成立的口径：1D 唯一方向里，离子扫期间 P_e 停在 U^n（deposit
    在同方向伪扫之后），各档 pass A/B 输入逐位相同；dt 只读储存的 rho 与
    P_eff，也逐位相同。ghost 在步首由同一份初值填充、扫只写内区，故关系
    对整块（含 ghost）成立——ghost 上 d = 0。
    """
    runs = _run_tiers(lib, TIERS)
    try:
        # dt 各档逐位相同（CFL 读同一份 rho 与 P_eff = P_s + w*pe）
        dts = {t: r.dt for t, r in runs.items()}
        assert dts["ions-only"] == dts["by-charge"] == dts["electron"], (
            f"各档 dt 应逐位相同：{dts}"
        )

        pe_io = _copy_pe(lib, runs["ions-only"])
        ps_io = _unpack_block(lib, runs["ions-only"], 0, SLOT_P)
        for tier in ("by-charge", "electron"):
            ps_t = _unpack_block(lib, runs[tier], 0, SLOT_P)
            d = ps_io - ps_t
            assert np.all(d >= 0.0), (
                f"{tier}：d = P_s^io - P_s^{tier} 应处处 >= 0，"
                f"实测最小 {float(np.min(d)):.3e}"
            )
            assert float(np.max(d)) > 1.0e-6 and np.count_nonzero(d) > 0, (
                f"{tier}：激波耗散应真的进份额（max d = {float(np.max(d)):.3e}）"
            )
            pe_t = _copy_pe(lib, runs[tier])
            assert np.array_equal(pe_t, pe_io + d), (
                f"{tier}：P_e^tier == P_e^io + d 应逐位成立，"
                f"max|diff|={float(np.max(np.abs(pe_t - (pe_io + d)))):.3e}"
            )

        # 单离子 w = 1.0 逐位 ⇒ by-charge 与 electron 全槽逐位相同
        for f in range(2):
            for var in range(5):
                a = _unpack_block(lib, runs["by-charge"], f, var)
                b = _unpack_block(lib, runs["electron"], f, var)
                assert np.array_equal(a, b), (
                    f"单离子：槽 ({f},{var}) by-charge 与 electron 应逐位相同"
                )
        assert np.array_equal(
            _copy_pe(lib, runs["by-charge"]), _copy_pe(lib, runs["electron"])
        ), "单离子：P_e 块 by-charge 与 electron 应逐位相同"

        # 中性支 w=0 走普通单遍扫：五槽各档逐位不变
        for var in range(5):
            ref = _unpack_block(lib, runs["ions-only"], 1, var)
            for tier in ("by-charge", "electron"):
                got = _unpack_block(lib, runs[tier], 1, var)
                assert np.array_equal(got, ref), (
                    f"中性支槽 {var} 在 {tier} 下应与 ions-only 逐位相同"
                )
    finally:
        for run in runs.values():
            run.close()


def test_tier_relations_multi_ion_by_charge(lib):
    """双离子（HII q=1 / HEII q=2，质量不同）+ 中性：by-charge 的 P_e
    增量 == 各离子 d 之和（按流体下标序相加）逐位；electron 档每支离子
    拿走的份额 >= by-charge（f=1 >= w_s）。

    deposit 在核内按流体下标序累加进 heat（0+dep 逐位无舍入，第二次加
    法单次舍入），测试按同一顺序 d_HII + d_HEII 相加，关系逐位成立。
    """
    fluids = [
        {"name": "HI", "mass": MASS_ION, "charge": 0, "kind": "neutral"},
        {"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"},
        {"name": "HEII", "mass": MASS_HEII, "charge": 2, "kind": "ion"},
    ]
    runs = {}
    for tier in TIERS:
        run = xmf.setup(_tube_cfg(shock_heat=tier, fluids=fluids, n=64))
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        rho = _step_along_x1(shape, RHO_L, RHO_R)
        p_s = _step_along_x1(shape, PS_L, PS_R)
        pe = _step_along_x1(shape, PE_L, PE_R)
        # 三支都带台阶（两支离子都过激波）；中性支均匀
        _pack_block(lib, run, 0, SLOT_N, np.full(shape, RHO_N))
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, PN))
        for f in (1, 2):
            _pack_block(lib, run, f, SLOT_N, rho)
            _pack_block(lib, run, f, SLOT_P, p_s)
        _write_pe(lib, run, pe)
        run.step()
        runs[tier] = run
    try:
        pe_io = _copy_pe(lib, runs["ions-only"])
        d_bc = []
        d_el = []
        for f in (1, 2):
            ps_io = _unpack_block(lib, runs["ions-only"], f, SLOT_P)
            d_bc.append(ps_io - _unpack_block(lib, runs["by-charge"], f, SLOT_P))
            d_el.append(ps_io - _unpack_block(lib, runs["electron"], f, SLOT_P))
        for tag, d in (("by-charge", d_bc), ("electron", d_el)):
            assert np.all(d[0] >= 0.0) and np.all(d[1] >= 0.0), (
                f"{tag}：两支离子的 d 都应 >= 0"
            )
            assert float(np.max(d[0])) > 1.0e-6 or float(np.max(d[1])) > 1.0e-6, (
                f"{tag}：至少一支离子应拿到非零耗散份额"
            )
        # electron 档份额 1 >= w_s ⇒ 每支离子的 d 不小于 by-charge
        assert np.all(d_el[0] >= d_bc[0] - 0.0) and np.all(d_el[1] >= d_bc[1] - 0.0), (
            "electron 档每支离子拿走的增量应 >= by-charge（f=1 >= w_s）"
        )
        # by-charge 的 P_e 增量 == 各离子 d 之和（流体下标序），逐位
        pe_bc = _copy_pe(lib, runs["by-charge"])
        assert np.array_equal(pe_bc, pe_io + (d_bc[0] + d_bc[1])), (
            "by-charge：P_e 增量应逐位等于两支离子 d 之和"
        )
        pe_el = _copy_pe(lib, runs["electron"])
        assert np.array_equal(pe_el, pe_io + (d_el[0] + d_el[1])), (
            "electron：P_e 增量应逐位等于两支离子 d 之和"
        )
        # 中性支五槽各档逐位不变
        for var in range(5):
            ref = _unpack_block(lib, runs["ions-only"], 0, var)
            for tier in ("by-charge", "electron"):
                assert np.array_equal(
                    _unpack_block(lib, runs[tier], 0, var), ref
                ), f"中性支槽 {var} 在 {tier} 下应逐位不变"
    finally:
        for run in runs.values():
            run.close()


# --- 2b. 手算期望值：pass B 公式在核外复现（不是跑核比核） ----------------------


def _hand_pass_b(lib, run, gamma):
    """纯 Python 复现 pass B（rho*P_s/P_eff, v, P_s）的 pcm/hll/rk1 单步
    扫——不经核，期望值完全在核外。

    输入只用管测初值（各档逐位相同的那份）、跑实际用的 dt 与网格读口读出的
    A/dV（网格是输入参数，不是被钉量）：
      - 线态 = 块状台阶；reflect ghost 镜像同台阶值的内区，逐位不变；
      - 单离子 w = q rho/q rho = 1.0 逐位（无 phys 口 ⇒ mass_code = 1），
        scale = q[4]/(q[4] + w*pe) 按核内装置式同一运算序；
      - HLL 面通量（Davis 波速）、散射顺序与 line_flux_update 逐位一致：
        面 m 先对左格 -= cl*F、再对右格 += cr*F，m 从 ng-1 走到 ng+n-1，
        q 全程不动、只有 U 累加；
      - cons->prim 用与核相同的 1e-30 地板。

    返回内区（长 n）的 pB、rho1、rho0 三个 Python float 列表：pB 是扫后
    伪压强（ions-only 的 P 槽本身），rho1 是扫后伪密度，rho0 是装置原式
    q[0]*scale。"""
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    assert (ny, nz) == (1, 1), "手算复现只针对 1D 管"
    dt = run.dt
    Af = np.empty(nxt + 1, dtype=np.float64)
    dV = np.empty(nxt, dtype=np.float64)
    assert lib.xmf_mesh_copy_A(run._h, 0, Af.ctypes.data_as(_PTR)) == nxt + 1
    assert lib.xmf_mesh_copy_dV(run._h, 0, dV.ctypes.data_as(_PTR)) == nxt

    mid = nxt // 2  # _step_along_x1 的分界
    rho = [RHO_L if m < mid else RHO_R for m in range(nxt)]
    ps = [PS_L if m < mid else PS_R for m in range(nxt)]
    pe = [PE_L if m < mid else PE_R for m in range(nxt)]
    # 装置式：peff = q[4] + w*pe（w = 1.0 逐位），scale = q[4]/peff
    scale = [ps[m] / (ps[m] + pe[m]) for m in range(nxt)]
    q = [(rho[m] * scale[m], 0.0, 0.0, 0.0, ps[m]) for m in range(nxt)]

    gmm1 = gamma - 1.0

    def p2c(q5):
        r, vn, vt1, vt2, p = q5
        return (r, r * vn, r * vt1, r * vt2,
                p / gmm1 + 0.5 * r * (vn * vn + vt1 * vt1 + vt2 * vt2))

    def c2p(U5):
        r = U5[0] if U5[0] > 1.0e-30 else 1.0e-30
        ek = 0.5 * (U5[1] * U5[1] + U5[2] * U5[2] + U5[3] * U5[3]) / r
        p = gmm1 * (U5[4] - ek)
        if p < 1.0e-30:
            p = 1.0e-30
        return (r, U5[1] / r, U5[2] / r, U5[3] / r, p)

    def flux5(q5):
        r, vn, p = q5[0], q5[1], q5[4]
        E = p / gmm1 + 0.5 * r * (vn * vn + q5[2] * q5[2] + q5[3] * q5[3])
        return (r * vn, r * vn * vn + p, r * vn * q5[2], r * vn * q5[3],
                (E + p) * vn)

    def hll(ql, qr):
        UL, UR = p2c(ql), p2c(qr)
        FL, FR = flux5(ql), flux5(qr)
        rl = ql[0] if ql[0] > 1.0e-30 else 1.0e-30
        rr = qr[0] if qr[0] > 1.0e-30 else 1.0e-30
        pl = ql[4] if ql[4] > 1.0e-30 else 1.0e-30
        pr = qr[4] if qr[4] > 1.0e-30 else 1.0e-30
        cl = math.sqrt(gamma * pl / rl)
        cr = math.sqrt(gamma * pr / rr)
        SL = min(ql[1] - cl, qr[1] - cr)
        SR = max(qr[1] + cr, ql[1] + cl)
        if SL >= 0.0:
            return FL
        if SR <= 0.0:
            return FR
        den = SR - SL
        return tuple(
            (SR * FL[c] - SL * FR[c] + SL * SR * (UR[c] - UL[c])) / den
            for c in range(5)
        )

    U = [p2c(q5) for q5 in q]
    for m in range(ng - 1, ng + nx):
        F = hll(q[m], q[m + 1])
        cl = dt * Af[m + 1] / dV[m]
        cr = dt * Af[m + 1] / dV[m + 1]
        lm, rm = list(U[m]), list(U[m + 1])
        for v in range(5):
            lm[v] -= cl * F[v]
            rm[v] += cr * F[v]
        U[m] = tuple(lm)
        U[m + 1] = tuple(rm)

    q1 = [c2p(U[m]) for m in range(ng, ng + nx)]
    pB = [q5[4] for q5 in q1]
    rho1 = [q5[0] for q5 in q1]
    rho0 = [rho[m] * scale[m] for m in range(ng, ng + nx)]
    return pB, rho1, rho0, mid


def test_tier_hand_computed_increment_formula(lib):
    """手算期望值钉（卡片 §3「至少一档用手算期望值对，不是跑核比核」）：
    在核外按 pass B 装置复现整条扫，再用核外数值检验 keep 环公式
        H    = max(P_B - P_s^n (rho'_1/rho'_0)^gamma, 0)
        P_s' = max(P_B - f_s H, 1e-30)，deposit = P_B - P_s'
    钉的内容：
      1) ions-only 的 P 槽 == 手算 pB 逐位（keep 环原样写 pass-B 压强，
         且复现本身正确——pass B 扫只有 +,-,*,/ 与 IEEE 精确的 sqrt，
         无超越函数，逐位可复现）；
      2) by-charge / electron 的 P 槽对 max(pB - f*H, 1e-30) 1e-12
         （单离子 f = 1：by-charge 的 w = 1.0 逐位、electron 恒 1；
         放宽的只是 pow 跨 libm 实现的末位 ulp，公式写错的偏差是
         O(1e-2) 量级，远大于此容差）；
      3) P_e 增量 pe_tier - pe_io 对手算 deposit 1e-12（伪扫的可逆
         PdV 部分各档逐位相同，在差里相消——既有逐位关系钉的旁证）；
      4) 手算 H > 0 只出现在间断两侧邻格（只有跨间断面产熵），其余
         格至多 ulp 噪声；max H > 1e-6（耗散真的非零）。
    """
    runs = _run_tiers(lib, TIERS)
    try:
        run_io = runs["ions-only"]
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run_io)
        shape = (nzt, nyt, nxt)
        row = (ng, ng, slice(ng, ng + nx))
        gamma = run_io._cfg.gamma

        pB, rho1, rho0, mid = _hand_pass_b(lib, run_io, gamma)
        exp_pB = np.array(pB)
        exp_rho1 = np.array(rho1)
        exp_rho0 = np.array(rho0)
        ps0 = np.array(
            [PS_L if m < mid else PS_R for m in range(ng, ng + nx)]
        )

        # 手算 H 与份额后的 P_s'、deposit（单离子 f = 1.0）
        H = exp_pB - ps0 * (exp_rho1 / exp_rho0) ** gamma
        H = np.where(H > 0.0, H, 0.0)
        assert float(np.max(H)) > 1.0e-6, (
            f"激波处手算 H 应有真耗散，实测 max H = {float(np.max(H)):.3e}"
        )
        adj = np.array([mid - 1 - ng, mid - ng])  # 间断两侧邻格的内区下标
        hot = np.argwhere(H > 1.0e-10).ravel()
        assert 1 <= hot.size <= 2 and np.all(np.isin(hot, adj)), (
            f"H > 0 应集中在间断邻格 {adj.tolist()}，实测 {hot.tolist()}"
        )
        ps_exp = np.maximum(exp_pB - H, 1.0e-30)
        dep_exp = exp_pB - ps_exp

        # 1) ions-only：P 槽就是 pass-B 压强，逐位
        ps_io = _unpack_block(lib, run_io, 0, SLOT_P).reshape(shape)[row]
        assert np.array_equal(ps_io, exp_pB), (
            "ions-only keep 环应原样写 pass-B 压强；max|diff|="
            f"{float(np.max(np.abs(ps_io - exp_pB))):.3e}"
        )

        # 2)/3) by-charge 与 electron（单离子 f 同为 1，期望值相同）
        pe_io = _copy_pe(lib, run_io).reshape(shape)[row]
        for tier in ("by-charge", "electron"):
            ps_t = _unpack_block(lib, runs[tier], 0, SLOT_P).reshape(shape)[row]
            err = float(np.max(np.abs(ps_t - ps_exp)))
            assert err <= 1.0e-12, (
                f"{tier}：P_s' 应对手算 max(P_B - f*H, 1e-30) 1e-12，"
                f"实测 max|diff| = {err:.3e}"
            )
            dep = _copy_pe(lib, runs[tier]).reshape(shape)[row] - pe_io
            derr = float(np.max(np.abs(dep - dep_exp)))
            assert derr <= 1.0e-12, (
                f"{tier}：P_e 增量应对手算 deposit = P_B - P_s' 1e-12，"
                f"实测 max|diff| = {derr:.3e}"
            )
    finally:
        for run in runs.values():
            run.close()


# --- 3. 关电子回退现空表 Sod 钉（三档都惰性） -----------------------------------


@pytest.mark.parametrize("tier", TIERS)
def test_electron_off_fallback_reproduces_empty_sod_pins(tier):
    """electron.mode=off 时三档全部惰性（无 P_e 块、无电荷表、heat 不
    分配）：advance() 到底逐位复现 test_step_empty 的空表基线。"""
    from test_step_empty import PIN_L1_N, PIN_L1_P, PIN_L1_V, PIN_T

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.riemann import exact_hi

    cfg = xmf.shock_tube()
    cfg["electron"] = {"mode": "off", "shock_heat": tier}
    p = normalize(cfg)
    assert p.electron.shock_heat == tier  # 前端收该名，但关电子时惰性

    run = xmf.setup(cfg)
    try:
        run.advance()
        assert run.t == PIN_T, (
            f"tier={tier} 关电子：最终 t 应逐位等于空表钉 {PIN_T!r}，"
            f"实测 {run.t!r}"
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
        assert err_n == PIN_L1_N, f"tier={tier}：L1(n) 偏离空表钉（{err_n!r}）"
        assert err_v == PIN_L1_V, f"tier={tier}：L1(v) 偏离空表钉（{err_v!r}）"
        assert err_p == PIN_L1_P, f"tier={tier}：L1(P) 偏离空表钉（{err_p!r}）"
        assert "P_e" not in run.fields_full(), "关电子不得出现 P_e 读口"
    finally:
        run.close()


# --- 4. 拒绝面：前端归一化 + C 级枚举校验 ---------------------------------------


def test_normalize_rejects_unknown_shock_heat():
    cfg = _tube_cfg()
    cfg["electron"]["shock_heat"] = "banana"
    with pytest.raises(ValueError, match="shock_heat"):
        cfgmod.normalize(cfg)


def _uniform_faces(blob):
    """直调 C 的用例补三向等距活动界面（同 test_abi 的机械适配）。"""
    mins = (blob.x1min, blob.x2min, blob.x3min)
    maxs = (blob.x1max, blob.x2max, blob.x3max)
    arrs = []
    for d in range(3):
        nd = int(blob.n[d]) if int(blob.n[d]) >= 1 else 1
        dx = (maxs[d] - mins[d]) / nd
        arrs.append(
            np.array([mins[d] + k * dx for k in range(nd + 1)], dtype=np.float64)
        )
    return arrs


def _setup_c(lib, blob):
    arrs = _uniform_faces(blob)
    return lib.xmf_setup(
        ctypes.byref(blob),
        arrs[0].ctypes.data_as(_PTR),
        arrs[1].ctypes.data_as(_PTR),
        arrs[2].ctypes.data_as(_PTR),
    )


def test_c_setup_rejects_out_of_range_shock_heat(lib):
    """blob.shock_heat 越界（绕过前端）必须在 C 级 xmf_setup 拒成 NULL；
    合法三档放行。字节位置是原 reserved0 槽（改名不改位）。"""
    for bad in (7, -1, 3):
        p = normalize(_tube_cfg(shock_heat="ions-only"))
        blob = problem_to_c(p)
        assert blob.shock_heat == 0, "normalize 后该槽应落 ions-only=0"
        blob.shock_heat = bad
        handle = _setup_c(lib, blob)
        assert handle is None or handle == 0, (
            f"shock_heat={bad} 应被 C 级拒成 NULL"
        )
    for good, code in (("ions-only", 0), ("by-charge", 1), ("electron", 2)):
        p = normalize(_tube_cfg(shock_heat=good))
        blob = problem_to_c(p)
        assert blob.shock_heat == code, f"{good} 应落枚举 {code}"
        handle = _setup_c(lib, blob)
        assert handle, f"合法档 {good} 应被 C 级接受"
        lib.xmf_finalize(handle)
