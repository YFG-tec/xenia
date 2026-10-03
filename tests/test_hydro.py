"""A4 hydro 模块契约单测（第一道门，无图；对照 docs/TODO.md → A4 第三行原文）。

第三行原文：
  单测 test_hydro：接触间断（u=0, ∇p=0）密度台阶不移动；1D Sod 对 Toro，L1 有上界；
  闲置 vy 全程为 0。

卡片（docs/SESSION.md）「交什么」第 2/3/4 条补充：
  - 接触间断建议 riemann=hllc；HLL 允许抹，但台阶中心不得漂移。
  - 界面带三速度；1D 闲置 vy、vz 全程保持 0（含 ghost 段）。
  - 禁止只测 VX1：薄 2D 必须碰非主轴速度（法向 x2 组激发 vy；法向 x1 组
    携带均匀横向 vy 并钉闲置 vz 逐位 0，见本文件薄 2D 两组）。
  - HD/MHD 分路：kind=mhd 必须在 setup 层拒绝（不得 B=0 偷梁）。

约定（docs/TODO.md §A）：一块一文件；断言只打 hydro 模块契约，不靠整条激波管
「看起来像」；维数覆盖 1D（nx2=nx3=1）与 nx2>1 的薄 2D。不读 pluto.ini、
不用 XUV_FACTOR==0 哨兵、不只测单速度分量。

口径说明：unpack 出的是核内码单位（rho 码 = n*mass*AMU/units.density，
v 码 = v/units.velocity，p 码 = p/(units.density*units.velocity^2)），
块长按 xmf_state_ncells 含 ghost，reshape 顺序 (nztot, nytot, nxtot)，
索引 (k,j,i)，i 最快——与 xmf_core.c 的 cidx 一致。
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU, KB

# 槽序 (n, vx, vy, vz, P) == 0..4（A2 契约）
SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

_PTR = ctypes.POINTER(ctypes.c_double)

# 接触间断用例的左右数密度（cm^-3）；左右压强取同一 float，保证 ∇p = 0 逐位成立
N_L, N_R = 1.0e13, 1.25e12
P0 = 13.806505  # dyn/cm^2 == N_L*KB*1e4，直接给 p 避免 n*KB*T 两侧舍入差
X0 = 5.5        # shock_tube 默认 x1 域 [1,10] 的中点
W0_TRANSVERSE = 5.0e-2  # 薄 2D 法向 x1 挤出携带的均匀横向 vy（码单位）


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（state/mesh 查询符号已绑定）。"""
    return load()


# --- 只读小助手 --------------------------------------------------------------


def _geom(run):
    """从 run 的已归一化 config 取 (ng, nx, ny, nz, nxtot, nytot, nztot)。"""
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _unpack(lib, run, var):
    """unpack 流体 0 槽 var 的整块（含 ghost），返回 flat 缓冲。"""
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, 0, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _unpack3(lib, run, var):
    """unpack 并按 (nztot, nytot, nxtot) 重塑（k,j,i 布局）。"""
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    return _unpack(lib, run, var).reshape((nzt, nyt, nxt)), (ng, nx, ny, nz)


def _axis(lib, run, d):
    """方向 d（0=x1, 1=x2）的体心坐标，含 ghost，长度 n_d+2ng。"""
    ntot = lib.xmf_mesh_ncell(run._h, d)
    buf = np.empty(ntot, dtype=np.float64)
    assert lib.xmf_mesh_copy_x(run._h, d, buf.ctypes.data_as(_PTR)) == ntot
    return buf


def _code_scales(run):
    """码单位 ↔ cgs 换算因子（按 run 的 units）。"""
    u = run._cfg.units
    mass = run._cfg.mass
    n_to_rho = mass * AMU / u.density
    return n_to_rho, u.density / (mass * AMU), u.velocity, u.density * u.velocity**2


def _contact_cfg(riemann: str) -> dict:
    """u=0、∇p=0 的密度台阶：左右压强逐位相等（直接给 p，不走 n*KB*T）。"""
    cfg = xmf.shock_tube()
    cfg["numerics"]["riemann"] = riemann
    cfg["ic"] = {
        "kind": "riemann",
        "x0": X0,
        "left": {"n": N_L, "v": 0.0, "p": P0},
        "right": {"n": N_R, "v": 0.0, "p": P0},
    }
    return cfg


def _iso_position(x_int, n_int, mid):
    """密度 mid 等值面位置：第一次下穿处线性内插（台阶初始在两胞元之间）。"""
    below = np.where(n_int < mid)[0]
    assert len(below) > 0, "剖面里应存在低于 mid 的胞（台阶还在）"
    i = int(below[0])
    assert i > 0, "等值面不应贴到最左胞"
    frac = (mid - n_int[i - 1]) / (n_int[i] - n_int[i - 1])
    return float(x_int[i - 1] + frac * (x_int[i] - x_int[i - 1]))


# --- 1. 接触间断不移动：HLLC（第三行原文第 1 条，卡片建议求解器） ------------


def test_contact_hllc_density_step_does_not_move(lib):
    """u=0、∇p=0 的密度台阶，hllc 跑 20 步：等值面位置零漂移、状态几乎不变。

    静止接触（SM=0）对 HLLC 是离散稳态：界面质量/能量通量为 0，动量通量两侧
    同为 p 相消。实测 20 步后逐位不变（漂移恰为 0）；断言给 1/4 胞元裕量。
    """
    run = xmf.setup(_contact_cfg("hllc"))
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        n2r, r2n, _, _ = _code_scales(run)
        dx1 = (run._cfg.x1max - run._cfg.x1min) / nx

        # IC 自检：压强逐位均匀（∇p=0），速度为 0
        p0, _ = _unpack3(lib, run, SLOT_P)
        assert float(np.max(p0) - np.min(p0)) == 0.0, "接触间断初值压强必须逐位均匀"

        n_before, _ = _unpack3(lib, run, SLOT_N)
        for _ in range(20):
            run.step()
        assert run.t > 0.0, "step 应真正推进时间"
        n_after, _ = _unpack3(lib, run, SLOT_N)
        vx, _ = _unpack3(lib, run, SLOT_VX)

        x = _axis(lib, run, 0)[ng : ng + nx]
        row = (ng, slice(ng, ng + ny), slice(ng, ng + nx))
        nb = n_before[row][0] if ny == 1 else n_before[row].mean(axis=0)
        na = n_after[row][0] if ny == 1 else n_after[row].mean(axis=0)
        mid = 0.5 * (N_L * n2r + N_R * n2r)  # 码单位的台阶中值
        iso0 = _iso_position(x, nb, mid)
        iso1 = _iso_position(x, na, mid)

        assert abs(iso0 - X0) <= 0.25 * dx1, f"初始等值面应就在 x0={X0} 处（得 {iso0}）"
        assert abs(iso1 - iso0) <= 0.25 * dx1, (
            f"hllc 静止接触 20 步后台阶中心漂移 {iso1 - iso0:+.3e}，超过 1/4 胞元"
        )
        # 稳态：密度几乎逐位不变（实测恰为 0）
        err = float(np.mean(np.abs(na - nb))) * r2n / N_L
        assert err <= 1.0e-9, f"hllc 静止接触应是离散稳态，L1(n) 变化 {err:.3e}"
        # 速度保持 0（没有假的压力不平衡驱动流动）
        v_int = vx[(ng, slice(ng, ng + ny), slice(ng, ng + nx))]
        assert float(np.max(np.abs(v_int))) <= 1.0e-12, "u=0 初值不得生出流动"
    finally:
        run.close()


# --- 2. 接触间断：HLL 允许抹平，中心不得漂 ------------------------------------


def test_contact_hll_smears_but_centre_holds(lib):
    """同一台阶换 hll：允许数值抹平（实测 L1(n)~6e-3），等值面漂移仍须 < 1 胞元。

    HLL 对静止接触有对称数值扩散（波速估计关于界面反对称），50% 等值面与
    质量心都钉在 x0。实测：漂移 -7.4e-3（≈0.2 胞元），质心偏移 ~7e-4。
    """
    run = xmf.setup(_contact_cfg("hll"))
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        n2r, r2n, _, _ = _code_scales(run)
        dx1 = (run._cfg.x1max - run._cfg.x1min) / nx

        n_before, _ = _unpack3(lib, run, SLOT_N)
        for _ in range(20):
            run.step()
        assert run.t > 0.0
        n_after, _ = _unpack3(lib, run, SLOT_N)

        x = _axis(lib, run, 0)[ng : ng + nx]
        row = (ng, slice(ng, ng + ny), slice(ng, ng + nx))
        nb = n_before[row][0] if ny == 1 else n_before[row].mean(axis=0)
        na = n_after[row][0] if ny == 1 else n_after[row].mean(axis=0)
        mid = 0.5 * (N_L * n2r + N_R * n2r)
        iso0 = _iso_position(x, nb, mid)
        iso1 = _iso_position(x, na, mid)

        assert abs(iso1 - iso0) <= 1.0 * dx1, (
            f"hll 允许抹平但台阶中心不得漂移：实测 {iso1 - iso0:+.3e}（dx1={dx1:.3e}）"
        )
        # 台阶仍完整存在（两端值未被抹到消失）
        assert float(np.max(na)) > mid > float(np.min(na)), "20 步后台阶应仍可分辨"
    finally:
        run.close()


# --- 3. 1D Sod 对 Toro 有上界（第三行原文第 2 条） ---------------------------


def test_sod_1d_l1_against_toro_bounded(lib):
    """默认 shock_tube()（hll/pcm/rk1）跑到 tstop，对 Toro 精确解报 L1/scale。

    口径：
      scale = (n_l, max|v_ex|, max p_ex)，L1 = mean|num-exact|/scale。
    实测基线 (3.515e-3, 9.716e-3, 4.029e-3)，与旧占位核逐位同值（无回退）；
    TODO「已完成」里的 ~1e-3 是旧树拉伸网格窗口 [1,2.2] 的另一口径。上界取
    ≈2× 实测基线：卡片说「与现 demo 同量级即可，不收紧」，但必须有上界。
    """
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.riemann import exact_hi

    cfg = xmf.shock_tube()
    run = xmf.setup(cfg)
    try:
        run.advance()
        assert run.t >= run.tstop, "advance 应跑到 tstop（可略过冲）"
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

        assert err_n < 8.0e-3, f"L1(n)/scale={err_n:.3e} 超出上界 8e-3（基线 3.5e-3）"
        assert err_v < 2.0e-2, f"L1(v)/scale={err_v:.3e} 超出上界 2e-2（基线 9.7e-3）"
        assert err_p < 8.0e-3, f"L1(P)/scale={err_p:.3e} 超出上界 8e-3（基线 4.0e-3）"
    finally:
        run.close()


# --- 4. 1D 闲置分量全程 0（第三行原文第 3 条） --------------------------------


def test_1d_idle_vy_vz_remain_zero_including_ghosts(lib):
    """1D Sod 跑到 tstop 后，槽 2/3（vy、vz）整块严格为 0——含 ghost 段。

    块长 = xmf_state_ncells = nxtot*nytot*nztot 含三向 ghost；x1 sweep 的
    切向槽只随质量通量平流（初值为 0 ⇒ 通量为 0），fill_bc 的 reflect 对
    切向槽不翻号，故全块逐位保持 0。
    """
    run = xmf.setup(xmf.shock_tube())
    try:
        run.advance()
        assert run.t >= run.tstop
        ncells = lib.xmf_state_ncells(run._h)
        for var, name in ((SLOT_VY, "vy"), (SLOT_VZ, "vz")):
            buf = _unpack(lib, run, var)
            assert buf.shape == (ncells,), "unpack 块长应含全部 ghost"
            assert np.all(buf == 0.0), (
                f"1D 闲置分量 {name} 在整块（含 ghost，ncells={ncells}）应严格为 0，"
                f"实测 max|{name}|={float(np.max(np.abs(buf))):.3e}"
            )
    finally:
        run.close()


# --- 5. 薄 2D 必须碰到 vy / x2 sweep（卡片第 4 条，禁止只测 VX1） ------------


def test_thin2d_x2_sweep_creates_vy_and_keeps_x1_uniform(lib):
    """薄 2D（8×32）沿 x2 放 Sod 台阶：10 步后 vy 非零、变化只来自 x2 方向。

    证据链：
      (a) 内区 max|vy| 明显非零（实测 ~0.74 码单位）——vy 只能由 x2 sweep
          产生（x1 sweep 只平流已有的切向分量，初值为 0）；
      (b) 每一 j 行内沿 x1 逐位一致（实测偏差恰为 0）⇒ 变化来自 x2 方向；
      (c) vx 保持 0（x2 sweep 里 vx 是切向，只随质量通量平流）；
      (d) 密度确实演化（L1 变化 > 1e-3，排除「空转也报非零」的假象）；
      (e) 闲置 vz 整块（含 ghost）逐位 0——nz=1 退化方向不扫、不影响结果
          （C2sup 维数合同的退化侧镜像断言）。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [8, 32, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        n2r, r2n, _, _ = _code_scales(run)
        y0 = 0.5 * (run._cfg.x2min + run._cfg.x2max)

        # 沿 x2 的台阶：x2 < y0 取左态（高压高密），否则右态；含 ghost 整块写
        p_l = N_L * KB * 1.0e4
        p_r = N_R * KB * 8.0e3
        y = _axis(lib, run, 1)
        m = (y < y0).reshape(1, nyt, 1)
        ones = np.ones((nzt, nyt, nxt))
        for var, vl, vr in (
            (SLOT_N, N_L * n2r, N_R * n2r),
            (SLOT_VX, 0.0, 0.0),
            (SLOT_VY, 0.0, 0.0),
            (SLOT_VZ, 0.0, 0.0),
            (SLOT_P, p_l, p_r),  # 默认单位下 unit_p = 1，cgs 值即码值
        ):
            arr = np.ascontiguousarray(np.where(m, vl, vr) * ones)
            assert lib.xmf_state_pack(run._h, 0, var, arr.ctypes.data_as(_PTR)) == arr.size

        n_before = _unpack(lib, run, SLOT_N).copy()
        for _ in range(10):
            run.step()
        assert run.t > 0.0

        n_after, _ = _unpack3(lib, run, SLOT_N)
        vx, _ = _unpack3(lib, run, SLOT_VX)
        vy, _ = _unpack3(lib, run, SLOT_VY)
        inner = (ng, slice(ng, ng + ny), slice(ng, ng + nx))
        vy_i = vy[inner]
        vx_i = vx[inner]
        n_b = n_before.reshape((nzt, nyt, nxt))[inner]
        n_a = n_after[inner]

        # (a) vy 真的被 x2 sweep 激发
        assert float(np.max(np.abs(vy_i))) > 5.0e-2, (
            f"薄 2D 沿 x2 的台阶 10 步后 vy 应明显非零，实测 max|vy|="
            f"{float(np.max(np.abs(vy_i))):.3e}（码单位）"
        )
        # (b) 沿 x1 各列一致：每一 j 行内 i 方向偏差为 0（变化只来自 x2）
        dev = float(np.max(np.abs(vy_i - vy_i.mean(axis=1, keepdims=True))))
        assert dev <= 1.0e-12, f"vy 应沿 x1 逐列一致（实测最大偏差 {dev:.3e}）"
        # (c) 切向 vx 不被无中生有
        assert float(np.max(np.abs(vx_i))) <= 1.0e-12, "x2 台阶不得生出 vx"
        # (d) 状态确实在演化
        dl1 = float(np.mean(np.abs(n_a - n_b))) * r2n / N_L
        assert dl1 > 1.0e-3, f"x2 sweep 应真正改变状态，L1(n) 变化仅 {dl1:.3e}"
        # (e) 闲置 vz 整块（含 ghost）逐位 0：退化方向（nz=1）不扫、不影响结果
        vz = _unpack(lib, run, SLOT_VZ)
        assert np.all(vz == 0.0), (
            f"薄 2D 闲置分量 vz 在整块（含 ghost，ncells={vz.size}）应严格为 0"
            f"（退化 x3 方向不扫），实测 max|vz|={float(np.max(np.abs(vz))):.3e}"
        )
    finally:
        run.close()


# --- 6. 薄 2D 法向 x1 挤出：横向 vy 只随质量通量携带（C2sup 维数合同） --------


def test_thin2d_x1_sweep_keeps_uniform_transverse_vy(lib):
    """薄 2D（32×8）沿 x1 放 Sod 台阶 + 均匀横向 vy：挤出真实、横向 vy 只随
    质量通量携带、闲置 vz 逐位 0、状态不随 j 变。

    与上一组互补：卡片允许挤出法向为 x 或 y，上一组法向 = x2（vy）；本组
    法向 = x1（vx），但同样禁止只测 VX1——在挤出之上叠一个非零均匀横向 vy，
    验证 x1 sweep「切向分量只随质量通量平流」（均匀切向场不产生通量散度，
    不被抹掉也不放大），x2 sweep 对 j 均匀态是 no-op。证据链：
      (a) 内区 max|vx| 明显非零（实测 ~0.57 码单位）——x1 sweep 真在多行
          （nx×ny）面上迭代；
      (b) 整块（含 ghost）|vy - W0| <= 1e-12（实测 ~6e-17，上界只给数值
          表示误差）——横向动量随质量通量携带，不被挤出抹掉或放大；
      (c) 闲置 vz 整块（含 ghost）严格为 0——nz=1 退化方向不扫；
      (d) 内区 n、vx、p 逐行一致（行偏差 <= 1e-12）⇒ 无跨行（j）混杂；
      (e) 密度确实演化（L1 变化 > 1e-3）。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [32, 8, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "reflect", "x2": "copy"}
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        n2r, r2n, _, _ = _code_scales(run)
        x0 = 0.5 * (run._cfg.x1min + run._cfg.x1max)

        # 沿 x1 的台阶 + 均匀横向 vy；含 ghost 整块写
        p_l = N_L * KB * 1.0e4
        p_r = N_R * KB * 8.0e3
        x = _axis(lib, run, 0)
        m = (x < x0).reshape(1, 1, nxt)
        ones = np.ones((nzt, nyt, nxt))
        for var, vl, vr in (
            (SLOT_N, N_L * n2r, N_R * n2r),
            (SLOT_VX, 0.0, 0.0),
            (SLOT_VY, W0_TRANSVERSE, W0_TRANSVERSE),
            (SLOT_VZ, 0.0, 0.0),
            (SLOT_P, p_l, p_r),  # 默认单位下 unit_p = 1，cgs 值即码值
        ):
            arr = np.ascontiguousarray(np.where(m, vl, vr) * ones)
            assert lib.xmf_state_pack(run._h, 0, var, arr.ctypes.data_as(_PTR)) == arr.size

        n_before = _unpack(lib, run, SLOT_N).copy()
        for _ in range(10):
            run.step()
        assert run.t > 0.0, "step 应真正推进时间"

        n_after, _ = _unpack3(lib, run, SLOT_N)
        vx, _ = _unpack3(lib, run, SLOT_VX)
        vy, _ = _unpack3(lib, run, SLOT_VY)
        p_after, _ = _unpack3(lib, run, SLOT_P)
        inner = (ng, slice(ng, ng + ny), slice(ng, ng + nx))
        vx_i = vx[inner]
        n_a = n_after[inner]
        n_b = n_before.reshape((nzt, nyt, nxt))[inner]

        # (a) x1 sweep 真的在挤出
        assert float(np.max(np.abs(vx_i))) > 5.0e-2, (
            f"薄 2D 沿 x1 的台阶 10 步后 vx 应明显非零，实测 max|vx|="
            f"{float(np.max(np.abs(vx_i))):.3e}（码单位）"
        )
        # (b) 均匀横向 vy 被质量通量原样携带（整块含 ghost）
        dev_vy = float(np.max(np.abs(vy - W0_TRANSVERSE)))
        assert dev_vy <= 1.0e-12, (
            f"均匀横向 vy 应被 x1 sweep 整块（含 ghost）携带不动，"
            f"实测 max|vy-W0|={dev_vy:.3e}"
        )
        # (c) 闲置 vz 逐位 0（退化方向不扫）
        vz = _unpack(lib, run, SLOT_VZ)
        assert np.all(vz == 0.0), (
            f"薄 2D 闲置分量 vz 在整块（含 ghost）应严格为 0，实测 max|vz|="
            f"{float(np.max(np.abs(vz))):.3e}"
        )
        # (d) 各 j 行彼此一致：无跨行混杂（内区）
        for name, arr in (("n", n_a), ("vx", vx_i), ("p", p_after[inner])):
            dev = float(np.max(np.abs(arr - arr.mean(axis=0, keepdims=True))))
            assert dev <= 1.0e-12, f"薄 2D x1 挤出状态不得随 j 变：{name} 最大行偏差 {dev:.3e}"
        # (e) 状态确实在演化
        dl1 = float(np.mean(np.abs(n_a - n_b))) * r2n / N_L
        assert dl1 > 1.0e-3, f"x1 sweep 应真正改变状态，L1(n) 变化仅 {dl1:.3e}"
    finally:
        run.close()


# --- 7. HD/MHD 分路：kind=mhd 在 setup 层拒绝（卡片第 3 条） -----------------


def test_mhd_kind_rejected_at_setup():
    """任何 fluids[].kind != 'hd' 都必须失败：不许静默当 HD，不许 B=0 偷梁。"""
    cfg = xmf.shock_tube()
    cfg["fluids"] = [{"name": "HI", "mass": 1.00794, "charge": 0, "kind": "mhd"}]
    with pytest.raises(ValueError, match="hd"):
        xmf.setup(cfg)
