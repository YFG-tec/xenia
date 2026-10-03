"""A4 CFL 契约单测（对照 docs/TODO.md → A4 第四行原文）。

第四行原文：
  单测 test_cfl：加密一倍，Δt 约减半。

卡片（docs/SESSION.md）「交什么」第 1 条写死的口径（本文件按此断言）：
  Δt = cfl / Σ_d (s_d / Δx_d)，s_d = 全域内区 max(|v_d| + c_s)；
  **约定：n_d == 1 的方向不进求和**（卡片推荐并写死的那种）——于是 1D 恰退化为
  旧式 `cfl*dx1/s`；`dt_max > 0` 时封顶。x2/x3 只有在 n_d > 1 时才贡献。
  该约定在下方用例 2/3 的断言里直接检验（1D 与旧式严格相等；2D 恰为两项之和，
  若把 nz==1 的 z 也算进去，等式即破坏）。

「加密一倍 Δt 约减半」的口径说明：只加密有长度的方向（x1）；波速取全域
max(|v|+c)，同样初值下两分辨率的 s 相同，比值理论上恰为 0.5，卡片允许 ~10%
相对裕量（防波速取 max 的口径差异），本文件按 10% 断言并写明。
"""
from __future__ import annotations

import math

import pytest

import xmf
from xmf.config import AMU

GAMMA = 5.0 / 3.0
MASS = 1.00794
UD, UV = 1.0e-12, 1.0e6  # shock_tube 默认 units（density, velocity）

# 均匀态用例的初值（左右相同）：密度 1e13 cm^-3、压强 13.806505、x 方向流速 3e5
N_U, V_U, P_U = 1.0e13, 3.0e5, 13.806505


def _uniform_cfg(ndim=1, n=None) -> dict:
    """左右态全同的均匀笛卡尔配置（波速可解析，CFL 可逐位对照）。"""
    cfg = xmf.shock_tube()
    if n is not None:
        cfg["grid"]["n"] = list(n)
    cfg["grid"]["ndim"] = ndim
    state = {"n": N_U, "v": V_U, "p": P_U}
    cfg["ic"] = {"kind": "riemann", "x0": 5.5, "left": dict(state), "right": dict(state)}
    return cfg


def _uniform_speeds(run):
    """按核内口径重算均匀态的每方向波速：s_d = |v_d| + c_s（码单位）。

    与 xmf_core.c max_signal_speed 的运算顺序一致：先 gamma*p/rho 再 sqrt，
    均匀态下全域相同。vx = V_U/UV，vy = vz = 0（ic 的 v 只给 x1 法向）。
    """
    # 默认单位下 unit_p = density*velocity^2 = 1，cgs 压强值即码单位压强值
    rho = N_U * MASS * AMU / UD
    cs = math.sqrt(GAMMA * P_U / rho)
    vx = V_U / UV
    s1 = abs(vx) + cs
    s2 = cs  # vy = 0
    s3 = cs  # vz = 0
    return s1, s2, s3


# --- 1. 加密一倍，首步 Δt 约减半（第四行原文） --------------------------------


def test_refining_active_direction_halves_first_dt():
    """只加密有长度的方向（x1：256→512），首步 Δt 比值 ≈ 0.5（±10%）。

    首步 Δt 取 setup 后的 run.dt（step 开头按同一状态重算，二者相同）；
    并顺手断言第一步确实以该 dt 推进（t1 == dt0 逐位）。
    """
    cfg_a = xmf.shock_tube()
    cfg_b = xmf.shock_tube()
    cfg_b["grid"]["n"][0] = 512  # 只加密 x1

    run_a = xmf.setup(cfg_a)
    run_b = xmf.setup(cfg_b)
    try:
        dt_a, dt_b = run_a.dt, run_b.dt
        assert dt_a > 0.0 and dt_b > 0.0, "setup 后应已算出首步 dt"
        ratio = dt_b / dt_a
        assert abs(ratio - 0.5) <= 0.05, (
            f"nx 256→512 后首步 dt 应减半（±10%），实测比值 {ratio:.4f}"
            f"（dt256={dt_a:.6e}, dt512={dt_b:.6e}）"
        )
        # 第一步确实使用该 dt（t 从 0 推进恰一个 dt）
        run_a.step()
        assert run_a.t == dt_a, f"首步推进量 {run_a.t:.12e} 应等于首步 dt {dt_a:.12e}"
    finally:
        run_a.close()
        run_b.close()


# --- 2. 1D 与旧式公式严格一致（n_d==1 不进求和的直接证据） --------------------


def test_1d_dt_is_exactly_classic_cfl_dx_over_s():
    """均匀 1D：run.dt 与旧式 cfl*dx1/s 逐位一致（相对误差 1e-12 内）。

    这直接检验「n_d==1 方向不进求和」的约定：ny=nz=1 时三项求和退化为单项，
    Δt = cfl/(s1/dx1) == cfl*dx1/s1，与 A4 之前的占位公式完全相同。
    """
    cfg = _uniform_cfg(ndim=1, n=(256, 1, 1))
    run = xmf.setup(cfg)
    try:
        s1, _, _ = _uniform_speeds(run)
        dx1 = (cfg["grid"]["x1max"] - cfg["grid"]["x1min"]) / 256
        expected = cfg["time"]["cfl"] / (s1 / dx1)  # 与 C 的 inv 累加同序
        assert run.dt == pytest.approx(expected, rel=1.0e-12), (
            f"1D 首步 dt={run.dt:.12e} 应严格等于 cfl*dx1/s={expected:.12e}"
        )
    finally:
        run.close()


# --- 3. 2D：两项求和，退化方向不进 --------------------------------------------


def test_2d_dt_sums_only_active_directions():
    """均匀 2D（16×16，nz=1）：dt == cfl/(s1/dx1 + s2/dx2)，s_d = |v_d| + c。

    - x2 项必须真的进了求和（否则 dt 会大一个 dx2 因子倍）；
    - nz==1 的 z 不进：若误加入（dx3 按 (x3max-x3min)/1），等式即破坏；
    - 每方向波速用该方向速度分量（vx≠0 ⇒ s1 > s2），口径同核内。
    """
    cfg = _uniform_cfg(ndim=2, n=(16, 16, 1))
    cfg["grid"]["x2min"], cfg["grid"]["x2max"] = 0.0, 1.0
    run = xmf.setup(cfg)
    try:
        s1, s2, _ = _uniform_speeds(run)
        dx1 = (cfg["grid"]["x1max"] - cfg["grid"]["x1min"]) / 16
        dx2 = (cfg["grid"]["x2max"] - cfg["grid"]["x2min"]) / 16
        expected = cfg["time"]["cfl"] / (s1 / dx1 + s2 / dx2)  # 与 C 的 d=0,1 累加同序
        assert run.dt == pytest.approx(expected, rel=1.0e-12), (
            f"2D 首步 dt={run.dt:.12e} 应为 cfl/(s1/dx1+s2/dx2)={expected:.12e}"
        )
        # 判别性：x2 项确实压低了 dt（只算 x1 的旧式会大 ~8×）
        dt_x1_only = cfg["time"]["cfl"] / (s1 / dx1)
        assert run.dt < 0.5 * dt_x1_only, (
            "x2 方向应真实进入 CFL 求和（n_d>1 的方向不得漏）"
        )
    finally:
        run.close()


# --- 4. A1 补：CFL 只读逐格度量（非均匀 x1 由最小格定 dt） --------------------


def test_nonuniform_x1_dt_set_by_smallest_cell():
    """均匀态 + 非均匀 x1：run.dt == cfl/(s1/dx_min)，逐位口径（rel 1e-12）。

    均匀态每格 s 相同，max_cell Σ_d s/dx 在最小宽格取到；若误用平均宽或
    全域标量 dx，dt 会偏 (dx_mean/dx_min)=2.75 倍，此钉必红。口径同卡片
    「交什么」§3：Δt = cfl / max_cell Σ_d (s_{d,cell}/dx_{d,cell})。
    """
    faces = [1.0, 1.5, 2.25, 3.25, 4.5, 6.0, 7.75, 9.75, 12.0]  # 宽度 0.5..2.25
    cfg = _uniform_cfg(ndim=1, n=(8, 1, 1))
    cfg["grid"]["x1min"] = faces[0]
    cfg["grid"]["x1max"] = faces[-1]
    cfg["grid"]["x1"] = list(faces)
    run = xmf.setup(cfg)
    try:
        s1, _, _ = _uniform_speeds(run)
        widths = [b - a for a, b in zip(faces[:-1], faces[1:])]
        dx_min = min(widths)
        expected = cfg["time"]["cfl"] / (s1 / dx_min)
        assert run.dt == pytest.approx(expected, rel=1.0e-12), (
            f"非均匀 1D 首步 dt={run.dt:.12e} 应为 cfl/(s1/dx_min)={expected:.12e}"
            f"（dx_min={dx_min}）"
        )
        # 判别性：平均宽口径会偏大，证明 dt 确实由最小格定
        dx_mean = (faces[-1] - faces[0]) / len(widths)
        dt_mean_based = cfg["time"]["cfl"] / (s1 / dx_mean)
        assert expected < 0.5 * dt_mean_based, "用例本身要能区分最小格与平均宽"
    finally:
        run.close()


# --- 5. dt_max 封顶 ------------------------------------------------------------


def test_dt_max_caps_dt():
    """配 dt_max=1e-3（远小于 CFL dt≈1.2e-2）：run.dt 恰为 dt_max，步长按它走。"""
    cap = 1.0e-3
    cfg = xmf.shock_tube()
    cfg["time"]["dt_max"] = cap
    run = xmf.setup(cfg)
    try:
        # 先确认未封顶时的 CFL dt 确实更大（否则本用例无判别力）
        ref = xmf.setup(xmf.shock_tube())
        assert ref.dt > 2.0 * cap, "封顶用例要求 CFL dt 本身显著大于 dt_max"
        ref.close()

        assert run.dt == cap, f"setup 后 dt={run.dt:.6e} 应恰为 dt_max={cap:.1e}"
        run.step()
        assert run.t == cap, f"首步推进量 {run.t:.6e} 应恰为 dt_max={cap:.1e}"
        assert run.dt == cap, f"下一步 dt={run.dt:.6e} 仍应被 dt_max 封顶"
    finally:
        run.close()
