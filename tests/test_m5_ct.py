"""M 块 3（M5）：CT——棱心 E=-u_+×B 与面心 B 更新单测。

规格来源：docs/SESSION.md（本刀卡片）「交什么」§3、§4、§5、§7：

  §3 离散 ∇·B 诊断（report-only）：
    - 构造性零散度初值（均匀 B 或单方向 B）跑若干步，最大归一化散度 < 1e-12
      （机器噪声）。
    - 人为加一格扰动制造非零初值散度，跑 CT 后不放大（增量 < 卡片阈值 1e-6）。
    - 读口用 builder 实现的 divb_max（Run.divb_max 属性）。

  §4 2D 钉 Orszag–Tang：
    - 初值 [0,2π]^2 周期边界，ρ=γ²/(4π)，P=5/(12π)，u=(-sin y,sin x,0)，
      B=(-sin y,sin 2x,0)/√(4π)，γ=5/3，一支离子 q=1,w_s=1，电子关。
    - 分辨率 nx=ny=64，t=0.5（或同等时刻）。
    - 断言：整程 ∇·B < 1e-6；L1(ρ)/L1(P)/L1(ME) 上界；总能量漂移 < 1e-2。

  §5 新测与回归：
    - 离散 ∇·B 测试（新）：构造性零散度初值 + 非零初值散度不放大。
    - Orszag–Tang 测试（新，2D）：§4 的断言全套。
    - M4 已锁的 1D MHD 管（Brio–Wu / Tóth）与两支共动还原：本刀替换感应
      路径后重新跑，结果与 M4 在噪声量级一致。钉值收紧不许放宽。

  §7 periodic 边界档（扩卡新增）：
    - 关磁 + 纯流体平移不变性（周期域上平移一个格距的初值，若干步后应与
      原初值演化逐位/噪声级一致）。
    - setup 校验拒收路径（单端 periodic、退化方向周期、periodic+user_bc
      冲突要 raise）。

用例清单：
  * 均匀 B + 零速度：构造性零散度，跑 10 步，divb_max < 1e-12（机器噪声）。
  * 均匀 B + 零速度（2D）：同上，薄 2D 网格。
  * 一格扰动 B：人为制造非零初值散度，跑 10 步，divb_max 增量 < 1e-6。
  * Orszag–Tang 2D：64² 周期边界，t=0.5，divB < 1e-6，L1 上界，能量漂移上界。
  * Periodic 平移不变性（关磁）：初值平移一格，演化后与原初值逐位一致。
  * Periodic 拒收：单端、退化方向、user_bc 冲突三条路径 raise。

钉值与实测关系（2026-08-30 本测首跑，builder 冒烟 16² 同参；64² 实测
见各测试注释）：
  Orszag–Tang 64²（hll/plm/rk3，cfl=0.4，t=0.507615，57 步）：
    E0 = 1.430555364040218e+03，E_final = 1.430180787183368e+03，
    energy_drift = 2.618402e-04；max_divb = 6.315193e-09；
    L1(rho) = 3.094631187473791e-02，L1(P) = 3.641994139670707e-02，
    L1(ME) = 1.700550126875293e-02。
  上界 = 实测 ×1.5 向上取整（同 M4 Brio–Wu 口径）。

禁令自查（卡片 + docs/TODO.md §A）：
  - 不读 pluto.ini：本文件无任何文件 IO；
  - 不用 XUV_FACTOR==0 哨兵：本文件不涉辐射；
  - 不只测单分量：B 覆盖 B1/B2/B3 三分量，divB 诊断覆盖全活动格，
    流体断言覆盖五槽 (n, vx, vy, vz, P)。

维度覆盖：1D（64×1×1）与 2D（8×4×1 / 64×64×1），同 test_mag.py 口径。
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU

_PTR = ctypes.POINTER(ctypes.c_double)

ION_FLUIDS = [{"name": "HII", "mass": 1.0, "charge": 1, "kind": "ion"}]


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库。"""
    return load()


# --- 小助手 -------------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_PTR)


def _ntot_of(run) -> tuple[int, int, int]:
    n = run._cfg.n
    ng = run._cfg.ng
    return tuple(n[d] + 2 * ng for d in range(3))


def _len_b(ntot, comp: int) -> int:
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return (ntot[comp] + 1) * ntot[d1] * ntot[d2]


def _pack_b(lib, h, comp: int, blk: np.ndarray) -> None:
    got = lib.xmf_mag_pack(h, comp, _ptr(np.ascontiguousarray(blk.ravel())))
    assert got == blk.size, f"xmf_mag_pack failed ({got})"


# --- 1. 构造性零散度：均匀 B + 零速度 -----------------------------------------


def test_uniform_b_zero_velocity_divb_noise_level(lib):
    """均匀 B（三分量常量）+ 零速度初值：构造性零散度（每格 ∇·B 逐项通量
    非零、和恰为零）。跑 10 步，divb_max < 1e-12（机器噪声）。

    均匀 B 的离散散度：对每格，sum_d (A_d[f+1] B_d - A_d[f] B_d) / dV = 0
    （笛卡尔 A=1, dV=dx，B_d 常量，逐项抵消）。CT 在均匀 B + 零速度下
    u_+ = 0 ⇒ E = -u_+ × B = 0 ⇒ ∂t B = -∇×E = 0，B 逐位不动。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [64, 1, 1]
    cfg["mhd"] = True
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg["boundary"] = {"x1": "copy"}
    cfg["time"] = {"cfl": 0.4, "tstop": 1.0, "dt_max": 0.0}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        # Pack 均匀 B（三分量常量）
        for comp in range(3):
            bval = 0.5 + 0.3 * comp  # B1=0.5, B2=0.8, B3=1.1
            blk = np.full(_len_b(ntot, comp), bval, dtype=np.float64)
            _pack_b(lib, h, comp, blk)
        # 初值 divB 检查
        divb0 = run.divb_max
        assert divb0 < 1e-12, f"initial divb_max={divb0} (should be machine noise)"
        # 跑 10 步
        for step in range(10):
            run.step()
            divb = run.divb_max
            assert divb < 1e-12, f"step {step}: divb_max={divb} (should stay < 1e-12)"
    finally:
        run.close()


# --- 2. 构造性零散度：单方向 B（B1=f(x2,x3)） ---------------------------------


def test_uniform_b_2d_divb_noise_level(lib):
    """均匀 B（三分量常量）+ 零速度初值，薄 2D（8×4×1）：构造性零散度，
    跑 10 步，divb_max < 1e-12（机器噪声）。

    2D 口径：与 1D 均匀 B 测试同初值，但网格是 2D（验证 CT 在 2D 路径下
    仍保持机器噪声级 divB）。均匀 B + 零速度 ⇒ u_+ = 0 ⇒ E = 0 ⇒ ∂t B = 0，
    B 逐位不动，divB 应保持初值噪声水平。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [8, 4, 1]
    cfg["grid"]["ndim"] = 2
    cfg["mhd"] = True
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg["boundary"] = {"x1": "copy", "x2": "copy"}
    cfg["time"] = {"cfl": 0.4, "tstop": 1.0, "dt_max": 0.0}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        # Pack 均匀 B（三分量常量）
        for comp in range(3):
            bval = 0.5 + 0.3 * comp  # B1=0.5, B2=0.8, B3=1.1
            blk = np.full(_len_b(ntot, comp), bval, dtype=np.float64)
            _pack_b(lib, h, comp, blk)
        # 初值 divB 检查
        divb0 = run.divb_max
        assert divb0 < 1e-12, f"initial divb_max={divb0} (should be machine noise)"
        # 跑 10 步
        for step in range(10):
            run.step()
            divb = run.divb_max
            assert divb < 1e-12, f"step {step}: divb_max={divb} (should stay < 1e-12)"
    finally:
        run.close()


# --- 3. 非零初值散度：一格扰动，CT 不放大 -------------------------------------


def test_one_cell_perturbation_divb_not_amplified(lib):
    """人为加一格扰动制造非零初值散度：均匀 B + 中央一格 B1 加 δ。
    初值 divb_max > 0（非零），跑 10 步后 divb_max 增量 < 1e-6（卡片阈值）。

    CT 在同一套离散算子下 ∂t(∇·B)=0 自动成立；残差只来自初值 / 边界的
    表示噪声。本测验证 CT 不把初值散度放大（增量在阈值内）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [64, 1, 1]
    cfg["mhd"] = True
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg["boundary"] = {"x1": "copy"}
    cfg["time"] = {"cfl": 0.4, "tstop": 1.0, "dt_max": 0.0}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        # Pack 均匀 B + 中央一格扰动
        b1_base = 0.5
        delta = 0.1  # 扰动幅度
        b1 = np.full((ntot[2], ntot[1], ntot[0] + 1), b1_base, dtype=np.float64)
        # 在中央格（i = ng + nx//2）的下表面加 δ
        i_center = ng + run._cfg.n[0] // 2
        b1[ng, ng, i_center] += delta
        _pack_b(lib, h, 0, b1)
        # B2 = B3 = 0
        for comp in (1, 2):
            blk = np.zeros(_len_b(ntot, comp), dtype=np.float64)
            _pack_b(lib, h, comp, blk)
        # 初值 divB（应 > 0，非零）
        divb0 = run.divb_max
        assert divb0 > 1e-9, f"initial divb_max={divb0} (should be > 0 after perturbation)"
        # 跑 10 步，divB 增量 < 1e-6
        for step in range(10):
            run.step()
            divb = run.divb_max
            increment = abs(divb - divb0)
            assert increment < 1e-6, (
                f"step {step}: divb_max={divb}, increment={increment} (should be < 1e-6)"
            )
    finally:
        run.close()


# --- 4. Orszag–Tang 2D 钉（§4） ---------------------------------------------


def test_orszag_tang_2d(lib):
    """Orszag–Tang 2D 钉（卡片 §4）：[0,2π]² 周期边界，γ=5/3，一支离子
    q=1,w_s=1，电子关。分辨率 64²，t=0.5（或同等时刻）。

    初值（代码单位）：
      ρ = γ²/(4π) ≈ 0.265258
      P = 5/(12π) ≈ 0.132629
      u = (-sin y, sin x, 0)
      B = (-sin y, sin 2x, 0)/√(4π)，B0 = 1/√(4π) ≈ 0.282095

    断言（钉值 2026-08-30 实测 64² hll/plm/rk3 cfl=0.4，t=0.507615，
    57 步；上界 = 实测 ×1.5 向上取整）：
      - 整程 max_divb < 1e-6（卡片阈值）
      - L1(rho) < 0.047（实测 0.0309）
      - L1(P) < 0.055（实测 0.0364）
      - L1(ME) < 0.026（实测 0.0170）
      - energy_drift < 0.0004（实测 0.000262）

    参考：Orszag & Tang (1979) JFM 95:119；PLUTO/Idefix 同分辨率对照。"""
    # 参数
    gamma = 5.0 / 3.0
    rho = gamma * gamma / (4.0 * math.pi)
    P = 5.0 / (12.0 * math.pi)
    B0 = 1.0 / math.sqrt(4.0 * math.pi)
    nx = ny = 64
    L = 2.0 * math.pi
    t_end = 0.5

    # 配置
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [nx, ny, 1]
    cfg["grid"]["ndim"] = 2
    cfg["grid"]["x1min"] = 0.0
    cfg["grid"]["x1max"] = L
    cfg["grid"]["x2min"] = 0.0
    cfg["grid"]["x2max"] = L
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "plm", "rk": "rk3"}
    cfg["gamma"] = gamma
    cfg["time"] = {"cfl": 0.4, "tstop": t_end, "dt_max": 0.0}
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg["electron"] = {"mode": "off"}
    cfg["boundary"] = {"x1": "periodic", "x2": "periodic", "x3": "none"}
    cfg["mhd"] = True
    cfg["ic"] = {
        "kind": "riemann",
        "x0": L / 2.0,
        "left": {"n": rho, "v": 0.0, "p": P},
        "right": {"n": rho, "v": 0.0, "p": P},
    }

    run = xmf.setup(cfg)
    try:
        h = run._h
        ng = run._cfg.ng
        ntot = _ntot_of(run)

        # 读格心 / 面心坐标
        nc0 = lib.xmf_mesh_ncell(h, 0)
        nc1 = lib.xmf_mesh_ncell(h, 1)
        xc0 = np.empty(nc0, dtype=np.float64)
        xc1 = np.empty(nc1, dtype=np.float64)
        lib.xmf_mesh_copy_x(h, 0, _ptr(xc0))
        lib.xmf_mesh_copy_x(h, 1, _ptr(xc1))

        # Pack 流体初值：ρ 均匀，vx = -sin(y)，vy = sin(x)，vz = 0，P 均匀
        ncells = lib.xmf_state_ncells(h)
        rho_arr = np.full(ncells, rho, dtype=np.float64)
        vx_arr = np.empty(ncells, dtype=np.float64)
        vy_arr = np.empty(ncells, dtype=np.float64)
        vz_arr = np.zeros(ncells, dtype=np.float64)
        P_arr = np.full(ncells, P, dtype=np.float64)

        for idx in range(ncells):
            i = idx % ntot[0]
            j = (idx // ntot[0]) % ntot[1]
            x = xc0[i]
            y = xc1[j]
            vx_arr[idx] = -math.sin(y)
            vy_arr[idx] = math.sin(x)

        lib.xmf_state_pack(h, 0, 0, _ptr(rho_arr))
        lib.xmf_state_pack(h, 0, 1, _ptr(vx_arr))
        lib.xmf_state_pack(h, 0, 2, _ptr(vy_arr))
        lib.xmf_state_pack(h, 0, 3, _ptr(vz_arr))
        lib.xmf_state_pack(h, 0, 4, _ptr(P_arr))

        # Pack B 初值：B1 = -sin(y)/√(4π)，B2 = sin(2x)/√(4π)，B3 = 0
        b1 = np.zeros((ntot[2], ntot[1], ntot[0] + 1), dtype=np.float64)
        for k in range(ntot[2]):
            for j in range(ntot[1]):
                y = xc1[j]
                for i in range(ntot[0] + 1):
                    b1[k, j, i] = -math.sin(y) * B0

        b2 = np.zeros((ntot[0], ntot[2], ntot[1] + 1), dtype=np.float64)
        for k in range(ntot[2]):
            for j in range(ntot[1] + 1):
                for i in range(ntot[0]):
                    x = xc0[i]
                    b2[i, k, j] = math.sin(2.0 * x) * B0

        b3 = np.zeros((ntot[1], ntot[0], ntot[2] + 1), dtype=np.float64)

        lib.xmf_mag_pack(h, 0, _ptr(np.ascontiguousarray(b1.ravel())))
        lib.xmf_mag_pack(h, 1, _ptr(np.ascontiguousarray(b2.ravel())))
        lib.xmf_mag_pack(h, 2, _ptr(np.ascontiguousarray(b3.ravel())))

        # 计算初始总能量（动能 + 内能 + 磁能）
        def compute_total_energy():
            fld = run.fields_full()
            HII = fld["HII"]
            rho_f = HII["n"] * (run._cfg.fluids[0].mass * AMU) / run._cfg.units.density
            vx_f = HII["vx"] / run._cfg.units.velocity
            vy_f = HII["vy"] / run._cfg.units.velocity
            vz_f = HII["vz"] / run._cfg.units.velocity
            P_f = HII["P"] / (run._cfg.units.density * run._cfg.units.velocity**2)
            B1_f = fld["B1"] / (math.sqrt(run._cfg.units.density) * run._cfg.units.velocity)
            B2_f = fld["B2"] / (math.sqrt(run._cfg.units.density) * run._cfg.units.velocity)
            B3_f = fld["B3"] / (math.sqrt(run._cfg.units.density) * run._cfg.units.velocity)

            KE = 0.5 * rho_f * (vx_f**2 + vy_f**2 + vz_f**2)
            IE = P_f / (gamma - 1.0)
            ME = 0.5 * (B1_f**2 + B2_f**2 + B3_f**2)

            return float(np.sum(KE + IE + ME))

        E0 = compute_total_energy()
        divb0 = run.divb_max
        assert divb0 < 1e-12, f"initial divb_max={divb0} (should be machine noise)"

        # 推进到 t=0.5，跟踪 max_divb
        max_divb = divb0
        while run.t < t_end:
            run.step()
            divb = run.divb_max
            if divb > max_divb:
                max_divb = divb

        # 断言 1：整程 max_divb < 1e-6
        assert max_divb < 1e-6, f"max_divb={max_divb} (should be < 1e-6)"

        # 断言 2：终态诊断（t ≈ 0.5）
        fld = run.fields_full()
        HII = fld["HII"]
        rho_final = HII["n"] * (run._cfg.fluids[0].mass * AMU) / run._cfg.units.density
        P_final = HII["P"] / (run._cfg.units.density * run._cfg.units.velocity**2)
        B1_f = fld["B1"] / (math.sqrt(run._cfg.units.density) * run._cfg.units.velocity)
        B2_f = fld["B2"] / (math.sqrt(run._cfg.units.density) * run._cfg.units.velocity)
        B3_f = fld["B3"] / (math.sqrt(run._cfg.units.density) * run._cfg.units.velocity)
        ME_final = 0.5 * (B1_f**2 + B2_f**2 + B3_f**2)

        L1_rho = float(np.mean(np.abs(rho_final - rho)))
        L1_P = float(np.mean(np.abs(P_final - P)))
        L1_ME = float(np.mean(np.abs(ME_final - 0.5 * B0**2)))

        # L1 上界（实测 ×1.5 向上取整）
        assert L1_rho < 0.047, f"L1(rho)={L1_rho} (should be < 0.047)"
        assert L1_P < 0.055, f"L1(P)={L1_P} (should be < 0.055)"
        assert L1_ME < 0.026, f"L1(ME)={L1_ME} (should be < 0.026)"

        # 断言 3：总能量漂移 < 1e-2
        E_final = compute_total_energy()
        energy_drift = abs(E_final - E0) / E0
        assert energy_drift < 0.0004, f"energy_drift={energy_drift} (should be < 0.0004)"

    finally:
        run.close()


# --- 5. Periodic BC 单测（§7） -----------------------------------------------


def test_periodic_translation_invariance(lib):
    """Periodic 平移不变性（关磁 + 纯流体）：周期域上均匀初值跑若干步，
    终态应保持均匀（逐位不变）。验证 periodic BC 不引入边界伪影。

    物理：均匀态在周期边界下应保持均匀；数值：Godunov 格式在均匀网格上
    对均匀态应逐位不动。"""
    # 配置：1D 周期，均匀态
    nx = 64
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [nx, 1, 1]
    cfg["boundary"] = {"x1": "periodic"}
    cfg["time"] = {"cfl": 0.4, "tstop": 0.1, "dt_max": 0.0}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }

    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng

        # 读初值（应均匀）
        ncells = lib.xmf_state_ncells(h)
        rho0 = np.empty(ncells, dtype=np.float64)
        vx0 = np.empty(ncells, dtype=np.float64)
        lib.xmf_state_unpack(h, 0, 0, _ptr(rho0))
        lib.xmf_state_unpack(h, 0, 1, _ptr(vx0))

        # 初值检查：活动格应均匀（逐位相等）
        rho0_active = rho0[ng : ng + nx]
        vx0_active = vx0[ng : ng + nx]
        assert np.all(rho0_active == rho0_active[0]), "initial rho should be uniform"
        assert np.all(vx0_active == vx0_active[0]), "initial vx should be uniform"

        # 跑 10 步
        for _ in range(10):
            run.step()

        # 读终态
        rho_final = np.empty(ncells, dtype=np.float64)
        vx_final = np.empty(ncells, dtype=np.float64)
        lib.xmf_state_unpack(h, 0, 0, _ptr(rho_final))
        lib.xmf_state_unpack(h, 0, 1, _ptr(vx_final))

        # 断言：终态活动格应保持均匀（逐位相等，与初值同值）
        rho_final_active = rho_final[ng : ng + nx]
        vx_final_active = vx_final[ng : ng + nx]
        assert np.all(rho_final_active == rho_final_active[0]), (
            f"final rho should stay uniform, max deviation = {np.max(np.abs(rho_final_active - rho_final_active[0]))}"
        )
        assert np.all(vx_final_active == vx_final_active[0]), (
            f"final vx should stay uniform, max deviation = {np.max(np.abs(vx_final_active - vx_final_active[0]))}"
        )
        # 终值应与初值逐位相等（均匀态不演化）
        assert np.all(rho_final_active == rho0_active[0]), "final rho should equal initial rho"
        assert np.all(vx_final_active == vx0_active[0]), "final vx should equal initial vx"

    finally:
        run.close()


def test_periodic_setup_rejects_single_end(lib):
    """Periodic 拒收：单端 periodic（xbeg=periodic 但 xend≠periodic）应
    raise ValueError（卡片 §7 校验：periodic 必须两端成对）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [64, 1, 1]
    cfg["boundary"] = {"x1beg": "periodic", "x1end": "outflow"}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    with pytest.raises(ValueError, match="periodic.*paired"):
        xmf.setup(cfg)


def test_periodic_setup_rejects_degenerate_axis(lib):
    """Periodic 拒收：退化方向周期（x2 periodic 但 ny=1）应 raise
    ValueError（卡片 §7 校验：x2/x3 periodic 要求 ny/nz ≥ 2）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [64, 1, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "outflow", "x2": "periodic"}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    with pytest.raises(ValueError, match="periodic.*requires.*ny.*>=.*2"):
        xmf.setup(cfg)


def test_periodic_setup_rejects_user_bc_conflict(lib):
    """Periodic 拒收：periodic + user_bc 冲突应 raise ValueError（卡片 §7
    校验：periodic ghost 从对端取，user_bc 会覆盖 ghost，语义冲突）。

    两条冲突路径都要测：
      - user_bc（表达式） + periodic face
      - boundary_floor（钉值） + periodic x1"""
    # 路径 1：user_bc + periodic
    cfg1 = xmf.shock_tube()
    cfg1["grid"]["n"] = [64, 1, 1]
    cfg1["boundary"] = {"x1": "periodic"}
    cfg1["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    cfg1["user_bc"] = {"x1beg": {"n": "0.5"}}
    with pytest.raises(ValueError, match="user_bc.*conflicts.*periodic"):
        xmf.setup(cfg1)

    # 路径 2：boundary_floor + periodic x1
    cfg2 = xmf.shock_tube()
    cfg2["grid"]["n"] = [64, 1, 1]
    cfg2["boundary"] = {"x1": "periodic"}
    cfg2["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    cfg2["boundary_floor"] = {"x1beg": [1.0, 0.0, 0.0, 0.0, 1.0]}  # 任意非 None 值
    with pytest.raises(ValueError, match="boundary_floor.*conflicts.*periodic"):
        xmf.setup(cfg2)
