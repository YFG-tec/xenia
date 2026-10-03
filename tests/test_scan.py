"""A5 辐射扫描合同单测（test_scan）：RadiationScan 模式 1（grid_line）。

合同来自本刀卡片（docs/SESSION.md §交什么 2/6）与 docs/cmd/PROMPT_A5.md §6：
  1. 真空 F 常数：无吸收体（rho=0）与 n→地板（1e-30）时，各格各束
     F_b == F∞_b 逐位。
  2. 均匀 κ 指数衰减：单束均匀数密度，沿内向
     F_i = F∞·exp(−σ n Δs·(外向内累计格数))；Δs 用 mesh 的 Δr 按厘米算
     （球面，units 全 1 → 码值即厘米）。
  3. nx2=3 三条径向射线互不串（nx1≥8）：中间射线给不同吸收，两侧不受影响。
  4. 厚格光子守恒：Σ_i ΔF_i = F∞(1−e^{−τ_total})；无专门 ΔF 读口，用
     透射后 F 的望远镜和 ΔF_i = F_{i+1} − F_i（外→内，F_{nx} := F∞，
     索引方向已对照 xmf_scan.c：i 从 ng+nx−1 递减到 ng，flux[c] 存透射后）。
  5. 53 路互不串：每束各自的 σ 独立衰减，含一束 σ=0（逐位 = F∞）。
  6. 关辐射一步与纯 hydro 一致：radiation 表加载但 enabled=false（F∞=0）
     与「整表不加载」的同配置（重力+复合两边同开）一步 step 后逐位相同。
  7. B3 多吸收体（追加辅助口 xmf_rad_load_absorbers）：
     a) 两吸收体 κ 相加：tau_b = (σ_H,b n_H + σ_He,b n_He) Δs 逐格对解析；
        且与两条单吸收体路径交叉：F_AB == F_A·F_B/F∞（同一均匀介质）；
     b) 53 路 × 两吸收体不串：每束按自己那行 σ 独立衰减，nx2>1 各径向
        射线仍互不串；
     c) 配置层：normalize() 不再静默丢 absorbers[1:]——两吸收体给行表
        （每行 bins.n 个），载荷带全部下标与行主序 σ；扁平行数不符、
        未知组分名一律拒。单吸收体旧路径（扁平行）不受影响。

扫描语义（xmf_scan.c 锁）：tau_b = sigma_b*n_abs*ds，ds = dx*unit_length；
F_{b,i} = F_{b,i+1}·exp(−tau_{b,i})，F_in = F∞ 从 x1end 外侧面注入；
n_abs = rho_abs/mass_code（负数裁 0）。scan 在每个 RK 阶的 hydro 之前跑；
本文件全部用 rk1（一步一扫描），密度均匀或零时多阶也不会污染。

口径：unpack/flux 块布局 (nztot, nytot, nxtot)，索引 (k,j,i)，i 最快；
flux() 吐内区 (nbeam, nz, ny, nx)。n_abs 的期望值按核内同一条算式复现：
rho = n_target*m_code 打包，n_abs = rho/m_code（同一对 double）。

禁令自查：不读 pluto.ini（束表是测试自带的合成数，不读旧树）；不用
XUV_FACTOR==0 哨兵（关辐射用 enabled=false + 表内 F∞=0，照工厂合同）；
不只测 VX1——本文件测的是束通量场，附带钉密度不变与五槽逐位比对。
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU, KB

_PTR = ctypes.POINTER(ctypes.c_double)

# 槽序 (n, vx, vy, vz, P) == 0..4（A2 契约）
SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

# 合成束表常数（刻意非 1、非物理表值，避免与任何换算碰巧重合）
SIGMA_1 = 2.0e-18     # cm^2
HNU_1 = 2.0e-11       # erg
FINF_1 = 4.5          # erg/cm^2/s
MASS_AMU = 1.00794


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（rad/state/mesh 符号已绑定）。"""
    return load()


# --- 只读小助手 --------------------------------------------------------------


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _dx1(lib, run) -> np.ndarray:
    """x1 格宽（含 ghost），长度 nx+2ng；scan 用的就是它乘 unit_length。"""
    ntot = lib.xmf_mesh_ncell(run._h, 0)
    buf = np.empty(ntot, dtype=np.float64)
    assert lib.xmf_mesh_copy_dx(run._h, 0, buf.ctypes.data_as(_PTR)) == ntot
    return buf


def _mass_code(run, f: int) -> float:
    """mass_code = mass_amu*XMF_AMU/unit_density（xmf_phys_load 的换算）。"""
    return run._cfg.fluids[f].mass * AMU / run._cfg.units.density


def _pack_block(lib, run, f: int, var: int, block: np.ndarray) -> None:
    """整块（含 ghost）pack 进流体 f 槽 var。"""
    ncells = lib.xmf_state_ncells(run._h)
    data = np.ascontiguousarray(block, dtype=np.float64).ravel()
    assert data.size == ncells
    assert lib.xmf_state_pack(run._h, f, var, data.ctypes.data_as(_PTR)) == ncells


def _unpack(lib, run, f: int, var: int) -> np.ndarray:
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, f, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _scan_cfg(*, sigma, hnu, finf, geometry="spherical", nx=16, ny=1,
              x1min=1.0e10, x1max=1.8e10, enabled=True):
    """单吸收体（HI = 流体 0）的扫描用例：无 sources，ic.kind='none'，
    units 全 1（码值即 cgs），rk1（一步恰一次扫描）。"""
    nb = len(sigma)
    assert len(hnu) == nb and len(finf) == nb
    cfg = xmf.shock_tube()
    cfg["grid"]["geometry"] = geometry
    cfg["grid"]["n"] = [nx, ny, 1]
    cfg["grid"]["ndim"] = 2 if ny > 1 else 1
    cfg["grid"]["x1min"] = x1min
    cfg["grid"]["x1max"] = x1max
    if ny > 1:
        cfg["grid"]["x2min"] = 0.0
        cfg["grid"]["x2max"] = 1.0
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["fluids"] = [{"name": "HI", "mass": MASS_AMU, "charge": 0, "kind": "hd"}]
    cfg["sources"] = []
    cfg["radiation"] = {
        "enabled": enabled,
        "absorbers": ["HI"],
        "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
        "bins": {
            "n": nb,
            "sigma": [float(v) for v in sigma],
            "hnu": [float(v) for v in hnu],
            "flux_inf": [float(v) for v in finf],
        },
    }
    cfg["ic"] = {"kind": "none"}
    return cfg


def _expected_transmitted(finf_b: float, sigma_b: float, n_abs: np.ndarray,
                          ds: np.ndarray) -> np.ndarray:
    """逐字复现 xmf_rad_scan 的行走（外→内，透射后存）：
        f = F∞; for i = nx-1 .. 0: tau = (sigma*n_abs)*ds; f *= exp(-tau)
    返回每格透射后 F（内下标 0..nx-1）。"""
    nx = len(n_abs)
    f = float(finf_b)
    out = np.empty(nx, dtype=np.float64)
    for iq in range(nx - 1, -1, -1):
        tau = (float(sigma_b) * float(n_abs[iq])) * float(ds[iq])
        f *= math.exp(-tau)
        out[iq] = f
    return out


# --- 1. 真空 / 地板密度：各格各束 F == F∞ 逐位（合同第 1 条） -----------------


@pytest.mark.parametrize("n_abs_target", [0.0, 1.0e-30])
def test_vacuum_flux_is_finf_bitwise(lib, n_abs_target):
    """无吸收体（rho=0）与 n→地板（1e-30）：tau≈0 → 各格各束 F == F∞ 逐位。"""
    finf = [3.75, 5.5, 7.25]  # 三束，各不相同
    sigma = [1.0e-18, 2.5e-18, 0.0]
    hnu = [1.0e-11, 2.0e-11, 4.0e-11]
    cfg = _scan_cfg(sigma=sigma, hnu=hnu, finf=finf, nx=8)
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.nbeam == 3, "束表应加载为 3 束"

        m = _mass_code(run, 0)
        rho_val = n_abs_target * m  # n→地板 / 0 仍按核内口径走 mass_code
        block = np.full((nzt, nyt, nxt), rho_val, dtype=np.float64)
        _pack_block(lib, run, 0, SLOT_N, block)

        run.step()  # scan -> hydro -> source（无带子）；scan 读打包密度
        assert run.t > 0.0, "step 应真正推进时间"

        flx = run.flux()
        assert flx.shape == (3, nz, ny, nx)
        for b in range(3):
            expect = np.full((nz, ny, nx), finf[b], dtype=np.float64)
            assert np.array_equal(flx[b], expect), (
                f"束 {b}：真空/地板密度下每格透射通量应逐位等于 F∞="
                f"{finf[b]!r}，实测最大差 {float(np.max(np.abs(flx[b] - expect))):.3e}"
            )
    finally:
        run.close()


# --- 2. 均匀 κ：F_i = F∞·exp(−σ n Δs·(外向内累计格数))（合同第 2 条） ---------


def test_uniform_kappa_exp_decay_spherical_cm(lib):
    """球面 1D、units 全 1：Δs = mesh Δr 按厘米；均匀数密度单束衰减对解析。"""
    nx = 16
    x1min, x1max = 1.0e10, 1.8e10  # cm；dx = 5e8 cm
    n0 = 2.0e8                      # cm^-3
    cfg = _scan_cfg(sigma=[SIGMA_1], hnu=[HNU_1], finf=[FINF_1],
                    nx=nx, x1min=x1min, x1max=x1max)
    run = xmf.setup(cfg)
    try:
        ng, nx_, ny, nz, nxt, nyt, nzt = _geom(run)
        assert nx_ == nx
        m = _mass_code(run, 0)
        rho0 = n0 * m
        P0 = n0 * KB * 1500.0  # T=1500 K，防密度/压强掉进 hydro 地板

        rho_block = np.full((nzt, nyt, nxt), rho0, dtype=np.float64)
        p_block = np.full((nzt, nyt, nxt), P0, dtype=np.float64)
        _pack_block(lib, run, 0, SLOT_N, rho_block)
        _pack_block(lib, run, 0, SLOT_P, p_block)

        # Δs 钉法：核读回的 mesh dx（球面仍沿 x1 走），乘 unit_length=1 即厘米；
        # 并与配置网格交叉核对（均匀）。
        dx_full = _dx1(lib, run)
        dx_cfg = (x1max - x1min) / nx
        assert np.allclose(dx_full, dx_cfg, rtol=1.0e-14, atol=0.0), (
            "mesh dx 应与配置均匀网格一致（Δs 的厘米口径从这里来）"
        )
        ds = dx_full[ng : ng + nx] * run._cfg.units.length

        n_abs = np.full(nx, rho0 / m, dtype=np.float64)  # 与核内同一对 double
        run.step()
        assert run.t > 0.0

        # 均匀密度在 v=0 下必须被 hydro 逐位保持（质量通量为 0）——
        # 保证扫描看到的正是打包密度，解析式才有意义
        rho_after = _unpack(lib, run, 0, SLOT_N).reshape((nzt, nyt, nxt))
        assert np.array_equal(rho_after[ng, ng, ng : ng + nx],
                              np.full(nx, rho0)), "均匀密度一步后应逐位不变"

        flx = run.flux()[0, 0, 0, :]
        # (a) 对逐字复现核内行走的乘积（同序浮点，应几乎逐位）
        expect_walk = _expected_transmitted(FINF_1, SIGMA_1, n_abs, ds)
        rel_walk = float(np.max(np.abs((flx - expect_walk) / expect_walk)))
        assert rel_walk <= 1.0e-12, (
            f"透射通量与核内行走复现的最大相对差 {rel_walk:.3e} 超过 1e-12"
        )
        # (b) 对闭式解析：F_i = F∞·exp(−σ n Δs·(外向内累计格数))
        tau = (SIGMA_1 * n_abs[0]) * ds[0]
        cum = nx - np.arange(nx, dtype=np.float64)  # 外→内累计格数（含本格）
        expect_closed = FINF_1 * np.exp(-tau * cum)
        rel = float(np.max(np.abs((flx - expect_closed) / expect_closed)))
        assert rel <= 1.0e-12, (
            f"均匀 κ 指数衰减对闭式解析的最大相对差 {rel:.3e} 超过 1e-12"
        )
        # 形状哨兵：向内严格单调衰减，最外格恰为一个 τ 的衰减
        assert np.all(np.diff(flx) > 0.0), "向内（下标增大→外）通量应递增"
        assert abs(flx[-1] / FINF_1 - math.exp(-tau)) <= 1.0e-12
    finally:
        run.close()


# --- 3. nx2=3 三条径向射线互不串（合同第 3 条） --------------------------------


def test_nx2_three_radial_rays_no_crosstalk(lib):
    """n=[8,3,1]：三条 x1 射线各带不同均匀吸收；每条只对解析自己的 τ，
    中间射线的特殊吸收不得渗进两侧（互不串）。"""
    nx, ny = 8, 3
    x1min, x1max = 1.0e10, 1.8e10
    n0 = 1.0e8
    cfg = _scan_cfg(sigma=[SIGMA_1], hnu=[HNU_1], finf=[FINF_1],
                    geometry="cartesian", nx=nx, ny=ny,
                    x1min=x1min, x1max=x1max)
    run = xmf.setup(cfg)
    try:
        ng, nx_, ny_, nz, nxt, nyt, nzt = _geom(run)
        assert (nx_, ny_) == (nx, ny)
        m = _mass_code(run, 0)
        dx = (x1max - x1min) / nx
        P0 = n0 * KB * 1500.0

        # 每条射线一个吸收密度：中间射线刻意不同（3 倍）
        n_ray = np.array([n0, 3.0 * n0, n0])
        rho_block = np.zeros((nzt, nyt, nxt), dtype=np.float64)
        p_block = np.zeros((nzt, nyt, nxt), dtype=np.float64)
        for jj in range(ny):
            rho_block[ng, ng + jj, :] = n_ray[jj] * m
            p_block[ng, ng + jj, :] = max(n_ray[jj], n0) * KB * 1500.0
        _pack_block(lib, run, 0, SLOT_N, rho_block)
        _pack_block(lib, run, 0, SLOT_P, p_block)

        run.step()
        assert run.t > 0.0

        flx = run.flux()[0]  # (nz, ny, nx)
        assert flx.shape == (nz, ny, nx)
        ds = np.full(nx, dx * run._cfg.units.length)
        expect = []
        for jj in range(ny):
            n_abs = np.full(nx, (n_ray[jj] * m) / m)
            expect.append(_expected_transmitted(FINF_1, SIGMA_1, n_abs, ds))
        for jj in range(ny):
            got = flx[0, jj, :]
            rel = float(np.max(np.abs((got - expect[jj]) / expect[jj])))
            assert rel <= 1.0e-12, (
                f"射线 j={jj}：通量应对自己的 τ 解析（n={n_ray[jj]:.1e}），"
                f"最大相对差 {rel:.3e}——若串线则必偏"
            )
        # 两侧射线不受中间影响：与「三条全同密度」的参考解逐位一致
        n_abs_ref = np.full(nx, (n0 * m) / m)
        ref = _expected_transmitted(FINF_1, SIGMA_1, n_abs_ref, ds)
        for jj in (0, 2):
            assert np.array_equal(flx[0, jj, :], ref), (
                f"射线 j={jj} 应与未受扰参考解逐位相同（中间射线的特殊吸收"
                "不得渗入）"
            )
        # 中间射线确实不同（防空对空）
        assert not np.allclose(flx[0, 1, :], ref, rtol=1.0e-6), (
            "中间射线吸收更强，通量应明显更低"
        )
        assert flx[0, 1, 0] < flx[0, 0, 0], "中间射线更吸收 → 最内格通量更低"
    finally:
        run.close()


# --- 4. 厚格光子守恒（合同第 4 条） -------------------------------------------


def test_thick_cell_photon_conservation(lib):
    """非均匀密度、两束：望远镜和 Σ_i ΔF_i（ΔF_i = F_{i+1}−F_i，外→内，
    F_{nx}:=F∞）= F∞ − F_0 = F∞(1−e^{−τ_total})（解析到 1e-12）。"""
    nx = 16
    x1min, x1max = 1.0e10, 1.8e10
    n0 = 2.0e8
    sigma = [SIGMA_1, 3.0e-18]
    hnu = [HNU_1, 4.0e-11]
    finf = [FINF_1, 2.25]
    cfg = _scan_cfg(sigma=sigma, hnu=hnu, finf=finf,
                    nx=nx, x1min=x1min, x1max=x1max)
    run = xmf.setup(cfg)
    try:
        ng, nx_, ny, nz, nxt, nyt, nzt = _geom(run)
        m = _mass_code(run, 0)
        dx_full = _dx1(lib, run)
        ds = dx_full[ng : ng + nx] * run._cfg.units.length
        P0 = n0 * KB * 1500.0

        # 沿射线变化的密度：n_iq = n0*(1 + 0.07*iq)
        iq = np.arange(nx, dtype=np.float64)
        n_line = n0 * (1.0 + 0.07 * iq)
        rho_block = np.zeros((nzt, nyt, nxt), dtype=np.float64)
        rho_block[ng, ng, ng : ng + nx] = n_line * m
        p_block = np.full((nzt, nyt, nxt), P0, dtype=np.float64)
        _pack_block(lib, run, 0, SLOT_N, rho_block)
        _pack_block(lib, run, 0, SLOT_P, p_block)

        run.step()
        assert run.t > 0.0

        n_abs = (n_line * m) / m  # 与核内同一对 double
        flx = run.flux()  # (2, nz, ny, nx)
        for b in range(2):
            F = flx[b, 0, 0, :]
            # 望远镜和（外→内）：ΔF_iq = F_{iq+1} − F_iq，F_{nx} := F∞
            dF = np.empty(nx, dtype=np.float64)
            for iqq in range(nx):
                f_outer = finf[b] if iqq == nx - 1 else F[iqq + 1]
                dF[iqq] = f_outer - F[iqq]
            assert np.all(dF >= 0.0), f"束 {b}：每格吸收量 ΔF 应非负"
            s_loop = 0.0
            for v in dF:
                s_loop += v  # 顺序累加，与卡片求和口径一致

            # (a) 望远镜和 == F∞ − F_0（浮点顺序误差内）
            assert abs(s_loop - (finf[b] - F[0])) <= 1.0e-13 * finf[b], (
                f"束 {b}：ΣΔF 应望远镜收缩为 F∞−F_0"
            )
            # (b) 光子守恒对闭式解析：ΣΔF = F∞(1−e^{−τ_total})
            tau_tot = 0.0
            for iqq in range(nx):
                tau_tot += (sigma[b] * n_abs[iqq]) * ds[iqq]
            expect = finf[b] * (1.0 - math.exp(-tau_tot))
            rel = abs(s_loop - expect) / expect
            assert rel <= 1.0e-12, (
                f"束 {b}：ΣΔF={s_loop:.6e} 对解析 F∞(1−e^−τ)={expect:.6e} "
                f"相对差 {rel:.3e} 超过 1e-12"
            )
            # (c) 与核内行走复现一致（吸收份额非零，防 τ→0 空转）
            walk = _expected_transmitted(finf[b], sigma[b], n_abs, ds)
            assert float(np.max(np.abs((F - walk) / walk))) <= 1.0e-12
            assert tau_tot > 0.5, "用例应真正光学厚（τ_total 足够大）"
    finally:
        run.close()


# --- 5. 53 路互不串（合同第 5 条） ---------------------------------------------


def test_53_beams_independent_attenuation(lib):
    """53 束（含一束 σ=0）各自独立衰减：逐束对解析，σ=0 束逐位 = F∞。"""
    nb = 53
    nx = 8
    x1min, x1max = 1.0e10, 1.4e10  # dx = 5e8 cm
    n0 = 1.0e8
    bidx = np.arange(nb, dtype=np.float64)
    sigma_arr = 6.3e-19 * (1.0 + bidx)         # 6.3e-19 .. 3.3e-17
    sigma_arr[17] = 0.0                        # 一束不吸收
    hnu_arr = 1.0e-11 * (1.0 + 0.05 * bidx)
    finf_arr = 0.05 + 0.002 * bidx
    cfg = _scan_cfg(sigma=sigma_arr.tolist(), hnu=hnu_arr.tolist(),
                    finf=finf_arr.tolist(), nx=nx, x1min=x1min, x1max=x1max)
    run = xmf.setup(cfg)
    try:
        ng, nx_, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.nbeam == nb
        m = _mass_code(run, 0)
        rho0 = n0 * m
        _pack_block(lib, run, 0, SLOT_N, np.full((nzt, nyt, nxt), rho0))
        _pack_block(lib, run, 0, SLOT_P, np.full((nzt, nyt, nxt), n0 * KB * 1500.0))

        dx_full = _dx1(lib, run)
        ds = dx_full[ng : ng + nx] * run._cfg.units.length
        n_abs = np.full(nx, rho0 / m)

        run.step()
        assert run.t > 0.0

        flx = run.flux()
        assert flx.shape == (nb, nz, ny, nx)
        # σ=0 束：逐位 = F∞（真空中行走）
        assert np.array_equal(flx[17], np.full((nz, ny, nx), finf_arr[17])), (
            "σ=0 的束应逐位等于 F∞（不吸收）"
        )
        for b in range(nb):
            walk = _expected_transmitted(finf_arr[b], sigma_arr[b], n_abs, ds)
            got = flx[b, 0, 0, :]
            rel = float(np.max(np.abs((got - walk) / walk)))
            assert rel <= 1.0e-12, (
                f"束 {b}：应对自己的 σ={sigma_arr[b]:.3e} 独立衰减，"
                f"最大相对差 {rel:.3e}——53 路互不串"
            )
            # 闭式交叉核对
            tau = (sigma_arr[b] * n_abs[0]) * ds[0]
            cum = nx - np.arange(nx, dtype=np.float64)
            closed = finf_arr[b] * np.exp(-tau * cum)
            assert float(np.max(np.abs((got - closed) / closed))) <= 1.0e-12
        # 束间确实不同（防「全表退化成一束」的假绿）
        assert not np.allclose(flx[0], flx[52], rtol=1.0e-3)
    finally:
        run.close()


# --- 6. 关辐射一步与纯 hydro 逐位相同（合同第 6 条） ---------------------------


def _off_cmp_cfg(with_rad_table: bool) -> dict:
    """同一配置的两份：重力 + 复合带子两边同开；区别只在辐射表是否加载。
    复合带读 T_HII → 两份都会触发 xmf_phys_load，mass_code 一致，逐位可比。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [16, 1, 1]
    cfg["grid"]["x1min"] = 1.0e10
    cfg["grid"]["x1max"] = 2.0e10
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["fluids"] = [
        {"name": "HI", "mass": MASS_AMU, "charge": 0, "kind": "hd"},
        {"name": "HII", "mass": MASS_AMU, "charge": 1, "kind": "hd"},
    ]
    cfg["constants"] = {
        "G": 6.6726e-8,
        "M_p": 1.309e30,
        "alpha_rec": 4.0e-12,
        "T_rec_ref": 300.0,
        "rec_exp": 0.64,
    }
    recomb = "alpha_rec * (T_rec_ref / T_HII)**rec_exp * n_HII * n_HII"
    cfg["sources"] = [
        {"name": "recomb_HI", "expr": {"dn_HI": recomb}},
        {"name": "recomb_HII", "expr": {"dn_HII": f"-({recomb})"}},
        {"name": "gravity", "expr": {"dvx": "-G * M_p / r**2"}},
    ]
    if with_rad_table:
        # 关辐射合同：表仍加载、enabled=false、F∞=0（工厂 off 模式同款）
        cfg["radiation"] = {
            "enabled": False,
            "absorbers": ["HI"],
            "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
            "bins": {"n": 1, "sigma": [SIGMA_1], "hnu": [HNU_1],
                     "flux_inf": [0.0]},
        }
    else:
        cfg["radiation"] = {"enabled": False, "absorbers": []}
    cfg["ic"] = {"kind": "none"}
    return cfg


def test_off_radiation_one_step_bitwise_pure_hydro(lib):
    """radiation.enabled=false（F∞=0，表加载）一步 step 后，与整表不加载的
    同配置逐位相同——五槽×两支 + t + dt（重力/复合两边同开）。"""
    run_off = xmf.setup(_off_cmp_cfg(with_rad_table=True))
    run_pure = xmf.setup(_off_cmp_cfg(with_rad_table=False))
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run_off)
        assert run_off.num_tapes == run_pure.num_tapes == 3
        assert run_off.nbeam == 1 and run_pure.nbeam == 0

        # 同一均匀双流体初值（两支都非零，T 带子才有定义）
        m = MASS_AMU * AMU  # unit_density = 1
        for run in (run_off, run_pure):
            for f, (n_f, t_f) in enumerate(((1.0e10, 1500.0), (1.0e8, 1500.0))):
                _pack_block(lib, run, f, SLOT_N,
                            np.full((nzt, nyt, nxt), n_f * m))
                _pack_block(lib, run, f, SLOT_P,
                            np.full((nzt, nyt, nxt), n_f * KB * t_f))

        run_off.step()
        run_pure.step()
        assert run_off.t > 0.0 and run_pure.t > 0.0

        for f in range(2):
            for s, name in enumerate("n vx vy vz P".split()):
                a = _unpack(lib, run_off, f, s)
                b = _unpack(lib, run_pure, f, s)
                assert np.array_equal(a, b), (
                    f"流体 {f} 槽 {name}：关辐射一步后与不加载辐射逐位不同"
                    f"（L∞ 差 {float(np.max(np.abs(a - b))):.3e}，码单位）"
                )
        assert run_off.t == run_pure.t, "t 应逐位相同"
        assert run_off.dt == run_pure.dt, "dt 应逐位相同"
        # 哨兵：源项确实在两边都跑了（不是拿纯 hydro 空转冒充）
        assert run_off.source_resid >= 0.0
    finally:
        run_off.close()
        run_pure.close()


# --- 7. B3 多吸收体：κ 相加、53 路不串、配置层不丢名 ----------------------------

MASS_HE_AMU = 4.002602  # 卡片锁的 HEI 质量（amu）


def _scan_cfg2(*, sigma_rows, hnu, finf, geometry="spherical", nx=16, ny=1,
               x1min=1.0e10, x1max=1.8e10, enabled=True):
    """两吸收体（HI=流体 0、HEI=流体 1）的扫描用例：sigma_rows 是
    [n_abs][nbeam] 的行表；其余口径与 _scan_cfg 相同。"""
    nb = len(hnu)
    assert len(finf) == nb
    assert len(sigma_rows) == 2 and all(len(r) == nb for r in sigma_rows)
    cfg = xmf.shock_tube()
    cfg["grid"]["geometry"] = geometry
    cfg["grid"]["n"] = [nx, ny, 1]
    cfg["grid"]["ndim"] = 2 if ny > 1 else 1
    cfg["grid"]["x1min"] = x1min
    cfg["grid"]["x1max"] = x1max
    if ny > 1:
        cfg["grid"]["x2min"] = 0.0
        cfg["grid"]["x2max"] = 1.0
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["fluids"] = [
        {"name": "HI", "mass": MASS_AMU, "charge": 0, "kind": "neutral"},
        {"name": "HEI", "mass": MASS_HE_AMU, "charge": 0, "kind": "neutral"},
    ]
    cfg["sources"] = []
    cfg["radiation"] = {
        "enabled": enabled,
        "absorbers": ["HI", "HEI"],
        "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
        "bins": {
            "n": nb,
            "sigma": [[float(v) for v in r] for r in sigma_rows],
            "hnu": [float(v) for v in hnu],
            "flux_inf": [float(v) for v in finf],
        },
    }
    cfg["ic"] = {"kind": "none"}
    return cfg


def _expected_transmitted_multi(finf_b: float, kappa_line: np.ndarray,
                                ds: np.ndarray) -> np.ndarray:
    """逐字复现多吸收体行走（外→内，透射后存）：tau = kappa*ds，
    kappa = Σ_s σ_s n_s 已按核内累加序（s=HI 后 s=HEI）合成。"""
    nx = len(kappa_line)
    f = float(finf_b)
    out = np.empty(nx, dtype=np.float64)
    for iq in range(nx - 1, -1, -1):
        tau = float(kappa_line[iq]) * float(ds[iq])
        f *= math.exp(-tau)
        out[iq] = f
    return out


def test_two_absorber_kappa_adds(lib):
    """两吸收体均匀混合：τ_b = (σ_H,b n_H + σ_He,b n_He) Δs 逐格对解析；
    并与两条单吸收体路径交叉钉 κ 相加：F_AB == F_A·F_B/F∞。"""
    nx = 16
    x1min, x1max = 1.0e10, 1.8e10
    n_h, n_he = 2.0e8, 0.5e8  # 不同量级，和才有判别力
    sig_h = [SIGMA_1, 0.0]        # 束 1 只被 He 吸收
    sig_he = [1.0e-18, 3.0e-18]
    hnu = [HNU_1, 4.0e-11]
    finf = [FINF_1, 2.25]
    cfg_ab = _scan_cfg2(sigma_rows=[sig_h, sig_he], hnu=hnu, finf=finf,
                        nx=nx, x1min=x1min, x1max=x1max)
    run_ab = xmf.setup(cfg_ab)
    try:
        ng, nx_, ny, nz, nxt, nyt, nzt = _geom(run_ab)
        assert nx_ == nx
        m_h = _mass_code(run_ab, 0)
        m_he = _mass_code(run_ab, 1)
        P_h = n_h * KB * 1500.0
        P_he = n_he * KB * 1500.0
        _pack_block(lib, run_ab, 0, SLOT_N, np.full((nzt, nyt, nxt), n_h * m_h))
        _pack_block(lib, run_ab, 0, SLOT_P, np.full((nzt, nyt, nxt), P_h))
        _pack_block(lib, run_ab, 1, SLOT_N, np.full((nzt, nyt, nxt), n_he * m_he))
        _pack_block(lib, run_ab, 1, SLOT_P, np.full((nzt, nyt, nxt), P_he))

        dx_full = _dx1(lib, run_ab)
        ds = dx_full[ng : ng + nx] * run_ab._cfg.units.length
        # 与核内同一对 double：n = (n*m)/m
        n_h_eff = (n_h * m_h) / m_h
        n_he_eff = (n_he * m_he) / m_he

        run_ab.step()
        assert run_ab.t > 0.0
        # 两支密度被 hydro 逐位保持（v=0 质量通量为 0），扫描读到的才是打包值
        rho_h_after = _unpack(lib, run_ab, 0, SLOT_N).reshape((nzt, nyt, nxt))
        rho_he_after = _unpack(lib, run_ab, 1, SLOT_N).reshape((nzt, nyt, nxt))
        assert np.array_equal(rho_h_after[ng, ng, ng : ng + nx],
                              np.full(nx, n_h * m_h)), "HI 密度一步后应逐位不变"
        assert np.array_equal(rho_he_after[ng, ng, ng : ng + nx],
                              np.full(nx, n_he * m_he)), "HEI 密度一步后应逐位不变"
        flx = run_ab.flux()
        assert flx.shape == (2, nz, ny, nx)
        for b in range(2):
            kappa = np.full(nx, sig_h[b] * n_h_eff + sig_he[b] * n_he_eff)
            expect = _expected_transmitted_multi(finf[b], kappa, ds)
            got = flx[b, 0, 0, :]
            rel = float(np.max(np.abs((got - expect) / expect)))
            assert rel <= 1.0e-12, (
                f"束 {b}：两吸收体 τ = (σ_H n_H + σ_He n_He) Δs 对解析的"
                f"最大相对差 {rel:.3e} 超过 1e-12"
            )
        # 哨兵：束 0 确实比「只吸收 H」衰减得更深（He 的 κ 真的加上了）
        kappa_h_only = np.full(nx, sig_h[0] * n_h_eff)
        f_h_only = _expected_transmitted_multi(finf[0], kappa_h_only, ds)
        assert flx[0, 0, 0, 0] < f_h_only[0]
    finally:
        run_ab.close()

    # 交叉钉 κ 相加：同一均匀介质，分别跑「只 H」「只 He」两条路径。
    # 只 H = 单吸收体旧路径（HI 单流体，σ 行 = sig_h）；只 He = 多吸收体
    # 口但 σ_H 行清零（0·n 逐位为 0，κ 只剩 He 项）——两条路径密度都
    # 保持正值，hydro 不进真空地板。F_AB 直接复用上面 run_ab 的 flx。
    cfg_a = _scan_cfg(sigma=sig_h, hnu=hnu, finf=finf,
                      nx=nx, x1min=x1min, x1max=x1max)
    run_a = xmf.setup(cfg_a)
    try:
        ng, nx_, ny, nz, nxt, nyt, nzt = _geom(run_a)
        m = _mass_code(run_a, 0)
        _pack_block(lib, run_a, 0, SLOT_N, np.full((nzt, nyt, nxt), n_h * m))
        _pack_block(lib, run_a, 0, SLOT_P, np.full((nzt, nyt, nxt), n_h * KB * 1500.0))
        run_a.step()
        flx_a = run_a.flux()
    finally:
        run_a.close()

    cfg_b = _scan_cfg2(sigma_rows=[[0.0, 0.0], sig_he], hnu=hnu, finf=finf,
                       nx=nx, x1min=x1min, x1max=x1max)
    run_b = xmf.setup(cfg_b)
    try:
        ng, nx_, ny, nz, nxt, nyt, nzt = _geom(run_b)
        m_h = _mass_code(run_b, 0)
        m_he = _mass_code(run_b, 1)
        _pack_block(lib, run_b, 0, SLOT_N, np.full((nzt, nyt, nxt), n_h * m_h))
        _pack_block(lib, run_b, 0, SLOT_P, np.full((nzt, nyt, nxt), n_h * KB * 1500.0))
        _pack_block(lib, run_b, 1, SLOT_N, np.full((nzt, nyt, nxt), n_he * m_he))
        _pack_block(lib, run_b, 1, SLOT_P, np.full((nzt, nyt, nxt), n_he * KB * 1500.0))
        run_b.step()
        flx_b = run_b.flux()
    finally:
        run_b.close()

    for b in range(2):
        prod = flx_a[b, 0, 0, :] * flx_b[b, 0, 0, :] / finf[b]
        rel = float(np.max(np.abs((flx[b, 0, 0, :] - prod) / prod)))
        assert rel <= 1.0e-13, (
            f"束 {b}：κ 相加合同 F_AB == F_A·F_B/F∞ 破坏，最大相对差 {rel:.3e}"
        )


def test_two_absorber_53_beams_no_crosstalk(lib):
    """53 束 × 两吸收体：每束按自己那行 σ 独立衰减（含各一支 σ=0 的束）；
    nx2=3 三条径向射线仍互不串（中间射线混合比不同）。"""
    nb = 53
    nx, ny = 8, 3
    x1min, x1max = 1.0e10, 1.4e10
    n_h0, n_he0 = 1.0e8, 4.0e7
    bidx = np.arange(nb, dtype=np.float64)
    sig_h = 6.3e-19 * (1.0 + bidx)
    sig_he = 3.1e-19 * (1.0 + 0.5 * bidx)
    sig_h[17] = 0.0   # 束 17 不被 H 吸收
    sig_he[42] = 0.0  # 束 42 不被 He 吸收
    hnu = 1.0e-11 * (1.0 + 0.05 * bidx)
    finf = 0.05 + 0.002 * bidx
    cfg = _scan_cfg2(sigma_rows=[sig_h.tolist(), sig_he.tolist()],
                     hnu=hnu.tolist(), finf=finf.tolist(),
                     geometry="cartesian", nx=nx, ny=ny,
                     x1min=x1min, x1max=x1max)
    run = xmf.setup(cfg)
    try:
        ng, nx_, ny_, nz, nxt, nyt, nzt = _geom(run)
        assert run.nbeam == nb
        m_h = _mass_code(run, 0)
        m_he = _mass_code(run, 1)
        dx = (x1max - x1min) / nx
        ds = np.full(nx, dx * run._cfg.units.length)

        # 每条射线一个混合比：中间射线 He 占比刻意加大
        mix = np.array([1.0, 3.0, 1.0])
        rho_block = np.zeros((nzt, nyt, nxt), dtype=np.float64)
        p_block = np.zeros((nzt, nyt, nxt), dtype=np.float64)
        he_rho = np.zeros((nzt, nyt, nxt), dtype=np.float64)
        he_p = np.zeros((nzt, nyt, nxt), dtype=np.float64)
        for jj in range(ny):
            n_he = mix[jj] * n_he0
            rho_block[ng, ng + jj, :] = n_h0 * m_h
            p_block[ng, ng + jj, :] = n_h0 * KB * 1500.0
            he_rho[ng, ng + jj, :] = n_he * m_he
            he_p[ng, ng + jj, :] = n_he * KB * 1500.0
        _pack_block(lib, run, 0, SLOT_N, rho_block)
        _pack_block(lib, run, 0, SLOT_P, p_block)
        _pack_block(lib, run, 1, SLOT_N, he_rho)
        _pack_block(lib, run, 1, SLOT_P, he_p)

        run.step()
        assert run.t > 0.0
        flx = run.flux()
        assert flx.shape == (nb, nz, ny, nx)

        n_h_eff = (n_h0 * m_h) / m_h
        for jj in range(ny):
            n_he_eff = ((mix[jj] * n_he0) * m_he) / m_he
            for b in range(nb):
                kappa = np.full(nx, sig_h[b] * n_h_eff + sig_he[b] * n_he_eff)
                expect = _expected_transmitted_multi(finf[b], kappa, ds)
                got = flx[b, 0, jj, :]
                rel = float(np.max(np.abs((got - expect) / expect)))
                assert rel <= 1.0e-12, (
                    f"射线 j={jj} 束 {b}：应对自己的 (σ_H,σ_He) 行独立衰减，"
                    f"最大相对差 {rel:.3e}——53 路 × 两吸收体不得串"
                )
        # σ=0 哨兵：束 17 只被 He 吸收、束 42 只被 H 吸收（两侧射线）
        kappa_17 = np.full(nx, sig_he[17] * n_he_eff)
        expect_17 = _expected_transmitted_multi(finf[17], kappa_17, ds)
        assert np.allclose(flx[17, 0, 0, :], expect_17, rtol=1.0e-12)
        # 束间确实不同（防全表退化）
        assert not np.allclose(flx[0, 0, 0, :], flx[52, 0, 0, :], rtol=1.0e-3)
        # 中间射线吸收更强 → 最内格通量更低
        assert flx[0, 0, 1, 0] < flx[0, 0, 0, 0]
    finally:
        run.close()


def test_multi_absorber_config_payload_and_validation():
    """normalize() 不再静默丢 absorbers[1:]：两吸收体给行表，载荷带全部
    下标与行主序 σ；行表形状不符 / 未知组分名一律拒。单吸收体扁平行不变。"""
    from xmf import config as cfgmod

    def base(sigma, absorbers=("HI", "HEI")):
        c = xmf.shock_tube()
        c["fluids"] = [
            {"name": "HI", "mass": MASS_AMU, "charge": 0, "kind": "neutral"},
            {"name": "HEI", "mass": MASS_HE_AMU, "charge": 0, "kind": "neutral"},
        ]
        c["radiation"] = {
            "enabled": True,
            "absorbers": list(absorbers),
            "bins": {"n": 2, "sigma": sigma, "hnu": [1.0e-11, 2.0e-11],
                     "flux_inf": [0.5, 0.75]},
        }
        c["ic"] = {"kind": "none"}
        return c

    # 行表（2×2）：两个下标都进载荷，σ 行主序摊平
    p = cfgmod.normalize(base([[2.0e-18, 0.0], [1.0e-18, 3.0e-18]]))
    assert p.extra["rad_load"]["absorber"] == 0
    assert p.extra["rad_load"]["sigma"] == [2.0e-18, 0.0]
    multi = p.extra["rad_load_multi"]
    assert multi["absorbers"] == [0, 1], "absorbers[1:] 不得被丢"
    assert multi["sigma"] == [2.0e-18, 0.0, 1.0e-18, 3.0e-18]

    # 单吸收体：扁平行口径不变，且不生成多吸收体载荷
    p1 = cfgmod.normalize(base([2.0e-18, 0.0], absorbers=("HI",)))
    assert p1.extra["rad_load"]["absorber"] == 0
    assert p1.extra["rad_load"]["sigma"] == [2.0e-18, 0.0]
    assert "rad_load_multi" not in p1.extra

    # 两吸收体 + 扁平行 → 拒（不得静默丢第二个名字）
    with pytest.raises(ValueError, match="row per absorber"):
        cfgmod.normalize(base([2.0e-18, 0.0]))
    # 行数对、列数错 → 拒
    with pytest.raises(ValueError, match="bins.n entries"):
        cfgmod.normalize(base([[2.0e-18], [1.0e-18, 3.0e-18]]))
    # 名字数对、行数少（三个名字两行表）→ 拒
    with pytest.raises(ValueError, match="row per absorber"):
        cfgmod.normalize(base([[2.0e-18, 0.0], [1.0e-18, 3.0e-18]],
                              absorbers=("HI", "HEI", "HI")))
    # 未知组分名 → 拒
    with pytest.raises(ValueError, match="absorbers"):
        cfgmod.normalize(base([[2.0e-18, 0.0], [1.0e-18, 3.0e-18]],
                              absorbers=("HI", "XXX")))


@pytest.mark.parametrize("n_abs_target", [0.0, 1.0e-30])
def test_two_absorber_vacuum_flux_is_finf_bitwise(lib, n_abs_target):
    """多吸收体路径的真空/地板合同（同 A5 合同第 1 条）：两支吸收体密度为 0
    或贴地板时 κ = σ_H·0 + σ_He·0 = 0，各格各束 F == F∞ 逐位。"""
    nx = 8
    x1min, x1max = 1.0e10, 1.4e10
    sig_h = [1.0e-18, 2.5e-18]     # 束 0 两支都吸收；
    sig_he = [3.0e-18, 0.0]        # 束 1 只被 H 吸收（σ_He=0 也不漏束）
    hnu = [HNU_1, 4.0e-11]
    finf = [FINF_1, 2.25]
    cfg = _scan_cfg2(sigma_rows=[sig_h, sig_he], hnu=hnu, finf=finf,
                     nx=nx, x1min=x1min, x1max=x1max)
    run = xmf.setup(cfg)
    try:
        ng, nx_, ny, nz, nxt, nyt, nzt = _geom(run)
        assert nx_ == nx and run.nbeam == 2
        m_h = _mass_code(run, 0)
        m_he = _mass_code(run, 1)
        _pack_block(lib, run, 0, SLOT_N,
                    np.full((nzt, nyt, nxt), n_abs_target * m_h))
        _pack_block(lib, run, 1, SLOT_N,
                    np.full((nzt, nyt, nxt), n_abs_target * m_he))

        run.step()
        assert run.t > 0.0

        flx = run.flux()
        assert flx.shape == (2, nz, ny, nx)
        for b in range(2):
            expect = np.full((nz, ny, nx), finf[b], dtype=np.float64)
            assert np.array_equal(flx[b], expect), (
                f"束 {b}：两吸收体真空/地板密度下每格透射通量应逐位等于 "
                f"F∞={finf[b]!r}，实测最大差 "
                f"{float(np.max(np.abs(flx[b] - expect))):.3e}"
            )
    finally:
        run.close()


def _off_pair_cfg(multi_table: bool) -> dict:
    """同一配置的两份（流体、初值口径完全相同）：两流体 + 两束 + F∞=0，
    辐射 enabled=false；差别只在束表是两吸收体行表（追加口）还是单吸收体
    扁平行（A5 旧口）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [16, 1, 1]
    cfg["grid"]["x1min"] = 1.0e10
    cfg["grid"]["x1max"] = 2.0e10
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["fluids"] = [
        {"name": "HI", "mass": MASS_AMU, "charge": 0, "kind": "neutral"},
        {"name": "HEI", "mass": MASS_HE_AMU, "charge": 0, "kind": "neutral"},
    ]
    cfg["sources"] = []
    bins = {"n": 2, "hnu": [HNU_1, 4.0e-11], "flux_inf": [0.0, 0.0]}
    if multi_table:
        bins["sigma"] = [[SIGMA_1, 2.5e-18], [1.0e-18, 3.0e-18]]
        absorbers = ["HI", "HEI"]
    else:
        bins["sigma"] = [SIGMA_1, 2.5e-18]
        absorbers = ["HI"]
    cfg["radiation"] = {
        "enabled": False,
        "absorbers": absorbers,
        "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
        "bins": bins,
    }
    cfg["ic"] = {"kind": "none"}
    return cfg


def test_two_absorber_disabled_bitwise_vs_single_absorber_off(lib):
    """enabled=false + 两吸收体行表在场：行为与 A5 关辐射完全相同——一步
    step 后与单吸收体（扁平行）关辐射的同配置逐位相同（五槽×两支 + t +
    dt）。多吸收体追加口在关辐射时不得有任何副作用。"""
    run_multi = xmf.setup(_off_pair_cfg(multi_table=True))
    run_single = xmf.setup(_off_pair_cfg(multi_table=False))
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run_multi)
        assert _geom(run_single) == (ng, nx, ny, nz, nxt, nyt, nzt)
        assert run_multi.nbeam == 2 and run_single.nbeam == 2

        # 同一 x1 台阶初值（两支同形）：hydro 真的在演化，等价钉才有力
        shape = (nzt, nyt, nxt)
        for f in range(2):
            rho = np.empty(shape, dtype=np.float64)
            prs = np.empty(shape, dtype=np.float64)
            rho[:, :, : nxt // 2] = 2.0e-16
            rho[:, :, nxt // 2:] = 5.0e-17
            prs[:, :, : nxt // 2] = 2.0e-10
            prs[:, :, nxt // 2:] = 6.0e-11
            for run in (run_multi, run_single):
                _pack_block(lib, run, f, SLOT_N, rho)
                _pack_block(lib, run, f, SLOT_P, prs)

        run_multi.step()
        run_single.step()
        assert run_multi.t > 0.0 and run_single.t > 0.0

        for f in range(2):
            for s, name in enumerate("n vx vy vz P".split()):
                a = _unpack(lib, run_multi, f, s)
                b = _unpack(lib, run_single, f, s)
                assert np.array_equal(a, b), (
                    f"流体 {f} 槽 {name}：关辐射时两吸收体行表与单吸收体"
                    f"扁平行一步后逐位不同（L∞ 差 "
                    f"{float(np.max(np.abs(a - b))):.3e}，码单位）"
                )
        assert run_multi.t == run_single.t, "t 应逐位相同"
        assert run_multi.dt == run_single.dt, "dt 应逐位相同"
    finally:
        run_multi.close()
        run_single.close()


def test_rad_load_absorbers_port_validates(lib):
    """C 追加口 xmf_rad_load_absorbers 自身的合同（对照
    xmf_core.h 注释）：无束表 / nabs<2 / NULL 表 / 越界下标一律 −1 且
    前态不动；合法两吸收体返回 nabs。核只认流体下标，不认组分名。"""
    # 无辐射束表的 run：追加口必须拒
    c0 = xmf.shock_tube()
    c0["fluids"] = [
        {"name": "HI", "mass": MASS_AMU, "charge": 0, "kind": "neutral"},
        {"name": "HEI", "mass": MASS_HE_AMU, "charge": 0, "kind": "neutral"},
    ]
    c0["ic"] = {"kind": "none"}
    run0 = xmf.setup(c0)
    try:
        ab = np.array([0, 1], dtype=np.int32)
        sg = np.array([1.0e-18, 2.0e-18], dtype=np.float64)
        assert lib.xmf_rad_load_absorbers(
            run0._h, 2, ab.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            sg.ctypes.data_as(_PTR)) == -1, "无束表必须拒"
    finally:
        run0.close()

    # 有束表（enabled=false 也加载，同 A5）的 run
    run = xmf.setup(_scan_cfg2(sigma_rows=[[SIGMA_1, 2.5e-18],
                                           [1.0e-18, 3.0e-18]],
                               hnu=[HNU_1, 4.0e-11], finf=[0.0, 0.0],
                               nx=8, enabled=False))
    try:
        assert run.nbeam == 2
        iptr = ctypes.POINTER(ctypes.c_int)

        def call(nabs, ab, sg):
            ab_p = ab.ctypes.data_as(iptr) if ab is not None else None
            sg_p = sg.ctypes.data_as(_PTR) if sg is not None else None
            return lib.xmf_rad_load_absorbers(run._h, nabs, ab_p, sg_p)

        ab2 = np.array([0, 1], dtype=np.int32)
        sg22 = np.array([1.0e-18, 2.0e-18, 3.0e-18, 4.0e-18], dtype=np.float64)
        assert call(1, np.array([0], dtype=np.int32),
                    np.array([1.0e-18, 2.0e-18])) == -1, "nabs<2 必须拒"
        assert call(2, ab2, None) == -1, "NULL sigma 必须拒"
        assert call(2, None, sg22) == -1, "NULL absorbers 必须拒"
        assert call(2, np.array([0, 5], dtype=np.int32), sg22) == -1, (
            "越界流体下标必须拒")
        assert call(2, np.array([-1, 1], dtype=np.int32), sg22) == -1, (
            "负流体下标必须拒")
        assert call(2, ab2, sg22) == 2, "合法两吸收体应返回 nabs"
        # 后一次成功调用整体替换：换成 [1, 0] 仍返回 2
        assert call(2, np.array([1, 0], dtype=np.int32), sg22) == 2
    finally:
        run.close()
