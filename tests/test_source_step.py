"""A5 源项与隐式链式合同单测（test_source_step）：Γ/加热/复合/重力 + ①③④。

合同来自本刀卡片（docs/SESSION.md §交什么 3/4/6）与 docs/cmd/PROMPT_A5.md：
  1. 真空 Γ_ph = 0：F=0 时光化/加热增量为 0（逐位），复合/重力仍开。
  2. 均匀薄极限：单束弱吸收，一步 step 的隐式增量对解析
     Γ = σ F n/hν 的线性后向欧拉精确解（rtol 1e-12），apply 速率对
     σ F n/hν；v、P 不被这两条带子动。
  3. CHE_PRS_TRANS + 全套工厂源项：真工厂 rad_config.hh_rad_1d(data,
     n_beam=1)（算例 examples/rad_hd_1d/rad_config.py，importlib 按绝对
     路径加载）一步 step 后，apply 增量对卡片公式一致（m 用克；速度继承、
     加热 η、重力+潮汐、复合全在）。若 m 误用 amu，动能项差 ~6e23 倍必红。
  4. 重力 apply 仍 1e-12：两支流体同一 −GM/r²，只改 v 不改 n、P。
  5. 链式 ①③④ 对解析：已知 (w, L_c) 下，核的一步隐式 step 与手算
     (S1·S2+S4)·S3 + 提升 R 的线性后向欧拉预测一致（核的 ② 用
     jacobian() 读回互验）；动量/能量的 ṅ、v̇ 混合项都在。
  6. ④ 专门探针：常数 dvx 带（J 全 0，②没有贡献）下，一步 step 的
     压强变化必须带上 ④ 的签名：隐式耦合把 P 推到
     P + (1/2)(γ−1)ρ(dt·g)²——漏 ④ 则落在 P − (1/2)(γ−1)ρ(dt·g)²，
     两者差 (γ−1)ρ(dt·g)²，1e-12 容差下必红。

两条锁定的口径（写期望式时必须照抄，不是放宽合同）：
  * apply 的 ADDS 约定是**顺序**的（xmf_ir.c 锁）：带子按加载顺序逐条
    求值并立刻写回，后面的带子读到的是前面带子已改过的状态。期望式
    必须按带序逐条推进，不能拿同一快照代所有带子。
  * dn 带子写进质量密度槽：rho += mass_code·dn。增量与库存密度相减的
    读回有一个表示噪声底 ≈ ulp(ρ)/mass_code（数密度单位 ~2^-53·n）——
    断言用「1e-12 相对 + 该噪声底的显式 atol」，atol 只覆盖存储往返，
    公式错（差系数、错质量、漏项）仍远大于它，必红。

其余约定：apply 一次 = 每条带子的瞬时值加到写目标，不乘 Δt；jacobian
是含 ghost 全格下标的辅助口。首步前 rad->flux 为 0（calloc）——要非零
F 的用例先走一步（scan 在每阶 hydro 前）。笛卡尔均匀态走 step 的用例
把 x1 边界设为 outflow（零梯度），默认 reflect 会在两端造墙。

口径：unpack 块长 = xmf_state_ncells（含 ghost），reshape (nztot,nytot,
nxtot)，(k,j,i) i 最快。units 全 1 的配置里码值即 cgs。

禁令自查：不读 pluto.ini（工厂读的是仓库内 examples/rad_hd_1d/data 的
随包数据表，不读旧树）；不用 XUV_FACTOR==0 哨兵；不只测 VX1——
n、P、vy 目标都进断言；不改旧测。
"""
from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU, KB

_PTR = ctypes.POINTER(ctypes.c_double)

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

# 工厂参考常数（与 rad_config.hh_rad_1d 同一份；测试期望式独立复现卡片公式）
G_P = 6.6726e-8
M_JUP = 1.8982e30
M_P = 0.69 * M_JUP
ALPHA_REC = 4.0e-12
T_REC_REF = 300.0
REC_EXP = 0.64
ETA = 0.20
GAMMA = 5.0 / 3.0
GM1 = GAMMA - 1.0
AMU_H = 1.00794
M_H_G = AMU_H * AMU          # 克

# 链式用例的速率常数（刻意非 1、互不相同）
K1 = 1.7e-3
K2 = 1.3e-1
K3 = 2.75e-2

# 重力专测常数（与 test_ir_ad 同一组非 1 数）
GRAV_G, GRAV_M = 2.5, 3.75

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "rad_hd_1d" / "data"

# 考试工厂是算例私有配置（examples/rad_hd_1d/rad_config.py），不再由
# xmf 导出：importlib 按绝对路径加载，模块名 rad_config，不动 sys.path。
_RAD_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "rad_hd_1d" / "rad_config.py"
)
_spec = importlib.util.spec_from_file_location("rad_config", _RAD_CONFIG_PATH)
rad_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rad_config)

# rho 槽存储往返的噪声底（相对库存量）：Δρ = fl(ρ + m·dn) 至多引入
# ulp(ρ)/2，读回再除 m、再做前后差，共 ~16·2^-53·|背景值| 足够覆盖。
# 这不是放宽合同：任何公式级错误（系数、质量单位、漏项、错带序）
# 都是 ≥1e-6 量级，远大于该底。
NOISE_F = 16.0 * 2.0**-53


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（rad/state/mesh/ir/source 符号已绑定）。"""
    return load()


# --- 只读小助手 --------------------------------------------------------------


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


def _unpack(lib, run, f: int, var: int) -> np.ndarray:
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, f, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _centres(lib, run) -> np.ndarray:
    ntot = lib.xmf_mesh_ncell(run._h, 0)
    buf = np.empty(ntot, dtype=np.float64)
    assert lib.xmf_mesh_copy_x(run._h, 0, buf.ctypes.data_as(_PTR)) == ntot
    return buf


def _mass_code(run, f: int) -> float:
    """mass_code = mass_amu*AMU/unit_density（xmf_phys_load 的换算）。"""
    return run._cfg.fluids[f].mass * AMU / run._cfg.units.density


def _base_cfg(nx=8, x1min=1.0e10, x1max=2.0e10) -> dict:
    """笛卡尔 1D 双流体空题：units 全 1（码值即 cgs）、rk1、ic none。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [nx, 1, 1]
    cfg["grid"]["x1min"] = x1min
    cfg["grid"]["x1max"] = x1max
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["fluids"] = [
        {"name": "HI", "mass": AMU_H, "charge": 0, "kind": "hd"},
        {"name": "HII", "mass": AMU_H, "charge": 1, "kind": "hd"},
    ]
    cfg["ic"] = {"kind": "none"}
    return cfg


def _rad_bins(sigma, hnu, finf) -> dict:
    return {
        "enabled": True,
        "absorbers": ["HI"],
        "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
        "bins": {"n": len(sigma), "sigma": list(sigma), "hnu": list(hnu),
                 "flux_inf": list(finf)},
    }


# --- 1. 真空 Γ = 0：光化/加热增量为 0，复合/重力仍开 --------------------------


def _photo_vacuum_cfg() -> dict:
    cfg = _base_cfg()
    cfg["constants"] = {
        "G": G_P, "M_p": M_P, "eta": ETA, "gm1": GM1,
        "alpha_rec": ALPHA_REC, "T_rec_ref": T_REC_REF, "rec_exp": REC_EXP,
        "sigma_0": 2.5e-18, "hnu_0": 3.1e-11,
    }
    gamma_ion = "n_HI * (sigma_0 * F_0 / hnu_0)"
    heat_vol = "n_HI * (sigma_0 * F_0)"
    recomb = "alpha_rec * (T_rec_ref / T_HII)**rec_exp * n_HII * n_HII"
    cfg["sources"] = [
        {"name": "photoion_HI", "expr": {"dn_HI": f"-({gamma_ion})"}},
        {"name": "photoion_HII", "expr": {"dn_HII": gamma_ion}},
        {"name": "photoheat_HII", "expr": {"dP_HII": f"gm1 * eta * ({heat_vol})"}},
        {"name": "recomb_HI", "expr": {"dn_HI": recomb}},
        {"name": "recomb_HII", "expr": {"dn_HII": f"-({recomb})"}},
        {"name": "gravity", "expr": {"dvx": "-G * M_p / r**2"}},
    ]
    # F∞=0 的表：F=0 必须让光化/加热归零，而不是靠不加载表
    cfg["radiation"] = _rad_bins([2.5e-18], [3.1e-11], [0.0])
    return cfg


def test_vacuum_photoion_heating_zero_rec_gravity_on(lib):
    """F=0（表内 F∞=0，首步前通量态亦为 0）：光化/加热增量逐位 0；
    复合 ±α(300/T)^0.64 n²、重力 −GM/r² 照常（对解析 1e-12）。

    数密度取 5e8 / T=300：复合增量 Δρ/ρ ≈ 2e-3，远高于质量密度槽的
    存储噪声底，1e-12 的相对断言是干净的。"""
    run = xmf.setup(_photo_vacuum_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 6
        assert run.nbeam == 1
        m = _mass_code(run, 0)
        unit_P = run._cfg.units.density * run._cfg.units.velocity ** 2

        n1, n2, T0 = 5.0e8, 5.0e8, 300.0
        for f, n_f in enumerate((n1, n2)):
            _pack_block(lib, run, f, SLOT_N, np.full((nzt, nyt, nxt), n_f * m))
            _pack_block(lib, run, f, SLOT_P,
                        np.full((nzt, nyt, nxt), n_f * KB * T0))

        before = {(f, s): _unpack(lib, run, f, s)
                  for f in range(2) for s in range(5)}
        run.apply_sources()  # 首步前：rad->flux 全 0，FLUX 读到真空值
        after = {(f, s): _unpack(lib, run, f, s)
                 for f in range(2) for s in range(5)}
        assert run.t == 0.0, "apply 不推进时间"

        r = _centres(lib, run)[ng : ng + nx]
        grav = -((G_P * M_P) / r**2)
        assert np.all(grav < -1.0), "重力增量应非零（防命题空转）"

        # 期望复合速率：T = P*unit_P/(n_safe*KB)（与核内同口径），
        # 带子左结合 ((α*(Tref/T)^e)*n)*n。带序下 tape3 之前只有值为 0
        # 的光化/加热带写过状态，故复合读到的就是打包初值。
        rho2 = before[(1, SLOT_N)].reshape((nzt, nyt, nxt))[ng, ng, ng : ng + nx]
        p2 = before[(1, SLOT_P)].reshape((nzt, nyt, nxt))[ng, ng, ng : ng + nx]
        n2v = rho2 / m
        n_safe = np.where(n2v > 1.0e-30, n2v, 1.0e-30)
        T = (p2 * unit_P) / (n_safe * KB)
        recomb = ((ALPHA_REC * ((T_REC_REF / T) ** REC_EXP)) * n2v) * n2v
        assert np.all(recomb > 0.0)
        # 增量必须远大于质量密度槽的存储噪声底，1e-12 才有判别力
        assert np.all(recomb / n2v > 1.0e-4)

        dn1 = (after[(0, SLOT_N)] - before[(0, SLOT_N)]).reshape(
            (nzt, nyt, nxt))[ng, ng, ng : ng + nx] / m
        dn2 = (after[(1, SLOT_N)] - before[(1, SLOT_N)]).reshape(
            (nzt, nyt, nxt))[ng, ng, ng : ng + nx] / m
        rel1 = float(np.max(np.abs((dn1 - recomb) / recomb)))
        rel2 = float(np.max(np.abs((dn2 + recomb) / recomb)))
        assert rel1 <= 1.0e-12, (
            f"F=0 时 Δn_HI 应只剩复合 +R，最大相对差 {rel1:.3e}"
        )
        assert rel2 <= 1.0e-12, (
            f"F=0 时 Δn_HII 应只剩 −R，最大相对差 {rel2:.3e}"
        )
        for f in range(2):
            dvx = (after[(f, SLOT_VX)] - before[(f, SLOT_VX)]).reshape(
                (nzt, nyt, nxt))[ng, ng, ng : ng + nx]
            relg = float(np.max(np.abs((dvx - grav) / grav)))
            assert relg <= 1.0e-12, (
                f"流体 {f}：重力增量应为 -GM/r²（两支同带），相对差 {relg:.3e}"
            )
        # 光化/加热归零：P 与横速度逐位不动；n 的差已被复合解释
        for f in range(2):
            assert np.array_equal(after[(f, SLOT_P)], before[(f, SLOT_P)]), (
                f"流体 {f}：F=0 时加热增量必须逐位为 0（P 不得动）"
            )
            assert np.array_equal(after[(f, SLOT_VY)], before[(f, SLOT_VY)])
            assert np.array_equal(after[(f, SLOT_VZ)], before[(f, SLOT_VZ)])
    finally:
        run.close()


# --- 2. 均匀薄极限：一步隐式增量对 Γ=σFn/hν 的精确解 --------------------------


def _thin_limit_cfg() -> dict:
    cfg = _base_cfg()
    cfg["constants"] = {"sigma_0": 2.5e-18, "hnu_0": 1.0e-13}
    gamma_ion = "n_HI * (sigma_0 * F_0 / hnu_0)"
    cfg["sources"] = [
        {"name": "photoion_HI", "expr": {"dn_HI": f"-({gamma_ion})"}},
        {"name": "photoion_HII", "expr": {"dn_HII": gamma_ion}},
    ]
    cfg["radiation"] = _rad_bins([2.5e-18], [1.0e-13], [4.716])
    return cfg


def test_thin_limit_photoion_rate_sigma_F_n_over_hnu(lib):
    """单束弱吸收（τ_tot≈0.025）：rk1 一步后
      n_HI -> n0/(1+dt·c_i)、n_HII -> n_HII0 + n0·dt·c_i/(1+dt·c_i)，
    c_i = σF_i/hν——线性系统的隐式步必须精确到 1e-12；v、P 不动；
    再 apply 一次的速率对 n·σF/hν。

    apply 按 ADDS 锁是顺序的：tape0 先写 dn_HI，tape1 用的是已减过的
    n_HI——d2 = (n − n·c)·c，比快照值小一个因子 (1−c)，本用例显式钉住。
    数密度增量 ~c·n 与库存密度同槽相减有 ~2^-53·n 的读回噪声底，断言
    带相应的显式 atol（公式错仍远大于它）。"""
    sigma, hnu, finf = 2.5e-18, 1.0e-13, 4.716
    run = xmf.setup(_thin_limit_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 2 and run.nbeam == 1
        m = _mass_code(run, 0)

        n0, n20, T0 = 1.0e6, 1.0e6, 1500.0
        for f, n_f in enumerate((n0, n20)):
            _pack_block(lib, run, f, SLOT_N, np.full((nzt, nyt, nxt), n_f * m))
            _pack_block(lib, run, f, SLOT_P,
                        np.full((nzt, nyt, nxt), n_f * KB * T0))

        run.step()
        assert run.t > 0.0
        dt = run.dt

        F = run.flux()[0, 0, 0, :]
        assert F.shape == (nx,)
        # 扫描真的在衰减：向内（下标增大→外）严格单调，最外格 < F∞
        assert np.all(np.diff(F) > 0.0), "向内（下标增大→外）通量应递增"
        assert F[-1] < finf and F[0] > 0.0

        c = (sigma * F) / hnu                    # 每格速率，σF/hν
        expect_n1 = n0 / (1.0 + dt * c)
        expect_n2 = n20 + n0 * dt * c / (1.0 + dt * c)

        got_n1 = _unpack(lib, run, 0, SLOT_N).reshape(
            (nzt, nyt, nxt))[ng, ng, ng : ng + nx] / m
        got_n2 = _unpack(lib, run, 1, SLOT_N).reshape(
            (nzt, nyt, nxt))[ng, ng, ng : ng + nx] / m
        rel1 = float(np.max(np.abs((got_n1 - expect_n1) / expect_n1)))
        rel2 = float(np.max(np.abs((got_n2 - expect_n2) / expect_n2)))
        assert rel1 <= 1.0e-12, (
            f"线性隐式步对解析精确解的 n_HI 相对差 {rel1:.3e} 超过 1e-12"
        )
        assert rel2 <= 1.0e-12, (
            f"线性隐式步对解析精确解的 n_HII 相对差 {rel2:.3e} 超过 1e-12"
        )
        # 这两条带子不得动 v 与 P（初值 v=0 / P 均匀）
        for f in range(2):
            vx = _unpack(lib, run, f, SLOT_VX)
            vy = _unpack(lib, run, f, SLOT_VY)
            vz = _unpack(lib, run, f, SLOT_VZ)
            assert np.all(vx == 0.0) and np.all(vy == 0.0) and np.all(vz == 0.0)
        p1 = _unpack(lib, run, 0, SLOT_P).reshape((nzt, nyt, nxt))
        assert np.allclose(p1[ng, ng, ng : ng + nx], n0 * KB * T0,
                           rtol=1.0e-13, atol=0.0), "P 不应被光化带改动"

        # 再 apply 一次：瞬时速率 = n·σF/hν（Γ 的薄极限口径），按带序
        gamma1 = got_n1 * c                       # tape0 用的还是当前 n
        exp_d1 = -gamma1
        exp_d2 = (got_n1 - gamma1) * c            # tape1 读到已减过的 n_HI
        atol_n = NOISE_F * got_n1                 # rho 槽读回噪声底
        before1 = _unpack(lib, run, 0, SLOT_N).copy()
        before2 = _unpack(lib, run, 1, SLOT_N).copy()
        run.apply_sources()
        d1 = (_unpack(lib, run, 0, SLOT_N) - before1).reshape(
            (nzt, nyt, nxt))[ng, ng, ng : ng + nx] / m
        d2 = (_unpack(lib, run, 1, SLOT_N) - before2).reshape(
            (nzt, nyt, nxt))[ng, ng, ng : ng + nx] / m
        assert np.all(np.abs(d1 - exp_d1) <= 1.0e-12 * np.abs(exp_d1) + atol_n), (
            f"apply 的 dn_HI 应为 −n·σF/hν，最大偏差 "
            f"{float(np.max(np.abs(d1 - exp_d1))):.3e}"
        )
        assert np.all(np.abs(d2 - exp_d2) <= 1.0e-12 * np.abs(exp_d2) + atol_n), (
            f"apply 的 dn_HII 应为 (n−Γ)·σF/hν（顺序 ADDS），最大偏差 "
            f"{float(np.max(np.abs(d2 - exp_d2))):.3e}"
        )
        # 顺序语义哨兵：tape1 的 Γ 必须比快照值小 ~因子 c（而非相等）
        snap = got_n1 * c
        deficit = (snap - d2) / snap
        assert np.all((deficit > 0.5 * c) & (deficit < 1.5 * c)), (
            "tape1 应读到被 tape0 减过的 n_HI——顺序 ADDS 约定没生效"
        )
        # 这两条带子仍不得动 v
        for f in range(2):
            assert np.all(_unpack(lib, run, f, SLOT_VX) == 0.0)
    finally:
        run.close()


# --- 3. 工厂全套源项：apply 增量对卡片公式（CHE_PRS_TRANS，m 用克） -----------


def _factory_sequential_expected(n1, n2, v1, v2, p1, p2, F, r, consts,
                                 unit_P):
    """按 ADDS 锁逐条顺序模拟工厂的 10 条带子（每条带子看到前面带子的
    写入）。表达式与结合序照抄 rad_config.hh_rad_1d 在 n_beam=1 下产出的
    字符串；返回 apply 后的 (n1, n2, v1, v2, p1, p2)。"""
    sig = consts["sigma_0"]
    hnu = consts["hnu_0"]
    alpha = consts["alpha_rec"]
    Tref = consts["T_rec_ref"]
    expo = consts["rec_exp"]
    nf = consts["n_floor"]
    gm1 = consts["gm1"]
    eta = consts["eta"]
    m_g = consts["m_HI_g"]
    G = consts["G"]
    Mp = consts["M_p"]
    Mstar = consts["M_star"]
    a_orb = consts["a"]

    n1 = n1.copy()
    n2 = n2.copy()
    v1 = v1.copy()
    v2 = v2.copy()
    p1 = p1.copy()
    p2 = p2.copy()

    def gamma_ion():
        return n1 * ((sig * F) / hnu)

    def heat_vol():
        return n1 * (sig * F)

    def recomb():
        n2_safe = np.where(n2 > 1.0e-30, n2, 1.0e-30)
        T = (p2 * unit_P) / (n2_safe * KB)
        Tfloor = consts["T_floor"]
        return ((alpha * ((Tref / (T + Tfloor)) ** expo)) * n2) * n2

    n1 = n1 - gamma_ion()                            # 1 photoion_HI
    n2 = n2 + gamma_ion()                            # 2 photoion_HII
    p2 = p2 + (gm1 * eta) * heat_vol()               # 3 photoheat_HII
    n1 = n1 + recomb()                               # 4 recomb_HI
    n2 = n2 - recomb()                               # 5 recomb_HII
    v1 = v1 + (recomb() / (n1 + nf)) * (v2 - v1)     # 6 drag_HI
    v2 = v2 + (gamma_ion() / (n2 + nf)) * (v1 - v2)  # 7 drag_HII（v1 已新）
    R = recomb()
    gam = gamma_ion()
    dv2 = (v2 - v1) * (v2 - v1)
    p1 = p1 + ((-(gam) / (n1 + nf)) * p1             # 8 che_HI
               + (R / (n2 + nf)) * p2
               + (((1.0 / 3.0) * R) * m_g) * dv2)
    R2 = recomb()
    gam2 = gamma_ion()
    dv1r = (v1 - v2) * (v1 - v2)
    p2 = p2 + ((-(R2) / (n2 + nf)) * p2              # 9 che_HII（p1 已新）
               + (gam2 / (n1 + nf)) * p1
               + (((1.0 / 3.0) * gam2) * m_g) * dv1r)
    grav = -((G * Mp) / r**2) + ((3 * G * Mstar) * r) / (a_orb ** 3)  # 10 gravity+tide
    v1 = v1 + grav
    v2 = v2 + grav
    return n1, n2, v1, v2, p1, p2


def test_factory_sources_apply_matches_card_formulas(lib):
    """rad_config.hh_rad_1d(data, n_beam=1) 一步 step 后 apply：Δn/Δvx/ΔP 对
    卡片公式逐项一致（Γ、η 加热、复合、CHE 压转+动能项、速度继承、重力），
    期望式按锁定 ADDS 带序逐条推进。动能项的 m 必须是克：误用 amu 会差
    ~1/AMU ≈ 6e23 倍，必红。

    容差 = 1e-12 相对 + 质量密度槽读回噪声底（显式，见 NOISE_F）：
    dn 的增量 ~速率·n 与库存密度同槽，读回有 ~2^-53·n 的表示噪声；
    任何公式级错误（错质量、漏项、错带序）都远大于该噪声底。"""
    assert (DATA_DIR / "hd209458b-1163.dat").exists(), "工厂数据表缺失"
    assert (DATA_DIR / "photoIon_sigma.dat").exists(), "工厂数据表缺失"
    # A1 补：球面 x1 任一界面（含 ghost）r<=0 → setup 失败。hh_rad_1d 的
    # r ∈ [R_p, 10 R_p]、ng=2 要求 nx > 2*9 = 18，原 nx=8 的左 ghost 面落在
    # 负半径，故抬到 32；断言全部逐格参数化，语义不变。
    cfg = rad_config.hh_rad_1d(DATA_DIR, radiation=True, nx=32, n_beam=1, tstop=1.0)
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 10, "工厂应编出 10 条带子"
        assert run.nbeam == 1

        # 均匀双流体初值（覆盖 floor 之上的整块；step 时 user 面会重钉）
        m = _mass_code(run, 0)
        n1, n2, T0 = 1.0e8, 1.0e8, 1500.0
        v1c, v2c = 1.0e5, -1.0e5
        _pack_block(lib, run, 0, SLOT_N, np.full((nzt, nyt, nxt), n1 * m))
        _pack_block(lib, run, 1, SLOT_N, np.full((nzt, nyt, nxt), n2 * m))
        _pack_block(lib, run, 0, SLOT_VX, np.full((nzt, nyt, nxt), v1c))
        _pack_block(lib, run, 1, SLOT_VX, np.full((nzt, nyt, nxt), v2c))
        _pack_block(lib, run, 0, SLOT_P, np.full((nzt, nyt, nxt), n1 * KB * T0))
        _pack_block(lib, run, 1, SLOT_P, np.full((nzt, nyt, nxt), n2 * KB * T0))

        run.step()  # rk2：两阶 scan→hydro→source
        assert run.t > 0.0
        assert np.isfinite(run.source_resid) and run.source_resid >= 0.0

        fld = run.fields_full()
        F = run.flux()[0, 0, 0, :]
        r = _centres(lib, run)[ng : ng + nx]
        assert F.shape == (nx,) and np.all(F >= 0.0)
        assert F[0] < F[-1], "最内格通量应低于最外格（floor 密度挡光）"

        n1v = fld["HI"]["n"][0, 0, :]
        n2v = fld["HII"]["n"][0, 0, :]
        vx1 = fld["HI"]["vx"][0, 0, :]
        vx2 = fld["HII"]["vx"][0, 0, :]
        p1 = fld["HI"]["P"][0, 0, :]
        p2 = fld["HII"]["P"][0, 0, :]
        unit_P = run._cfg.units.density * run._cfg.units.velocity ** 2

        en1, en2, ev1, ev2, ep1, ep2 = _factory_sequential_expected(
            n1v, n2v, vx1, vx2, p1, p2, F, r, cfg["constants"], unit_P)

        run.apply_sources()
        fld2 = run.fields_full()
        cases = (
            ("dn_HI", fld2["HI"]["n"][0, 0, :] - n1v, en1 - n1v, n1v),
            ("dn_HII", fld2["HII"]["n"][0, 0, :] - n2v, en2 - n2v, n2v),
            ("dvx_HI", fld2["HI"]["vx"][0, 0, :] - vx1, ev1 - vx1,
             np.abs(vx1) + 1.0),
            ("dvx_HII", fld2["HII"]["vx"][0, 0, :] - vx2, ev2 - vx2,
             np.abs(vx2) + 1.0),
            ("dP_HI", fld2["HI"]["P"][0, 0, :] - p1, ep1 - p1, p1),
            ("dP_HII", fld2["HII"]["P"][0, 0, :] - p2, ep2 - p2, p2),
        )
        for name, got, exp, bg in cases:
            tol = 1.0e-12 * np.abs(exp) + NOISE_F * np.abs(bg)
            bad = np.abs(got - exp) > tol
            assert not np.any(bad), (
                f"apply 增量 {name} 对卡片公式（顺序 ADDS）的最大偏差 "
                f"{float(np.max(np.abs(got - exp))):.3e} 超出容差；最坏格 "
                f"got={got[int(np.argmax(bad))]:.6e} "
                f"exp={exp[int(np.argmax(bad))]:.6e}"
            )

        # m 用克的显式哨兵：把动能项的克质量换成 amu 数，期望值必须偏出
        # 容差许多个量级——否则「对上了」可能只是动能项碰巧为 0
        iq = 3
        dv3 = float(vx2[iq] - vx1[iq])
        assert abs(dv3) > 1.0, "抽查格应有非零相对速度（动能项非零）"
        sig0 = cfg["constants"]["sigma_0"]
        hnu0 = cfg["constants"]["hnu_0"]
        gam3 = float(n1v[iq]) * ((sig0 * F[iq]) / hnu0)
        kin_g = (1.0 / 3.0) * gam3 * M_H_G * dv3 * dv3
        kin_amu = (1.0 / 3.0) * gam3 * AMU_H * dv3 * dv3
        assert abs(kin_g) > 0.0
        assert abs(kin_amu - kin_g) > 1.0e6 * abs(kin_g), (
            "克/amu 两种质量的动能项必须差许多个量级（否则哨兵失效）"
        )
        # 加热项与重力项非零（防空对空）
        heat = GM1 * ETA * float(n1v[iq]) * (sig0 * F[iq])
        assert heat > 0.0
        grav = -((G_P * M_P) / r**2)
        assert float(np.max(np.abs(grav))) > 1.0
    finally:
        run.close()


def test_rad_config_contract_public_constants_and_signature():
    """算例私有 rad_config.py 的自身合同（A5 补充卡片 §交什么 1/3）：

    * KB/AMU 从 xmf.config 进口（同一对象），文件里没有复制粘贴的
      玻尔兹曼/amu 数值字面量——防搬家时复制常数；
    * hh_rad_1d 签名仍 (data_dir, *, radiation, nx, n_beam, tstop,
      stretch, stretch_factor) -> dict（A1 补开的非均匀网格口，
      关键字参；未传的旋钮读 rad_config.RUN，stretch 为
      'none' 或 'log'）；
    * 加载方式是 importlib 按绝对路径：rad_config 不进 sys.modules
      （进了就说明走了裸 import，那必然动过 sys.path）。"""
    import inspect
    import sys

    assert rad_config.KB is KB and rad_config.AMU is AMU, (
        "rad_config 的 KB/AMU 必须从 xmf.config 进口，不得自持一份"
    )
    src = _RAD_CONFIG_PATH.read_text(encoding="utf-8")
    assert "from xmf.config import" in src
    for lit, name in ((repr(KB), "KB"), (repr(AMU), "AMU")):
        assert lit not in src, (
            f"rad_config.py 不得复制 {name} 的数值字面量 {lit}"
        )

    sig = inspect.signature(rad_config.hh_rad_1d)
    assert list(sig.parameters) == ["data_dir", "radiation", "nx",
                                    "n_beam", "tstop", "stretch",
                                    "stretch_factor"], (
        f"hh_rad_1d 参数表变了：{list(sig.parameters)}"
    )
    assert sig.parameters["data_dir"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("radiation", "nx", "n_beam", "tstop", "stretch",
                 "stretch_factor"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} 必须是关键字参"
        )
    assert sig.parameters["stretch"].default is None, (
        "stretch 未传时读 RUN，签名默认必须是 None"
    )
    assert rad_config.RUN.stretch in ("none", "log"), (
        "RUN.stretch 必须是 'none' 或 'log'"
    )
    assert str(sig.return_annotation).startswith("dict")

    assert "rad_config" not in sys.modules, (
        "rad_config 应由 importlib 按路径加载，不进 sys.modules（不动 sys.path）"
    )


# --- 4. 重力专测：两支同加速度，只改 v 不改 n、P ------------------------------


def _grav_two_fluid_cfg() -> dict:
    cfg = _base_cfg(nx=16, x1min=1.0, x1max=2.0)
    cfg["constants"] = {"G": GRAV_G, "M": GRAV_M}
    cfg["sources"] = [{"name": "g", "expr": {"dvx": "-G*M/r**2"}}]
    return cfg


def test_gravity_apply_two_fluids_only_velocity(lib):
    """裸 dvx 带：两支流体同一 −GM/r²（1e-12）；n、P（与 vy、vz）逐位不动。"""
    run = xmf.setup(_grav_two_fluid_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 1
        # 两支不同的密度/压强/速度：重力增量必须与状态无关、两支相同
        for f, (rho_f, vx_f, p_f) in enumerate(((1.5, 0.3, 1.1), (2.5, -0.2, 0.9))):
            _pack_block(lib, run, f, SLOT_N, np.full((nzt, nyt, nxt), rho_f))
            _pack_block(lib, run, f, SLOT_VX, np.full((nzt, nyt, nxt), vx_f))
            _pack_block(lib, run, f, SLOT_P, np.full((nzt, nyt, nxt), p_f))

        r = _centres(lib, run)[ng : ng + nx]
        exact = -GRAV_G * GRAV_M / r**2
        before = {(f, s): _unpack(lib, run, f, s)
                  for f in range(2) for s in range(5)}
        run.apply_sources()
        assert run.t == 0.0, "apply 不推进时间"

        for f in range(2):
            dvx = (_unpack(lib, run, f, SLOT_VX) - before[(f, SLOT_VX)]).reshape(
                (nzt, nyt, nxt))[ng, ng, ng : ng + nx]
            rel = float(np.max(np.abs((dvx - exact) / exact)))
            assert rel <= 1.0e-12, (
                f"流体 {f}：apply 增量应为 -GM/r²，最大相对差 {rel:.3e}"
            )
            for s, name in ((SLOT_N, "n"), (SLOT_P, "P"),
                            (SLOT_VY, "vy"), (SLOT_VZ, "vz")):
                assert np.array_equal(_unpack(lib, run, f, s), before[(f, s)]), (
                    f"流体 {f}：重力带不得改动 {name}（整块含 ghost 逐位）"
                )
    finally:
        run.close()


# --- 5. 链式 ①③④ 对解析：手算 (S1·S2+S4)·S3 预测整步 -------------------------


def _chain_cfg() -> dict:
    """两流体、无辐射、无 T/F 读口（mass_code 保持 1）：
    cx 对（dn）、流体 0 的拖曳（dvx_HI）、流体 1 的加热（dP_HII）。
    x1 边界 outflow（零梯度）：均匀态走 hydro 逐位不动，步后差异全部
    来自源步。"""
    cfg = _base_cfg(x1min=1.0, x1max=2.0)
    cfg["boundary"] = {"x1": "outflow"}
    cfg["constants"] = {"k1": K1, "k2": K2, "k3": K3}
    cfg["sources"] = [
        {"name": "cx_HI", "expr": {"dn_HI": "-k1*n_HI*n_HII"}},
        {"name": "cx_HII", "expr": {"dn_HII": "k1*n_HI*n_HII"}},
        {"name": "drag_HI", "expr": {"dvx_HI": "-k2*n_HII"}},
        {"name": "heat_HII", "expr": {"dP_HII": "k3*n_HII"}},
    ]
    return cfg


def _segments_from_card(m, w, lc, gmm1):
    """①④ 与提升 R：按卡片公式手算（ρ̇=mṅ，(ρv)̇=mṅv+ρv̇，
    Ė=Ṗ/(γ−1)+½|v|²ρ̇+ρ v·v̇）。w=(n,vx,vy,vz,P)，lc=(ṅ,v̇x,v̇y,v̇z,Ṗ)。"""
    n, vx, vy, vz, _ = w
    ndot, dx_, dy_, dz_, pdot = lc
    rho = m * n
    v2 = vx * vx + vy * vy + vz * vz
    vdv = vx * dx_ + vy * dy_ + vz * dz_
    S1 = np.zeros((5, 5))
    S1[0, 0] = m
    S1[1, 0] = m * vx
    S1[1, 1] = rho
    S1[2, 0] = m * vy
    S1[2, 2] = rho
    S1[3, 0] = m * vz
    S1[3, 3] = rho
    S1[4, 0] = 0.5 * m * v2
    S1[4, 1] = rho * vx
    S1[4, 2] = rho * vy
    S1[4, 3] = rho * vz
    S1[4, 4] = 1.0 / gmm1
    S4 = np.zeros((5, 5))
    S4[1, 0] = m * dx_
    S4[1, 1] = m * ndot
    S4[2, 0] = m * dy_
    S4[2, 2] = m * ndot
    S4[3, 0] = m * dz_
    S4[3, 3] = m * ndot
    S4[4, 0] = m * vdv
    S4[4, 1] = m * (ndot * vx + n * dx_)
    S4[4, 2] = m * (ndot * vy + n * dy_)
    S4[4, 3] = m * (ndot * vz + n * dz_)
    R = np.array([
        m * ndot,
        m * ndot * vx + rho * dx_,
        m * ndot * vy + rho * dy_,
        m * ndot * vz + rho * dz_,
        pdot / gmm1 + 0.5 * m * v2 * ndot + rho * vdv,
    ])
    return S1, S4, R


def _s3_from_card(m, w, gmm1):
    """③ dw/dU：n=ρ/m，v=mom/ρ，P=(γ−1)(E−ek)。"""
    _, vx, vy, vz, _ = w
    rho = m * w[0]
    rs = rho if rho > 1.0e-30 else 1.0e-30
    v2 = vx * vx + vy * vy + vz * vz
    S3 = np.zeros((5, 5))
    S3[0, 0] = 1.0 / m
    S3[1, 0] = -vx / rs
    S3[1, 1] = 1.0 / rs
    S3[2, 0] = -vy / rs
    S3[2, 2] = 1.0 / rs
    S3[3, 0] = -vz / rs
    S3[3, 3] = 1.0 / rs
    S3[4, 0] = 0.5 * gmm1 * v2
    S3[4, 1] = -gmm1 * vx
    S3[4, 2] = -gmm1 * vy
    S3[4, 3] = -gmm1 * vz
    S3[4, 4] = gmm1
    return S3


def _s2_chain_hand(a, b):
    """② dL_c/dw 手算（本用例带组；行/列 = f*5+slot）。"""
    S2 = np.zeros((10, 10))
    S2[0, 0] = -K1 * b
    S2[0, 5] = -K1 * a
    S2[5, 0] = K1 * b
    S2[5, 5] = K1 * a
    S2[1, 5] = -K2
    S2[9, 5] = K3  # dP_HII = 流体 1 槽 P = 下标 5+4=9
    return S2


def test_source_chain_step_matches_analytic_one_three_four(lib):
    """已知 (w, L_c)：核的一步隐式 step 对
    (I − dt·(S1·S2+S4)·S3) dU = dt·R 的手算预测（1e-12）；
    核的 ②（jacobian）与手算 S2 互验；ṅ 与 v̇ 的混合项（含 vy 目标）都在。"""
    run = xmf.setup(_chain_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 4
        assert run.nbeam == 0, "本用例无辐射：mass_code 保持 1，码值即原值"

        # O(1) 均匀初值：n、vx、vy、P 两支都非零（vz 保持 0）
        w0 = {0: (1.2, 0.9, 0.3, 0.0, 1.1), 1: (0.8, -0.4, -0.2, 0.0, 0.7)}
        for f in range(2):
            for s, val in enumerate(w0[f]):
                if val != 0.0:
                    _pack_block(lib, run, f, s, np.full((nzt, nyt, nxt), val))

        # 核的 ②（首步前可用：带子不读 F/T）对手算 S2
        a, b = w0[0][0], w0[1][0]
        S2_hand = _s2_chain_hand(a, b)
        J_ker = run.jacobian(ng, ng, ng + 3)
        assert J_ker.shape == (10, 10)
        assert np.allclose(J_ker, S2_hand, rtol=1.0e-12, atol=0.0), (
            "核的 jacobian（②）与手算 S2 不一致：max|diff|="
            f"{float(np.max(np.abs(J_ker - S2_hand))):.3e}"
        )

        # 手算 ①③④ 与 R（每支一块，块对角拼 10×10）
        gmm1 = GAMMA - 1.0
        lc = {
            0: (-K1 * a * b, -K2 * b, 0.0, 0.0, 0.0),
            1: (K1 * a * b, 0.0, 0.0, 0.0, K3 * b),
        }
        S1 = np.zeros((10, 10))
        S4 = np.zeros((10, 10))
        S3 = np.zeros((10, 10))
        R = np.zeros(10)
        for f in range(2):
            s1, s4, r5 = _segments_from_card(1.0, w0[f], lc[f], gmm1)
            s3 = _s3_from_card(1.0, w0[f], gmm1)
            S1[f * 5:f * 5 + 5, f * 5:f * 5 + 5] = s1
            S4[f * 5:f * 5 + 5, f * 5:f * 5 + 5] = s4
            S3[f * 5:f * 5 + 5, f * 5:f * 5 + 5] = s3
            R[f * 5:f * 5 + 5] = r5
        assert np.any(S4 != 0.0), "④ 块在本用例必须非零（ṅ、v̇ 都非零）"
        Jtot = (S1 @ S2_hand + S4) @ S3

        # U* = cons(w)（outflow 零梯度 + 均匀态：hydro 逐位不动，
        # 结合序与核内 xmf_source_step 相同）
        Ustar = np.zeros(10)
        for f in range(2):
            n, vx, vy, vz, P = w0[f]
            rho = n  # m = 1
            Ustar[f * 5 + 0] = rho
            Ustar[f * 5 + 1] = rho * vx
            Ustar[f * 5 + 2] = rho * vy
            Ustar[f * 5 + 3] = rho * vz
            Ustar[f * 5 + 4] = P / gmm1 + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

        run.step()
        assert run.t > 0.0
        dt = run.dt
        dU = np.linalg.solve(np.eye(10) - dt * Jtot, dt * R)
        Unew = Ustar + dU

        for f in range(2):
            rho = Unew[f * 5 + 0]
            pred = np.array([
                rho,
                Unew[f * 5 + 1] / rho,
                Unew[f * 5 + 2] / rho,
                Unew[f * 5 + 3] / rho,
                gmm1 * (Unew[f * 5 + 4]
                        - 0.5 * (Unew[f * 5 + 1] ** 2 + Unew[f * 5 + 2] ** 2
                                 + Unew[f * 5 + 3] ** 2) / rho),
            ])
            for s, name in enumerate("rho vx vy vz P".split()):
                got = _unpack(lib, run, f, s).reshape(
                    (nzt, nyt, nxt))[ng, ng, ng : ng + nx]
                scale = np.maximum(np.abs(pred[s]), 1.0e-300)
                rel = float(np.max(np.abs(got - pred[s]) / scale))
                assert rel <= 1.0e-12, (
                    f"流体 {f} 槽 {name}：step 结果与手算链式预测的"
                    f"最大相对差 {rel:.3e} 超过 1e-12"
                )
        # vz 应严格为 0（无 v̇z、无初值）
        for f in range(2):
            assert np.all(_unpack(lib, run, f, SLOT_VZ) == 0.0)

        # 哨兵：状态确实被链式推进了（n、vx、vy、P 目标都动过，不只 VX1）
        got_n2 = _unpack(lib, run, 1, SLOT_N).reshape((nzt, nyt, nxt))[ng, ng, ng]
        got_vy1 = _unpack(lib, run, 0, SLOT_VY).reshape((nzt, nyt, nxt))[ng, ng, ng]
        got_p2 = _unpack(lib, run, 1, SLOT_P).reshape((nzt, nyt, nxt))[ng, ng, ng]
        assert abs(got_n2 - w0[1][0]) > 1.0e-6, "n_HII 应被 cx 对推进"
        # 纯质量转移（无横向力）下横向速度不变：(ρvy)̇ = mṅvy 与
        # ρ̇ = mṅ 精确相消，vy = (ρvy)/ρ 保持原值。钉 ① 的 ṅ·v 横向
        # 动量提升项与 ρ̇ 的自洽——任一项系数写错，vy 会漂 O(|ṅ|dt·|vy|/ρ)
        # ~1e-5，远大于 1e-12。
        assert abs(got_vy1 - w0[0][2]) <= 1.0e-12, (
            "纯质量转移下 vy_HI 必须不变（① 的 ṅ·vy 提升项与 ρ̇ 相消）"
        )
        # 但守恒横向动量 ρ·vy 必须动（提升项非零，不是空对空）
        got_rho1 = _unpack(lib, run, 0, SLOT_N).reshape(
            (nzt, nyt, nxt))[ng, ng, ng]
        assert abs(got_rho1 * got_vy1 - w0[0][0] * w0[0][2]) > 1.0e-6, (
            "(ρvy)_HI 应被 mṅ·vy 提升项推进"
        )
        assert abs(got_p2 - w0[1][4]) > 1.0e-6, "P_HII 应被加热带推进"
        assert np.isfinite(run.source_resid)
        assert 0.0 <= run.source_resid < 1.0e-3, (
            f"近线性系统的单步残差应很小，实测 {run.source_resid:.3e}"
        )
    finally:
        run.close()


# --- 6. ④ 专门探针：常数 dvx 带下压强签名 -------------------------------------


def test_four_segment_exists_via_pressure_signature(lib):
    """常数带 dvx = g0：J（=S1·S2·S3 侧）全 0，②给不出任何贡献；
    但 ④ = ∂T/∂w|Lc 非零。对线性系统这就是精确后向欧拉：
        (ρvx)' = ρvx + dt·ρg0，dE = dt·ρvx·g0 + dt·g0·d(ρvx)，
    故一步 step 必须给出
        v -> v + dt·g0，ρ 不变，
        P -> P + (1/2)(γ−1)ρ(dt·g0)²。
    漏 ④ 时能量只显式加 dt·ρvx·g0，P 落在
    P − (1/2)(γ−1)ρ(dt·g0)²，差 (γ−1)ρ(dt·g0)²，1e-12 容差下必红。
    边界 outflow：均匀态 hydro 逐位不动，签名全来自源步。"""
    g0 = 1.0
    cfg = _base_cfg(x1min=1.0, x1max=2.0)
    cfg["boundary"] = {"x1": "outflow"}
    cfg["fluids"] = [{"name": "HI", "mass": AMU_H, "charge": 0, "kind": "hd"}]
    cfg["constants"] = {"g0": g0}
    cfg["sources"] = [{"name": "acc", "expr": {"dvx": "g0"}}]
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 1
        rho0, vx0, p0 = 1.0, 1.0, 1.0
        _pack_block(lib, run, 0, SLOT_N, np.full((nzt, nyt, nxt), rho0))
        _pack_block(lib, run, 0, SLOT_VX, np.full((nzt, nyt, nxt), vx0))
        _pack_block(lib, run, 0, SLOT_P, np.full((nzt, nyt, nxt), p0))

        # ② 侧全 0：常数带没有状态依赖列——④ 若存在只能来自提升本身
        J = run.jacobian(ng, ng, ng + nx // 2)
        assert J.shape == (5, 5)
        assert np.all(J == 0.0), (
            f"常数带的 jacobian 应逐位全 0，实测 max|J|={float(np.max(np.abs(J))):.3e}"
        )

        run.step()
        assert run.t > 0.0
        dt = run.dt
        assert dt > 0.0

        got_rho = _unpack(lib, run, 0, SLOT_N).reshape((nzt, nyt, nxt))[ng, ng, ng : ng + nx]
        got_vx = _unpack(lib, run, 0, SLOT_VX).reshape((nzt, nyt, nxt))[ng, ng, ng : ng + nx]
        got_p = _unpack(lib, run, 0, SLOT_P).reshape((nzt, nyt, nxt))[ng, ng, ng : ng + nx]

        exp_v = vx0 + dt * g0
        exp_p_with4 = p0 + 0.5 * GM1 * rho0 * (dt * g0) ** 2   # 含 ④
        exp_p_no4 = p0 - 0.5 * GM1 * rho0 * (dt * g0) ** 2     # 漏 ④ 的错值

        assert float(np.max(np.abs(got_rho - rho0))) <= 1.0e-13, (
            "常数体力不改密度"
        )
        rel_v = float(np.max(np.abs((got_vx - exp_v) / exp_v)))
        assert rel_v <= 1.0e-12, (
            f"v 应为 v + dt·g0，最大相对差 {rel_v:.3e}"
        )
        rel_p = float(np.max(np.abs((got_p - exp_p_with4) / exp_p_with4)))
        assert rel_p <= 1.0e-12, (
            f"P 应为 P + (1/2)(γ−1)ρ(dt·g)²（④ 签名），最大相对差 {rel_p:.3e}"
        )
        # 判别力哨兵：含 ④ 与漏 ④ 的预测必须相隔许多个容差，
        # 且实测值贴近含 ④ 一侧
        gap = abs(exp_p_with4 - exp_p_no4)
        assert gap > 1.0e6 * (1.0e-12 * abs(exp_p_with4)), (
            "两种预测间隔不够，探针判别力不足"
        )
        assert float(np.max(np.abs(got_p - exp_p_no4))) > 0.9 * gap, (
            "实测 P 若贴近漏 ④ 的预测，说明 ④ 没进链式"
        )
        # 横速度保持 0
        assert np.all(_unpack(lib, run, 0, SLOT_VY) == 0.0)
        assert np.all(_unpack(lib, run, 0, SLOT_VZ) == 0.0)
    finally:
        run.close()
