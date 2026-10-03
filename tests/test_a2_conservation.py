"""A.2 回归：四流体源组的守恒律（核数 / 准中性 / 总动量）。

合同来自 docs/SESSION.md B3 卡片 §4（A.2 三条回归，作模块单测，短积分、
粗笛卡尔均匀盒，不叠在激波管上）：
  (a) H 核数 n_HI + n_HII 与 He 核数 n_HEI + n_HEII 的盒积分相对漂移
      低于 NUC_DRIFT_TOL（reflect 闭盒，无 outflow）；
  (b) 准中性投影噪声：核内派生读 n_e（电荷加权和，只认 q_s != 0，从不
      认名字）与 n_HII + n_HEII 之差不超过 QNEUT_TOL（相对 n_e 尺度）；
  (c) 只有碰撞 / CX / 库仑，无外力、无辐射：四流体总动量相对漂移低于
      MOM_DRIFT_TOL。

设计说明：
  - 16 胞笛卡尔均匀盒、dt_max = 0.5 s、20 步。源组镜像
    examples/rad_hhe_pe_1d/hhe_config.py 的带形：CX 反对称四带 +
    速度继承 drag 四带 + CHE_PRS_TRANS + 碰撞动量/压强带（反对称差）+
    库仑三带（离子侧两条、电子侧精确反号）。速率的具体函数形式由
    test_hhe_factory / test_source_step 覆盖，此处碰撞率取常数、CX 率
    保留工厂形式并把 k_cx1 放大（物理值 1.25e-15 在 10 s 内转换量太
    小），保证 20 步内有可见电荷交换，验的是反对称记账而不是速率值。
  - 初态取单一温度（四组分与电子同温）＝压强平衡初值：单步线性化后
    向欧拉源步在 P_e 弛豫瞬态上会破坏反对称和（诊断实测：非平衡初态
    下核数伪漂移最高 ~4e-5、饱和不增长，与样例例锁 dt_max 的瞬态同
    源）。从平衡初态出发，源步残差 ~1e-10，守恒回到舍入级。
  - (a) 用 reflect 闭盒 + 全流体静止：每支流体看到自己的镜像鬼胞，
    壁面质量通量恒为零，核数不因边界漏；静止初态下 hydro 是 no-op，
    断言只压源项记账。卡片口径「周期/反射盒，无 outflow」——核内 BC
    表无周期（config.BC_TYPES），故选 reflect。
  - (c) 用零梯度 copy 边界 + 均匀非零速度：反射壁对净运动流体施加
    冲量（动量汇，设计使然，静止初态又无动量交换可测），不能用于动
    量断言；copy 鬼胞在均匀态下与原胞相同，各界面通量相同、通量差
    逐位为零，hydro 对 Σp 的贡献精确为零，残差只剩源项记账——即离
    散周期盒的等价物。
  - 动量守恒的解析依据：碰撞带 Δp_f = m_f,amu·AMU·n_f·(ν/m_f,amu)
    ·n_other·Δv = ν·AMU·n_f·n_other·Δv，两侧因子相同、对任意质量对
    精确抵消；CX + drag 对（HI<->HII 同质量、速度继承补齐）逐位守恒；
    库仑/碰撞压强带只动 P，不动 vx。

容差依据（实测本底 -> 钉死值）：
  - 核数：平衡初态下逐位 0（求和舍入 ~1e-16）；真实泄漏（缺反对称
    伴带、符号错）>= 1e-2 -> 钉 1e-12。
  - 准中性：往返舍入 ~1e-15 相对 -> 钉 1e-12。
  - 动量：速度槽更新舍入本底 ~1.6e-11（20 步，|Δv| = 2e4）；真实泄
    漏（漏继承带 ~2.4e-4、反对称符号错 >= 1e-3）-> 钉 1e-9。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；钉 n 与 P、四支
流体全查，不只测 VX1。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU, KB

_PTR = ctypes.POINTER(ctypes.c_double)

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

# ---- 钉死的容差（依据见文件头） ------------------------------------------------
NUC_DRIFT_TOL = 1.0e-12   # (a) 盒积分核数相对漂移上限
QNEUT_TOL = 1.0e-12       # (b) |n_e - (n_HII+n_HEII)| / (n_HII+n_HEII) 上限
MOM_DRIFT_TOL = 1.0e-9    # (c) 四流体总动量相对漂移上限

NSTEP = 20
DT_MAX = 0.5              # s
T_ALL = 2000.0            # 单一温度：压强平衡初值，躲开源步弛豫瞬态

# ---- 组分表（B3 卡锁死） -----------------------------------------------------
M_HI_AMU = 1.00794
M_HEI_AMU = 4.002602
M_HII_AMU = 1.00794
M_HEII_AMU = 4.002053
M_E_AMU = 5.4858e-4

FLUIDS = [
    {"name": "HI", "mass": M_HI_AMU, "charge": 0, "kind": "neutral"},
    {"name": "HEI", "mass": M_HEI_AMU, "charge": 0, "kind": "neutral"},
    {"name": "HII", "mass": M_HII_AMU, "charge": 1, "kind": "ion"},
    {"name": "HEII", "mass": M_HEII_AMU, "charge": 1, "kind": "ion"},
]
MASS_AMU = (M_HI_AMU, M_HEI_AMU, M_HII_AMU, M_HEII_AMU)

# ---- 均匀初态（n 为数密度，cm^-3） -------------------------------------------
N_HI = 1.0e8
N_HEI = 7.76441e6          # 0.0776441 * N_HI（B3 卡 He 丰度）
N_HII = 1.0e6
N_HEII = 7.76441e4
N_E = N_HII + N_HEII       # 准中性
VX_HI = 1.0e4              # (c) 用：HI/HII 异速 -> 碰撞/继承动量交换可见
VX_HII = -1.0e4            # m_HI == m_HII -> CX+drag 对逐位守恒


@pytest.fixture(scope="module")
def lib():
    return load()


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


def _box_cfg(sources, boundary: str):
    """16 胞笛卡尔均匀盒；radiation 关但装两吸收体行表（B3 辅助口带载）。"""
    c = xmf.shock_tube()
    c["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    c["grid"]["n"] = [16, 1, 1]
    c["grid"]["geometry"] = "cartesian"
    c["grid"]["x1min"] = 0.0
    c["grid"]["x1max"] = 1.6e7
    c["boundary"] = {"x1beg": boundary, "x1end": boundary}
    c["fluids"] = FLUIDS
    c["electron"] = {"mode": "pressure-only"}
    c["sources"] = sources
    c["constants"] = {
        # CX：工厂形式（hhe_config.py :381-395），k_cx1 放大见文件头注。
        "k_cx1": 1.0e-10,
        "k_cx2": 1.75e-11,
        "cx1_exp": -0.25,
        "cx2_exp": 0.75,
        "T_cx2_act": 128000.0,
        "T_rec_ref": 300.0,
        "T_floor": 1.0,
        "n_floor": 1.0e-30,
        # 碰撞率取常数（反对称记账与函数形式无关；放大到 10 s 可见）。
        "nu_hh": 1.0e-9,
        "nu_hehe": 1.0e-9,
        "nu_hhe": 1.0e-9,
        "nu_ii": 1.0e-9,
        # 库仑（B2 同形）：nu_e = k_coul / T_e^{3/2}。
        "k_coul": 0.0299,
        "m_HI_amu": M_HI_AMU,
        "m_HEI_amu": M_HEI_AMU,
        "m_HII_amu": M_HII_AMU,
        "m_HEII_amu": M_HEII_AMU,
        "m_e_amu": M_E_AMU,
        "m_HI_g": M_HI_AMU * AMU,
        "m_HEI_g": M_HEI_AMU * AMU,
        "m_HII_g": M_HII_AMU * AMU,
        "m_HEII_g": M_HEII_AMU * AMU,
        "m_e_g": M_E_AMU * AMU,
    }
    c["time"] = {"cfl": 0.4, "tstop": 1.0e9, "dt_max": DT_MAX}
    # 关辐射、两吸收体行表：开 phys 口让 n_e / T_e 以真实质量转换，
    # 扫描不会跑（enabled=False），FLUX 读恒 0。
    c["radiation"] = {
        "enabled": False,
        "absorbers": ["HI", "HEI"],
        "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
        "bins": {
            "n": 1,
            "sigma": [[1.0e-18], [1.0e-18]],
            "hnu": [2.0e-11],
            "flux_inf": [0.0],
        },
    }
    c["ic"] = {"kind": "none"}
    return c


def _cx_coul_sources():
    """碰撞/CX/库仑源组：镜像工厂带形，去掉辐射相关带与重力。"""
    t_cx1 = ("(T_HI * m_HEII_amu + T_HEII * m_HI_amu) "
             "/ (m_HI_amu + m_HEII_amu)")
    t_cx2 = ("(T_HII * m_HEI_amu + T_HEI * m_HII_amu) "
             "/ (m_HII_amu + m_HEI_amu)")
    cx1 = f"(k_cx1 * (T_rec_ref / (({t_cx1}) + T_floor))**cx1_exp) * n_HI * n_HEII"
    cx2 = (
        f"(k_cx2 * (T_rec_ref / (({t_cx2}) + T_floor))**cx2_exp "
        f"* exp(-T_cx2_act / (({t_cx2}) + T_floor))) * n_HII * n_HEI"
    )

    def che_prs_trans(f1: str, f2: str, nloss: str, nget: str) -> str:
        dv = f"(vx_{f2} - vx_{f1})"
        return (
            f"-({nloss}) / (n_{f1} + n_floor) * P_{f1}"
            f" + ({nget}) / (n_{f2} + n_floor) * P_{f2}"
            f" + (1.0 / 3.0) * ({nget}) * m_{f1}_g * ({dv} * {dv})"
        )

    def col_prs_trans(f1: str, f2: str) -> str:
        dv = f"(vx_{f1} - vx_{f2})"
        return (
            f"2.0 * (n_{f1} * P_{f2} - n_{f2} * P_{f1})"
            f" + (2.0 / 3.0) * m_{f2}_g * n_{f1} * n_{f2} * ({dv} * {dv})"
        )

    te_floor = "(T_e + T_floor)"
    nu_e = f"k_coul / ({te_floor}**0.5 * {te_floor})"
    v_e = "(n_HII * vx_HII + n_HEII * vx_HEII) / (n_e + n_floor)"

    def col_prs_trans_e(fion: str) -> str:
        dv = f"(vx_{fion} - ({v_e}))"
        return (
            f"2.0 * (n_{fion} * P_e - n_e * P_{fion})"
            f" + (2.0 / 3.0) * m_e_g * n_{fion} * n_e * ({dv} * {dv})"
        )

    coul_hii = f"({nu_e}) / (m_HII_amu + m_e_amu) * ({col_prs_trans_e('HII')})"
    coul_heii = f"({nu_e}) / (m_HEII_amu + m_e_amu) * ({col_prs_trans_e('HEII')})"

    return [
        # CX 化学：反对称四带（核数按元素逐格守恒）。
        {"name": "chem_HI", "expr": {"dn_HI": f"-({cx1}) + ({cx2})"}},
        {"name": "chem_HII", "expr": {"dn_HII": f"({cx1}) - ({cx2})"}},
        {"name": "chem_HEI", "expr": {"dn_HEI": f"-({cx2}) + ({cx1})"}},
        {"name": "chem_HEII", "expr": {"dn_HEII": f"({cx2}) - ({cx1})"}},
        # CX 速度继承（工厂同形：rec 带携源流体速度进来）。
        {"name": "drag_HI",
         "expr": {"dvx_HI": f"({cx2}) / (n_HI + n_floor) * (vx_HII - vx_HI)"}},
        {"name": "drag_HII",
         "expr": {"dvx_HII": f"({cx1}) / (n_HII + n_floor) * (vx_HI - vx_HII)"}},
        {"name": "drag_HEI",
         "expr": {"dvx_HEI": f"({cx1}) / (n_HEI + n_floor) * (vx_HEII - vx_HEI)"}},
        {"name": "drag_HEII",
         "expr": {"dvx_HEII": f"({cx2}) / (n_HEII + n_floor) * (vx_HEI - vx_HEII)"}},
        # CHE 压强转移。
        {"name": "che_HI", "expr": {"dP_HI": che_prs_trans("HI", "HII", cx1, cx2)}},
        {"name": "che_HII", "expr": {"dP_HII": che_prs_trans("HII", "HI", cx2, cx1)}},
        {"name": "che_HEI", "expr": {"dP_HEI": che_prs_trans("HEI", "HEII", cx2, cx1)}},
        {"name": "che_HEII", "expr": {"dP_HEII": che_prs_trans("HEII", "HEI", cx1, cx2)}},
        # 碰撞动量交换：反对称差 -> 总动量守恒（常数 ν）。
        {"name": "coll_mom_HI", "expr": {"dvx_HI": (
            "nu_hh / m_HI_amu * n_HII * (vx_HII - vx_HI)"
            " + nu_hhe / m_HI_amu * n_HEI * (vx_HEI - vx_HI)")}},
        {"name": "coll_mom_HEI", "expr": {"dvx_HEI": (
            "nu_hehe / m_HEI_amu * n_HEII * (vx_HEII - vx_HEI)"
            " + nu_hhe / m_HEI_amu * n_HI * (vx_HI - vx_HEI)")}},
        {"name": "coll_mom_HII", "expr": {"dvx_HII": (
            "nu_hh / m_HII_amu * n_HI * (vx_HI - vx_HII)"
            " + nu_ii / m_HII_amu * n_HEII * (vx_HEII - vx_HII)")}},
        {"name": "coll_mom_HEII", "expr": {"dvx_HEII": (
            "nu_hehe / m_HEII_amu * n_HEI * (vx_HEI - vx_HEII)"
            " + nu_ii / m_HEII_amu * n_HII * (vx_HII - vx_HEII)")}},
        # 碰撞压强交换（COL_PRS_TRANS 形，净 dn = 0）。
        {"name": "coll_prs_HI", "expr": {"dP_HI": (
            f"nu_hh / (m_HI_amu + m_HII_amu) * ({col_prs_trans('HI', 'HII')})"
            f" + nu_hhe / (m_HI_amu + m_HEI_amu) * ({col_prs_trans('HI', 'HEI')})")}},
        {"name": "coll_prs_HEI", "expr": {"dP_HEI": (
            f"nu_hehe / (m_HEI_amu + m_HEII_amu) * ({col_prs_trans('HEI', 'HEII')})"
            f" + nu_hhe / (m_HEI_amu + m_HI_amu) * ({col_prs_trans('HEI', 'HI')})")}},
        {"name": "coll_prs_HII", "expr": {"dP_HII": (
            f"nu_hh / (m_HII_amu + m_HI_amu) * ({col_prs_trans('HII', 'HI')})"
            f" + nu_ii / (m_HII_amu + m_HEII_amu) * ({col_prs_trans('HII', 'HEII')})")}},
        {"name": "coll_prs_HEII", "expr": {"dP_HEII": (
            f"nu_hehe / (m_HEII_amu + m_HEI_amu) * ({col_prs_trans('HEII', 'HEI')})"
            f" + nu_ii / (m_HEII_amu + m_HII_amu) * ({col_prs_trans('HEII', 'HII')})")}},
        # 离子-电子库仑：只动压强；电子侧精确反号。
        {"name": "coul_prs_HII", "expr": {"dP_HII": coul_hii}},
        {"name": "coul_prs_HEII", "expr": {"dP_HEII": coul_heii}},
        {"name": "coul_prs_e", "expr": {"dP_e": f"-(({coul_hii}) + ({coul_heii}))"}},
    ]


def _pack_uniform(lib, run, vxs):
    """整块（含鬼胞）填同一均匀态：单一温度，各流体指定 vx。"""
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    shape = (nzt, nyt, nxt)
    ns = (N_HI, N_HEI, N_HII, N_HEII)
    for f, (n, vx) in enumerate(zip(ns, vxs)):
        m_g = MASS_AMU[f] * AMU
        _pack_block(lib, run, f, SLOT_N, np.full(shape, n * m_g))
        _pack_block(lib, run, f, SLOT_VX, np.full(shape, vx))
        _pack_block(lib, run, f, SLOT_VY, np.zeros(shape))
        _pack_block(lib, run, f, SLOT_VZ, np.zeros(shape))
        _pack_block(lib, run, f, SLOT_P, np.full(shape, n * KB * T_ALL))
    _write_pe(lib, run, np.full(shape, N_E * KB * T_ALL))


def _inner(run):
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    return (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))


def _reshape(run, buf):
    g = _geom(run)
    return buf.reshape(g[6], g[5], g[4])


def _nuclei_sums(lib, run):
    """盒积分核数：N_H = Σ (rho_HI/m_HI + rho_HII/m_HII)，He 同。"""
    inner = _inner(run)
    tot_h = tot_he = 0.0
    for f in range(4):
        rho = _reshape(run, _unpack_block(lib, run, f, SLOT_N))
        n = float(np.sum(rho[inner] / (MASS_AMU[f] * AMU)))
        if f in (0, 2):
            tot_h += n
        else:
            tot_he += n
    return tot_h, tot_he


def _species_sum(lib, run, f: int):
    inner = _inner(run)
    rho = _reshape(run, _unpack_block(lib, run, f, SLOT_N))
    return float(np.sum(rho[inner] / (MASS_AMU[f] * AMU)))


def _total_momentum(lib, run):
    """盒积分总动量 Σ_f Σ_cells rho_f * vx_f（体积因子相对漂移里消掉）。"""
    inner = _inner(run)
    p_tot = 0.0
    for f in range(4):
        rho = _reshape(run, _unpack_block(lib, run, f, SLOT_N))
        vx = _reshape(run, _unpack_block(lib, run, f, SLOT_VX))
        p_tot += float(np.sum(rho[inner] * vx[inner]))
    return p_tot


def _mean_vx(lib, run, f: int):
    inner = _inner(run)
    vx = _reshape(run, _unpack_block(lib, run, f, SLOT_VX))
    return float(np.mean(vx[inner]))


def _run_box(lib, boundary: str, vxs):
    cfg = _box_cfg(_cx_coul_sources(), boundary)
    run = xmf.setup(cfg)
    try:
        _pack_uniform(lib, run, vxs)
        h0, he0 = _nuclei_sums(lib, run)
        p0 = _total_momentum(lib, run)
        heii0 = _species_sum(lib, run, 3)
        hii0 = _species_sum(lib, run, 2)
        vx_hii0 = _mean_vx(lib, run, 2)
        for _ in range(NSTEP):
            run.step()
        return {
            "h0": h0, "he0": he0,
            "h1": _nuclei_sums(lib, run)[0], "he1": _nuclei_sums(lib, run)[1],
            "p0": p0, "p1": _total_momentum(lib, run),
            "heii0": heii0, "heii1": _species_sum(lib, run, 3),
            "hii0": hii0, "hii1": _species_sum(lib, run, 2),
            "vx_hii0": vx_hii0, "vx_hii1": _mean_vx(lib, run, 2),
            "resid": run.source_resid,
        }
    finally:
        run.close()


# --- (a) reflect 闭盒、静止初态：H / He 核数守恒 ------------------------------


def test_a2_nuclei_conserved_reflect_box(lib):
    """CX 真实转换下，盒积分 N_H 与 N_He 相对漂移 < NUC_DRIFT_TOL。"""
    r = _run_box(lib, "reflect", (0.0, 0.0, 0.0, 0.0))
    # 防空转：20 步内 CX1 必须真的转换了可见量的 HEII -> HEI / HI -> HII。
    assert r["heii1"] < 0.99 * r["heii0"], (
        f"CX 没在跑：HEII 盒积分 {r['heii0']:.6e} -> {r['heii1']:.6e}")
    assert r["hii1"] > 1.001 * r["hii0"], (
        f"CX 没在跑：HII 盒积分 {r['hii0']:.6e} -> {r['hii1']:.6e}")
    drift_h = abs(r["h1"] - r["h0"]) / r["h0"]
    drift_he = abs(r["he1"] - r["he0"]) / r["he0"]
    assert drift_h <= NUC_DRIFT_TOL, (
        f"H 核数不守恒：{r['h0']:.12e} -> {r['h1']:.12e}，"
        f"相对漂移 {drift_h:.3e} > {NUC_DRIFT_TOL:.1e}")
    assert drift_he <= NUC_DRIFT_TOL, (
        f"He 核数不守恒：{r['he0']:.12e} -> {r['he1']:.12e}，"
        f"相对漂移 {drift_he:.3e} > {NUC_DRIFT_TOL:.1e}")


# --- (b) 准中性投影噪声 --------------------------------------------------------


def test_a2_quasi_neutrality_projection_noise(lib):
    """派生读 n_e 与 n_HII+n_HEII 的差在 QNEUT_TOL 噪声级。

    探针带把 (n_e) - ((n_HII) + (n_HEII)) 写进 dP_e：P_e 初值 0，
    apply_sources 直加，读回即差值本身。
    """
    probe = {"name": "qneutral_probe",
             "expr": {"dP_e": "(n_e) - ((n_HII) + (n_HEII))"}}
    cfg = _box_cfg([probe], "reflect")
    run = xmf.setup(cfg)
    try:
        _pack_uniform(lib, run, (0.0, 0.0, 0.0, 0.0))
        g = _geom(run)
        _write_pe(lib, run, np.zeros((g[6], g[5], g[4])))
        run.apply_sources()
        pe = _reshape(run, _copy_pe(lib, run))
        inner = _inner(run)
        diff = float(np.max(np.abs(pe[inner])))
        scale = N_E
        assert diff <= QNEUT_TOL * scale, (
            f"准中性投影噪声超容差：max|n_e-(n_HII+n_HEII)| = {diff:.3e}，"
            f"容差 {QNEUT_TOL * scale:.3e}（尺度 n_e = {scale:.3e}）")
    finally:
        run.close()


# --- (c) copy 盒、均匀异速：总动量守恒 ------------------------------------------


def test_a2_total_momentum_conserved_copy_box(lib):
    """碰撞/CX/库仑、无外力无辐射：四流体总动量相对漂移 < MOM_DRIFT_TOL。"""
    r = _run_box(lib, "copy", (VX_HI, 0.0, VX_HII, 0.0))
    # 防空转：动量交换必须真的发生了（HII 被 drag/碰撞显著加速）。
    assert abs(r["vx_hii1"] - r["vx_hii0"]) > 1.0, (
        f"碰撞/继承没在跑：vx_HII {r['vx_hii0']:.3e} -> {r['vx_hii1']:.3e}")
    assert r["heii1"] < 0.99 * r["heii0"], "CX 没在跑（HEII 未减少）"
    drift = abs(r["p1"] - r["p0"]) / max(abs(r["p0"]), 1.0e-30)
    assert drift <= MOM_DRIFT_TOL, (
        f"总动量不守恒：{r['p0']:.12e} -> {r['p1']:.12e}，"
        f"相对漂移 {drift:.3e} > {MOM_DRIFT_TOL:.1e}")
