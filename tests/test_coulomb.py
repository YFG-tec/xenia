"""B2 补管测：HII–电子库仑热交换（只压强、反号、热从热侧流向冷侧）。

合同来自 docs/SESSION.md B2 补 §交什么 2：
  - 笛卡尔均匀静态、一支离子 + pressure-only 电子，只有库仑带；
  - apply：同一快照上 \\dot P_HII + \\dot P_e = 0 到 1e-12（摩擦省略后
    逐位反号；两条带子分两次 apply，避免顺序写回污染第二带）；
  - 源步：T_e > T_HII 时 P_e 降、P_HII 升；反过来亦然。若干步后
    |T_e - T_HII| 单调下降。不要把一步牛顿残差当平衡判据。

公式与工厂同一尺子（SSH mmd_model_src.c:164,192,207）：
  nu = 0.0299 / ((T_e+T_floor)**0.5 * (T_e+T_floor)),
  Delta = 2 (n_HII P_e - n_e P_HII),
  dP_HII = nu/(m_HII+m_e) * Delta,  dP_e = -dP_HII。
不写 dvx_*。单位全 1，码值即 cgs。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；钉的是 P_HII 与
P_e，不只测 VX1。
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

K_COUL = 0.0299
M_E_AMU = 5.4858e-4
M_HII_AMU = 1.00794
T_FLOOR = 1.0
N_NUM = 1.0e8  # cm^-3；配 dt_max 让库仑时标跨过几步，而不是一步抽平


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


def _coulomb_expr_pair():
    coul_T = "(T_e + T_floor)"
    coul_rate = f"k_coul / (({coul_T})**0.5 * ({coul_T}))"
    coul_delta = "2.0 * (n_HII * P_e - n_e * P_HII)"
    coul_dP_HII = f"({coul_rate}) / (m_HII_amu + m_e_amu) * ({coul_delta})"
    return coul_dP_HII, f"-({coul_dP_HII})"


def _coulomb_cfg(sources):
    """16 胞笛卡尔 1D：单支 HII + P_e；两端 outflow，均匀静态下 hydro 是 no-op。"""
    c = xmf.shock_tube()
    c["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    c["grid"]["n"] = [16, 1, 1]
    c["grid"]["geometry"] = "cartesian"
    c["grid"]["x1min"] = 0.0
    c["grid"]["x1max"] = 1.6e7
    c["boundary"] = {"x1beg": "outflow", "x1end": "outflow"}
    c["fluids"] = [
        {"name": "HII", "mass": M_HII_AMU, "charge": 1, "kind": "ion"},
    ]
    c["electron"] = {"mode": "pressure-only"}
    c["sources"] = sources
    c["constants"] = {
        "k_coul": K_COUL,
        "T_floor": T_FLOOR,
        "m_HII_amu": M_HII_AMU,
        "m_e_amu": M_E_AMU,
    }
    c["time"] = {"cfl": 0.4, "tstop": 1.0, "dt_max": 5.0e-3}
    # Empty beam table: opens the phys port so n_e / T_e / n_HII convert
    # with real masses. Scan never runs; FLUX reads stay 0.
    c["radiation"] = {
        "enabled": False,
        "absorbers": ["HII"],
        "bins": {"n": 1, "sigma": [1.0e-18], "hnu": [2.0e-11], "flux_inf": [0.0]},
    }
    c["ic"] = {"kind": "none"}
    return c


def _pack_static(lib, run, *, T_hii, T_e):
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    shape = (nzt, nyt, nxt)
    m_g = M_HII_AMU * AMU
    rho = N_NUM * m_g
    p_hii = N_NUM * KB * T_hii
    p_e = N_NUM * KB * T_e
    _pack_block(lib, run, 0, SLOT_N, np.full(shape, rho))
    _pack_block(lib, run, 0, SLOT_VX, np.zeros(shape))
    _pack_block(lib, run, 0, SLOT_VY, np.zeros(shape))
    _pack_block(lib, run, 0, SLOT_VZ, np.zeros(shape))
    _pack_block(lib, run, 0, SLOT_P, np.full(shape, p_hii))
    _write_pe(lib, run, np.full(shape, p_e))
    return shape, p_hii, p_e


def _analytic_dP(T_hii, T_e):
    n_hii = N_NUM
    n_e = N_NUM  # q=1, 准中性
    p_hii = n_hii * KB * T_hii
    p_e = n_e * KB * T_e
    t = T_e + T_FLOOR
    nu = K_COUL / (t**0.5 * t)
    delta = 2.0 * (n_hii * p_e - n_e * p_hii)
    dP = nu / (M_HII_AMU + M_E_AMU) * delta
    return dP, -dP


# --- 1. apply：同一快照上速率反号到 1e-12 ---------------------------------------


def test_coulomb_apply_rates_sum_to_zero(lib):
    """两条库仑带分两次 apply（同一初态）：增量之和到 1e-12，且各边对解析。"""
    ion_expr, ele_expr = _coulomb_expr_pair()
    T_hii, T_e = 1500.0, 3000.0
    expect_ion, expect_ele = _analytic_dP(T_hii, T_e)
    assert expect_ion + expect_ele == 0.0
    increments = []
    for dest, expr in (("dP_HII", ion_expr), ("dP_e", ele_expr)):
        cfg = _coulomb_cfg([{"name": "coul", "expr": {dest: expr}}])
        run = xmf.setup(cfg)
        try:
            shape, p_hii0, p_e0 = _pack_static(lib, run, T_hii=T_hii, T_e=T_e)
            ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
            inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
            run.apply_sources()
            p_hii1 = _unpack_block(lib, run, 0, SLOT_P).reshape(shape)
            p_e1 = _copy_pe(lib, run).reshape(shape)
            d_ion = float(np.mean(p_hii1[inner] - p_hii0))
            d_ele = float(np.mean(p_e1[inner] - p_e0))
            increments.append((d_ion, d_ele))
        finally:
            run.close()
    d_ion = increments[0][0]
    d_ele = increments[1][1]
    scale = max(abs(expect_ion), abs(expect_ele), 1.0e-30)
    assert abs(d_ion - expect_ion) / scale <= 1.0e-12
    assert abs(d_ele - expect_ele) / scale <= 1.0e-12
    assert abs(d_ion + d_ele) / scale <= 1.0e-12


# --- 2. 源步：热从热侧流向冷侧，|ΔT| 随步下降 -----------------------------------


def _source_walk(lib, T_hii0, T_e0, nstep=8):
    ion_expr, ele_expr = _coulomb_expr_pair()
    cfg = _coulomb_cfg(
        [
            {"name": "coul_prs_HII", "expr": {"dP_HII": ion_expr}},
            {"name": "coul_prs_e", "expr": {"dP_e": ele_expr}},
        ]
    )
    run = xmf.setup(cfg)
    try:
        shape, p_hii0, p_e0 = _pack_static(lib, run, T_hii=T_hii0, T_e=T_e0)
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
        gaps = [abs(T_e0 - T_hii0)]
        p_hii_hist = [p_hii0]
        p_e_hist = [p_e0]
        for _ in range(nstep):
            run.step()
            p_hii = float(np.mean(_unpack_block(lib, run, 0, SLOT_P).reshape(shape)[inner]))
            p_e = float(np.mean(_copy_pe(lib, run).reshape(shape)[inner]))
            t_hii = p_hii / (N_NUM * KB)
            t_e = p_e / (N_NUM * KB)
            p_hii_hist.append(p_hii)
            p_e_hist.append(p_e)
            gaps.append(abs(t_e - t_hii))
        return {
            "p_hii": p_hii_hist,
            "p_e": p_e_hist,
            "gaps": gaps,
            "resid": run.source_resid,
        }
    finally:
        run.close()


def test_coulomb_heat_flows_hot_to_cold_and_gap_falls(lib):
    """T_e > T_HII：P_e 降、P_HII 升；T_e < T_HII 反过来。|ΔT| 逐步不升。"""
    hot_e = _source_walk(lib, T_hii0=1500.0, T_e0=3000.0)
    assert hot_e["p_e"][1] < hot_e["p_e"][0], "热电子应把热量交给离子：P_e 降"
    assert hot_e["p_hii"][1] > hot_e["p_hii"][0], "热电子应把热量交给离子：P_HII 升"
    hot_i = _source_walk(lib, T_hii0=3000.0, T_e0=800.0)
    assert hot_i["p_hii"][1] < hot_i["p_hii"][0], "热离子应把热量交给电子：P_HII 降"
    assert hot_i["p_e"][1] > hot_i["p_e"][0], "热离子应把热量交给电子：P_e 升"
    for walk, label in ((hot_e, "Te>Ti"), (hot_i, "Te<Ti")):
        gaps = walk["gaps"]
        for i in range(1, len(gaps)):
            assert gaps[i] <= gaps[i - 1] + 1.0e-9 * max(gaps[0], 1.0), (
                f"{label}: |T_e-T_HII| 应单调下降，"
                f"step {i-1}->{i}: {gaps[i-1]:.6e} -> {gaps[i]:.6e}"
            )
        assert gaps[-1] < 0.5 * gaps[0], (
            f"{label}: 若干步后温差应明显收窄，{gaps[0]:.4e} -> {gaps[-1]:.4e}"
        )
