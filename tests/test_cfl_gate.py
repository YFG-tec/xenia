"""B3 硬门槛单测：电子伪扫信号速 |u_e,d| + c_e 并进 CFL 逐格最大值。

合同来自 docs/cmd/PROMPT_B3_XING.md「硬门槛」：
  - (a) 离子 CFL 读 P_eff = P_s + w_s P_e 是 B1 已实现部分
    （test_cfl_reads_peff 钉），本文件不重做；
  - (b) 伪扫自己的 |u_{e,d}| + c_e（c_e = sqrt(gamma P_e / rho_e)，
    rho_e / u_e 与 sweep_elec_line 同一份拼装）此前没进逐格信号和：
    两支离子 + P_e 主导时 dt 必须比只取真实流体的旧口径更紧；
    单离子被离子 P_eff 声速盖住，dt 逐位不变（核内以带电支 >= 2 为门）。

断言口径：
  1. 多离子（HII + HEII，q 均为 1、质量不同，P_e 主导）：
     run.dt 对解析 cfl / ((|u_e| + c_e) / dx) 到 1e-9——电子信号真的
     定了 max；且严格小于只取真实流体（P_eff 口径）的旧公式
     （本用例解析比 ~0.951，断言阈值 0.99）。
  2. 单离子（HII + HI 中性，P_e 主导、v != 0）：run.dt 对 B1 的
     P_eff 公式到 1e-12；并用开/关电子等价钉（关电子侧 P 槽打包
     numpy 的 P_s + P_e，与核内 fp + w*pe 在 w=1 时同一加法）证明
     dt、t 与一步后的 n / 三速度逐位相同——nchg>=2 的门没有开。

口径同 test_pe / test_coulomb：pack 含 ghost 全块，块长 =
xmf_state_ncells，reshape (nztot, nytot, nxtot)。units 全 1，码值即
cgs；空束表只用来触发 phys 口让 mass_code 是真实质量（多离子钉里
m_He != m_H 的加权才有判别力）。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；钉的是 dt / P_e
口径，不只测 VX1。
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

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

GAMMA = 5.0 / 3.0
M_H_AMU = 1.00794
M_HE_AMU = 4.002053

# 均匀态参数（cgs；units 全 1 时码值同值）
V_X = 3.0e5  # 两离子同速：u_e = v，电子支的 |u| 项有判别力
T_ION = 10.0  # 离子冷：P_s 远小于 P_e，c_e 不被 P_eff 盖住
T_ELE = 1.5e4


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


def _dx1(lib, run) -> float:
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    buf = np.empty(nxt, dtype=np.float64)
    assert lib.xmf_mesh_copy_dx(run._h, 0, buf.ctypes.data_as(_PTR)) == nxt
    return float(buf[ng])


def _gate_cfg(fluids, electron_mode: str) -> dict:
    """16 胞均匀笛卡尔 1D：两端 outflow，hll/pcm/rk1（等价钉口径）。
    空束表（enabled=false）只为触发 phys 口，扫描不跑。"""
    c = xmf.shock_tube()
    c["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    c["grid"]["n"] = [16, 1, 1]
    c["grid"]["geometry"] = "cartesian"
    c["grid"]["x1min"] = 0.0
    c["grid"]["x1max"] = 1.6e7
    c["boundary"] = {"x1beg": "outflow", "x1end": "outflow"}
    c["fluids"] = fluids
    c["electron"] = {"mode": electron_mode}
    c["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    c["time"] = {"cfl": 0.4, "tstop": 1.0, "dt_max": 0.0}
    c["radiation"] = {
        "enabled": False,
        "absorbers": [fluids[0]["name"]],
        "bins": {"n": 1, "sigma": [1.0e-18], "hnu": [2.0e-11], "flux_inf": [0.0]},
    }
    c["ic"] = {"kind": "none"}
    return c


# --- 1. 多离子 + P_e 主导：电子信号定 max，dt 严格更紧 ---------------------------


def test_multi_ion_pe_dominated_dt_tighter_than_real_fluids(lib):
    """HII + HEII（q=1/q=1，质量 1.00794 / 4.002053）：
    n_HEII/n_HII = 0.5 时 c_e^2/c_HII(P_eff)^2 = (1+x)^2/(1+mu x^2) > 1，
    电子信号速超过两支离子的 P_eff 声速；并入后 dt 由 |u_e| + c_e 定。"""
    fluids = [
        {"name": "HII", "mass": M_H_AMU, "charge": 1, "kind": "ion"},
        {"name": "HEII", "mass": M_HE_AMU, "charge": 1, "kind": "ion"},
    ]
    cfg = _gate_cfg(fluids, "pressure-only")
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        n_h, n_he = 1.0e13, 0.5e13
        rho_h = n_h * M_H_AMU * AMU
        rho_he = n_he * M_HE_AMU * AMU
        p_h = n_h * KB * T_ION
        p_he = n_he * KB * T_ION
        pe = (n_h + n_he) * KB * T_ELE
        _pack_block(lib, run, 0, SLOT_N, np.full(shape, rho_h))
        _pack_block(lib, run, 0, SLOT_VX, np.full(shape, V_X))
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, p_h))
        _pack_block(lib, run, 1, SLOT_N, np.full(shape, rho_he))
        _pack_block(lib, run, 1, SLOT_VX, np.full(shape, V_X))
        _pack_block(lib, run, 1, SLOT_P, np.full(shape, p_he))
        _write_pe(lib, run, np.full(shape, pe))
        dx = _dx1(lib, run)

        # 均匀态扫是 no-op；步首 compute_dt 用打包态重算
        run.step()

        # 解析波速（与核内口径同一结合序：w = n_s/n_e，
        # rho_e = sum w_s rho_s，c_e = sqrt(gamma P_e / rho_e)）
        w_h = n_h / (n_h + n_he)
        w_he = n_he / (n_h + n_he)
        rho_e = w_h * rho_h + w_he * rho_he
        s_e = abs(V_X) + math.sqrt(GAMMA * pe / rho_e)
        s_h = abs(V_X) + math.sqrt(GAMMA * (p_h + w_h * pe) / rho_h)
        s_he = abs(V_X) + math.sqrt(GAMMA * (p_he + w_he * pe) / rho_he)
        s_fluid = max(s_h, s_he)
        cfl = cfg["time"]["cfl"]
        dt_expect = cfl / (s_e / dx)
        dt_fluids_only = cfl / (s_fluid / dx)

        # 用例自身判别力：电子信号速比离子 P_eff 最大速大 >1%
        assert s_e > 1.01 * s_fluid, (
            f"用例构造失效：s_e={s_e:.6e} 应显著大于 s_fluid={s_fluid:.6e}"
        )
        # 电子信号真的定了 max（1e-9：核内质量换算往返有 ~ulp 级差异）
        assert run.dt == pytest.approx(dt_expect, rel=1.0e-9), (
            f"多离子 dt={run.dt!r} 应为 cfl/((|u_e|+c_e)/dx)={dt_expect!r}"
        )
        # 硬门槛：比只取真实流体的旧口径严格更紧
        assert run.dt < 0.99 * dt_fluids_only, (
            f"dt={run.dt!r} 应严格紧于真实流体口径 {dt_fluids_only!r}"
        )
    finally:
        run.close()


# --- 2. 单离子回归：并入后 dt 逐位不变（被离子 P_eff 盖住） ----------------------


def test_single_ion_dt_bitwise_unchanged(lib):
    """HII + HI 中性、P_e 主导、v != 0：nchg = 1 时护栏不开。

    两条证据：
      a) run.dt 对 B1 公式 cfl / ((|v| + sqrt(gamma (P_s+P_e)/rho))/dx)
         到 1e-12（w = 1 逐位）；
      b) 开/关电子等价钉：关电子侧 P 槽打包 numpy 的 P_s + P_e（与核内
         fp + w*pe 在 w=1 时同一加法逐位相同）——dt 与 t 逐位相同，
         一步后两支流体的 n 与三个速度槽逐位相同，P 槽必然分裂。
         test_pe 的等价钉是 v=0 的；这里 v != 0 + P_e 主导，专打
         「(ns*v)/ns 舍入把 |u_e| 垫高 1 ulp」的角落。
    """
    fluids = [
        {"name": "HI", "mass": M_H_AMU, "charge": 0, "kind": "neutral"},
        {"name": "HII", "mass": M_H_AMU, "charge": 1, "kind": "ion"},
    ]
    n_i, n_n = 1.0e13, 0.7e13
    rho_i = n_i * M_H_AMU * AMU
    rho_n = n_n * M_H_AMU * AMU
    ps = n_i * KB * T_ION
    pn = n_n * KB * T_ION
    pe = n_i * KB * T_ELE  # P_e 主导

    run_on = xmf.setup(_gate_cfg(fluids, "pressure-only"))
    run_off = xmf.setup(_gate_cfg(fluids, "off"))
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run_on)
        shape = (nzt, nyt, nxt)
        assert _geom(run_off) == (ng, nx, ny, nz, nxt, nyt, nzt)

        for run in (run_on, run_off):
            _pack_block(lib, run, 0, SLOT_N, np.full(shape, rho_n))
            _pack_block(lib, run, 0, SLOT_VX, np.full(shape, V_X))
            _pack_block(lib, run, 0, SLOT_P, np.full(shape, pn))
            _pack_block(lib, run, 1, SLOT_N, np.full(shape, rho_i))
            _pack_block(lib, run, 1, SLOT_VX, np.full(shape, V_X))
        _pack_block(lib, run_on, 1, SLOT_P, np.full(shape, ps))
        _write_pe(lib, run_on, np.full(shape, pe))
        # 关电子侧 HII 的 P 槽 = P_s + P_e（numpy 逐元素一次加法）
        _pack_block(lib, run_off, 1, SLOT_P, np.full(shape, ps) + pe)
        dx = _dx1(lib, run_on)

        run_on.step()
        run_off.step()

        # b) dt / t 逐位相同（门槛没开，CFL 走的还是 B1 那份）
        assert run_on.dt == run_off.dt, (
            f"单离子 dt 应逐位不变：on={run_on.dt!r} off={run_off.dt!r}"
        )
        assert run_on.t == run_off.t

        # a) B1 P_eff 公式钉（w = 1 逐位，puse = ps + 1.0*pe）
        s = abs(V_X) + math.sqrt(GAMMA * (ps + 1.0 * pe) / rho_i)
        cfl = run_on._cfg.numerics.cfl
        expect = cfl / (s / dx)
        assert run_on.dt > 0.0
        assert abs(run_on.dt - expect) <= 1.0e-12 * expect, (
            f"dt={run_on.dt!r} 应为 cfl/(s/dx)={expect!r}（单离子读 P_eff）"
        )

        # b) 一步后：n 与三速度逐位相同，P 槽分裂（A 只留 P_s 增量）
        for var in (SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ):
            a = _unpack_block(lib, run_on, 1, var)
            b = _unpack_block(lib, run_off, 1, var)
            assert np.array_equal(a, b), f"HII 槽 {var} 一步后应逐位相同"
            a0 = _unpack_block(lib, run_on, 0, var)
            b0 = _unpack_block(lib, run_off, 0, var)
            assert np.array_equal(a0, b0), f"HI 槽 {var} 应逐位相同（w=0）"
        pa = _unpack_block(lib, run_on, 1, SLOT_P)
        pb = _unpack_block(lib, run_off, 1, SLOT_P)
        assert np.any(pa != pb), "开/关电子的 HII P 槽演化应已分裂"
    finally:
        run_on.close()
        run_off.close()
