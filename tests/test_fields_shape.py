"""A5 读口合同单测（test_fields_shape）：fields_full() 与 flux() 的形状与口径。

合同来自本刀卡片（docs/SESSION.md §交什么 3/6）：
  1. fields_full() 形状 (nfluid,nz,ny,nx)（实现为 {流体名: {槽: (nz,ny,nx)}}，
     nfluid 维展开成字典键）；槽 0 的 "n" 是数密度 = ρ/m，与 state 读回
     一致；速度/压强按单位三元组换算。
  2. 2D 不展平错位：薄 2D（nx2=4 ≥3、nx1=8）下按 (k,j,i) 编码的模式值必须
     逐位落回同一 (k,j,i)，抽查 ≥4 点含非对角；两支流体互不串。
  3. 能读 F_b：scan 之后 flux() 形状 (nbeam,nz,ny,nx)，值与解析指数衰减
     一致；无辐射态的 run 上 flux() 必须报错而不是返回假数据。

口径：unpack/flux 块布局 (nztot, nytot, nxtot)，索引 (k,j,i)，i 最快；
fields_full / flux 只吐内区。数密度换算按 api 的同一条算式复现：
raw * u.density / (mass_amu * AMU)（units 全 1 时与核内 rho/mass_code
逐位相同）。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；不只测 VX1——
n / vx / P 三种槽都钉，vy/vz 也断言。
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

NX1, NX2 = 8, 4          # 薄 2D：nx2=4 ≥ 3，nx1 踩核的 sanity 下限 8
MASS_HI = 1.00794
MASS_HII = 4.0026        # 两支质量刻意不同：换算错用别支质量必红


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（state/mesh/rad 符号已绑定）。"""
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


def _thin2d_cfg() -> dict:
    """薄 2D 双流体空题：无 sources、无辐射、ic 全零（稍后用助手打包）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1, NX2, 1]
    cfg["grid"]["ndim"] = 2
    cfg["fluids"] = [
        {"name": "HI", "mass": MASS_HI, "charge": 0, "kind": "hd"},
        {"name": "HII", "mass": MASS_HII, "charge": 1, "kind": "hd"},
    ]
    cfg["sources"] = []
    cfg["ic"] = {"kind": "none"}
    return cfg


def _scan_cfg_1beam() -> dict:
    """球面 1D 单束扫描用例（与 test_scan 同口径）：units 全 1 → 码值即
    cgs；均匀数密度、无 sources；rk1 一步恰一次扫描。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["geometry"] = "spherical"
    cfg["grid"]["n"] = [8, 1, 1]
    cfg["grid"]["x1min"] = 1.0e10
    cfg["grid"]["x1max"] = 1.8e10
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["sources"] = []
    cfg["radiation"] = {
        "enabled": True,
        "absorbers": ["HI"],
        "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
        "bins": {"n": 1, "sigma": [2.5e-18], "hnu": [2.0e-11],
                 "flux_inf": [4.5]},
    }
    cfg["ic"] = {"kind": "none"}
    return cfg


# --- 1. 形状与数密度换算（合同第 1 条） ---------------------------------------


def test_fields_full_shape_and_number_density(lib):
    """(nfluid,nz,ny,nx) 形状 + 槽 0 换算成数密度 ρ/m + 单位三元组换算。

    每支每槽打包一个均匀常数；fields_full 的 n 必须等于
    raw*u.density/(mass_amu*AMU)（逐位，同结合序），且圆回 ρ = n*m。
    两支质量不同：换算错用质量或错用流体必红。
    """
    run = xmf.setup(_thin2d_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert (nx, ny, nz) == (NX1, NX2, 1)
        u = run._cfg.units

        rho_c = (2.5, 5.0)     # 码单位密度（刻意不同）
        vx_c = (0.5, 1.5)
        p_c = (3.75, 7.5)
        for f in range(2):
            _pack_block(lib, run, f, SLOT_N, np.full((nzt, nyt, nxt), rho_c[f]))
            _pack_block(lib, run, f, SLOT_VX, np.full((nzt, nyt, nxt), vx_c[f]))
            _pack_block(lib, run, f, SLOT_P, np.full((nzt, nyt, nxt), p_c[f]))

        fld = run.fields_full()
        assert set(fld.keys()) == {"HI", "HII"}, (
            f"fields_full 的键应为全部流体名，实测 {sorted(fld.keys())}"
        )
        for f, name in enumerate(("HI", "HII")):
            slots = fld[name]
            assert set(slots.keys()) == {"n", "vx", "vy", "vz", "P"}
            for key, arr in slots.items():
                assert arr.shape == (nz, ny, nx), (
                    f"{name}.{key} 形状应为 (nz,ny,nx)={(nz, ny, nx)}，"
                    f"实测 {arr.shape}"
                )
            mass = run._cfg.fluids[f].mass
            # n = raw * u.density / (mass_amu * AMU)：与 api 同结合序，逐位
            n_exp = np.full((nz, ny, nx), rho_c[f]) * u.density / (mass * AMU)
            assert np.array_equal(slots["n"], n_exp), (
                f"{name}.n 应为 ρ/m 的数密度（换算逐位一致）"
            )
            # 语义圆回：n*m*AMU/u.density == 打包的码密度
            got_rho = slots["n"] * mass * AMU / u.density
            assert np.allclose(got_rho, rho_c[f], rtol=1.0e-15, atol=0.0), (
                f"{name}.n 圆不回打包密度：不是数密度口径"
            )
            vx_exp = np.full((nz, ny, nx), vx_c[f]) * u.velocity
            assert np.array_equal(slots["vx"], vx_exp)
            p_exp = np.full((nz, ny, nx), p_c[f]) * u.density * u.velocity * u.velocity
            assert np.array_equal(slots["P"], p_exp), (
                f"{name}.P 应为 raw*u.density*u.velocity²（同结合序逐位）"
            )
            # 未打包的横速度槽保持 0
            assert np.all(slots["vy"] == 0.0) and np.all(slots["vz"] == 0.0)
    finally:
        run.close()


# --- 2. 2D 不展平错位（合同第 2 条） ------------------------------------------


def test_fields_full_2d_positions_no_flatten_mixup(lib):
    """薄 2D 按 (j,i) 编码的模式值：整张数组逐位落回同一 (k,j,i)，
    非对角抽查 ≥4 点；两支流体各带不同 salt，互不串。"""
    run = xmf.setup(_thin2d_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        u = run._cfg.units

        salt = (0.0, 5000.0)
        for f in range(2):
            block = np.zeros((nzt, nyt, nxt), dtype=np.float64)
            for jj in range(ny):
                for ii in range(nx):
                    block[ng, ng + jj, ng + ii] = (jj + 1) * 100.0 + (ii + 1) + salt[f]
            _pack_block(lib, run, f, SLOT_N, block)

        fld = run.fields_full()
        for f, name in enumerate(("HI", "HII")):
            mass = run._cfg.fluids[f].mass
            expect = np.empty((nz, ny, nx), dtype=np.float64)
            for jj in range(ny):
                for ii in range(nx):
                    # 与 api 同一结合序：raw*u.density/(mass*AMU)（逐位）
                    expect[0, jj, ii] = (
                        ((jj + 1) * 100.0 + (ii + 1) + salt[f])
                        * u.density / (mass * AMU)
                    )
            got = fld[name]["n"]
            assert np.array_equal(got, expect), (
                f"{name}.n 的 (k,j,i) 位置错位：max|diff|="
                f"{float(np.max(np.abs(got - expect))):.3e}"
                "（展平/转置/跨流体串扰都会打红这条）"
            )
        # 非对角抽查 ≥4 点（(j,i) 均取非对角位置）
        mass0 = run._cfg.fluids[0].mass
        for jj, ii in ((0, 1), (2, 0), (3, 7), (1, 5)):
            want = ((jj + 1) * 100.0 + (ii + 1)) * u.density / (mass0 * AMU)
            assert fld["HI"]["n"][0, jj, ii] == want, (
                f"非对角点 (j={jj}, i={ii}) 的值错位"
            )
        # 流体维不串：HI 的对角值 != HII 的同位值（salt 不同）
        assert fld["HI"]["n"][0, 1, 2] != fld["HII"]["n"][0, 1, 2]
    finally:
        run.close()


# --- 3. 能读 F_b：flux() 形状与解析衰减（合同第 3 条） -------------------------


def test_flux_read_port_matches_analytic_decay(lib):
    """球面单束均匀吸收：一步 step 后 flux() 形状 (nbeam,nz,ny,nx)，
    值对 F∞·exp(−σ n Δs·外向内累计格数) 相对差 ≤1e-12。"""
    sigma, hnu, finf = 2.5e-18, 2.0e-11, 4.5
    n0 = 2.0e8
    run = xmf.setup(_scan_cfg_1beam())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.nbeam == 1
        m = run._cfg.fluids[0].mass * AMU / run._cfg.units.density
        _pack_block(lib, run, 0, SLOT_N, np.full((nzt, nyt, nxt), n0 * m))
        _pack_block(lib, run, 0, SLOT_P, np.full((nzt, nyt, nxt), n0 * KB * 1500.0))

        run.step()
        assert run.t > 0.0

        flx = run.flux()
        assert flx.shape == (1, nz, ny, nx), (
            f"flux() 形状应为 (nbeam,nz,ny,nx)={(1, nz, ny, nx)}，实测 {flx.shape}"
        )
        dx = (run._cfg.x1max - run._cfg.x1min) / nx
        tau = (sigma * n0) * dx  # units 全 1：ds 即厘米
        cum = nx - np.arange(nx, dtype=np.float64)
        expect = finf * np.exp(-tau * cum)
        got = flx[0, 0, 0, :]
        rel = float(np.max(np.abs((got - expect) / expect)))
        assert rel <= 1.0e-12, (
            f"flux() 与解析指数衰减的最大相对差 {rel:.3e} 超过 1e-12"
        )
        assert got[-1] <= finf and got[0] < got[-1], (
            "向内应严格衰减（最内格 < 最外格 ≤ F∞）"
        )
    finally:
        run.close()


def test_flux_without_radiation_state_raises(lib):
    """无辐射态（普通激波管）上调 flux() 必须抛错，不得返回全 0 假数据。"""
    run = xmf.setup(xmf.shock_tube())
    try:
        assert run.nbeam == 0
        with pytest.raises(RuntimeError):
            run.flux()
    finally:
        run.close()
