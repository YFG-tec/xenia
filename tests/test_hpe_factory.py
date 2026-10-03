"""B2 算例工厂合同单测：examples/rad_h_pe_1d/hpe_config.py 的钉。

合同来自本刀卡片（docs/SESSION.md B2 §交什么 4/5）：
  - 工厂私有：不进 xmf.config、`import xmf` 不导出（只按绝对路径加载）；
  - RUN 钉死在文件顶：nx=128、n_beam=53、tstop=2.0e5、stretch=log、
    stretch_factor=1.025；
  - 电子：mode=pressure-only、share=by-charge、shock_heat 显式钉
    （ions-only：激波增量全留离子压，光电加热走 dP_e 源带）；
  - 流体：HI + HII（q=0 / q=1）；球坐标 1D，r ∈ [R_p, 10 R_p]，log 拉伸；
  - 源项：含光电离化学（dn_HI / dn_HII 反号）、加热必须写 dP_e（不得写
    进 dP_HII 投影）、复合用派生读 n_e（不是 n_HII*n_HII）、HI–HII
    共振碰撞（yardstick 系数 2.67e-10，净 dn=0，动量/内能互换）、
    HII–电子库仑热交换（0.0299 T_e^{-3/2}，dP_HII / dP_e 反号，无
    dvx_*，无 He）；
  - 辐射表：SED / σ 从 ../rad_hd_1d/data/ 读（引用不拷贝）；关门跑
    flux_inf 全 0；
  - 开门配置能真正 setup：N = 5*nfluid + 1，53 束。

按 test_source_step 的先例，用 importlib 绝对路径加载工厂（不进
sys.modules 注册、不是包）。
"""
from __future__ import annotations

import ctypes
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import xmf
from xmf import config as cfgmod
from xmf._native import load

_PTR = ctypes.POINTER(ctypes.c_double)
SLOT_N, SLOT_VX, SLOT_P = 0, 1, 4

_REPO = Path(__file__).resolve().parents[1]
_FACTORY_PATH = _REPO / "examples" / "rad_h_pe_1d" / "hpe_config.py"
_DATA_DIR = _REPO / "examples" / "rad_hd_1d" / "data"  # 只引用，不拷贝

_spec = importlib.util.spec_from_file_location("hpe_config", _FACTORY_PATH)
hpe_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hpe_config)


def _names_of(sources) -> list[str]:
    return [s["name"] for s in sources]


def _exprs_by_dest(sources) -> dict[str, list[str]]:
    """dest 名 -> 表达式列表（同一 dest 多条带子全收）。"""
    out: dict[str, list[str]] = {}
    for s in sources:
        for dest, expr in s["expr"].items():
            out.setdefault(dest, []).append(expr)
    return out


# --- 1. 工厂私有：不进公开池 ----------------------------------------------------


def test_factory_is_private():
    """工厂不进 xmf.config、`import xmf` 不导出；KB/AMU/SCHEMA 从公开池
    引入而不是另立一份。"""
    assert not hasattr(xmf, "hpe_1d"), "hpe_1d 不得被 import xmf 导出"
    assert not hasattr(cfgmod, "hpe_1d"), "hpe_1d 不得进 xmf.config 公开池"
    assert hpe_config.KB is cfgmod.KB
    assert hpe_config.AMU is cfgmod.AMU
    assert hpe_config.SCHEMA is cfgmod.SCHEMA


def test_data_referenced_not_copied():
    """SED / σ 表引用 ../rad_hd_1d/data/：算例目录不得自带 data/ 拷贝。"""
    assert _DATA_DIR.is_dir(), "被引用的 rad_hd_1d/data 必须在场"
    assert not (_REPO / "examples" / "rad_h_pe_1d" / "data").exists(), (
        "rad_h_pe_1d 不得拷贝一份自己的 SED/σ 表（合同：引用不拷贝）"
    )


# --- 2. RUN 钉 -----------------------------------------------------------------


def test_run_knobs_pinned():
    R = hpe_config.RUN
    assert R.nx == 128
    assert R.n_beam == 53
    assert R.tstop == 2.0e5, "评估跑：tstop = 2.0e5 s"
    assert R.stretch == "log"
    assert R.stretch_factor == 1.025


# --- 3. 开门配置合同 ------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg_on():
    return hpe_config.hpe_1d(_DATA_DIR, radiation=True)


def test_electron_block_pinned(cfg_on):
    ele = cfg_on["electron"]
    assert ele["mode"] == "pressure-only"
    assert ele["share"] == "by-charge"
    assert ele["shock_heat"] == "ions-only", (
        "shock_heat 必须显式钉档（Tóth 约定：无一是物理真值，全是可切换选项）"
    )


def test_fluids_and_grid(cfg_on):
    names = [f["name"] for f in cfg_on["fluids"]]
    assert "HI" in names and "HII" in names
    charges = {f["name"]: f["charge"] for f in cfg_on["fluids"]}
    assert charges["HI"] == 0 and charges["HII"] == 1
    grid = cfg_on["grid"]
    assert grid["geometry"] == "spherical"
    assert grid["ndim"] == 1
    assert grid["n"][0] == hpe_config.RUN.nx
    assert grid["stretch"] == "log"
    r_p = cfg_on["constants"]["R_p"]
    assert grid["x1min"] == r_p and grid["x1max"] == 10.0 * r_p
    assert cfg_on["time"]["tstop"] == hpe_config.RUN.tstop
    assert cfg_on["boundary"] == {"x1beg": "user", "x1end": "user"}
    assert "x1beg" in cfg_on["user_bc"], "x1beg 走 A5 形状的解析 user_bc 带"
    assert "x1end" in cfg_on["user_bc"], "开门 x1end 走旧树 userdef outflow"


def test_x1end_old_outflow_userdef(cfg_on):
    """外边界是旧树 outflow_BC：n/P 幂律、vx = MAX(0, 线性外推)；
    常数 x1end_grid_rate = x[IEND]/x[IEND-1]；无 max() 操作码。"""
    end = cfg_on["user_bc"]["x1end"]
    g = cfg_on["constants"]["x1end_grid_rate"]
    assert g > 1.0, "log 拉伸外区格子变宽，末两格心之比应 > 1"
    expect_g = hpe_config._x1end_grid_rate(
        cfg_on["grid"]["x1min"],
        cfg_on["grid"]["x1max"],
        cfg_on["grid"]["n"][0],
        cfg_on["grid"]["stretch"],
        cfg_on["grid"]["stretch_factor"],
    )
    assert g == pytest.approx(expect_g)
    for name in ("HI", "HII"):
        assert end[f"n_{name}"] == (
            f"n_{name}@1**(1.0 + x1end_grid_rate) / n_{name}@2**x1end_grid_rate"
        )
        assert end[f"P_{name}"] == (
            f"P_{name}@1**(1.0 + x1end_grid_rate) / P_{name}@2**x1end_grid_rate"
        )
        vx = end[f"vx_{name}"]
        assert "max(" not in vx.lower()
        assert f"vx_{name}@1" in vx and f"vx_{name}@2" in vx
        assert "**0.5" in vx, "MAX(0,v) 必须写成 0.5*(v+(v*v)**0.5)"
        assert f"vy_{name}" not in end and f"vz_{name}" not in end


def test_photoion_and_heating_write_dpe(cfg_on):
    """光电离化学成对（dn_HI 与 dn_HII 反号）；加热写 dP_e——直写电子块，
    不允许写进离子 P 再投影。"""
    sources = cfg_on["sources"]
    names = _names_of(sources)
    assert "photoion_HI" in names and "photoion_HII" in names
    by_dest = _exprs_by_dest(sources)
    assert "dP_e" in by_dest, "光致加热必须以 dP_e 写名进核（B2 追加写名）"
    heat = next(s for s in sources if s["name"] == "photoheat_Pe")
    expr = heat["expr"]["dP_e"]
    assert "n_HI" in expr and "eta" in expr and "gm1" in expr
    assert "F_0" in expr, "加热应含束通量 Σ σ_b F_b（53 束展开）"
    # 同一份光子吸收率驱动电离与加热：两带共用 sigma_b/hnu_b 常数
    ion_src = next(s for s in sources if s["name"] == "photoion_HII")
    assert "sigma_0" in ion_src["expr"]["dn_HII"]
    assert "hnu_0" in ion_src["expr"]["dn_HII"]
    # 电离/复合粒子数成对
    rec_hi = next(s for s in sources if s["name"] == "recomb_HI")
    rec_hii = next(s for s in sources if s["name"] == "recomb_HII")
    assert rec_hii["expr"]["dn_HII"] == f"-({rec_hi['expr']['dn_HI']})"


def test_recombination_reads_ne(cfg_on):
    """复合用派生读 n_e，不是 n_HII*n_HII。"""
    rec = next(s for s in cfg_on["sources"] if s["name"] == "recomb_HI")
    expr = rec["expr"]["dn_HI"]
    assert "n_e" in expr, "复合率必须读派生 n_e"
    assert "n_HII * n_HII" not in expr.replace(" ", ""), (
        "不得写成 n_HII*n_HII（B2 合同：用 n_e 派生读）"
    )


def test_hi_hii_resonance_collision(cfg_on):
    """HI–HII 共振碰撞：yardstick 系数 2.67e-10、T = 0.5*(T_HI+T_HII)、
    净 dn = 0、动量与内能互换；平方按显式 X*X 写（AD 的 POW 拒穿零底）。"""
    sources = cfg_on["sources"]
    names = _names_of(sources)
    coll = [n for n in names if n.startswith("coll")]
    assert coll, "应有碰撞带（动量 + 内能）"
    by_dest = _exprs_by_dest(sources)
    assert "dvx_HI" in by_dest and "dvx_HII" in by_dest
    assert "dP_HI" in by_dest and "dP_HII" in by_dest
    # 净 dn = 0：碰撞带不得写 dn_*（光电离/复合带才写）
    for s in sources:
        if s["name"].startswith("coll"):
            assert not any(k.startswith("dn_") for k in s["expr"]), (
                f"{s['name']} 不得写 dn_*（共振碰撞净 dn=0）"
            )
    c = cfg_on["constants"]
    assert c["k_coll"] == pytest.approx(2.67e-10), "yardstick 系数钉"
    assert c["coll_beta"] == pytest.approx(0.083)
    assert c["inv_log10"] == pytest.approx(1.0 / math.log(10.0))
    # 速率串含 T_mid^0.5 与 (1 - beta*log(T_mid)*inv_log10)^2 的显式平方
    mom = next(s for s in sources if s["name"] == "coll_mom_HI")["expr"]["dvx_HI"]
    assert "0.5 * (T_HI + T_HII)" in mom
    assert "**0.5" in mom
    assert "**2" not in mom, "平方必须写成显式 X*X，不得用 **2"


def test_hii_electron_coulomb(cfg_on):
    """HII–电子库仑：yardstick 0.0299 与 m_e=5.4858e-4、T^{-3/2} 写成
    1/(sqrt(T)*T)、dP_HII 与 dP_e 同一式反号、无库仑动量/粒子数带、无 He。"""
    sources = cfg_on["sources"]
    names = _names_of(sources)
    assert "coul_prs_HII" in names and "coul_prs_e" in names
    assert not any("HE" in n or n.startswith("he") for n in names)
    fluids = [f["name"] for f in cfg_on["fluids"]]
    assert "HeI" not in fluids and "HEI" not in fluids and "HEII" not in fluids
    c = cfg_on["constants"]
    assert c["k_coul"] == pytest.approx(0.0299)
    assert c["m_e_amu"] == pytest.approx(5.4858e-4)
    ion = next(s for s in sources if s["name"] == "coul_prs_HII")
    ele = next(s for s in sources if s["name"] == "coul_prs_e")
    assert list(ion["expr"]) == ["dP_HII"]
    assert list(ele["expr"]) == ["dP_e"]
    assert not any(k.startswith("dvx") or k.startswith("dn_") for k in ion["expr"])
    assert not any(k.startswith("dvx") or k.startswith("dn_") for k in ele["expr"])
    dP_ion = ion["expr"]["dP_HII"]
    assert ele["expr"]["dP_e"] == f"-({dP_ion})", "电子侧必须是离子侧同一式的反号"
    assert "T_e" in dP_ion and "n_HII" in dP_ion and "P_e" in dP_ion
    assert "n_e" in dP_ion and "P_HII" in dP_ion
    assert "**(-1.5)" not in dP_ion and "**-1.5" not in dP_ion
    assert "**1.5" not in dP_ion
    assert "**0.5" in dP_ion
    assert "m_HII_amu" in dP_ion and "m_e_amu" in dP_ion
    # 库仑带不得写动量：全表里带 coul 前缀的 dest 只能是 dP_*
    for s in sources:
        if s["name"].startswith("coul"):
            assert all(k.startswith("dP") for k in s["expr"]), (
                f"{s['name']} 不得写 dvx_* / dn_*（yardstick HII 动量不跟电子换）"
            )


# --- 4. 辐射表引用与开关 --------------------------------------------------------


def test_radiation_table_and_off_switch():
    cfg_on = hpe_config.hpe_1d(_DATA_DIR, radiation=True)
    bins = cfg_on["radiation"]["bins"]
    assert bins["n"] == 53 and len(bins["sigma"]) == 53
    assert cfg_on["radiation"]["enabled"] is True
    assert cfg_on["radiation"]["absorbers"] == ["HI"]
    cfg_off = hpe_config.hpe_1d(_DATA_DIR, radiation=False)
    assert cfg_off["radiation"]["enabled"] is False
    assert all(v == 0.0 for v in cfg_off["radiation"]["bins"]["flux_inf"]), (
        "关门跑：束表保留、边界通量清零（真空极限），不用哨兵改写"
    )
    # 源带与电子闭合在关门跑里仍然在场（复合 / CHE / 碰撞 / 库仑 / 重力保留）
    assert _names_of(cfg_off["sources"]) == _names_of(cfg_on["sources"])
    assert cfg_on["boundary"]["x1end"] == "user" and "x1end" in cfg_on["user_bc"]
    assert cfg_off["boundary"]["x1end"] == "outflow"
    assert "x1end" not in cfg_off["user_bc"], (
        "关门对照只用默认 copy-outflow，不得挂对数外插带子"
    )


# --- 5. 开门配置真的能 setup：N = 5*nfluid + 1 ----------------------------------


def test_open_setup_num_states_and_beams():
    """小 nx 冒烟：开门配置经 normalize + setup，N = 5*2 + 1、53 束、
    四步可读（ic.kind=none，step 留给 run.py）。"""
    cfg = hpe_config.hpe_1d(_DATA_DIR, radiation=True, nx=16)
    run = xmf.setup(cfg)
    try:
        assert run.num_states == 5 * 2 + 1, "开门：P_e 殿后，N = 11"
        assert run.nbeam == 53
        assert run.num_tapes == len(cfg["sources"])  # 每条源带各一条
        assert "P_e" in run.fields_full(), "P_e 读口必须开"
    finally:
        run.close()


def test_x1end_fill_matches_old_first_ghost_formula():
    """fill_bc 后 x1end 两层 ghost 对旧树 IEND+1 式（同一 @1/@2）到 1e-12。
    vx 外推为负时钉到 0（MAX(0,v)）。"""
    lib = load()
    cfg = hpe_config.hpe_1d(_DATA_DIR, radiation=True, nx=16)
    g = cfg["constants"]["x1end_grid_rate"]
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        p = run._cfg
        ng, nx, ny, nz = p.ng, p.n[0], p.n[1], p.n[2]
        nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
        shape = (nzt, nyt, nxt)
        assert ncells == nzt * nyt * nxt
        n1, n2 = 4.0e4, 2.5e4
        p1, p2 = 3.0e-8, 1.5e-8
        v1, v2 = -1.0e5, 2.0e5  # 线性外推 (1+g)*v1 - g*v2 < 0 → 钉 0
        for f in (0, 1):
            nblk = np.ones(shape)
            nblk[:, :, ng + nx - 1] = n1
            nblk[:, :, ng + nx - 2] = n2
            pblk = np.ones(shape)
            pblk[:, :, ng + nx - 1] = p1
            pblk[:, :, ng + nx - 2] = p2
            vblk = np.zeros(shape)
            vblk[:, :, ng + nx - 1] = v1
            vblk[:, :, ng + nx - 2] = v2
            for var, blk in ((SLOT_N, nblk), (SLOT_P, pblk), (SLOT_VX, vblk)):
                data = np.ascontiguousarray(blk, dtype=np.float64).ravel()
                assert lib.xmf_state_pack(h, f, var, data.ctypes.data_as(_PTR)) == ncells
        assert lib.xmf_fill_bc(h) == 0
        n_expect = n1 ** (1.0 + g) / n2**g
        p_expect = p1 ** (1.0 + g) / p2**g
        v_ext = (1.0 + g) * v1 - g * v2
        v_expect = 0.5 * (v_ext + abs(v_ext))
        assert v_expect == 0.0
        for f in (0, 1):
            for var, expect in ((SLOT_N, n_expect), (SLOT_P, p_expect), (SLOT_VX, v_expect)):
                buf = np.empty(ncells, dtype=np.float64)
                assert lib.xmf_state_unpack(h, f, var, buf.ctypes.data_as(_PTR)) == ncells
                got = buf.reshape(shape)[ng, ng, ng + nx :]
                rel = float(np.max(np.abs(got - expect) / max(abs(expect), 1.0e-30)))
                assert rel <= 1.0e-12, f"fluid {f} slot {var}: {got} vs {expect}"
    finally:
        run.close()
