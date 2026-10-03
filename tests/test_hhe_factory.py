"""B3 算例工厂合同单测：examples/rad_hhe_pe_1d/hhe_config.py 的钉。

合同来自本刀卡片（docs/SESSION.md B3 §交什么 3）：
  - 工厂私有：不进 xmf.config、`import xmf` 不导出（只按绝对路径加载）；
  - RUN 钉死在文件顶：nx=128、n_beam=53、tstop=2.0e5、stretch=log、
    stretch_factor=1.025、dt_max=10.0（四流体化学组单步线性化源步的稳
    定帽，见工厂内注释）；
  - 电子：mode=pressure-only、share=by-charge、shock_heat=ions-only；
  - 流体：HI/HEI/HII/HEII 四组分（质量 1.00794 / 4.002602 / 1.00794 /
    4.002053 amu，电荷 0/0/1/1，kind neutral/neutral/ion/ion）；球坐标
    1D，r ∈ [R_p, 10 R_p]，log 拉伸；无第五电子动量流体、无 vx_e 读名；
  - 辐射：两吸收体 ["HI","HEI"] 行表（B3 追加多吸收体口），σ_He 与
    σ_H 同读 ../rad_hd_1d/data/photoIon_sigma.dat（引用不拷贝），束表
    永远整表 53 束（无单束回退）；关门 flux_inf 全 0；
  - 源项：每元素对一份化学和（光电离 + CX + 电子碰撞 / 复合 + CX），
    四带反对称；CHE / drag 用「和速率」而不是逐道；CX 交叉记账
    （cx1 进 H 电离与 He 复合，cx2 反之）；库仑三带 dP_e 精确反号、
    无 dvx_*；无 **2 平方（重力带的 r**2 除外，r 恒正）；T^{-3/2} 写
    成 1/(sqrt(T)*T)；
  - x1end 开关分裂：开门旧树对数外插（四流体全带），关门默认
    copy-outflow；
  - 开门配置能真正 setup：N = 5*4 + 1 = 21，53 束，26 条源带。

按 test_hpe_factory 的先例，用 importlib 绝对路径加载工厂（不进
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
_EXAMPLE_DIR = _REPO / "examples" / "rad_hhe_pe_1d"
_FACTORY_PATH = _EXAMPLE_DIR / "hhe_config.py"
_DATA_DIR = _REPO / "examples" / "rad_hd_1d" / "data"  # 只引用，不拷贝

_spec = importlib.util.spec_from_file_location("hhe_config", _FACTORY_PATH)
hhe_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hhe_config)

FOUR_FLUIDS = ("HI", "HEI", "HII", "HEII")


def _names_of(sources) -> list[str]:
    return [s["name"] for s in sources]


def _tape(sources, name: str):
    return next(s for s in sources if s["name"] == name)


def _split_top(expr: str, op: str) -> tuple[str, str]:
    """按括号深度 0 处的第一个 op 切分（顶层算子才切）。"""
    depth = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and expr[i : i + len(op)] == op:
            return expr[:i], expr[i + len(op):]
        i += 1
    raise AssertionError(f"顶层未找到 {op!r}：{expr[:120]}...")


# --- 1. 工厂私有：不进公开池 ----------------------------------------------------


def test_factory_is_private():
    """工厂不进 xmf.config、`import xmf` 不导出；KB/AMU/SCHEMA 从公开池
    引入而不是另立一份。"""
    assert not hasattr(xmf, "hhe_1d"), "hhe_1d 不得被 import xmf 导出"
    assert not hasattr(cfgmod, "hhe_1d"), "hhe_1d 不得进 xmf.config 公开池"
    assert hhe_config.KB is cfgmod.KB
    assert hhe_config.AMU is cfgmod.AMU
    assert hhe_config.SCHEMA is cfgmod.SCHEMA


def test_data_referenced_not_copied():
    """SED / σ 表引用 ../rad_hd_1d/data/：算例目录不得自带 data/ 拷贝。"""
    assert _DATA_DIR.is_dir(), "被引用的 rad_hd_1d/data 必须在场"
    assert not (_EXAMPLE_DIR / "data").exists(), (
        "rad_hhe_pe_1d 不得拷贝一份自己的 SED/σ 表（合同：引用不拷贝）"
    )


def test_no_electron_momentum_read_names(cfg_on):
    """无第五电子动量流体、无 vx_e / dvx_e 读/写名：电子速度用电荷加权
    均值在带子里拼出来。查生成的配置（源带、流体表、user_bc），不查
    文件文本（docstring 里会提到这些名字）。"""
    assert len(cfg_on["fluids"]) == 4, "电子不得作为第五流体出现"
    names = " ".join(f["name"] for f in cfg_on["fluids"])
    assert "ELCI" not in names and "ELEC" not in names
    exprs = []
    for s in cfg_on["sources"]:
        exprs.extend(s["expr"].keys())
        exprs.extend(s["expr"].values())
    for face in cfg_on["user_bc"].values():
        exprs.extend(face.keys())
        exprs.extend(face.values())
    for e in exprs:
        assert "vx_e" not in e, f"禁止 vx_e 读名（电荷加权拼）：{e[:80]}"
        assert "dvx_e" not in e, f"禁止 dvx_e 写名：{e[:80]}"


# --- 2. RUN 钉 -----------------------------------------------------------------


def test_run_knobs_pinned():
    R = hhe_config.RUN
    assert R.nx == 128
    assert R.n_beam == 53
    assert R.tstop == 2.0e5, "评估跑：tstop = 2.0e5 s"
    assert R.stretch == "log"
    assert R.stretch_factor == 1.025
    assert R.dt_max == 10.0, (
        "四流体化学组单步线性化源步的稳定帽：裸 CFL dt ≈ 22 s 发散、"
        "18 s 崩步，10 s 通过（见工厂注释）"
    )


# --- 3. 开门配置合同 ------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg_on():
    return hhe_config.hhe_1d(_DATA_DIR, radiation=True)


def test_electron_block_pinned(cfg_on):
    ele = cfg_on["electron"]
    assert ele["mode"] == "pressure-only"
    assert ele["share"] == "by-charge"
    assert ele["shock_heat"] == "ions-only", (
        "shock_heat 必须显式钉档（Tóth 约定：无一是物理真值，全是可切换选项）"
    )


def test_fluids_and_grid(cfg_on):
    fluids = cfg_on["fluids"]
    names = [f["name"] for f in fluids]
    assert names == list(FOUR_FLUIDS), "四组分顺序锁 HI/HEI/HII/HEII"
    mass = {f["name"]: f["mass"] for f in fluids}
    charge = {f["name"]: f["charge"] for f in fluids}
    kind = {f["name"]: f["kind"] for f in fluids}
    assert mass["HI"] == pytest.approx(1.00794)
    assert mass["HEI"] == pytest.approx(4.002602)
    assert mass["HII"] == pytest.approx(1.00794)
    assert mass["HEII"] == pytest.approx(4.002053), "HEII = He 原子减一个电子质量"
    assert charge["HI"] == 0 and charge["HEI"] == 0
    assert charge["HII"] == 1 and charge["HEII"] == 1
    assert kind["HI"] == "neutral" and kind["HEI"] == "neutral"
    assert kind["HII"] == "ion" and kind["HEII"] == "ion"
    grid = cfg_on["grid"]
    assert grid["geometry"] == "spherical"
    assert grid["ndim"] == 1
    assert grid["n"][0] == hhe_config.RUN.nx
    assert grid["stretch"] == "log"
    r_p = cfg_on["constants"]["R_p"]
    assert grid["x1min"] == r_p and grid["x1max"] == 10.0 * r_p
    assert cfg_on["time"]["tstop"] == hhe_config.RUN.tstop
    assert cfg_on["time"]["cfl"] == 0.4
    assert cfg_on["time"]["dt_max"] == hhe_config.RUN.dt_max
    assert cfg_on["boundary"] == {"x1beg": "user", "x1end": "user"}
    assert "x1beg" in cfg_on["user_bc"]
    assert "x1end" in cfg_on["user_bc"], "开门 x1end 走旧树 userdef outflow"


def test_dt_max_override():
    cfg = hhe_config.hhe_1d(_DATA_DIR, radiation=True, dt_max=7.5)
    assert cfg["time"]["dt_max"] == 7.5


def test_x1beg_user_bc_four_fluids(cfg_on):
    """内边界钉行星基态：四组分的 n/P 常数 + 三速度清零。"""
    beg = cfg_on["user_bc"]["x1beg"]
    for name in FOUR_FLUIDS:
        assert beg[f"n_{name}"] == f"n_{name}_beg"
        assert beg[f"P_{name}"] == f"P_{name}_beg"
        assert beg[f"vx_{name}"] == "0.0"
        assert beg[f"vy_{name}"] == "0.0"
        assert beg[f"vz_{name}"] == "0.0"


def test_ic_constants_locked(cfg_on):
    """IC 配方钉死：n_beg = 1e13 是 H+He 总量；He 丰度 / 电离份额 /
    P_e,ic 按卡片公式。"""
    c = cfg_on["constants"]
    assert c["n_beg"] == pytest.approx(1.0e13)
    assert c["HE_RATIO"] == pytest.approx(0.0776441)
    assert c["HE_SH_COEF"] == pytest.approx(1.0)
    assert c["T_beg"] == pytest.approx(1500.0)
    y = c["HE_RATIO"]
    assert c["frac_HI"] == pytest.approx(1.0 - y)
    assert c["frac_HEI"] == pytest.approx(y)
    assert c["frac_HII"] == pytest.approx(1.0e-3 * (1.0 - y))
    assert c["frac_HEII"] == pytest.approx(1.0e-3 * y)
    kb = cfgmod.KB
    assert c["P_e_beg"] == pytest.approx(
        c["n_beg"] * (c["frac_HII"] + c["frac_HEII"]) * kb * c["T_beg"]
    )
    assert c["P_HEI_beg"] == pytest.approx(
        c["n_beg"] * y * c["HE_SH_COEF"] * kb * c["T_beg"]
    )


def _chem_sums(cfg):
    """从化学四带里反解出每元素对的和速率 (ion, rec)。"""
    sources = cfg["sources"]
    ion_h, rec_h = _split_top(_tape(sources, "chem_HI")["expr"]["dn_HI"], " + ")
    # dn_HI = -(ion) + (rec)：左半是 "-(ion)"
    assert ion_h.startswith("-(") and ion_h.endswith(")")
    ion_h = ion_h[2:-1]
    assert rec_h.startswith("(") and rec_h.endswith(")")
    rec_h = rec_h[1:-1]
    lhs, rec_h2 = _split_top(_tape(sources, "chem_HII")["expr"]["dn_HII"], " - ")
    assert lhs.startswith("(") and lhs.endswith(")")
    assert lhs[1:-1] == ion_h, "H 对：电离和速率必须同一份"
    assert rec_h2.startswith("(") and rec_h2.endswith(")")
    assert rec_h2[1:-1] == rec_h, "H 对：复合和速率必须同一份（反对称）"
    ion_he, rec_he = _split_top(
        _tape(sources, "chem_HEI")["expr"]["dn_HEI"], " + ")
    assert ion_he.startswith("-(")
    ion_he = ion_he[2:-1]
    rec_he = rec_he[1:-1]
    lhs, rec_he2 = _split_top(
        _tape(sources, "chem_HEII")["expr"]["dn_HEII"], " - ")
    assert lhs[1:-1] == ion_he and rec_he2[1:-1] == rec_he, "He 对反对称"
    return ion_h, rec_h, ion_he, rec_he


def test_chemistry_summed_pairs_antisymmetric(cfg_on):
    """每元素对一份化学和（光 + CX + 电子碰撞 / 复合 + CX），四带反对称；
    CX 交叉记账：cx1 进 ion_H 与 rec_He，cx2 进 rec_H 与 ion_He。"""
    ion_h, rec_h, ion_he, rec_he = _chem_sums(cfg_on)
    # 光电离分量：53 束展开、H/He 各用各的 σ。
    assert "sigma_H_0 * F_0 / hnu_0" in ion_h
    assert "sigma_H_52 * F_52 / hnu_52" in ion_h
    assert "sigma_He_0 * F_0 / hnu_0" in ion_he
    # 电子碰撞电离读派生 n_e。
    assert "n_HI * n_e" in ion_h and "n_HEI * n_e" in ion_he
    # 复合读派生 n_e（不是 n_ion*n_ion）。
    assert "n_HII * n_e" in rec_h and "n_HEII * n_e" in rec_he
    assert "n_HII * n_HII" not in rec_h.replace(" ", "")
    # CX 交叉：cx1 ∝ n_HI n_HEII 进 ion_H 与 rec_He；
    #          cx2 ∝ n_HII n_HEI 进 rec_H 与 ion_He。
    assert "n_HI * n_HEII" in ion_h and "n_HI * n_HEII" in rec_he
    assert "n_HII * n_HEI" in rec_h and "n_HII * n_HEI" in ion_he
    # CX 率常数与激活温度钉死（yardstick :84-94）。
    c = cfg_on["constants"]
    assert c["k_cx1"] == pytest.approx(1.25e-15)
    assert c["k_cx2"] == pytest.approx(1.75e-11)
    assert c["cx1_exp"] == pytest.approx(-0.25)
    assert c["cx2_exp"] == pytest.approx(0.75)
    assert c["T_cx2_act"] == pytest.approx(128000.0)


def test_drag_and_che_use_summed_rates(cfg_on):
    """速度继承与 CHE 压强转移必须用「和速率」（yardstick :115-142 的
    记账先求和再继承），不得逐道拆。"""
    ion_h, rec_h, ion_he, rec_he = _chem_sums(cfg_on)
    sources = cfg_on["sources"]

    def lead_rate(tape_name: str, dest: str) -> str:
        expr = _tape(sources, tape_name)["expr"][dest]
        lhs, _ = _split_top(expr, " / ")
        assert lhs.startswith("(") and lhs.endswith(")")
        return lhs[1:-1]

    assert lead_rate("drag_HI", "dvx_HI") == rec_h
    assert lead_rate("drag_HII", "dvx_HII") == ion_h
    assert lead_rate("drag_HEI", "dvx_HEI") == rec_he
    assert lead_rate("drag_HEII", "dvx_HEII") == ion_he
    # CHE 带以 - (nloss)/(n+n_floor)*P 开头，nloss 就是和速率。
    che = _tape(sources, "che_HI")["expr"]["dP_HI"]
    assert che.startswith(f"-({ion_h}) / (n_HI + n_floor) * P_HI")
    che_he = _tape(sources, "che_HEII")["expr"]["dP_HEII"]
    assert che_he.startswith(f"-({rec_he}) / (n_HEII + n_floor) * P_HEII")


def test_heating_and_cooling_write_dpe(cfg_on):
    """光致加热与 Ly-α 冷却直写 dP_e（gm1 因子），不进离子压投影。"""
    sources = cfg_on["sources"]
    heat = _tape(sources, "photoheat_Pe")["expr"]["dP_e"]
    assert "gm1" in heat and "eta" in heat
    assert "sigma_H_0 * F_0" in heat and "sigma_He_0 * F_0" in heat
    assert "hnu" not in heat, "加热按 Σ σ_b F_b（能量通量），不再除 hnu"
    cool = _tape(sources, "cool_Pe")["expr"]["dP_e"]
    assert "cool_coef" in cool and "n_HI" in cool and "n_e" in cool
    assert "exp(-cool_T_act" in cool
    c = cfg_on["constants"]
    assert c["cool_coef"] == pytest.approx(7.5e-19)
    assert c["cool_T_act"] == pytest.approx(118348.0)
    assert c["eta"] == pytest.approx(0.20)


def test_collision_bands_structure(cfg_on):
    """碰撞带：四流体动量交换（amu 除数、反对称差）+ 压强交换；
    净 dn = 0；共振对系数钉 yardstick；无 **2 平方。"""
    sources = cfg_on["sources"]
    names = _names_of(sources)
    for name in ("coll_mom_HI", "coll_mom_HEI", "coll_mom_HII", "coll_mom_HEII",
                 "coll_prs_HI", "coll_prs_HEI", "coll_prs_HII", "coll_prs_HEII"):
        assert name in names
    # 反对称差：每对里两侧的速度差互为反号写法。
    mom_hi = _tape(sources, "coll_mom_HI")["expr"]["dvx_HI"]
    mom_hii = _tape(sources, "coll_mom_HII")["expr"]["dvx_HII"]
    assert "(vx_HII - vx_HI)" in mom_hi and "(vx_HI - vx_HII)" in mom_hii
    assert "/ m_HI_amu" in mom_hi and "/ m_HII_amu" in mom_hii
    mom_hei = _tape(sources, "coll_mom_HEI")["expr"]["dvx_HEI"]
    mom_heii = _tape(sources, "coll_mom_HEII")["expr"]["dvx_HEII"]
    assert "(vx_HEII - vx_HEI)" in mom_hei and "(vx_HEI - vx_HEII)" in mom_heii
    # 碰撞带不得写 dn_*（净 dn = 0）。
    for s in sources:
        if s["name"].startswith("coll"):
            assert not any(k.startswith("dn_") for k in s["expr"]), (
                f"{s['name']} 不得写 dn_*")
            for expr in s["expr"].values():
                assert "**2" not in expr, (
                    f"{s['name']}：平方必须显式 X*X（AD POW 拒穿零底）")
    c = cfg_on["constants"]
    assert c["k_coll_HH"] == pytest.approx(2.67e-10)
    assert c["coll_beta_HH"] == pytest.approx(0.083)
    assert c["k_coll_HeHe"] == pytest.approx(3.50e-10)
    assert c["coll_beta_HeHe"] == pytest.approx(0.093)
    assert c["k_coll_ii"] == pytest.approx(1.15)
    assert c["inv_log10"] == pytest.approx(1.0 / math.log(10.0))
    # 共振对速率串：0.5*(T1+T2) 平均温 + **0.5 + 显式平方。
    assert "0.5 * (T_HI + T_HII)" in mom_hi
    assert "**0.5" in mom_hi


def test_coulomb_three_bands_exact_negation(cfg_on):
    """库仑三带：HII/HEII 侧全 COL_PRS_TRANS（含电荷加权 v_e 的摩擦热），
    电子侧是两条离子带的精确反号；无动量/粒子数带；T^{-3/2} 写成
    1/(sqrt(T)*T)。"""
    sources = cfg_on["sources"]
    ion_hii = _tape(sources, "coul_prs_HII")["expr"]["dP_HII"]
    ion_heii = _tape(sources, "coul_prs_HEII")["expr"]["dP_HEII"]
    ele = _tape(sources, "coul_prs_e")["expr"]["dP_e"]
    assert ele == f"-(({ion_hii}) + ({ion_heii}))", (
        "电子侧必须是两条离子库仑带之和的精确反号（同除数）")
    for expr in (ion_hii, ion_heii):
        assert "n_e" in expr and "P_e" in expr
        assert "T_e" in expr, "nu_e 读派生 T_e"
        assert "**0.5" in expr
        assert "**1.5" not in expr and "**(-1.5)" not in expr, (
            "T^{-3/2} 必须写成 1/(sqrt(T)*T)")
        assert "m_e_amu" in expr
    # 电荷加权电子速度拼写（无 vx_e 读名）。
    assert "(n_HII * vx_HII + n_HEII * vx_HEII) / (n_e + n_floor)" in ion_hii
    assert "(n_HII * vx_HII + n_HEII * vx_HEII) / (n_e + n_floor)" in ion_heii
    # B2 省略的摩擦热必须写回压强带（两离子时 u_e != u_ion）：
    # (2/3) m_e n_ion n_e (v_ion - v_e)^2，m_e 用克（m_e_g）。
    assert "(2.0 / 3.0) * m_e_g * n_HII * n_e" in ion_hii, (
        "HII-e 库仑带必须含摩擦热项（克质量）")
    assert "(2.0 / 3.0) * m_e_g * n_HEII * n_e" in ion_heii, (
        "HEII-e 库仑带必须含摩擦热项（克质量）")
    # 库仑带不得写动量/粒子数。
    for s in sources:
        if s["name"].startswith("coul"):
            assert all(k.startswith("dP") for k in s["expr"]), (
                f"{s['name']} 不得写 dvx_* / dn_*")
    c = cfg_on["constants"]
    assert c["k_coul"] == pytest.approx(0.0299)
    assert c["m_e_amu"] == pytest.approx(5.4858e-4)


def test_rate_constants_locked_to_yardstick(cfg_on):
    """指令卡锁死的速率系数逐一对表（mmd_model_src.c :67-107）：
    复合 4.0e-12 / 4.6e-12 · (300/T_e)^0.64；电子碰撞电离 2.91e-8 /
    1.75e-8，χ = E_ion/(1.5 kB T_e)，E_ion = 13.6 / 24.6 eV。
    禁止另发明系数。"""
    c = cfg_on["constants"]
    ev = hhe_config.EV_ERG
    # 复合
    assert c["alpha_rec_H"] == pytest.approx(4.0e-12)
    assert c["alpha_rec_He"] == pytest.approx(4.6e-12)
    assert c["rec_exp"] == pytest.approx(0.64)
    assert c["T_rec_ref"] == pytest.approx(300.0)
    # 电子碰撞电离
    assert c["e_imp_H"] == pytest.approx(2.91e-8)
    assert c["e_imp_He"] == pytest.approx(1.75e-8)
    assert c["chi_H_num"] == pytest.approx(13.6 * ev), "χ_H = 13.6 eV（erg）"
    assert c["chi_He_num"] == pytest.approx(24.6 * ev), "χ_He = 24.6 eV（erg）"
    assert c["chi_den_fac"] == pytest.approx(1.5), "χ = E/(1.5 kB T_e)"
    assert c["chi_H_den_add"] == pytest.approx(0.232)
    assert c["chi_He_den_add"] == pytest.approx(0.180)
    # 幂次写进带子：H 0.39 / He 0.35（旧 :99-103）
    sources = cfg_on["sources"]
    ion_h = _tape(sources, "chem_HI")["expr"]["dn_HI"]
    ion_he = _tape(sources, "chem_HEI")["expr"]["dn_HEI"]
    assert "**0.39" in ion_h, "H 电子碰撞电离幂次应为 0.39"
    assert "**0.35" in ion_he, "He 电子碰撞电离幂次应为 0.35"
    assert "chi_den_fac" in ion_h and "chi_den_fac" in ion_he


def test_no_time_or_gdt_read_names(cfg_on):
    """CX 启动门锁成 1（IR 无时间读名，禁止 g_time）；IR 不乘 G_DT：
    任何源带表达式都不得出现 g_time / G_DT 读名。"""
    for s in cfg_on["sources"]:
        for key, expr in s["expr"].items():
            assert "g_time" not in expr, (
                f"{s['name']}/{key} 不得读 g_time（CX 门锁锁成 1）")
            assert "G_DT" not in expr, (
                f"{s['name']}/{key} 不得乘 G_DT（核自己乘 dt）")


def test_gravity_is_only_body_force(cfg_on):
    """外力只有重力+潮汐（公开 dvx 写名，不带流体后缀）；禁止核内特化
    力进带子。"""
    sources = cfg_on["sources"]
    grav = _tape(sources, "gravity")["expr"]["dvx"]
    assert "-G * M_p / r**2" in grav and "M_star" in grav
    dvx_all = [k for s in sources for k in s["expr"] if k == "dvx"]
    assert dvx_all == ["dvx"], "只有重力带写公开 dvx"


# --- 4. 辐射表引用与开关 --------------------------------------------------------


def test_radiation_table_two_absorbers_and_off_switch():
    cfg_on = hhe_config.hhe_1d(_DATA_DIR, radiation=True)
    rad = cfg_on["radiation"]
    assert rad["enabled"] is True
    assert rad["absorbers"] == ["HI", "HEI"], "B3：第二吸收体 HEI"
    bins = rad["bins"]
    assert bins["n"] == 53, "束表永远整表 53 束（无单束回退）"
    assert len(bins["sigma"]) == 2, "每吸收体一行 σ"
    assert all(len(row) == 53 for row in bins["sigma"])
    assert len(bins["hnu"]) == 53 and len(bins["flux_inf"]) == 53
    assert any(v > 0.0 for v in bins["flux_inf"]), "开门：边界通量非零"
    cfg_off = hhe_config.hhe_1d(_DATA_DIR, radiation=False)
    assert cfg_off["radiation"]["enabled"] is False
    assert all(v == 0.0 for v in cfg_off["radiation"]["bins"]["flux_inf"]), (
        "关门跑：束表保留、边界通量清零（真空极限），不用哨兵改写")
    # 源带与电子闭合在关门跑里仍然在场。
    assert _names_of(cfg_off["sources"]) == _names_of(cfg_on["sources"])
    assert cfg_on["boundary"]["x1end"] == "user" and "x1end" in cfg_on["user_bc"]
    assert cfg_off["boundary"]["x1end"] == "outflow"
    assert "x1end" not in cfg_off["user_bc"], (
        "关门对照只用默认 copy-outflow，不得挂对数外插带子")


# --- 5. 开门配置真的能 setup：N = 5*4 + 1 ---------------------------------------


def test_open_setup_num_states_and_beams():
    """小 nx 冒烟：开门配置经 normalize + setup，N = 21、53 束、26 带、
    四步可读（ic.kind=none，step 留给 run.py）。"""
    cfg = hhe_config.hhe_1d(_DATA_DIR, radiation=True, nx=16)
    run = xmf.setup(cfg)
    try:
        assert run.num_states == 5 * 4 + 1, "开门：P_e 殿后，N = 21"
        assert run.nbeam == 53
        assert run.num_tapes == len(cfg["sources"]) == 26
        assert "P_e" in run.fields_full(), "P_e 读口必须开"
    finally:
        run.close()


def test_x1end_fill_matches_old_formula_four_fluids():
    """fill_bc 后 x1end 两层 ghost 对旧树 IEND+1 式（同一 @1/@2）到
    1e-12，四流体全带；vx 外推为负时钉到 0（MAX(0,v)）。"""
    lib = load()
    cfg = hhe_config.hhe_1d(_DATA_DIR, radiation=True, nx=16)
    g = cfg["constants"]["x1end_grid_rate"]
    assert g == pytest.approx(hhe_config._x1end_grid_rate(
        cfg["grid"]["x1min"], cfg["grid"]["x1max"], cfg["grid"]["n"][0],
        cfg["grid"]["stretch"], cfg["grid"]["stretch_factor"]))
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
        for f in range(4):
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
        for f in range(4):
            for var, expect in ((SLOT_N, n_expect), (SLOT_P, p_expect), (SLOT_VX, v_expect)):
                buf = np.empty(ncells, dtype=np.float64)
                assert lib.xmf_state_unpack(h, f, var, buf.ctypes.data_as(_PTR)) == ncells
                got = buf.reshape(shape)[ng, ng, ng + nx :]
                rel = float(np.max(np.abs(got - expect) / max(abs(expect), 1.0e-30)))
                assert rel <= 1.0e-12, f"fluid {f} slot {var}: {got} vs {expect}"
    finally:
        run.close()
