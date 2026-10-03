"""C1a Kokkos Serial 基准钉：三处循环改上 Kokkos 界面后必须复现改前串行 C 基准。

基准出处（仓库根 .tmp/，契约测试运行时**不读** .tmp/——所有数字已作为字面量
钉死在本文件，来源标注如下）：
  * .tmp/c1a_baseline_empty.txt / .tmp/c1a_baseline_empty.npz
    —— 空表 pcm Sod：25 步定步长短积分的 t/dt 摘要 + 场 dump；全推进
       t 与 L1(n,v,P) 三重。
  * .tmp/c1a_baseline_source.txt / .tmp/c1a_baseline_source.npz
    —— 非空开源项：2 流体 HI/HII、6 条带子（光化对、光加热、复合对、重力）、
       1 束辐射、x1 outflow，均匀初值 10 步定步长积分的 t/dt/resid 摘要、
       内区逐槽 L2/max 摘要、通量（含 ghost）L2。
  两份基准由 builder 在 C1a 刀改循环**之前**于现串行 C 核上采集（见
  .tmp/c1a_baseline.py）；改完后的逐位对照记录在 .tmp/c1a_compare_result.txt
  （BITWISE MATCH）。numerics-review 只认这份基准。

钉位与容差策略（卡片：容差只给数值表示误差，不给物理漂移空间）：
  * 空表 t/dt、全推进 t、L1 三重：全精度 ==（逐位同值钉）。
  * 源项短积分 t / dt：rtol = 1e-12、atol = 0；source_resid：atol = 1e-14
    纯绝对（resid 是往零收敛的残差，绝对尺子合身；其水位 1.6e-5 会把
    相对尺子无谓放大）——只覆盖平台数学库固差（libdevice pow(0.2,0.64)
    与 UCRT 差 1 ulp；定性与实测 archive/C2_cuda.md）。同一份测试两树
    跑，不拆平台；禁止以此为先例放宽其它钉。
  * 场内区 L2 / max|.|（由基准 npz 摘要算出的字面量）：rtol = 1e-13——
    只覆盖 norm 归约的表示噪声；任何循环顺序/公式级改动引起的物理漂移
    远大于此（且源项三标量容差钉先行报警）。
  * 恒 0 槽（1D 的 vy/vz）：逐位 == 0。

三处循环的分工覆盖：空表短积分 + 全推进钉住 hydro 通量循环；开源项短积分
同时走 hydro、scan（i 顺序前缀积）与源项+局部 LU 循环。单吸收体 scan、
关电子 CFL/hydro 的钉位由 tests/test_scan.py、tests/test_cfl.py、
tests/test_hydro.py、tests/test_step_empty.py 各自持有（本文件不重复）。
全推进 L1 三重与 test_step_empty.py（A5）/ test_core_split.py（A4.5）同值
同口径，本文件持一份 C1a 自己的验收副本（卡片点名的 3.515e-3 / 9.716e-3 /
4.029e-3 钉不动）。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；钉的是五槽×双流体
（含 n、P、vy/vz 恒 0 钉）加 t/dt/resid，不只测 VX1。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

import xmf
from xmf._native import load
from xmf.config import AMU, KB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.riemann import exact_hi  # noqa: E402

_PTR = ctypes.POINTER(ctypes.c_double)

NG = 2  # 两份基准配置的 ghost 数（shock_tube 默认 nghost=2）

# norm 类钉位的容差：只给数值表示误差（归约顺序/舍入），不给物理漂移。
NORM_RTOL = 1.0e-13

# 源项短积分 t / dt 两标量钉：显式相对容差，atol = 0（不叠绝对容差项）。
# 只覆盖平台数学库固差：libdevice pow(0.2, 0.64) 与 UCRT 差 1 ulp，10 步
# 积分放大到相对漂移 ≤6.3e-15；rtol=1e-12 留三个量级余量。定性证据：
# archive/C2_cuda.md。同一份测试两树跑，不拆平台。纯流体钉
# （空表 t、Sod L1 三重、全推进）保持全精度 ==，禁止以此为先例放宽
# 其它钉。
SRC_SCALAR_RTOL = 1.0e-12

# 源项短积分 source_resid 钉：纯绝对容差（不带相对项）。resid 是隐式源
# 求解往零收敛的残差，绝对尺子合身；其水位 ~1.6e-5 会把相对尺子无谓
# 放大（CUDA 对基准 1.0e-11 的相对漂即此因）。实测绝对漂 1.63e-16，
# 三次运行逐位复现（archive/C2_cuda.md）；
# atol=1e-14 留 60 倍余量。根因仍是平台数学库固差，非物理漂移。
SRC_RESID_ATOL = 1.0e-14

# ---------------------------------------------------------------------------
# 空表基准钉（.tmp/c1a_baseline_empty.txt / .npz，C1a 刀）
# ---------------------------------------------------------------------------

NSTEPS_EMPTY = 25  # 短积分定步数（基准同）

# 25 步定步长积分后的时间标量（逐位钉）
PIN_EMPTY_T_DUMP = 0.19526979292962213
PIN_EMPTY_DT_DUMP = 0.007110866715904685

# 全推进（advance 到 tstop）的 t 与 Sod L1(n,v,P)/scale 三重（逐位钉）。
# 与 tests/test_step_empty.py、tests/test_core_split.py 同值同口径。
PIN_EMPTY_T_FULL = 0.15224840316529667
PIN_EMPTY_L1_N = 0.0035152402455663314
PIN_EMPTY_L1_V = 0.0097162878019579242
PIN_EMPTY_L1_P = 0.0040293309314733058

# 25 步场内区（去 2 层 ghost）逐槽 (L2, max|.|)，由 c1a_baseline_empty.npz
# 的场 dump 摘要算出。槽序：n, vx, vy, vz, P。1D 的 vy/vz 基准即恒 0。
PIN_EMPTY_SLOTS = {
    "n": (189.04166755346037, 16.737235385484002),
    "vx": (2.4870851108934047, 0.7700617125092953),
    "vy": (0.0, 0.0),
    "vz": (0.0, 0.0),
    "P": (154.65393108501854, 13.806505000000001),
}
# 内区 + 活动面鬼格（1D 中央行 k=ng,j=ng 全 i）单一 L2 指纹。
# C2补3 口径变更：退化轴鬼格（k!=ng 或 j!=ng）不再由设备填皮（无读者），
# 故原全场 L2 不再可复现；改为只钉中央行（内区 + x1 两端活动面鬼格）。
# 钉值来源：.tmp/c2sup3_empty_short.npz 的 fld[:, :, 2, 2, :] 的 L2。
PIN_EMPTY_FLD_GHOST_L2 = 246.20083658059934

# ---------------------------------------------------------------------------
# 开源项基准钉（.tmp/c1a_baseline_source.txt / .npz，C1a 刀）
# ---------------------------------------------------------------------------

NSTEPS_SRC = 10  # 短积分定步数（基准同）

PIN_SRC_T = 804.3403544436965
PIN_SRC_DT = 56.08765032631123
PIN_SRC_RESID = 1.6288990348893032e-05

# 内区（去 2 层 ghost）逐流体逐槽 (L2, max|.|)，照抄基准摘要（全精度值取自
# npz 复核）。槽序：n, vx, vy, vz, P。
PIN_SRC_SLOTS = {
    (0, "n"): (1.2002305248770211e-17, 1.521493253172928e-18),
    (0, "vx"): (2950864.8029298275, 661983.0707883952),
    (0, "vy"): (0.0, 0.0),
    (0, "vz"): (0.0, 0.0),
    (0, "P"): (1.6717748255596875e-06, 2.1806565175089555e-07),
    (1, "n"): (1.4307775227138066e-17, 1.817927906507626e-18),
    (1, "vx"): (3008007.1065987283, 674926.8994790241),
    (1, "vy"): (0.0, 0.0),
    (1, "vz"): (0.0, 0.0),
    (1, "P"): (1.6854682312317106e-06, 2.2080281703897351e-07),
}
# 辐射通量（xmf_rad_flux_copy，全格含 ghost，shape (1, 1700)）的 L2
PIN_SRC_FLUX_L2 = 37.29698101070999

# 源用例常数（与 .tmp/c1a_baseline.py::source_cfg 同一份）
G_P = 6.6726e-8
M_JUP = 1.8982e30
M_P = 0.69 * M_JUP
ALPHA_REC = 4.0e-12
T_REC_REF = 300.0
REC_EXP = 0.64
ETA = 0.20
AMU_H = 1.00794
SIGMA0 = 2.5e-18
HNU0 = 1.0e-13
FINF = 4.716
N0 = 1.0e6
T0 = 1500.0
NX_SRC = 64


# --- 只读小助手（与基准脚本同口径） ------------------------------------------


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _unpack_all(lib, run) -> np.ndarray:
    """全部流体五槽的核内平铺块（含 ghost，码单位），
    shape (nfluid, 5, nztot, nytot, nxtot)。与基准脚本 _unpack_all 同口径。"""
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    ncells = lib.xmf_state_ncells(run._h)
    nfluid = lib.xmf_state_nfluid(run._h)
    out = np.empty((nfluid, 5, nzt, nyt, nxt), dtype=np.float64)
    buf = np.empty(ncells, dtype=np.float64)
    for f in range(nfluid):
        for s in range(5):
            got = lib.xmf_state_unpack(run._h, f, s, buf.ctypes.data_as(_PTR))
            assert got == ncells, f"unpack failed ({got})"
            out[f, s] = buf.reshape((nzt, nyt, nxt))
    return out


def _pack_block(lib, run, f: int, var: int, value: float) -> None:
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    ncells = lib.xmf_state_ncells(run._h)
    data = np.full((nzt, nyt, nxt), value, dtype=np.float64).ravel()
    assert data.size == ncells
    assert lib.xmf_state_pack(run._h, f, var, data.ctypes.data_as(_PTR)) == ncells


def _flux_all(lib, run) -> np.ndarray:
    """每束全格（含 ghost）通量，shape (nbeam, ncells)。同基准脚本。"""
    nb = lib.xmf_rad_nbeam(run._h)
    assert nb > 0
    ncells = lib.xmf_state_ncells(run._h)
    out = np.empty((nb, ncells), dtype=np.float64)
    buf = np.empty(ncells, dtype=np.float64)
    for b in range(nb):
        got = lib.xmf_rad_flux_copy(run._h, b, buf.ctypes.data_as(_PTR))
        assert got == ncells
        out[b] = buf
    return out


def _assert_slot_pins(tag, got, pin_l2, pin_max):
    """单槽 (L2, max) 钉：基准为 0 时逐位钉 0，否则相对容差只给表示误差。"""
    l2 = float(np.linalg.norm(got))
    mx = float(np.max(np.abs(got)))
    if pin_l2 == 0.0:
        assert l2 == 0.0 and mx == 0.0, (
            f"{tag}：基准为恒 0 槽，实测 L2={l2!r} max={mx!r}（1D 横速度/闲槽"
            "不得被 Kokkos 化引入非零值）"
        )
    else:
        assert abs(l2 - pin_l2) <= NORM_RTOL * abs(pin_l2), (
            f"{tag}：L2={l2!r} 偏离基准钉 {pin_l2!r} 超出表示容差"
        )
        assert abs(mx - pin_max) <= NORM_RTOL * abs(pin_max), (
            f"{tag}：max|.|={mx!r} 偏离基准钉 {pin_max!r} 超出表示容差"
        )


# --- 1. 空表短积分（25 步）：t/dt 逐位 + 场逐槽摘要钉 ---------------------------


def test_empty_short_integration_matches_serial_baseline():
    """xmf.shock_tube() 空表 pcm/hll/rk1 定步 25 步：与改前串行 C 基准
    （.tmp/c1a_baseline_empty）的 t/dt 逐位一致；场内区逐槽 (L2, max) 与
    全场（含 ghost）L2 对基准钉死。空路径哨兵：num_tapes==0、nbeam==0。"""
    cfg = xmf.shock_tube()
    assert cfg["numerics"] == {"riemann": "hll", "reconstruction": "pcm",
                               "rk": "rk1"}, "本钉只对 pcm/hll/rk1 有效"
    assert cfg["sources"] == [], "本钉只对空 sources 表有效"

    lib = load()
    run = xmf.setup(cfg)
    try:
        assert run.num_tapes == 0, "空表不得加载带子"
        assert run.nbeam == 0, "无辐射配置不得有束表"

        for _ in range(NSTEPS_EMPTY):
            run.step()

        assert run.t == PIN_EMPTY_T_DUMP, (
            f"空表 25 步后 t={run.t!r}，基准钉 {PIN_EMPTY_T_DUMP!r}"
            "（hydro 循环 Kokkos 化动了时间推进）"
        )
        assert run.dt == PIN_EMPTY_DT_DUMP, (
            f"空表 25 步后 dt={run.dt!r}，基准钉 {PIN_EMPTY_DT_DUMP!r}"
            "（CFL 波速口径被循环改造动了）"
        )

        fld = _unpack_all(lib, run)
        assert fld.shape == (1, 5, 5, 5, 260), (
            f"场块形状 {fld.shape} 与基准 dump (1,5,5,5,260) 不一致"
        )
        inner = fld[:, :, NG:-NG, NG:-NG, NG:-NG]
        for s, name in enumerate(("n", "vx", "vy", "vz", "P")):
            pin_l2, pin_max = PIN_EMPTY_SLOTS[name]
            _assert_slot_pins(f"空表 f0 {name}", inner[0, s], pin_l2, pin_max)
        # C2补3 口径：只钉内区 + 活动面鬼格（1D 中央行 k=ng,j=ng 全 i）；
        # 退化轴鬼格（k!=ng 或 j!=ng）不再由设备填皮，不钉其具体值。
        central_row = fld[:, :, NG, NG, :]
        active_l2 = float(np.linalg.norm(central_row))
        assert abs(active_l2 - PIN_EMPTY_FLD_GHOST_L2) <= NORM_RTOL * PIN_EMPTY_FLD_GHOST_L2, (
            f"空表内区+活动面鬼格 L2={active_l2!r}，基准钉 {PIN_EMPTY_FLD_GHOST_L2!r}"
        )
    finally:
        run.close()


# --- 2. 空表全推进：t 与 Sod L1 三重逐位钉（卡片点名） --------------------------


def test_empty_full_advance_sod_l1_triple_bitwise():
    """空表 pcm Sod 全推进：t 与 L1(n,v,P)/scale 三重逐位等于基准
    （3.515e-3 / 9.716e-3 / 4.029e-3）。口径与 test_step_empty /
    test_core_split 逐字相同（scale = (n_l, max|v_ex|, max p_ex)，
    精确解取 advance 实际到达的 t）。"""
    cfg = xmf.shock_tube()
    run = xmf.setup(cfg)
    try:
        run.advance()
        assert run.t >= run.tstop, "advance 应跑到 tstop（可略过冲）"
        assert run.t == PIN_EMPTY_T_FULL, (
            f"空表全推进 t={run.t!r}，基准钉 {PIN_EMPTY_T_FULL!r}"
        )

        hi = run.fields["HI"]
        ic = cfg["ic"]
        p_l = ic["left"]["n"] * KB * ic["left"]["T"]
        p_r = ic["right"]["n"] * KB * ic["right"]["T"]
        n_ex, v_ex, p_ex = exact_hi(
            hi.x, run.t, ic["x0"], ic["left"]["n"], ic["right"]["n"],
            0.0, 0.0, p_l, p_r, gamma=run._cfg.gamma,
        )
        scale_n = ic["left"]["n"]
        scale_v = max(float(np.max(np.abs(v_ex))), 1.0)
        scale_p = float(np.max(p_ex))
        err_n = float(np.mean(np.abs(hi.n - n_ex)) / scale_n)
        err_v = float(np.mean(np.abs(hi.v - v_ex)) / scale_v)
        err_p = float(np.mean(np.abs(hi.p - p_ex)) / scale_p)

        assert err_n == PIN_EMPTY_L1_N, (
            f"空表 Sod L1(n)/scale={err_n!r}，C1a 基准钉 {PIN_EMPTY_L1_N!r}（钉不动）"
        )
        assert err_v == PIN_EMPTY_L1_V, (
            f"空表 Sod L1(v)/scale={err_v!r}，C1a 基准钉 {PIN_EMPTY_L1_V!r}（钉不动）"
        )
        assert err_p == PIN_EMPTY_L1_P, (
            f"空表 Sod L1(P)/scale={err_p!r}，C1a 基准钉 {PIN_EMPTY_L1_P!r}（钉不动）"
        )
    finally:
        run.close()


# --- 3. 开源项短积分（10 步）：t/dt rtol=1e-12、resid atol=1e-14 + 逐槽 + 通量 ---


def _c1a_source_cfg() -> dict:
    """与 .tmp/c1a_baseline.py::source_cfg 逐字同构：2 流体 HI/HII、6 条带子
    （光化对 + 光加热 + 复合对 + 重力）、1 束辐射、x1 outflow、units 全 1、
    ic none（初值靠 pack）、nx=64。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX_SRC, 1, 1]
    cfg["grid"]["x1min"] = 1.0e10
    cfg["grid"]["x1max"] = 2.0e10
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["fluids"] = [
        {"name": "HI", "mass": AMU_H, "charge": 0, "kind": "hd"},
        {"name": "HII", "mass": AMU_H, "charge": 1, "kind": "hd"},
    ]
    cfg["electron"] = {"mode": "off"}
    cfg["ic"] = {"kind": "none"}
    cfg["boundary"] = {"x1": "outflow"}
    cfg["constants"] = {
        "G": G_P, "M_p": M_P, "eta": ETA, "gm1": 5.0 / 3.0 - 1.0,
        "alpha_rec": ALPHA_REC, "T_rec_ref": T_REC_REF, "rec_exp": REC_EXP,
        "sigma_0": SIGMA0, "hnu_0": HNU0,
    }
    gamma_ion = "n_HI * (sigma_0 * F_0 / hnu_0)"
    heat_vol = "n_HI * (sigma_0 * F_0)"
    recomb = "alpha_rec * (T_rec_ref / T_HII)**rec_exp * n_HII * n_HII"
    cfg["sources"] = [
        {"name": "photoion_HI", "expr": {"dn_HI": f"-({gamma_ion})"}},
        {"name": "photoion_HII", "expr": {"dn_HII": gamma_ion}},
        {"name": "photoheat_HII", "expr": {"dP_HII": "gm1 * eta * "
                                                    f"({heat_vol})"}},
        {"name": "recomb_HI", "expr": {"dn_HI": recomb}},
        {"name": "recomb_HII", "expr": {"dn_HII": f"-({recomb})"}},
        {"name": "gravity", "expr": {"dvx": "-G * M_p / r**2"}},
    ]
    cfg["radiation"] = {
        "enabled": True,
        "absorbers": ["HI"],
        "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
        "bins": {"n": 1, "sigma": [SIGMA0], "hnu": [HNU0],
                 "flux_inf": [FINF]},
    }
    return cfg


def test_source_short_integration_matches_serial_baseline():
    """非空开源项 10 步定步长积分（hydro + scan + 隐式源步全走）：
    t / dt 在 rtol=1e-12（atol=0）内、source_resid 在 atol=1e-14（纯
    绝对）内等于基准——只覆盖平台数学库固差（定性证据
    archive/C2_cuda.md）；内区逐流体逐槽 (L2, max) 与
    通量全格 L2 对基准钉死。这是对三处 Kokkos 化循环（含 scan 的
    i 顺序前缀积与格内局部 LU）的数值验收。"""
    lib = load()
    run = xmf.setup(_c1a_source_cfg())
    try:
        assert run.num_tapes == 6, "基准配置应编出 6 条带子"
        assert run.nbeam == 1, "基准配置应只有 1 束"

        # 均匀初值（与基准脚本相同：units 全 1 → mass_code = AMU_H*AMU）
        m = AMU_H * AMU
        for f in range(2):
            _pack_block(lib, run, f, 0, N0 * m)        # rho 槽
            _pack_block(lib, run, f, 4, N0 * KB * T0)  # P 槽

        for _ in range(NSTEPS_SRC):
            run.step()

        # 三标量钉：t / dt 为 rtol=1e-12、atol=0（显式相对容差，无绝对
        # 容差加项）；source_resid 为 atol=1e-14 纯绝对（往零收敛的残差，
        # 绝对尺子合身，水位 1.6e-5 会无谓放大相对尺子）。定性证据
        # archive/C2_cuda.md：libdevice pow(0.2,0.64) 与 UCRT
        # 差 1 ulp，非物理漂移。同一份测试两树跑，不拆平台；
        # 纯流体钉仍逐位 ==。
        assert abs(run.t - PIN_SRC_T) <= SRC_SCALAR_RTOL * abs(PIN_SRC_T), (
            f"开源项 10 步后 t={run.t!r}，基准钉 {PIN_SRC_T!r}"
            "（rtol=1e-12、atol=0，平台数学库固差容差；"
            "循环改造动了时间推进）"
        )
        assert abs(run.dt - PIN_SRC_DT) <= SRC_SCALAR_RTOL * abs(PIN_SRC_DT), (
            f"开源项 10 步后 dt={run.dt!r}，基准钉 {PIN_SRC_DT!r}"
            "（rtol=1e-12、atol=0，平台数学库固差容差；"
            "波速/步长口径被动）"
        )
        assert abs(run.source_resid - PIN_SRC_RESID) <= SRC_RESID_ATOL, (
            f"source_resid={run.source_resid!r}，基准钉 {PIN_SRC_RESID!r}"
            "（atol=1e-14 纯绝对，平台数学库固差容差；"
            "隐式源步/局部 LU 的残差超限）"
        )

        fld = _unpack_all(lib, run)
        assert fld.shape == (2, 5, 5, 5, 68), (
            f"场块形状 {fld.shape} 与基准 dump (2,5,5,5,68) 不一致"
        )
        inner = fld[:, :, NG:-NG, NG:-NG, NG:-NG]
        for f in range(2):
            for s, name in enumerate(("n", "vx", "vy", "vz", "P")):
                pin_l2, pin_max = PIN_SRC_SLOTS[(f, name)]
                _assert_slot_pins(f"开源项 f{f} {name}", inner[f, s],
                                  pin_l2, pin_max)

        flux = _flux_all(lib, run)
        assert flux.shape == (1, 1700), (
            f"通量形状 {flux.shape} 与基准 (1, 1700) 不一致"
        )
        flux_l2 = float(np.linalg.norm(flux))
        assert abs(flux_l2 - PIN_SRC_FLUX_L2) <= NORM_RTOL * PIN_SRC_FLUX_L2, (
            f"通量（含 ghost）L2={flux_l2!r}，基准钉 {PIN_SRC_FLUX_L2!r}"
            "（scan 的射线前缀积被改序）"
        )
        # 哨兵：物理确实在跑——束向内衰减、流体被重力加速（防拿初值空转冒充）
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        line = flux[0].reshape((nzt, nyt, nxt))[ng, ng, ng:ng + nx]
        assert np.all(np.diff(line) > 0.0), "向内（下标增大→外）通量应递增"
        assert float(np.max(inner[0, 1]) - np.min(inner[0, 1])) > 0.0, (
            "vx 应被重力/源步推出非零结构"
        )
    finally:
        run.close()


# --- 4. 薄 3D x3 台阶（test_hydro_3d.py 退役后唯一持有点：x3 sweep 合同） -------


def test_thin3d_x3_step_vz_excited_idle_vx_vy_zero():
    """薄 3D (8x8x32) 沿 x3 放台阶：10 步后 vz 非零、闲置 vx/vy 整块为 0。

    test_hydro_3d.py 退役后，本条是 x3 sweep 合同的唯一持有点。
    与 test_hydro.py 的 x1/x2 sweep 互补：三个方向都必须真扫。
    闲置分量整块（含 ghost）逐位 0 的证据链同 test_hydro.py。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [8, 8, 32]
    cfg["grid"]["ndim"] = 3
    cfg["boundary"] = {"x1": "copy", "x2": "copy", "x3": "reflect"}
    lib = load()
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        n2r = run._cfg.mass * AMU / run._cfg.units.density
        z0 = 0.5 * (run._cfg.x3min + run._cfg.x3max)
        p_l = 1.0e13 * KB * 1.0e4
        p_r = 1.25e12 * KB * 8.0e3

        # z-axis coordinates for step initialization
        ntot_z = lib.xmf_mesh_ncell(run._h, 2)
        zbuf = np.empty(ntot_z, dtype=np.float64)
        assert lib.xmf_mesh_copy_x(run._h, 2, zbuf.ctypes.data_as(_PTR)) == ntot_z
        m = (zbuf < z0).reshape(nzt, 1, 1)
        ones = np.ones((nzt, nyt, nxt))
        n_l, n_r = 1.0e13, 1.25e12
        for var, vl, vr in (
            (0, n_l * n2r, n_r * n2r),
            (1, 0.0, 0.0),
            (2, 0.0, 0.0),
            (3, 0.0, 0.0),
            (4, p_l, p_r),
        ):
            arr = np.ascontiguousarray(np.where(m, vl, vr) * ones)
            assert lib.xmf_state_pack(run._h, 0, var,
                                      arr.ctypes.data_as(_PTR)) == arr.size

        for _ in range(10):
            run.step()
        assert run.t > 0.0

        inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
        # vz excited by x3 sweep
        vz_buf = np.empty(lib.xmf_state_ncells(run._h), dtype=np.float64)
        assert lib.xmf_state_unpack(run._h, 0, 3,
                                    vz_buf.ctypes.data_as(_PTR)) == vz_buf.size
        vz_i = vz_buf.reshape(nzt, nyt, nxt)[inner]
        assert float(np.max(np.abs(vz_i))) > 5.0e-2, (
            f"x3 sweep: vz should be non-zero, got max|vz|="
            f"{float(np.max(np.abs(vz_i))):.3e}"
        )
        # idle vx/vy remain zero (whole block including ghost)
        for var, name in ((1, "vx"), (2, "vy")):
            buf = np.empty(lib.xmf_state_ncells(run._h), dtype=np.float64)
            assert lib.xmf_state_unpack(run._h, 0, var,
                                        buf.ctypes.data_as(_PTR)) == buf.size
            assert np.all(buf == 0.0), (
                f"idle {name} should be zero everywhere, "
                f"max|{name}|={float(np.max(np.abs(buf))):.3e}"
            )
    finally:
        run.close()
