"""M 块 2（M4）：离子 MHD 通量 + 一次有限体积感应单测。

规格来源：docs/SESSION.md（本刀卡片）「交什么」§5 与
docs/archive/MHD_REVIEW.md R-9（本刀只锁两条）：

  1. 空表 1D Brio–Wu / Tóth 管（钉 A：hll+plm；钉 B：hlld+pcm）；
  2. 两支共动离子（同速度、Σw_s=1）逐支通量 + 一次感应还原单流体
     MHD（R-9 候选首条）。

用例清单：
  * 闸门：磁开 + hllc 拒（无静默回退）；磁开 + 无 charge>0 流体拒。
    两者都落在 xmf_mag_load(-1) → RuntimeError，断到错误文本；
    磁开 + hlld（有离子）：setup 过、一步能跑。
  * 零 B + hll 与关磁流体五槽逐位相等（np.array_equal，12 步 ≥10）：
    B=0 ⇒ MHD HLL 退回 HD HLL、感应为 0，dt / t 逐步逐位相同。
    （本对照只存这一份，test_mag.py 不重复。）
  * 空表 1D Brio–Wu：NX=512、γ=2、cfl=0.4、outflow、BX=0.75、
    t_end=0.25、单 HII（q=1,m=1）、电子关、x∈[0,1] 间断 0.5，
    左(ρ=1,By=+1,P=1) 右(ρ=0.125,By=−1,P=0.1)。
    断言 L1(n,vx,vy,P,B2)/scale 上界（1.5× 实测起步）、vz/B3 逐位 0、
    法向 B1 全块（含鬼区）逐步逐位等于初值、步数在区间内（防 dt 暴涨）。
  * 两支共动还原（钉 C）：NX=128、rk3/plm、40 步；hll 与 hlld 各一条；
    逐支五槽 + B 对单支参考解相对差 ≤1e-12，dt 同量级。
  * 磁开四步链 setup → step → fields_full → close 通 + 同进程多轮。
  * B 读口：fields_full 的 B1/B2/B3 三分量（活动格下表面）与
    xmf_mag_pack/unpack 完整面心块并用；断言从不只取单分量。

钉值与实测关系（2026-08-29 本测首跑复现，builder 探针 .tmp/m4_*.py
同参数逐位一致；上界 = 实测 ×1.5 向上取整）：
  钉 A hll/plm/rk3：steps=1517，L1(n,vx,vy,P,B2) =
    0.23821777638107583 / 0.5015876017094985 / 0.639908693461778 /
    0.3389802335734843 / 0.32671029276583363；L1(vz)=L1(B3)=0；
    max|B1−B1_ic|=0（逐位）。
  钉 B hlld/pcm/rk3：steps=1441，L1(n,vx,vy,P,B2) =
    0.23603066180926174 / 0.49601629578843165 / 0.6337390886221774 /
    0.33739049566182405 / 0.33386173217385984；B1 漂移=0（逐位）。
  钉 C（128 格 40 步）：最差相对差 hll 2.7e-14（P）、hlld 2.4e-14
    （vx）；dt 相对差 ≤2e-15。断言用 1e-12 相对容差（约两量级裕量）。
  已知弱点（不绕过、不改核）：hlld + plm/weno3 在 NX=512 右行快波
  出 outflow 边界时长棋盘排空层（odd-even 类，已定性归档），钉 B
  因此用 pcm。

L1 口径（同 builder 探针）：L1 = mean(|final − ic|) 活动排、代码
单位；scale 取初值特征量 ρ_L = 1、P_L = 1、|By_L| = 1、v 参考 = 1
（域长亦为 1），故 L1/scale 数值即 L1 本身。

禁令自查（卡片 + docs/TODO.md §A）：
  - 不读 pluto.ini：本文件无任何文件 IO；
  - 不用 XUV_FACTOR==0 哨兵：本文件不涉辐射；
  - 不只测单分量：流体断言覆盖五槽 (n, vx, vy, vz, P)，磁断言覆盖
    B1/B2/B3 三分量（法向漂移、切向 L1、读口三分量同读）。

运行时长设计：512 格两管各 ~22 s（实测），共动题留在 128 格（<1 s），
全套 < 1 min，不需降参数。
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU, UNIT_DENSITY

_PTR = ctypes.POINTER(ctypes.c_double)

# --- Brio–Wu 管参数（钉 A / B，与 builder 探针逐字同参） --------------------
NX = 512
NG = 2
TEND = 0.25
BX = 0.75
GAMMA = 2.0
CFL = 0.4
STEPS_CAP = 2400          # 早退护栏（实测 1441 / 1517；防 dt 异常小死循环）
PIN_STEP_RANGE = (1300, 1800)  # 步数区间（实测 1517 / 1441），防 dt 暴涨

# 钉 A / B 上界 = builder 实测 ×1.5 向上取整（关系见文件头）
PIN_A_BOUNDS = dict(n=0.36, vx=0.76, vy=0.96, P=0.51, B2=0.50)   # hll/plm
PIN_B_BOUNDS = dict(n=0.36, vx=0.75, vy=0.96, P=0.51, B2=0.51)   # hlld/pcm

# --- 两支共动还原参数（钉 C） ------------------------------------------------
NX_PAIR = 128
NSTEPS_PAIR = 40
REL_TOL = 1.0e-12         # 实测 ≤2.7e-14，留约两量级裕量

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


def _unpack_slot(lib, h, fluid: int, var: int) -> np.ndarray:
    ncells = lib.xmf_state_ncells(h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(h, fluid, var, _ptr(buf)) == ncells
    return buf


def _unpack_b(lib, h, comp: int, ntot) -> np.ndarray:
    """One whole face-centred block (ghosts included), reshaped to the
    kernel's (d2, d1, comp) axis order."""
    nb = _len_b(ntot, comp)
    buf = np.empty(nb, dtype=np.float64)
    assert lib.xmf_mag_unpack(h, comp, _ptr(buf)) == nb
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return buf.reshape(ntot[d2], ntot[d1], ntot[comp] + 1)


def _pack_b(lib, h, comp: int, blk: np.ndarray) -> None:
    got = lib.xmf_mag_pack(h, comp, _ptr(np.ascontiguousarray(blk.ravel())))
    assert got == blk.size, f"xmf_mag_pack failed ({got})"


def _cell_row(arr3: np.ndarray, nx: int, ng: int = NG) -> np.ndarray:
    """Interior cell row of a (k, j, i) block (1D tube)."""
    return arr3[ng, ng, ng : ng + nx]


def _face_row(blk: np.ndarray, comp: int, nx: int, ng: int = NG) -> np.ndarray:
    """Active lower-face row of one B block (face index == cell index)."""
    if comp == 0:  # block axes (k, j, iface)
        return blk[ng, ng, ng : ng + nx]
    if comp == 1:  # block axes (i, k, jface)
        return blk[ng : ng + nx, ng, ng]
    return blk[ng, ng : ng + nx, ng]  # comp == 2, block axes (j, i, kface)


def _pack_tube_b(lib, h, ntot, nx: int, ng: int = NG) -> None:
    """Brio–Wu 型 B 初值：法向均匀 BX；By 在间断处 ±1；Bz 恒零。
    左态占格子 0..nx/2-1（其下表面即面 ng..ng+nx/2-1），域顶面续右态。"""
    b1 = np.full((ntot[2], ntot[1], ntot[0] + 1), BX)
    b2 = np.zeros((ntot[0], ntot[2], ntot[1] + 1))
    b2[: ng + nx // 2, :, :] = 1.0
    b2[ng + nx // 2 :, :, :] = -1.0
    b3 = np.zeros((ntot[1], ntot[0], ntot[2] + 1))
    _pack_b(lib, h, 0, b1)
    _pack_b(lib, h, 1, b2)
    _pack_b(lib, h, 2, b3)


# --- 1. 闸门：hllc 拒 / 无离子拒 / hlld 过 -----------------------------------


def _ion_tube_cfg(riemann: str = "hll") -> dict:
    """小磁开管：64 格单 HII、outflow、短 tstop（链路 / 闸门用例共用）。"""
    cfg = xmf.shock_tube()
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg["numerics"]["riemann"] = riemann
    cfg["boundary"] = {"x1": "outflow"}
    cfg["grid"]["n"] = [64, 1, 1]
    cfg["time"] = {"cfl": 0.4, "tstop": 0.02, "dt_max": 0.0}
    cfg["mhd"] = True
    return cfg


def test_mag_on_hllc_refused_at_setup():
    """磁开 + hllc：setup 拒。现成 hllc 波系不含磁波，离子 MHD 不复用；
    无静默回退、不把 blob 里的 1 偷换成 3。拒收落在 xmf_mag_load(-1)，
    配置层本身通过（闸门是内核闸门）。"""
    cfg = _ion_tube_cfg("hllc")
    packed = xmf.normalize(cfg)
    assert packed.numerics.riemann == "hllc"
    assert "mhd" in packed.extra
    with pytest.raises(RuntimeError, match=r"xmf_mag_load failed \(-1\)"):
        xmf.setup(cfg)


def test_mag_on_without_positive_charge_refused_at_setup():
    """磁开 + 全表无 charge>0 流体：u_+ 无定义、全员 B_s=0 却挂着门——
    setup 拒（同样落 xmf_mag_load(-1)）。"""
    cfg = xmf.shock_tube()  # 默认单支中性 HI
    cfg["mhd"] = True
    packed = xmf.normalize(cfg)
    assert all(f.charge <= 0.0 for f in packed.fluids)
    with pytest.raises(RuntimeError, match=r"xmf_mag_load failed \(-1\)"):
        xmf.setup(cfg)


def test_mag_on_hlld_with_ion_setup_and_step(lib):
    """磁开 + hlld（有离子）：M4 起 setup 过、一步能跑、B 读口在场。"""
    run = xmf.setup(_ion_tube_cfg("hlld"))
    try:
        run.step()
        assert run.dt > 0.0
        fld = run.fields_full()
        assert {"B1", "B2", "B3"} <= set(fld.keys())
    finally:
        run.close()


# --- 2. 零 B + hll 与关磁流体五槽逐位相等 ------------------------------------


def test_zero_b_hll_bitwise_matches_mag_off(lib):
    """同一套离子初值 / 边界 / 步数：磁开（B 块全零，xmf_mag_load 的
    零初值，不 pack）与关磁各跑 12 步（≥10），每步流体五槽
    (n, vx, vy, vz, P) 逐位相等、dt / t 逐位相等；跑完三个 B 块仍恒零。
    B=0 ⇒ 快磁声速退回声速（dt 同）、MHD HLL 退回 HD HLL、感应
    u_+×B=0。关磁三档不受影响的口径在 test_config，不在此重复。"""
    nsteps = 12
    cfg_off = xmf.shock_tube()
    cfg_off["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg_on = xmf.shock_tube()
    cfg_on["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg_on["mhd"] = True
    assert cfg_off["numerics"]["riemann"] == "hll"
    run_off = xmf.setup(cfg_off)
    run_on = xmf.setup(cfg_on)
    try:
        h_off, h_on = run_off._h, run_on._h
        ncells = lib.xmf_state_ncells(h_off)
        assert lib.xmf_state_ncells(h_on) == ncells
        ntot = _ntot_of(run_on)
        buf_a = np.empty(ncells, dtype=np.float64)
        buf_b = np.empty(ncells, dtype=np.float64)
        for step in range(nsteps):
            run_off.step()
            run_on.step()
            assert run_off.dt == run_on.dt, step
            assert run_off.t == run_on.t, step
            for var in range(5):  # 五槽全比，不只 VX1
                assert lib.xmf_state_unpack(h_off, 0, var, _ptr(buf_a)) == ncells
                assert lib.xmf_state_unpack(h_on, 0, var, _ptr(buf_b)) == ncells
                assert np.array_equal(buf_a, buf_b), f"step {step} slot {var}"
        for comp in range(3):  # 感应为 0：B 块逐位不动（恒零）
            assert np.all(_unpack_b(lib, h_on, comp, ntot) == 0.0), comp
    finally:
        run_off.close()
        run_on.close()


# --- 3. 磁开四步链 + 同进程多轮 ----------------------------------------------


def test_four_step_chain_mag_on_multi_round(lib):
    """同进程多轮 setup → step → fields_full → close：磁开轮 pack 非零
    B1 常量后跑两步，B 读口三分量齐全且有限；穿插一轮关磁（无 B 键）。"""
    for _round in range(2):
        run = xmf.setup(_ion_tube_cfg("hll"))
        try:
            h = run._h
            ntot = _ntot_of(run)
            nb1 = _len_b(ntot, 0)
            b1 = np.full(nb1, 0.3, dtype=np.float64)
            assert lib.xmf_mag_pack(h, 0, _ptr(b1)) == nb1
            for _ in range(2):
                run.step()
                assert run.dt > 0.0
            fld = run.fields_full()
            assert {"B1", "B2", "B3"} <= set(fld.keys())
            assert "HII" in fld
            for key in ("B1", "B2", "B3"):  # 三分量都读，不只单分量
                assert np.all(np.isfinite(fld[key])), key
            for key in ("n", "vx", "vy", "vz", "P"):
                assert np.all(np.isfinite(fld["HII"][key])), key
        finally:
            run.close()
        run_off = xmf.setup(xmf.shock_tube())
        try:
            run_off.step()
            assert not any(k.startswith("B") for k in run_off.fields_full())
        finally:
            run_off.close()


# --- 4. 两支共动离子还原单流体 MHD（R-9 首条，钉 C） -------------------------


def _comoving_cfg(riemann: str, two: bool) -> dict:
    """128 格 Brio–Wu 型初值；two 时两枚 q=1,m=1 离子各占 ρ/2、P/2、
    同速、同 B（w_s 各 1/2，Σw_s=1；a_s、b_s 与单支全量同值）。"""
    cfg = xmf.shock_tube()
    cfg["gamma"] = GAMMA
    cfg["numerics"] = {"riemann": riemann, "reconstruction": "plm", "rk": "rk3"}
    cfg["grid"] = {
        "geometry": "cartesian",
        "ndim": 1,
        "n": [NX_PAIR, 1, 1],
        "stretch": "none",
        "x1min": 0.0,
        "x1max": 1.0,
        "nghost": NG,
    }
    cfg["time"] = {"cfl": CFL, "tstop": 10.0, "dt_max": 0.0}
    if two:
        cfg["fluids"] = [
            {"name": "HII", "mass": 1.0, "charge": 1, "kind": "ion"},
            {"name": "HeII", "mass": 1.0, "charge": 1, "kind": "ion"},
        ]
    else:
        cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg["electron"] = {"mode": "off"}
    cfg["boundary"] = {"x1": "outflow"}
    half = 0.5 if two else 1.0
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": half * UNIT_DENSITY / (1.0 * AMU), "v": 0.0, "p": half * 1.0},
        "right": {
            "n": half * 0.125 * UNIT_DENSITY / (1.0 * AMU),
            "v": 0.0,
            "p": half * 0.1,
        },
    }
    cfg["mhd"] = True
    return cfg


def _pack_second_ion(lib, run, ntot) -> None:
    """内核 IC 只填流体 0：第二支按同一台阶（ρ/2、P/2、零速）手填，
    含鬼区；台阶判定复刻内核的 x1[i] < x0（x1 是含鬼区格心坐标）。"""
    ncells = lib.xmf_state_ncells(run._h)
    ntot0 = ntot[0]
    dx = 1.0 / NX_PAIR
    n = np.empty(ncells, dtype=np.float64)
    p = np.empty(ncells, dtype=np.float64)
    v = np.zeros(ncells, dtype=np.float64)
    for idx in range(ncells):
        i = idx % ntot0
        left = (i - NG + 0.5) * dx < 0.5
        n[idx] = 0.5 if left else 0.0625
        p[idx] = 0.5 if left else 0.05
    assert lib.xmf_state_pack(run._h, 1, 0, _ptr(n)) == ncells
    for var in (1, 2, 3):
        assert lib.xmf_state_pack(run._h, 1, var, _ptr(v)) == ncells
    assert lib.xmf_state_pack(run._h, 1, 4, _ptr(p)) == ncells


@pytest.mark.parametrize("riemann", ["hll", "hlld"])
def test_two_comoving_ions_reduce_to_single_fluid(lib, riemann):
    """两支共动还原（本刀锁死）：逐支通量 + 一次感应后，复合态对单支
    参考解在噪声量级重合——密度 / 压强按两支和比，速度逐支对单支、
    两支互比，B 场逐分量对单支；dt 每步同量级。实测最差相对差
    ≤2.7e-14，断言用 1e-12。"""
    run_one = xmf.setup(_comoving_cfg(riemann, two=False))
    run_two = xmf.setup(_comoving_cfg(riemann, two=True))
    try:
        ntot = _ntot_of(run_one)
        assert _ntot_of(run_two) == ntot
        _pack_tube_b(lib, run_one._h, ntot, NX_PAIR)
        _pack_tube_b(lib, run_two._h, ntot, NX_PAIR)
        _pack_second_ion(lib, run_two, ntot)
        shape = (ntot[2], ntot[1], ntot[0])
        for step in range(NSTEPS_PAIR):
            run_one.step()
            run_two.step()
            dt_rel = abs(run_one.dt - run_two.dt) / run_one.dt
            assert dt_rel < REL_TOL, (step, run_one.dt, run_two.dt)
            for var, key in enumerate(("n", "vx", "vy", "vz", "P")):
                one = _cell_row(_unpack_slot(lib, run_one._h, 0, var).reshape(shape), NX_PAIR)
                two0 = _cell_row(_unpack_slot(lib, run_two._h, 0, var).reshape(shape), NX_PAIR)
                two1 = _cell_row(_unpack_slot(lib, run_two._h, 1, var).reshape(shape), NX_PAIR)
                scale = max(float(np.max(np.abs(one))), 1.0)
                if key in ("vx", "vy", "vz"):
                    # 速度：两支互比 + 对单支比
                    assert float(np.max(np.abs(two1 - two0))) / scale < REL_TOL, (step, key)
                    assert float(np.max(np.abs(one - two0))) / scale < REL_TOL, (step, key)
                else:
                    # 密度 / 压强：单支 == 两支和
                    assert float(np.max(np.abs(one - (two0 + two1)))) / scale < REL_TOL, (step, key)
            for comp in range(3):  # B 三分量都比，不只单分量
                one = _face_row(_unpack_b(lib, run_one._h, comp, ntot), comp, NX_PAIR)
                two = _face_row(_unpack_b(lib, run_two._h, comp, ntot), comp, NX_PAIR)
                scale = max(float(np.max(np.abs(one))), 1.0)
                assert float(np.max(np.abs(one - two))) / scale < REL_TOL, (step, comp)
    finally:
        run_one.close()
        run_two.close()


# --- 5. 空表 1D Brio–Wu 管（钉 A / B） ----------------------------------------


def _briowu_cfg(riemann: str, recon: str) -> dict:
    cfg = xmf.shock_tube()
    cfg["gamma"] = GAMMA
    cfg["numerics"] = {"riemann": riemann, "reconstruction": recon, "rk": "rk3"}
    cfg["grid"] = {
        "geometry": "cartesian",
        "ndim": 1,
        "n": [NX, 1, 1],
        "stretch": "none",
        "x1min": 0.0,
        "x1max": 1.0,
        "nghost": NG,
    }
    cfg["time"] = {"cfl": CFL, "tstop": TEND, "dt_max": 0.0}
    cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
    cfg["electron"] = {"mode": "off"}
    cfg["boundary"] = {"x1": "outflow"}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.5,
        "left": {"n": UNIT_DENSITY / (1.0 * AMU), "v": 0.0, "p": 1.0},
        "right": {"n": 0.125 * UNIT_DENSITY / (1.0 * AMU), "v": 0.0, "p": 0.1},
    }
    cfg["mhd"] = True
    return cfg


def _run_briowu(lib, riemann: str, recon: str, bounds: dict) -> None:
    """一条管：pack B 初值 → 推进到 t_end（每步钉 B1 逐位不动）→
    L1 上界 + vz/B3 零钉 + 步数区间 + 读口三分量。"""
    run = xmf.setup(_briowu_cfg(riemann, recon))
    try:
        h = run._h
        ntot = _ntot_of(run)
        assert ntot == (NX + 2 * NG, 1 + 2 * NG, 1 + 2 * NG), ntot
        _pack_tube_b(lib, h, ntot, NX)
        # IC 快照（代码单位）：五槽活动排 + B2 活动面排 + B1 全块
        ic = {}
        for var, name in enumerate(("n", "vx", "vy", "vz", "P")):
            blk = _unpack_slot(lib, h, 0, var).reshape(ntot[2], ntot[1], ntot[0])
            ic[name] = _cell_row(blk, NX).copy()
        ic["B2"] = _face_row(_unpack_b(lib, h, 1, ntot), 1, NX).copy()
        ic["B3"] = _face_row(_unpack_b(lib, h, 2, ntot), 2, NX).copy()
        b1_ic = _unpack_b(lib, h, 0, ntot).copy()
        # 初值自洽钉：左右态确是 ρ=1/0.125、P=1/0.1（码值）
        assert abs(ic["n"][0] - 1.0) < 1e-12 and abs(ic["n"][-1] - 0.125) < 1e-12
        assert abs(ic["P"][0] - 1.0) < 1e-12 and abs(ic["P"][-1] - 0.1) < 1e-12

        steps = 0
        while run.t < TEND:
            run.step()
            steps += 1
            if steps > STEPS_CAP:
                raise AssertionError(
                    f"{riemann}/{recon}: steps > {STEPS_CAP}，t={run.t!r}"
                    "（dt 异常偏小？）"
                )
            assert run.dt > 0.0
            # 法向 B1：1D 感应不写法向面——全块（含鬼区）逐步逐位等于
            # 初值（表示噪声级，不是容差）
            b1_now = _unpack_b(lib, h, 0, ntot)
            assert np.array_equal(b1_now, b1_ic), (
                f"step {steps}: B1 drifted, max|dB1|="
                f"{float(np.max(np.abs(b1_now - b1_ic)))!r}"
            )
        lo, hi = PIN_STEP_RANGE
        assert lo <= steps <= hi, (
            f"{riemann}/{recon}: steps={steps} 超出区间 [{lo}, {hi}]"
            "（dt 异常？）"
        )

        # 终态五槽 + 切向 B 活动面排
        fin = {}
        for var, name in enumerate(("n", "vx", "vy", "vz", "P")):
            blk = _unpack_slot(lib, h, 0, var).reshape(ntot[2], ntot[1], ntot[0])
            fin[name] = _cell_row(blk, NX)
        fin["B2"] = _face_row(_unpack_b(lib, h, 1, ntot), 1, NX)
        fin["B3"] = _face_row(_unpack_b(lib, h, 2, ntot), 2, NX)
        assert np.all(np.isfinite(fin["n"])) and np.all(fin["n"] > 0.0)
        assert np.all(np.isfinite(fin["P"])) and np.all(fin["P"] > 0.0)

        # L1 = mean|final-ic| / scale；scale 全是 1（ρ_L、P_L、|By_L|、
        # v 参考，见文件头），逐量给上界（1.5× 实测），禁止逐位 ==
        l1 = {key: float(np.mean(np.abs(fin[key] - ic[key]))) for key in ic}
        for key, bound in bounds.items():
            assert l1[key] < bound, (
                f"{riemann}/{recon}: L1({key})={l1[key]!r} 超上界 {bound}"
            )
        # 横向零钉：z 方向无激发，逐位 0（不是容差）
        assert l1["vz"] == 0.0, l1["vz"]
        assert l1["B3"] == 0.0, l1["B3"]
        # 法向 B1 活动面终检：逐位等于初值常量
        assert np.array_equal(
            _face_row(_unpack_b(lib, h, 0, ntot), 0, NX), np.full(NX, BX)
        )

        # B 读口：fields_full 三分量全读——形状 (1,1,NX)（活动格下表面），
        # 数值 = xmf_mag_unpack 下表面排 × unit_b（同 api.py 的切法与
        # 乘法，逐位相等）。默认单位下 unit_b = sqrt(1e-12)*1e6。
        fld = run.fields_full()
        assert set(k for k in fld if k.startswith("B")) == {"B1", "B2", "B3"}
        u = run._cfg.units
        unit_b = math.sqrt(u.density) * u.velocity
        for comp, key in enumerate(("B1", "B2", "B3")):
            assert fld[key].shape == (1, 1, NX), key
            expect = _face_row(_unpack_b(lib, h, comp, ntot), comp, NX) * unit_b
            assert np.array_equal(fld[key][0, 0, :], expect), key
    finally:
        run.close()


def test_briowu_hll_plm_l1_bounds(lib):
    """钉 A：hll / plm / rk3。实测 steps=1517，L1(n,vx,vy,P,B2) =
    0.23822 / 0.50159 / 0.63991 / 0.33898 / 0.32671（上界 1.5× 实测）。"""
    _run_briowu(lib, "hll", "plm", PIN_A_BOUNDS)


def test_briowu_hlld_pcm_l1_bounds(lib):
    """钉 B：hlld / pcm / rk3（hlld+plm 的已知棋盘弱点见文件头，不绕）。
    实测 steps=1441，L1(n,vx,vy,P,B2) = 0.23603 / 0.49602 / 0.63374 /
    0.33739 / 0.33386（上界 1.5× 实测）。"""
    _run_briowu(lib, "hlld", "pcm", PIN_B_BOUNDS)
