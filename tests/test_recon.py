"""A4 补：重建模块（LINEAR=plm / WENO3=weno3）契约单测。

合同来源：`docs/SESSION.md` 本刀卡片「交什么」第 5 条（模块契约）逐条落地，
第 4 条的 setup 拒绝与第 6 条的「四步仍通」也在这里锁。指令卡：
`docs/cmd/PROMPT_A4_RECON.md`。同一 `xmf.shock_tube()`（hll + rk1 + 256 格 +
默认 tstop），只把 `numerics.reconstruction` 换成 pcm / plm / weno3。

断言 ↔ 合同对应：
  1. test_plm_weno3_strictly_better_than_pcm_l1 —— 「严优于」。三家各自
     advance 到 tstop，对 `utils.riemann.exact_hi` 报 L1(n,v,P)/scale（口径 =
     test_shock_hd / test_rk 的 scale = (n_left, max|v_ex|∨1, max p_ex)；
     终时略不同，各自对各自 run.t 的精确解）。合同：plm / weno3 的 L1(n)、
     L1(P) **严格小于** pcm（用 `<`，不是比值容差）；L1(v) ≤ pcm。
  2. test_plm_weno3_not_silently_pcm_1d —— 「不再静默当 PCM」。
     **注意：分段常数初值下第 1 步 plm 与 pcm 逐位相同是应有性质，不是
     bug**——van Leer 斜率在台阶初值下处处为 0（Δ+Δ−≤0），WENO3 在台阶面
     的权重也退化回单侧格心。所以差异断言放在第 5 步（此时剖面已被整出
     结构），不在第 1 步上断「必须不同」。后人不要把这个性质误改成
     「第 1 步就必须不同」。
  3. test_idle_vy_vz_zero_full_block —— 闲置速度。1D 管跑到 tstop，
     plm / weno3 的 vy、vz 整块（内部 + ghost）精确为 0。
  4. test_thin2d_x2_reconstruction_excited + test_thin2d_x2_not_silently_pcm
     —— 薄 2D（8×32）必须走到 x2 重建：沿 x2 的 Sod 台阶 10 步后 vy 明显
     非零、变化只来自 x2、状态真演化；且 5 步后整块密度与 pcm 不同——
     若 x2 sweep 静默当 PCM，该初值沿 x1 均匀、全部动力学在 x2，plm 结果
     将与 pcm 逐位相同，差异断言正是钉这一点的。
  5. test_contact_hllc_step_does_not_drift —— `hllc` + plm / weno3：
     u=0、∇p=0 的密度台阶 20 步后台阶中心不漂移（容差口径照抄
     test_hydro 的 hllc 接触测法：漂移 ≤ 1/4 胞元、L1(n) 变化 ≤ 1e-9、
     max|vx| ≤ 1e-12）。
  6. setup 拒绝（卡片第 4 条）：recon 整数越界（绕过前端表，直接改
     blob）→ C setup 返回 NULL；未知重建名 → 前端 ValueError；
     plm / weno3 且 nghost=1 → 前端能过、C setup 拒（同 test_rk 的 rk4
     写法）；pcm + nghost=1 不拒（PCM 只需 1 层，实测可正常 step）。
  7. test_four_steps_all_recons —— 卡片第 6 条：pcm / plm / weno3 各一次
     setup → step → fields → advance → fields 仍通。

实测基线（本文件首跑，2026-08-22，`pip install .` 重装包，与指令卡参考
实测逐位一致）：
  pcm    t=0.1522  L1(n,v,P)/scale = 3.515e-3  9.716e-3  4.029e-3
  plm    t=0.1544  L1(n,v,P)/scale = 1.920e-3  3.402e-3  1.229e-3
  weno3  t=0.1557  L1(n,v,P)/scale = 2.393e-3  4.540e-3  1.750e-3
  v 三分量全部优于 pcm（比值 ~0.35 / ~0.47），`<=` 断言直接成立。
  1D 5 步差异：max|Δn| ≈ 0.87（plm）/ 0.85（weno3）码单位。
  薄 2D 10 步：max|vy| = 0.843（plm）/ 0.849（weno3）码单位
  （与 builder 自证 ≈0.84 同量级）。接触漂移 / ΔL1 / max|vx| 均为恰 0。

约定（docs/TODO.md §A）：一块一文件；断言只打重建模块契约。不读
`pluto.ini`、无 `XUV_FACTOR==0` 哨兵、不只测 `VX1`（本文件有 vy / vz 整块
断言与薄 2D x2 断言）。
"""
from __future__ import annotations

import copy
import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

import xmf
from xmf._native import load, problem_to_c
from xmf.config import AMU, KB, normalize

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from utils.riemann import exact_hi  # noqa: E402

# 槽序 (n, vx, vy, vz, P) == 0..4（A2 契约）
SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4
_PTR = ctypes.POINTER(ctypes.c_double)


def _uniform_faces(blob):
    """A1 补：直调 C 的用例按新签名补三向等距活动界面（n_d+1 个）。
    机械适配：界面由 blob 自身的 xdmin/xdmax/n 生成，断言语义不变。"""
    mins = (blob.x1min, blob.x2min, blob.x3min)
    maxs = (blob.x1max, blob.x2max, blob.x3max)
    arrs = []
    for d in range(3):
        nd = int(blob.n[d]) if int(blob.n[d]) >= 1 else 1
        dx = (maxs[d] - mins[d]) / nd
        arrs.append(
            np.array([mins[d] + k * dx for k in range(nd + 1)], dtype=np.float64)
        )
    return arrs


def _setup_c(lib, blob):
    """按新 xmf_setup(cfg, xf1, xf2, xf3) 签名直调 C。"""
    arrs = _uniform_faces(blob)
    return lib.xmf_setup(
        ctypes.byref(blob),
        arrs[0].ctypes.data_as(_PTR),
        arrs[1].ctypes.data_as(_PTR),
        arrs[2].ctypes.data_as(_PTR),
    )

GAMMA = 5.0 / 3.0
# Sod 左右态（cgs），与 xmf.shock_tube() 的 ic 一致
SOD_NL, SOD_TL = 1.0e13, 1.0e4
SOD_NR, SOD_TR = 1.25e12, 8.0e3
SOD_PL = SOD_NL * KB * SOD_TL
SOD_PR = SOD_NR * KB * SOD_TR
SOD_X0 = 5.5
NX = 256

# 接触间断（合同 5）左右数密度与恒定压强；压强直接给 p，保证 ∇p=0 逐位成立
N_L, N_R = 1.0e13, 1.25e12
P0 = 13.806505  # dyn/cm^2 == N_L*KB*1e4（同 test_hydro 口径）
X0 = 5.5

RECONS = ("pcm", "plm", "weno3")


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（state/mesh 查询符号已绑定）。"""
    return load()


# --- 只读小助手（口径照抄 test_rk / test_hydro） ------------------------------


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _unpack(lib, run, var):
    """unpack 流体 0 槽 var 的整块（含 ghost），返回 flat 缓冲。"""
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, 0, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _unpack3(lib, run, var):
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    return _unpack(lib, run, var).reshape((nzt, nyt, nxt)), (ng, nx, ny, nz)


def _axis(lib, run, d):
    ntot = lib.xmf_mesh_ncell(run._h, d)
    buf = np.empty(ntot, dtype=np.float64)
    assert lib.xmf_mesh_copy_x(run._h, d, buf.ctypes.data_as(_PTR)) == ntot
    return buf


def _code_scales(run):
    u = run._cfg.units
    mass = run._cfg.mass
    n_to_rho = mass * AMU / u.density
    return n_to_rho, u.density / (mass * AMU), u.velocity, u.density * u.velocity**2


def _tube_cfg(recon: str) -> dict:
    """默认 shock_tube()，只改 numerics.reconstruction。"""
    cfg = xmf.shock_tube()
    cfg["numerics"]["reconstruction"] = recon
    return cfg


def _run_to_tstop(recon: str):
    run = xmf.setup(_tube_cfg(recon))
    run.advance()
    assert run.t >= run.tstop, f"advance 应跑到 tstop（recon={recon}，t={run.t}）"
    return run


def _l1(num, ex, scale: float) -> float:
    return float(np.mean(np.abs(num - ex)) / scale)


def _l1_triple(run):
    """hd_eval / test_rk 口径：L1(n,v,P)/scale，scale = (n_l, max|v_ex|∨1, max p_ex)。"""
    hi = run.fields["HI"]
    n_ex, v_ex, p_ex = exact_hi(
        hi.x, run.t, SOD_X0, SOD_NL, SOD_NR, 0.0, 0.0, SOD_PL, SOD_PR, gamma=GAMMA
    )
    scales = (SOD_NL, max(float(np.max(np.abs(v_ex))), 1.0), float(np.max(p_ex)))
    return (
        _l1(hi.n, n_ex, scales[0]),
        _l1(hi.v, v_ex, scales[1]),
        _l1(hi.p, p_ex, scales[2]),
    )


def _pack_x2_step(lib, run):
    """薄 2D：整块（含 ghost）写入沿 x2 的 Sod 台阶（照抄 test_hydro 做法）。

    x2 默认域 [0,1]，台阶在 y0 = 域中点；x1 方向均匀。左态（y<y0）高压高密。
    """
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    n2r, _, _, _ = _code_scales(run)
    y0 = 0.5 * (run._cfg.x2min + run._cfg.x2max)
    p_l = N_L * KB * 1.0e4
    p_r = N_R * KB * 8.0e3
    y = _axis(lib, run, 1)
    m = (y < y0).reshape(1, nyt, 1)
    ones = np.ones((nzt, nyt, nxt))
    for var, vl, vr in (
        (SLOT_N, N_L * n2r, N_R * n2r),
        (SLOT_VX, 0.0, 0.0),
        (SLOT_VY, 0.0, 0.0),
        (SLOT_VZ, 0.0, 0.0),
        (SLOT_P, p_l, p_r),  # 默认单位下 unit_p = 1，cgs 值即码值
    ):
        arr = np.ascontiguousarray(np.where(m, vl, vr) * ones)
        assert lib.xmf_state_pack(run._h, 0, var, arr.ctypes.data_as(_PTR)) == arr.size


def _thin2d_cfg(recon: str) -> dict:
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [8, 32, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    cfg["numerics"]["reconstruction"] = recon
    return cfg


def _iso_position(x_int, n_int, mid):
    """密度 mid 等值面位置：第一次下穿处线性内插（照抄 test_hydro）。"""
    below = np.where(n_int < mid)[0]
    assert len(below) > 0, "剖面里应存在低于 mid 的胞（台阶还在）"
    i = int(below[0])
    assert i > 0, "等值面不应贴到最左胞"
    frac = (mid - n_int[i - 1]) / (n_int[i] - n_int[i - 1])
    return float(x_int[i - 1] + frac * (x_int[i] - x_int[i - 1]))


# --- 7. 四步仍通（卡片第 6 条）：三家各走 setup → step → fields ---------------


@pytest.mark.parametrize("recon", RECONS)
def test_four_steps_all_recons(lib, recon):
    run = xmf.setup(_tube_cfg(recon))
    try:
        t = run.step()
        assert t > 0.0, f"{recon} 单步应真正推进时间"
        hi = run.fields["HI"]
        ncell = int(lib.xmf_ncell(run._h))
        assert ncell == NX
        for arr in (hi.x, hi.n, hi.v, hi.p):
            assert len(arr) == ncell
            assert np.isfinite(arr).all()
        assert np.all(hi.n > 0.0), f"{recon} 单步后密度应保持正"
        run.advance()
        assert run.t >= run.tstop, f"{recon} 应能 advance 到 tstop"
        hi2 = run.fields["HI"]
        assert np.isfinite(hi2.n).all() and np.all(hi2.n > 0.0), (
            f"{recon} 全程后 fields 仍可读"
        )
    finally:
        run.close()


# --- 1. 严优于（合同第 1 条）：plm / weno3 的 L1(n)、L1(P) 严格小于 pcm --------


def test_plm_weno3_strictly_better_than_pcm_l1(lib):
    cfg0 = _tube_cfg("pcm")
    assert cfg0["numerics"]["riemann"] == "hll"
    assert cfg0["numerics"]["rk"] == "rk1"
    for recon in ("plm", "weno3"):
        # 同一初值：除 numerics.reconstruction 外逐键相等
        same = copy.deepcopy(_tube_cfg(recon))
        same["numerics"]["reconstruction"] = "pcm"
        assert same == cfg0, f"pcm 与 {recon} 配置只允许 reconstruction 不同"

    errs = {}
    times = {}
    for recon in RECONS:
        run = _run_to_tstop(recon)
        try:
            times[recon], errs[recon] = run.t, _l1_triple(run)
        finally:
            run.close()

    for recon in RECONS:
        e = errs[recon]
        print(f"[recon] {recon:6s} t={times[recon]:.4f} "
              f"L1(n,v,P)/scale = {e[0]:.3e} {e[1]:.3e} {e[2]:.3e}")

    ep = errs["pcm"]
    for recon in ("plm", "weno3"):
        e = errs[recon]
        assert e[0] < ep[0], (
            f"{recon} L1(n)/scale={e[0]:.3e} 应严格小于 pcm {ep[0]:.3e}"
        )
        assert e[2] < ep[2], (
            f"{recon} L1(P)/scale={e[2]:.3e} 应严格小于 pcm {ep[2]:.3e}"
        )
        assert e[1] <= ep[1], (
            f"{recon} L1(v)/scale={e[1]:.3e} 不应差于 pcm {ep[1]:.3e}"
            "（合同：v 若实测略差，报主会话裁断，不要自行改容差）"
        )


# --- 2. 不再静默当 PCM（合同第 2 条）：5 步后逐格必须与 pcm 不同 --------------
#
# 第 1 步的分段常数初值下 plm ≡ pcm（van Leer 斜率处处 0）、weno3 也只在
# ε=1e-6 权重舍入层面有 ~1e-10 级微差——这是应有性质。故差异断言取第 5 步。


@pytest.mark.parametrize("recon", ["plm", "weno3"])
def test_plm_weno3_not_silently_pcm_1d(lib, recon):
    def blocks_after(recon_name: str, steps: int):
        run = xmf.setup(_tube_cfg(recon_name))
        try:
            for _ in range(steps):
                run.step()
            assert run.t > 0.0
            return tuple(_unpack(lib, run, var).copy()
                         for var in (SLOT_N, SLOT_VX, SLOT_P))
        finally:
            run.close()

    steps = 5
    base = blocks_after("pcm", steps)
    cur = blocks_after(recon, steps)
    for name, b, c in zip(("n", "vx", "P"), base, cur):
        dmax = float(np.max(np.abs(c - b)))
        assert dmax > 0.0, (
            f"{recon} 跑 {steps} 步后 {name} 整块与 pcm 逐位相同——疑似静默当 PCM"
        )
        print(f"[recon] {recon} vs pcm, {steps} steps: max|Δ{name}|={dmax:.3e}（码单位）")


# --- 3. 闲置速度（合同第 3 条）：1D 的 vy / vz 整块（含 ghost）精确为 0 --------


@pytest.mark.parametrize("recon", ["plm", "weno3"])
def test_idle_vy_vz_zero_full_block(lib, recon):
    run = _run_to_tstop(recon)
    try:
        ncells = lib.xmf_state_ncells(run._h)
        for var, name in ((SLOT_VY, "vy"), (SLOT_VZ, "vz")):
            buf = _unpack(lib, run, var)
            assert buf.shape == (ncells,), "unpack 块长应含全部 ghost"
            assert np.all(buf == 0.0), (
                f"{recon} 1D 闲置分量 {name} 在整块（含 ghost，ncells={ncells}）"
                f"应严格为 0，实测 max|{name}|={float(np.max(np.abs(buf))):.3e}"
            )
    finally:
        run.close()


# --- 4a. 薄 2D 走到 x2 重建（合同第 4 条）：vy 被激发、变化只来自 x2 ----------


@pytest.mark.parametrize("recon", ["plm", "weno3"])
def test_thin2d_x2_reconstruction_excited(lib, recon):
    """薄 2D（8×32）沿 x2 放 Sod 台阶，10 步：证据链照抄 test_hydro 的
    薄 2D 测法，外加本刀专属的 vy 量级（实测 plm 0.843 / weno3 0.849 码单位，
    与 builder 自证 ≈0.84 一致）。"""
    run = xmf.setup(_thin2d_cfg(recon))
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        _, r2n, _, _ = _code_scales(run)
        assert ny >= 4, f"薄 2D 要求 n[1]>=4，实测 {ny}"
        _pack_x2_step(lib, run)
        n_before = _unpack(lib, run, SLOT_N).copy()
        for _ in range(10):
            run.step()
        assert run.t > 0.0

        n_after, _ = _unpack3(lib, run, SLOT_N)
        vx, _ = _unpack3(lib, run, SLOT_VX)
        vy, _ = _unpack3(lib, run, SLOT_VY)
        inner = (ng, slice(ng, ng + ny), slice(ng, ng + nx))
        vy_i = vy[inner]
        vx_i = vx[inner]
        n_b = n_before.reshape((nzt, nyt, nxt))[inner]
        n_a = n_after[inner]

        # (a) vy 真的被 x2 sweep 激发（vy 只能由 x2 sweep 产生）
        vmax = float(np.max(np.abs(vy_i)))
        assert vmax > 5.0e-2, (
            f"{recon} 薄 2D 沿 x2 的台阶 10 步后 vy 应明显非零，实测 max|vy|="
            f"{vmax:.3e}（码单位）"
        )
        # (b) 沿 x1 各列一致：每一 j 行内 i 方向偏差为 0（变化只来自 x2）
        dev = float(np.max(np.abs(vy_i - vy_i.mean(axis=1, keepdims=True))))
        assert dev <= 1.0e-12, f"vy 应沿 x1 逐列一致（实测最大偏差 {dev:.3e}）"
        # (c) 切向 vx 不被无中生有
        assert float(np.max(np.abs(vx_i))) <= 1.0e-12, "x2 台阶不得生出 vx"
        # (d) 状态确实在演化（x2 剖面不再是初始台阶的平移复制）
        dl1 = float(np.mean(np.abs(n_a - n_b))) * r2n / N_L
        assert dl1 > 1.0e-3, f"x2 sweep 应真正改变状态，L1(n) 变化仅 {dl1:.3e}"
        print(f"[recon] {recon} thin-2D 10 steps: max|vy|={vmax:.3e}, dl1(n)={dl1:.3e}")
    finally:
        run.close()


# --- 4b. 薄 2D 的 x2 sweep 不再静默当 PCM（合同第 4 条的差分钉法） ------------


@pytest.mark.parametrize("recon", ["plm", "weno3"])
def test_thin2d_x2_not_silently_pcm(lib, recon):
    """同一 x2 台阶初值，5 步后整块密度与 pcm 不同。该初值沿 x1 均匀、全部
    动力学在 x2：若 x2 sweep 静默当 PCM，本测逐位相同 ⇒ 红。"""
    def blocks_after(recon_name: str, steps: int):
        run = xmf.setup(_thin2d_cfg(recon_name))
        try:
            _pack_x2_step(lib, run)
            for _ in range(steps):
                run.step()
            assert run.t > 0.0
            return (_unpack(lib, run, SLOT_N).copy(),
                    _unpack(lib, run, SLOT_VY).copy())
        finally:
            run.close()

    steps = 5
    nb, vyb = blocks_after("pcm", steps)
    nc, vyc = blocks_after(recon, steps)
    dn = float(np.max(np.abs(nc - nb)))
    dvy = float(np.max(np.abs(vyc - vyb)))
    assert dn > 0.0, (
        f"{recon} 薄 2D 跑 {steps} 步后密度整块与 pcm 逐位相同——"
        "x2 sweep 疑似静默当 PCM"
    )
    assert dvy > 0.0, f"{recon} 薄 2D 跑 {steps} 步后 vy 整块与 pcm 逐位相同"
    print(f"[recon] {recon} thin-2D vs pcm, {steps} steps: "
          f"max|Δn|={dn:.3e} max|Δvy|={dvy:.3e}（码单位）")


# --- 5. 接触不漂移（合同第 5 条）：hllc + plm / weno3 -------------------------


@pytest.mark.parametrize("recon", ["plm", "weno3"])
def test_contact_hllc_step_does_not_drift(lib, recon):
    """u=0、∇p=0 的密度台阶，hllc + 重建，20 步：等值面零漂移。

    静止接触（SM=0）对 HLLC 是离散稳态；台阶面上 plm 的 van Leer 斜率为 0、
    weno3 权重退化回单侧格心，重建不改变面态，稳态保持。容差口径照抄
    test_hydro 的 hllc 接触测法（实测漂移 / ΔL1 / max|vx| 均恰为 0）。
    """
    cfg = xmf.shock_tube()
    cfg["numerics"]["riemann"] = "hllc"
    cfg["numerics"]["reconstruction"] = recon
    cfg["ic"] = {
        "kind": "riemann",
        "x0": X0,
        "left": {"n": N_L, "v": 0.0, "p": P0},
        "right": {"n": N_R, "v": 0.0, "p": P0},
    }
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        n2r, r2n, _, _ = _code_scales(run)
        dx1 = (run._cfg.x1max - run._cfg.x1min) / nx

        # IC 自检：压强逐位均匀（∇p=0）
        p0, _ = _unpack3(lib, run, SLOT_P)
        assert float(np.max(p0) - np.min(p0)) == 0.0, "接触间断初值压强必须逐位均匀"

        n_before, _ = _unpack3(lib, run, SLOT_N)
        for _ in range(20):
            run.step()
        assert run.t > 0.0, "step 应真正推进时间"
        n_after, _ = _unpack3(lib, run, SLOT_N)
        vx, _ = _unpack3(lib, run, SLOT_VX)

        x = _axis(lib, run, 0)[ng : ng + nx]
        row = (ng, slice(ng, ng + ny), slice(ng, ng + nx))
        nb = n_before[row][0] if ny == 1 else n_before[row].mean(axis=0)
        na = n_after[row][0] if ny == 1 else n_after[row].mean(axis=0)
        mid = 0.5 * (N_L * n2r + N_R * n2r)  # 码单位的台阶中值
        iso0 = _iso_position(x, nb, mid)
        iso1 = _iso_position(x, na, mid)

        assert abs(iso0 - X0) <= 0.25 * dx1, f"初始等值面应就在 x0={X0} 处（得 {iso0}）"
        assert abs(iso1 - iso0) <= 0.25 * dx1, (
            f"hllc+{recon} 静止接触 20 步后台阶中心漂移 {iso1 - iso0:+.3e}，"
            "超过 1/4 胞元"
        )
        # 稳态：密度几乎逐位不变（实测恰为 0）
        err = float(np.mean(np.abs(na - nb))) * r2n / N_L
        assert err <= 1.0e-9, f"hllc+{recon} 静止接触应是离散稳态，L1(n) 变化 {err:.3e}"
        # 速度保持 0（重建不得给静止接触造出假流动）
        v_int = vx[(ng, slice(ng, ng + ny), slice(ng, ng + nx))]
        assert float(np.max(np.abs(v_int))) <= 1.0e-12, "u=0 初值不得生出流动"
        print(f"[recon] hllc+{recon} contact: drift={iso1 - iso0:+.3e}, "
              f"dl1(n)={err:.3e}, max|vx|={float(np.max(np.abs(v_int))):.3e}")
    finally:
        run.close()


# --- 6. setup 拒绝 / 接受（合同第 6 条，卡片「交什么」第 4 条） ---------------


@pytest.mark.parametrize("bad", [-1, 3, 99])
def test_setup_rejects_out_of_range_recon_at_c(lib, bad):
    """recon 整数越界 → C setup 返回 NULL。前端表只认三个名字，越界整数
    只能绕过前端直接打 C 契约（同 test_abi 坏 schema / 坏 n[0] 写法）。"""
    blob = problem_to_c(normalize(xmf.shock_tube()))
    blob.recon = bad
    handle = _setup_c(lib, blob)
    assert handle is None or handle == 0, f"recon={bad} 必须被 xmf_setup 拒绝"


def test_setup_rejects_unknown_recon_name():
    """未实现的重建档（如 ppm）在前端就被拒：不允许静默当任何已实现档。"""
    cfg = xmf.shock_tube()
    cfg["numerics"]["reconstruction"] = "ppm"
    with pytest.raises(ValueError, match="reconstruction"):
        xmf.setup(cfg)


@pytest.mark.parametrize("recon", ["plm", "weno3"])
def test_plm_weno3_require_two_ghost_layers(lib, recon):
    """plm / weno3 且 nghost=1（<2）→ setup 失败。前端能过（同 test_rk 的
    rk4 写法），C setup 拒：Python 抛 RuntimeError('xmf_setup failed')。"""
    cfg = _tube_cfg(recon)
    cfg["grid"]["nghost"] = 1
    p = normalize(cfg)
    assert p.ng == 1 and p.numerics.reconstruction == recon, "前端应放行到 C"
    blob = problem_to_c(p)
    handle = _setup_c(lib, blob)
    assert handle is None or handle == 0, f"{recon} + ng=1 必须被 xmf_setup 拒绝"
    with pytest.raises(RuntimeError, match="xmf_setup failed"):
        xmf.setup(cfg)


def test_pcm_accepts_single_ghost_layer():
    """pcm + nghost=1 不拒：PCM 只需 1 层（实测可正常 setup 并 step）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["nghost"] = 1
    run = xmf.setup(cfg)
    try:
        t = run.step()
        assert t > 0.0, "pcm + ng=1 应能正常推进"
        hi = run.fields["HI"]
        assert np.isfinite(hi.n).all() and np.all(hi.n > 0.0)
    finally:
        run.close()
