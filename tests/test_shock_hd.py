"""第二道门：四道空表 HD 系统测（对照卡片 docs/SESSION.md「交什么」第 5 条）。

规格：默认 riemann=hll、recon=pcm、rk=rk1、electron=off、sources=[]、γ=5/3。
四道题（都是挤出的 1D Toro 黎曼，不是四象限 2D、不是 Brio-Wu / Orszag-Tang）：

  1D-A  现 shock_tube() Sod（沿 x1，回归不回退）
  1D-B  Toro 2（123，双稀疏；左右态写进测试）
  2D-A  1D-A 沿 x1 挤出（ndim=2，n[1]>=4，x2 边界 copy；各 j 线彼此接近）
  2D-B  同一组左右态沿 x2（间断在 y，沿 x 均匀，x1 边界 copy，x2 两端 reflect；
        法向速度是 vy，剖面沿 y 取，精确解坐标换 y）

口径：L1 = mean|num-exact|/scale，
scale = (n_left, max|v_exact|, max p_exact)。每道必须通过带上界的 L1 断言，
并各写一张「数值剖面 vs Toro 精确解」三子图（n, v_n, P）到
tests/output/（已 gitignore；matplotlib Agg，不弹窗）。
图是给人看的证据，不能代替 L1 断言。给人看的评估图在 examples/hd_eval。

实测基线（builder 版核，2026-08-22，本测试首跑记录）：
  1D-A  L1(n,v,P)/scale = 3.515e-3  9.716e-3  4.029e-3（与旧占位核逐位同值，
        经重编旧核对照确认无回退；TODO「已完成」的 ~1e-3 是旧树拉伸网格
        窗口 [1,2.2] 的另一口径，不是本口径）
  1D-B  L1(n,v,P)/scale = 1.237e-2  1.913e-2  2.515e-2（双稀中部密度平台
        ρ*/ρ_L≈6e-3 在 256 胞里只占 ~6 胞，PCM 一阶抹平主导误差）
  2D-A  3.639e-3  1.036e-2  4.205e-3（各 j 线逐位一致）
  2D-B  3.649e-3  1.028e-2  4.162e-3（各 i 线逐位一致，max|vy|≈7.7e5 cm/s）
上界取 ≈2× 实测：与现 demo 同量级、留回归裕量，不放宽到失去意义。
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 测试环境不弹窗
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU, KB

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from utils.riemann import exact_hi  # noqa: E402

OUT = _REPO / "tests" / "output"

GAMMA = 5.0 / 3.0
MASS = 1.00794
UD, UV = 1.0e-12, 1.0e6  # shock_tube 默认 units（unit_p = 1：cgs 压强即码值）
R2N = UD / (MASS * AMU)  # rho 码 -> n cgs
N2R = MASS * AMU / UD    # n cgs -> rho 码

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4
_PTR = ctypes.POINTER(ctypes.c_double)

# Sod（1D-A / 2D-A / 2D-B 共用的左右态，cgs）
SOD_NL, SOD_TL = 1.0e13, 1.0e4
SOD_NR, SOD_TR = 1.25e12, 8.0e3
SOD_PL = SOD_NL * KB * SOD_TL
SOD_PR = SOD_NR * KB * SOD_TR

# Toro 2（123，双稀疏）：ρ=1（码）、v=∓2（码）、p=0.4（码）⇒ 换算成 cgs
TORO2_N = UD / (MASS * AMU)
TORO2_V = 2.0e6
TORO2_P = 0.4

# 每道带上界的 L1 上界（≈2× 实测基线，理由见文件头）
B_1D_A = dict(n=8.0e-3, v=2.0e-2, p=8.0e-3)
B_1D_B = dict(n=3.0e-2, v=4.0e-2, p=5.0e-2)
B_2D = dict(n=8.0e-3, v=2.0e-2, p=8.0e-3)


@pytest.fixture(scope="module")
def lib():
    return load()


# --- 小助手 ------------------------------------------------------------------


def _l1(num, ex, scale):
    return float(np.mean(np.abs(num - ex)) / scale)


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _unpack3(lib, run, var):
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, 0, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf.reshape((nzt, nyt, nxt))


def _axis(lib, run, d):
    ntot = lib.xmf_mesh_ncell(run._h, d)
    buf = np.empty(ntot, dtype=np.float64)
    assert lib.xmf_mesh_copy_x(run._h, d, buf.ctypes.data_as(_PTR)) == ntot
    return buf


def _gate_defaults(cfg):
    """第二道门规格：hll / pcm / rk1 / 电子关 / 空源项。"""
    assert cfg["numerics"]["riemann"] == "hll"
    assert cfg["numerics"]["reconstruction"] == "pcm"
    assert cfg["numerics"]["rk"] == "rk1"
    assert cfg["electron"]["mode"] == "off"
    assert cfg["sources"] == []
    assert cfg["gamma"] == pytest.approx(GAMMA)


def _pack_y_step(lib, run, y0, left, right):
    """整块（含 ghost）写入沿 x2 的黎曼台阶。态为 (n, v_n, p) cgs；v_n 进 vy 槽。"""
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    y = _axis(lib, run, 1)
    m = (y < y0).reshape(1, nyt, 1)
    ones = np.ones((nzt, nyt, nxt))
    for var, vl, vr in (
        (SLOT_N, left[0] * N2R, right[0] * N2R),
        (SLOT_VX, 0.0, 0.0),
        (SLOT_VY, left[1] / UV, right[1] / UV),
        (SLOT_VZ, 0.0, 0.0),
        (SLOT_P, left[2], right[2]),  # unit_p == 1
    ):
        arr = np.ascontiguousarray(np.where(m, vl, vr) * ones)
        assert lib.xmf_state_pack(run._h, 0, var, arr.ctypes.data_as(_PTR)) == arr.size


def _plot_case(case, coord, num, ex, errs, t, png, xlab, x0):
    """三子图：n（对数）、v_n（km/s）、P；数值 vs Toro 精确解；图题带题号与 L1。"""
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.4), sharex=True)
    fig.suptitle(
        f"shock-HD {case}  ·  hll / pcm / rk1 / e-off / sources=[]  ·  t = {t:.3f}",
        fontsize=11,
    )
    axes[0].plot(coord, num[0], color="C0", lw=1.6, label="numerical")
    axes[0].plot(coord, ex[0], color="k", lw=1.0, ls="--", label="exact (Toro)")
    axes[0].set_ylabel(r"$n\ \mathrm{(cm^{-3})}$")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(coord, num[1] / 1e5, color="C1", lw=1.6)
    axes[1].plot(coord, ex[1] / 1e5, color="k", lw=1.0, ls="--")
    axes[1].set_ylabel(r"$v_\mathrm{n}\ \mathrm{(km\,s^{-1})}$")
    axes[2].plot(coord, num[2], color="C2", lw=1.6)
    axes[2].plot(coord, ex[2], color="k", lw=1.0, ls="--")
    axes[2].set_ylabel(r"$P\ \mathrm{(dyn\,cm^{-2})}$")
    axes[2].set_xlabel(xlab)
    axes[2].text(
        0.02,
        0.92,
        rf"$L_1(n,v,P)/{{\rm scale}}={errs[0]:.2e},\ {errs[1]:.2e},\ {errs[2]:.2e}$",
        transform=axes[2].transAxes,
        fontsize=8,
        va="top",
    )
    for ax in axes:
        ax.axvline(x0, color="0.75", lw=0.8, ls="--")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def _assert_l1(case, errs, bounds):
    (en, ev, ep), msg = errs, f"{case} L1/scale = {errs[0]:.3e} {errs[1]:.3e} {errs[2]:.3e}"
    assert en < bounds["n"], f"{msg}：L1(n) 超上界 {bounds['n']:.1e}"
    assert ev < bounds["v"], f"{msg}：L1(v) 超上界 {bounds['v']:.1e}"
    assert ep < bounds["p"], f"{msg}：L1(P) 超上界 {bounds['p']:.1e}"


# --- 1D-A：现 shock_tube() Sod，回归不回退 ------------------------------------


def test_sys_1d_a_sod(lib):
    """1D-A：默认 shock_tube()（沿 x1），vs run.py 同口径的 Toro 精确解。"""
    cfg = xmf.shock_tube()
    _gate_defaults(cfg)
    run = xmf.setup(cfg)
    try:
        run.advance()
        hi = run.fields["HI"]
        n_ex, v_ex, p_ex = exact_hi(
            hi.x, run.t, 5.5, SOD_NL, SOD_NR, 0.0, 0.0, SOD_PL, SOD_PR, gamma=GAMMA
        )
        scales = (SOD_NL, max(float(np.max(np.abs(v_ex))), 1.0), float(np.max(p_ex)))
        errs = (
            _l1(hi.n, n_ex, scales[0]),
            _l1(hi.v, v_ex, scales[1]),
            _l1(hi.p, p_ex, scales[2]),
        )
        print(f"[1D-A] t={run.t:.4f} L1(n,v,P)/scale = {errs[0]:.3e} {errs[1]:.3e} {errs[2]:.3e}")
        _assert_l1("1D-A", errs, B_1D_A)
        png = _plot_case(
            "1D-A Sod (x1)", hi.x, (hi.n, hi.v, hi.p), (n_ex, v_ex, p_ex),
            errs, run.t, OUT / "shock_hd_1d_a.png",
            r"$x_1$  (code units)", 5.5,
        )
        assert png.exists() and png.stat().st_size > 0, "1D-A 对照图应生成且非空"
    finally:
        run.close()


# --- 1D-B：Toro 2（123，双稀疏） ----------------------------------------------


def test_sys_1d_b_toro2_double_rarefaction(lib):
    """1D-B：Toro 2（ρ=1, v=∓2, p=0.4，码单位），同一套 Toro 精确解。

    边界取 x1 两端 copy（开放）：初值 ±2 码的相背流若配 reflect 壁面，壁面会
    立即产生真实的壁面激波（向内陆传播，t=0.15 已进内区），而自由空间精确解
    不含它——那是测试设计问题，不是核的问题。copy 边界下两端均为超声出流，
    波（|ξ|≤2.82 ⇒ |x|≤0.43）不触壁，精确解在内区严格有效。
    """
    cfg = xmf.shock_tube()
    _gate_defaults(cfg)
    cfg["grid"]["x1min"], cfg["grid"]["x1max"] = -1.0, 1.0
    cfg["boundary"] = {"x1": "copy"}
    cfg["ic"] = {
        "kind": "riemann",
        "x0": 0.0,
        "left": {"n": TORO2_N, "v": -TORO2_V, "p": TORO2_P},
        "right": {"n": TORO2_N, "v": +TORO2_V, "p": TORO2_P},
    }
    run = xmf.setup(cfg)
    try:
        run.advance()
        hi = run.fields["HI"]
        n_ex, v_ex, p_ex = exact_hi(
            hi.x, run.t, 0.0, TORO2_N, TORO2_N, -TORO2_V, TORO2_V,
            TORO2_P, TORO2_P, gamma=GAMMA,
        )
        scales = (TORO2_N, max(float(np.max(np.abs(v_ex))), 1.0), float(np.max(p_ex)))
        errs = (
            _l1(hi.n, n_ex, scales[0]),
            _l1(hi.v, v_ex, scales[1]),
            _l1(hi.p, p_ex, scales[2]),
        )
        print(f"[1D-B] t={run.t:.4f} L1(n,v,P)/scale = {errs[0]:.3e} {errs[1]:.3e} {errs[2]:.3e}"
              f"  min(n_num)={hi.n.min():.3e} min(n_ex)={n_ex.min():.3e}")
        _assert_l1("1D-B", errs, B_1D_B)
        png = _plot_case(
            "1D-B Toro 2 double rarefaction (x1)", hi.x,
            (hi.n, hi.v, hi.p), (n_ex, v_ex, p_ex),
            errs, run.t, OUT / "shock_hd_1d_b.png",
            r"$x_1$  (code units)", 0.0,
        )
        assert png.exists() and png.stat().st_size > 0, "1D-B 对照图应生成且非空"
    finally:
        run.close()


# --- 2D-A：1D-A 沿 x1 挤出（状态不随 j 变） ----------------------------------


def test_sys_2d_a_extruded_sod(lib):
    """2D-A：ndim=2、n=[256,8]，x2 边界 copy。每一行 j 的 x1 剖面都过 1D 上界，
    且各 j 线彼此接近（实测逐位一致）——多一层 j 不污染 x1。"""
    cfg = xmf.shock_tube()
    _gate_defaults(cfg)
    cfg["grid"]["n"] = [256, 8, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "reflect", "x2": "copy"}
    run = xmf.setup(cfg)
    try:
        run.advance()
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        inner = (ng, slice(ng, ng + ny), slice(ng, ng + nx))
        n_cgs = _unpack3(lib, run, SLOT_N)[inner] * R2N
        v_cgs = _unpack3(lib, run, SLOT_VX)[inner] * UV
        p_cgs = _unpack3(lib, run, SLOT_P)[inner]  # unit_p == 1
        x = _axis(lib, run, 0)[ng : ng + nx]
        n_ex, v_ex, p_ex = exact_hi(
            x, run.t, 5.5, SOD_NL, SOD_NR, 0.0, 0.0, SOD_PL, SOD_PR, gamma=GAMMA
        )
        scales = (SOD_NL, max(float(np.max(np.abs(v_ex))), 1.0), float(np.max(p_ex)))

        # 每一条 j 线各自报 L1 并过同一上界
        row_errs = [
            (
                _l1(n_cgs[j], n_ex, scales[0]),
                _l1(v_cgs[j], v_ex, scales[1]),
                _l1(p_cgs[j], p_ex, scales[2]),
            )
            for j in range(ny)
        ]
        worst = tuple(max(e[k] for e in row_errs) for k in range(3))
        print(f"[2D-A] t={run.t:.4f} worst-row L1(n,v,P)/scale = "
              f"{worst[0]:.3e} {worst[1]:.3e} {worst[2]:.3e}")
        _assert_l1("2D-A", worst, B_2D)

        # 各 j 线彼此接近：状态不随 j 变（实测恰为 0，给 1e-8 相对裕量）
        for arr, scale, name in ((n_cgs, scales[0], "n"),
                                 (v_cgs, scales[1], "v"),
                                 (p_cgs, scales[2], "P")):
            dev = float(np.max(np.abs(arr - arr.mean(axis=0, keepdims=True)))) / scale
            assert dev <= 1.0e-8, f"2D-A 状态不得随 j 变：{name} 最大行偏差 {dev:.3e}"

        # 图用 j 平均剖面（各行一致，平均即代表）
        errs = tuple(
            _l1(arr.mean(axis=0), ex, sc)
            for arr, ex, sc in ((n_cgs, n_ex, scales[0]),
                                (v_cgs, v_ex, scales[1]),
                                (p_cgs, p_ex, scales[2]))
        )
        png = _plot_case(
            "2D-A extruded Sod (x1, j-mean)", x,
            (n_cgs.mean(axis=0), v_cgs.mean(axis=0), p_cgs.mean(axis=0)),
            (n_ex, v_ex, p_ex), errs, run.t, OUT / "shock_hd_2d_a.png",
            r"$x_1$  (code units)", 5.5,
        )
        assert png.exists() and png.stat().st_size > 0, "2D-A 对照图应生成且非空"
    finally:
        run.close()


# --- 2D-B：同一组左右态沿 x2（x2 sweep 真在干活） -----------------------------


def test_sys_2d_b_transverse_sod(lib):
    """2D-B：间断在 y（x2∈[1,10]，y0=5.5），沿 x 均匀；x1 边界 copy、x2 两端
    reflect（与 1D-A 同类型）。法向速度 = vy；剖面沿 y（i 平均）对同一 1D Toro
    精确解（坐标换 y）。另断言状态不随 i 变、vy 真的被激发。"""
    cfg = xmf.shock_tube()
    _gate_defaults(cfg)
    cfg["grid"]["n"] = [8, 256, 1]
    cfg["grid"]["ndim"] = 2
    cfg["grid"]["x2min"], cfg["grid"]["x2max"] = 1.0, 10.0
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    cfg["ic"]["x0"] = 0.0  # 核内 x1 向填充（全右态）随后被整块 pack 覆盖
    run = xmf.setup(cfg)
    try:
        y0 = 5.5
        _pack_y_step(lib, run, y0, (SOD_NL, 0.0, SOD_PL), (SOD_NR, 0.0, SOD_PR))
        run.advance()

        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        inner = (ng, slice(ng, ng + ny), slice(ng, ng + nx))
        n_cgs = _unpack3(lib, run, SLOT_N)[inner] * R2N
        vy_cgs = _unpack3(lib, run, SLOT_VY)[inner] * UV
        p_cgs = _unpack3(lib, run, SLOT_P)[inner]
        y = _axis(lib, run, 1)[ng : ng + ny]
        n_ex, v_ex, p_ex = exact_hi(
            y, run.t, y0, SOD_NL, SOD_NR, 0.0, 0.0, SOD_PL, SOD_PR, gamma=GAMMA
        )
        scales = (SOD_NL, max(float(np.max(np.abs(v_ex))), 1.0), float(np.max(p_ex)))

        # 法向速度真是 vy，且被 x2 sweep 激发（实测 ~7.7e5 cm/s）
        assert float(np.max(np.abs(vy_cgs))) > 1.0e5, (
            f"2D-B 的 vy 应被真实激发，实测 max|vy|={float(np.max(np.abs(vy_cgs))):.3e} cm/s"
        )

        # 沿 x 均匀：各 i 线彼此接近（实测恰为 0）
        for arr, scale, name in ((n_cgs, scales[0], "n"),
                                 (vy_cgs, scales[1], "vy"),
                                 (p_cgs, scales[2], "P")):
            dev = float(np.max(np.abs(arr - arr.mean(axis=1, keepdims=True)))) / scale
            assert dev <= 1.0e-8, f"2D-B 状态不得随 i 变：{name} 最大列偏差 {dev:.3e}"

        # i 平均剖面 vs 同一 1D Toro 精确解（坐标换 y）
        nm, vm, pm = n_cgs.mean(axis=1), vy_cgs.mean(axis=1), p_cgs.mean(axis=1)
        errs = (_l1(nm, n_ex, scales[0]), _l1(vm, v_ex, scales[1]), _l1(pm, p_ex, scales[2]))
        print(f"[2D-B] t={run.t:.4f} L1(n,vy,P)/scale = {errs[0]:.3e} {errs[1]:.3e} {errs[2]:.3e}")
        _assert_l1("2D-B", errs, B_2D)
        png = _plot_case(
            "2D-B transverse Sod (x2, i-mean)", y,
            (nm, vm, pm), (n_ex, v_ex, p_ex), errs, run.t,
            OUT / "shock_hd_2d_b.png", r"$x_2$  (code units)", y0,
        )
        assert png.exists() and png.stat().st_size > 0, "2D-B 对照图应生成且非空"
    finally:
        run.close()
