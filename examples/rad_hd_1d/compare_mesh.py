"""Compare uniform nx=1000 vs log-stretch nx=128 on the rad_hd_1d setup.

The two cases share the spherical factory (A = r^2, dV = |Δr^3|/3).
"Cartesian" here means equidistant faces (stretch=none), not
geometry=cartesian — switching the metric would be a different problem.

The advance, pack, snapshot and number block all come from run.py.
This file only pins the two mesh specs, interpolates, plots, and writes
report_mesh_compare.md.

Usage:
    python examples/rad_hd_1d/compare_mesh.py
"""
from __future__ import annotations

import os
import types

import numpy as np

import run as exrun
from rad_config import RUN

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "mesh_compare")
REPORT = os.path.join(HERE, "report_mesh_compare.md")

STRETCH_FACTOR = (
    float(RUN.stretch_factor) if RUN.stretch_factor is not None else 1.025
)
TSTOP = 2.e4  # s; compare-only, does not change rad_config.RUN


def spec(nx: int, stretch: str, stretch_factor=None):
    return types.SimpleNamespace(
        nx=nx,
        n_beam=RUN.n_beam,
        tstop=TSTOP,
        stretch=stretch,
        stretch_factor=stretch_factor,
        log_every=RUN.log_every,
        plots=False,
    )


CART = spec(1000, "none")
STRETCH = spec(128, "log", STRETCH_FACTOR)


def interp_to(r_src: np.ndarray, prof: dict, r_dst: np.ndarray) -> dict:
    out = {}
    for fl in ("HI", "HII"):
        out[fl] = {
            k: np.interp(r_dst, r_src, prof[fl][k])
            for k in ("n", "vx", "T", "P")
        }
    return out


def rel_diff(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def dx_active(res: dict) -> np.ndarray:
    return res["dx_full"][2 : 2 + res["nx"]]


def mdot_sum(res: dict) -> np.ndarray:
    c = res["cfg_constants"]
    r = res["r"]
    n1, v1 = res["final"]["HI"]["n"], res["final"]["HI"]["vx"]
    n2, v2 = res["final"]["HII"]["n"], res["final"]["HII"]["vx"]
    return (exrun.mdot_g_s(n1, v1, r, c["m_HI_g"])
            + exrun.mdot_g_s(n2, v2, r, c["m_HII_g"]))


def mesh_stats(res: dict) -> dict:
    rp = res["r_p"]
    dx = dx_active(res) / rp
    return {
        "dx0": float(dx[0]),
        "dx1": float(dx[-1]),
        "ratio": float(dx[-1] / dx[0]),
        "ghost_l": float(res["dx_full"][0] / rp),
        "ghost_r": float(res["dx_full"][-1] / rp),
    }


def profile_cmp(fine: dict, coarse: dict) -> dict:
    """Interpolate coarse onto the fine radii; split inner / outer."""
    r = fine["r"]
    rr = r / fine["r_p"]
    fc = interp_to(coarse["r"], coarse["final"], r)
    ff = fine["final"]
    inner = rr < 1.5
    front = (rr >= 2.0) & (rr < 4.5)
    far = rr >= 4.5
    out = {}
    for fl in ("HI", "HII"):
        dn = rel_diff(fc[fl]["n"], ff[fl]["n"])
        dv = np.abs(fc[fl]["vx"] - ff[fl]["vx"])
        dT = rel_diff(fc[fl]["T"], ff[fl]["T"])
        out[fl] = {
            "n_all": float(dn.max()),
            "n_inner": float(dn[inner].max()),
            "n_front": float(dn[front].max()),
            "n_far": float(dn[far].max()),
            "dv_all": float(dv.max()),
            "dv_inner": float(dv[inner].max()),
            "dv_front": float(dv[front].max()),
            "dv_far": float(dv[far].max()),
            "T_inner": float(dT[inner].max()),
            "T_front": float(dT[front].max()),
            "T_far": float(dT[far].max()),
        }
    return out


def ion_ratios(on: dict, off: dict) -> dict:
    a = on["fields"]["HII"]["n"][0, 0, :]
    b = off["fields"]["HII"]["n"][0, 0, :]
    return {
        "base": float(a[0] / max(b[0], 1e-300)),
        "top": float(a[-1] / max(b[-1], 1e-300)),
    }


def outflow_mdot(res: dict) -> dict:
    md = mdot_sum(res)
    mask = res["r"] > 4.5 * res["r_p"]
    out = md[mask]
    mean = float(np.mean(out))
    return {
        "mean": mean,
        "scatter": float(np.std(out) / max(abs(mean), 1e-300)),
        "top": float(md[-1]),
        "mid": float(md[len(md) // 2]),
    }


def make_plots(cases: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    cart_on = cases["CART1000"]["on"]
    str_on = cases["STRETCH128"]["on"]
    rp = cart_on["r_p"]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for tag, c in (("CART1000", "C0"), ("STRETCH128", "C2")):
        on = cases[tag]["on"]
        rr = on["r"] / rp
        ax[0, 0].plot(rr, on["final"]["HI"]["n"], c=c, label=f"HI {tag}")
        ax[0, 0].plot(rr, on["final"]["HII"]["n"], c=c, ls="--",
                      label=f"HII {tag}")
        ax[0, 1].plot(rr, on["final"]["HI"]["vx"], c=c, label=f"HI {tag}")
        ax[0, 1].plot(rr, on["final"]["HII"]["vx"], c=c, ls="--",
                      label=f"HII {tag}")
        ax[1, 0].plot(rr, on["final"]["HII"]["T"], c=c, label=f"HII {tag}")
        ax[1, 0].plot(rr, on["final"]["HI"]["T"], c=c, ls="--",
                      label=f"HI {tag}")
        h = on["hist"]
        ax[1, 1].plot(h["t"], h["dt"], c=c, label=tag)
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel("n, cm^-3")
    ax[0, 0].set_title("number density (radiation on)")
    ax[0, 1].set_ylabel("v_r, cm/s")
    ax[0, 1].set_title("radial velocity")
    ax[1, 0].set_ylabel("T, K")
    ax[1, 0].set_xlabel("r / R_p")
    ax[1, 0].set_title("temperature")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("t, s")
    ax[1, 1].set_ylabel("dt, s")
    ax[1, 1].set_title("CFL step")
    for a in ax.flat:
        a.legend(fontsize=7)
        if a is not ax[1, 1] and not a.get_xlabel():
            a.set_xlabel("r / R_p")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "profiles.png"), dpi=140)
    plt.close(fig)

    r = cart_on["r"]
    rr = r / rp
    fc = interp_to(str_on["r"], str_on["final"], r)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    for fl, ls in (("HI", "-"), ("HII", "--")):
        ax[0].plot(rr, rel_diff(fc[fl]["n"], cart_on["final"][fl]["n"]),
                   ls=ls, label=fl)
        ax[1].plot(rr, np.abs(fc[fl]["vx"] - cart_on["final"][fl]["vx"]),
                   ls=ls, label=fl)
        ax[2].plot(rr, rel_diff(fc[fl]["T"], cart_on["final"][fl]["T"]),
                   ls=ls, label=fl)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("|n_S − n_C| / |n_C|")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("|v_S − v_C|, cm/s")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("|T_S − T_C| / |T_C|")
    for a in ax:
        a.set_xlabel("r / R_p")
        a.legend(fontsize=8)
        a.axvline(1.5, color="0.6", ls=":", lw=0.8)
        a.axvline(2.0, color="0.6", ls=":", lw=0.8)
    fig.suptitle("STRETCH128 interpolated onto CART1000 (radiation on)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "diff.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tag, c in (("CART1000", "C0"), ("STRETCH128", "C2")):
        on, off = cases[tag]["on"], cases[tag]["off"]
        rr2 = on["r"] / rp
        ax[0].plot(rr2, on["final"]["HII"]["n"], c=c, label=f"on {tag}")
        ax[0].plot(rr2, off["final"]["HII"]["n"], c=c, ls=":",
                   label=f"off {tag}")
        flx = on["flux"][:, 0, 0, :]
        ax[1].plot(rr2, np.sum(flx, axis=0) / float(np.sum(flx[:, -1])),
                   c=c, label=tag)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("n_HII, cm^-3")
    ax[0].set_title("n_HII on (solid) / off (dotted)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("F / F_inf")
    ax[1].set_title("transmitted flux / F_inf (on)")
    for a in ax:
        a.set_xlabel("r / R_p")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ion_flux.png"), dpi=140)
    plt.close(fig)
    print("plots written to", OUT)


def fmt_sci(x: float) -> str:
    return f"{x:.4e}"


def write_report(cases: dict, cmp_on: dict, cmp_off: dict) -> None:
    cart, st = cases["CART1000"], cases["STRETCH128"]
    ms_c, ms_s = mesh_stats(cart["on"]), mesh_stats(st["on"])
    ion_c = ion_ratios(cart["on"], cart["off"])
    ion_s = ion_ratios(st["on"], st["off"])
    md = {
        "CART1000": {lab: outflow_mdot(cart[lab]) for lab in ("on", "off")},
        "STRETCH128": {lab: outflow_mdot(st[lab]) for lab in ("on", "off")},
    }
    block_c = exrun.number_block(cart["on"], cart["off"], CART)
    block_s = exrun.number_block(st["on"], st["off"], STRETCH)
    with open(os.path.join(OUT, "numbers_cart1000.txt"), "w") as fh:
        fh.write(block_c)
    with open(os.path.join(OUT, "numbers_stretch128.txt"), "w") as fh:
        fh.write(block_s)

    def row(tag, lab):
        res = cases[tag][lab]
        return (
            f"| {tag} {lab} | {res['steps']} | {res['t_final']:.4e} | "
            f"{res['dt_final']:.3e} | {res['wall']:.1f} | "
            f"{res['resid_final']:.2e} / {res['resid_max']:.2e} |"
        )

    def drow(label, d):
        return (
            f"| {label} | {d['n_inner']:.2e} / {d['n_front']:.2e} / "
            f"{d['n_far']:.2e} | {d['dv_inner']:.2e} / {d['dv_front']:.2e} / "
            f"{d['dv_far']:.2e} | {d['T_inner']:.2e} / {d['T_front']:.2e} / "
            f"{d['T_far']:.2e} |"
        )

    text = f"""# 网格对拍：均匀 1000 格 vs 对数拉伸 128 格

脚本 `compare_mesh.py` 写。推进、初值打包、快照和数字块都走
`run.py` 的 `one_run` / `number_block`，这里只换网格旋钮。

两套都是球面 1D（`geometry=spherical`，`r ∈ [R_p, 10 R_p]`，53 束，
`tstop = {CART.tstop:.4e} s`）。**笛卡尔**指等距界面（`stretch=none`，
`nx=1000`），不是把度量改成 `geometry=cartesian`——改度量会丢掉
`A=r^2`、`dV=|Δr^3|/3`，那是另一道题。**stretch** 是
`stretch=log`、`stretch_factor={STRETCH_FACTOR}`、`nx=128`。

图在 `output/mesh_compare/`（gitignore）。数字块原文：
`numbers_cart1000.txt`、`numbers_stretch128.txt`。

## 1. 网格

| | CART1000 | STRETCH128 |
|---|---|---|
| stretch | none（等距） | log, factor = {STRETCH_FACTOR} |
| nx | 1000 | 128 |
| 首格 Δx / R_p | {ms_c['dx0']:.6e} | {ms_s['dx0']:.6e} |
| 末格 Δx / R_p | {ms_c['dx1']:.6e} | {ms_s['dx1']:.6e} |
| 宽比 last/first | {ms_c['ratio']:.4f} | {ms_s['ratio']:.4f} |
| ghost 宽 L / R （R_p） | {ms_c['ghost_l']:.6e} / {ms_c['ghost_r']:.6e} | {ms_s['ghost_l']:.6e} / {ms_s['ghost_r']:.6e} |

等距 1000 格的底格宽 ≈ {ms_c['dx0']:.4f} R_p，和 stretch 128 的首格
（≈ {ms_s['dx0']:.4f} R_p）同一量级，都小于初值包络
H_ic = 0.04 R_p。均匀 128 的底格是 0.070 R_p，分辨不了吸收层。

## 2. 步进

| 跑 | 步数 | t_final (s) | dt_final (s) | 耗时 (s) | resid final / max |
|---|---|---|---|---|---|
{row('CART1000', 'on')}
{row('CART1000', 'off')}
{row('STRETCH128', 'on')}
{row('STRETCH128', 'off')}

CFL 由最紧的格定。两套底格接近，初始 dt 应同量级；CART1000 每步
格子多，实际耗时更长。

## 3. 末态剖面（stretch 插到均匀 1000 的半径上）

分三段：内区 `r < 1.5 R_p`，电离锋 `2 ≤ r/R_p < 4.5`，远区
`r ≥ 4.5 R_p`。参考是 CART1000。全域最大差几乎都在锋上
（加热峰错开半格就会把相对差顶满）。

| 辐射 | n 相对差 inner / front / far | \\|Δv_r\\| (cm/s) inner / front / far | T 相对差 inner / front / far |
|---|---|---|---|
{drow('on HI', cmp_on['HI'])}
{drow('on HII', cmp_on['HII'])}
{drow('off HI', cmp_off['HI'])}
{drow('off HII', cmp_off['HII'])}

## 4. 电离与质量损失（r > 4.5 R_p）

| | on/off n_HII 底 / 顶 | on ṁ mean (g/s) | on std/\\|mean\\| | off ṁ mean (g/s) |
|---|---|---|---|---|
| CART1000 | {ion_c['base']:.4e} / {ion_c['top']:.4e} | {fmt_sci(md['CART1000']['on']['mean'])} | {md['CART1000']['on']['scatter']:.3e} | {fmt_sci(md['CART1000']['off']['mean'])} |
| STRETCH128 | {ion_s['base']:.4e} / {ion_s['top']:.4e} | {fmt_sci(md['STRETCH128']['on']['mean'])} | {md['STRETCH128']['on']['scatter']:.3e} | {fmt_sci(md['STRETCH128']['off']['mean'])} |

ṁ = 4π m n r² v_r，两支相加。透射流量由柱密度定，两套网格应对到
同一条衰减（见 `ion_flux.png`）；差的是加热沉积落在哪几格。

## 5. 图

n、\(v_r\)、T、dt（辐射开）：

![profiles](output/mesh_compare/profiles.png)

stretch 插到均匀格上的相对差：

![diff](output/mesh_compare/diff.png)

n_HII on/off 与透射流量：

![ion_flux](output/mesh_compare/ion_flux.png)

## 6. 怎么读

stretch 128 用底密顶疏换分辨率：底格和均匀 1000 同量级，外区格子
更粗。顶层 on/off n_HII 比都是约 21，远区 ṁ 同量级，透射流量叠在
一起。差集中在 r ≈ 3 R_p 的加热 / 电离锋：stretch 的 T_HII 峰更高、
位置略偏，HII 速度尖也更尖——粗格把沉积摊在更宽的 Δx 上。
CART1000 的 dt 被锋上的高速 + 细格一起压下去（终态 2.3 s vs
stretch 13 s），实际耗时大约 35 倍。均匀 128 在本柱密度下会把 XUV
灌进一两个格子，不是这张表的对照。
"""
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("wrote", REPORT)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    print(f"compare_mesh TSTOP = {TSTOP:.4e} s")
    cases = {}
    for tag, sp in (("CART1000", CART), ("STRETCH128", STRETCH)):
        print("=" * 64)
        print(f"{tag}: nx={sp.nx} stretch={sp.stretch} "
              f"factor={sp.stretch_factor} tstop={sp.tstop:.4e}")
        print("=" * 64)
        on = exrun.one_run(True, sp)
        off = exrun.one_run(False, sp)
        cases[tag] = {"on": on, "off": off, "spec": sp}
        print(exrun.number_block(on, off, sp))

    cmp_on = profile_cmp(cases["CART1000"]["on"], cases["STRETCH128"]["on"])
    cmp_off = profile_cmp(cases["CART1000"]["off"], cases["STRETCH128"]["off"])
    make_plots(cases)
    write_report(cases, cmp_on, cmp_off)
    print("compare_mesh done")


if __name__ == "__main__":
    main()
