"""Figures for the HD evaluation example. All written under output/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kernel import OUT

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.grid": False,
        "image.origin": "lower",
        "image.interpolation": "nearest",
    }
)


def _save(fig, path: Path) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_lineout(rec: dict, png: Path) -> Path:
    n, v, p = rec["num"]
    ne, ve, pe = rec["exact"]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.6), sharex=True)
    fig.suptitle(
        f"{rec['case']}  ·  {rec['riemann']} / pcm / rk1  ·  t={rec['t']:.3f}  ·  "
        f"L1(n,v,P)/scale = {rec['L1_n']:.2e}, {rec['L1_v']:.2e}, {rec['L1_p']:.2e}",
        fontsize=10,
    )
    axes[0].plot(rec["coord"], n, color="C0", lw=1.6, label="numerical")
    axes[0].plot(rec["coord"], ne, color="k", lw=1.0, ls="--", label="Toro exact")
    axes[0].set_ylabel(r"$n$")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, loc="best")
    axes[1].plot(rec["coord"], np.asarray(v) / 1e5, color="C1", lw=1.6)
    axes[1].plot(rec["coord"], np.asarray(ve) / 1e5, color="k", lw=1.0, ls="--")
    axes[1].set_ylabel(r"$v_n$ (km s$^{-1}$)")
    axes[2].plot(rec["coord"], p, color="C2", lw=1.6)
    axes[2].plot(rec["coord"], pe, color="k", lw=1.0, ls="--")
    axes[2].set_ylabel(r"$P$")
    axes[2].set_xlabel(rec["xlab"] + "  (code units)")
    for ax in axes:
        ax.axvline(rec["x0"], color="0.75", lw=0.8, ls="--")
        ax.grid(True, alpha=0.25)
    return _save(fig, png)


def plot_solver_overlay(recs: list[dict], png: Path) -> Path:
    """Three Riemann solvers on the same 1D Toro problem."""
    rec0 = recs[0]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.8), sharex=True)
    fig.suptitle(
        f"{rec0['case']}  ·  hll / hllc / tvdlf overlay vs Toro  ·  t={rec0['t']:.3f}",
        fontsize=10,
    )
    colors = {"hll": "C0", "hllc": "C1", "tvdlf": "C2"}
    ne, ve, pe = rec0["exact"]
    axes[0].plot(rec0["coord"], ne, "k--", lw=1.1, label="Toro exact")
    axes[1].plot(rec0["coord"], np.asarray(ve) / 1e5, "k--", lw=1.1)
    axes[2].plot(rec0["coord"], pe, "k--", lw=1.1)
    for rec in recs:
        c = colors[rec["riemann"]]
        n, v, p = rec["num"]
        lab = (
            f"{rec['riemann']}  "
            f"L1=({rec['L1_n']:.2e},{rec['L1_v']:.2e},{rec['L1_p']:.2e})"
        )
        axes[0].plot(rec["coord"], n, color=c, lw=1.3, label=lab)
        axes[1].plot(rec["coord"], np.asarray(v) / 1e5, color=c, lw=1.3)
        axes[2].plot(rec["coord"], p, color=c, lw=1.3)
    axes[0].set_ylabel(r"$n$")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    axes[1].set_ylabel(r"$v_n$ (km s$^{-1}$)")
    axes[2].set_ylabel(r"$P$")
    axes[2].set_xlabel(rec0["xlab"] + "  (code units)")
    for ax in axes:
        ax.axvline(rec0["x0"], color="0.75", lw=0.8, ls="--")
        ax.grid(True, alpha=0.25)
    return _save(fig, png)


def plot_extruded_2d(rec: dict, stem: Path) -> list[Path]:
    """2D maps for an extruded Riemann problem, plus the 1D lineout."""
    paths = [plot_lineout(rec, stem.parent / f"{stem.name}_lineout.png")]
    x, y = rec["x"], rec["y"]
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    fields = [
        (rec["field2d"], r"$n$", "n", "viridis"),
        (rec["vx2d"] / 1e5, r"$v_x$ (km s$^{-1}$)", "vx", "coolwarm"),
        (rec["vy2d"] / 1e5, r"$v_y$ (km s$^{-1}$)", "vy", "coolwarm"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    fig.suptitle(
        f"{rec['case']}  ·  {rec['riemann']}  ·  t={rec['t']:.3f}  ·  "
        f"L1(n,v,P)/scale = {rec['L1_n']:.2e}, {rec['L1_v']:.2e}, {rec['L1_p']:.2e}",
        fontsize=10,
    )
    for ax, (arr, lab, _key, cmap) in zip(axes, fields):
        im = ax.imshow(arr, extent=extent, cmap=cmap, aspect="auto")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
    paths.append(_save(fig, stem.parent / f"{stem.name}_2d.png"))
    return paths


def plot_classic_panel(recs: list[dict], png: Path, qty: str = "rho") -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8))
    title = recs[0]["title"]
    fig.suptitle(f"{title}  ·  ρ  ·  pcm / rk1  ·  t={recs[0]['t']:.3f}", fontsize=10)
    for ax, rec in zip(axes, recs):
        x, y = rec["x"], rec["y"]
        extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
        im = ax.imshow(rec[qty], extent=extent, cmap="inferno", aspect="equal")
        extra = ""
        if rec["case"] == "C-SEDOV" and np.isfinite(rec.get("circularity", np.nan)):
            extra = f"\ncirc={rec['circularity']:.3f}"
        if rec["case"] == "C-KH":
            extra = f"\nKE_y={rec['ke_y']:.3e}"
        ax.set_title(f"{rec['riemann']}{extra}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, png)


def plot_classic_one(rec: dict, png: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    fig.suptitle(
        f"{rec['case']}  ·  {rec['riemann']}  ·  t={rec['t']:.3f}  ·  {rec['title']}",
        fontsize=10,
    )
    x, y = rec["x"], rec["y"]
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    panels = [
        (rec["rho"], r"$\rho$", "inferno"),
        (rec["p"], r"$P$", "magma"),
        (np.hypot(rec["vx"], rec["vy"]), r"$|v|$", "viridis"),
    ]
    for ax, (arr, lab, cmap) in zip(axes, panels):
        im = ax.imshow(arr, extent=extent, cmap=cmap, aspect="equal")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
    return _save(fig, png)


def plot_cyl_rz(rec: dict, png: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4))
    fig.suptitle(
        f"{rec['case']}  ·  {rec['riemann']}  ·  cylindrical (R,z)  ·  "
        f"t={rec['t']:.3f}  ·  circ={rec.get('circularity', float('nan')):.3f}",
        fontsize=10,
    )
    x, y = rec["x"], rec["y"]
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    panels = [
        (rec["rho"], r"$\rho$", "inferno"),
        (rec["p"], r"$P$", "magma"),
        (np.hypot(rec["vx"], rec["vy"]), r"$|v_{Rz}|$", "viridis"),
    ]
    for ax, (arr, lab, cmap) in zip(axes, panels):
        im = ax.imshow(arr, extent=extent, cmap=cmap, aspect="equal")
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$z$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
    return _save(fig, png)


def plot_sph_rth(rec: dict, png: Path) -> Path:
    """Meridional view: x = r sinθ, z = r cosθ, plus a (r,θ) map."""
    r, th = rec["x"], rec["y"]
    R = np.outer(np.sin(th), r)
    Z = np.outer(np.cos(th), r)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.6))
    fig.suptitle(
        f"{rec['case']}  ·  {rec['riemann']}  ·  spherical (r,θ)  ·  "
        f"t={rec['t']:.3f}  ·  θ-dev={rec.get('theta_dev', float('nan')):.2e}",
        fontsize=10,
    )
    fields = [
        (rec["rho"], r"$\rho$", "inferno"),
        (rec["p"], r"$P$", "magma"),
        (rec["vx"], r"$v_r$", "coolwarm"),
    ]
    for ax, (arr, lab, cmap) in zip(axes, fields):
        im = ax.pcolormesh(R, Z, arr, cmap=cmap, shading="nearest")
        ax.set_aspect("equal")
        ax.set_xlabel(r"$r\sin\theta$")
        ax.set_ylabel(r"$r\cos\theta$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
    return _save(fig, png)


def plot_sph_lineout(rec: dict, png: Path) -> Path:
    r = rec["x"]
    rho_m = rec["rho"].mean(axis=0)
    vr_m = rec["vx"].mean(axis=0)
    p_m = rec["p"].mean(axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
    fig.suptitle(
        f"{rec['case']}  ·  θ-mean radial profiles  ·  t={rec['t']:.3f}",
        fontsize=10,
    )
    axes[0].plot(r, rho_m, color="C0", lw=1.5)
    axes[0].set_ylabel(r"$\langle\rho\rangle_\theta$")
    axes[1].plot(r, vr_m, color="C1", lw=1.5)
    axes[1].set_ylabel(r"$\langle v_r\rangle_\theta$")
    axes[2].plot(r, p_m, color="C2", lw=1.5)
    axes[2].set_ylabel(r"$\langle P\rangle_\theta$")
    axes[2].set_xlabel(r"$r$")
    for ax in axes:
        ax.axvline(rec.get("r_disc", 0.45), color="0.75", lw=0.8, ls="--")
        ax.grid(True, alpha=0.25)
    return _save(fig, png)


def plot_sedov_radial(recs: list[dict], png: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for rec, c in zip(recs, ("C0", "C1", "C2")):
        x, y = rec["x"], rec["y"]
        xx, yy = np.meshgrid(x, y, indexing="xy")
        r = np.sqrt(xx * xx + yy * yy).ravel()
        rho = rec["rho"].ravel()
        bins = np.linspace(0.0, 0.55, 64)
        idx = np.digitize(r, bins)
        prof = [float(np.mean(rho[idx == i])) if np.any(idx == i) else np.nan for i in range(1, len(bins))]
        rc = 0.5 * (bins[1:] + bins[:-1])
        ax.plot(rc, prof, color=c, lw=1.4, label=f"{rec['riemann']}  R={rec['shock_R']:.3f}")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\langle\rho\rangle_\theta$")
    ax.set_title("Sedov  ·  angle-averaged density")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    return _save(fig, png)
