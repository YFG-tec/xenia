"""Redraw cyl_2d figures from a saved dump. No xmf step.

    python examples/cyl_2d/plot.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def load(out: Path) -> tuple[dict[str, np.ndarray], dict]:
    dump = out / "fields.npz"
    summary = out / "summary.json"
    if not dump.is_file():
        raise FileNotFoundError(f"missing {dump}; run run.py first")
    raw = np.load(dump)
    code = {key: raw[key] for key in raw.files}
    meta = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
    return code, meta


def _vorticity(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dvx_dy, _ = np.gradient(vx, dy, dx)
    _, dvy_dx = np.gradient(vy, dy, dx)
    return dvy_dx - dvx_dy


def _disk(ax, D: float) -> None:
    circ = np.linspace(0.0, 2.0 * np.pi, 180)
    ax.fill(0.5 * D * np.cos(circ), 0.5 * D * np.sin(circ), color="0.15", zorder=3)
    ax.plot(0.5 * D * np.cos(circ), 0.5 * D * np.sin(circ), color="0.85", lw=0.8, zorder=4)


def plot_cyl(code: dict[str, np.ndarray], meta: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    x, y = code["x"], code["y"]
    D = float(meta.get("D", 1.0))
    speed = np.sqrt(code["vx"] ** 2 + code["vy"] ** 2)
    omg = _vorticity(x, y, code["vx"], code["vy"])
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.4), constrained_layout=True)
    fig.suptitle(
        f"cylinder  ·  {meta.get('riemann', 'hllc')}/"
        f"{meta.get('reconstruction', 'weno3')}  ·  "
        f"{int(meta.get('nx', x.size))}×{int(meta.get('ny', y.size))}  ·  "
        f"t={float(meta.get('t', 0.0)):.1f}  ·  "
        f"M={float(meta.get('Mach', 0.0)):.2f}",
        fontsize=11,
        color="0.15",
    )
    omg_lim = float(np.percentile(np.abs(omg), 99))
    panels = (
        (speed, dict(cmap="inferno"), r"$|u|$"),
        (omg, dict(cmap="RdBu_r", vmin=-omg_lim, vmax=omg_lim), r"$\omega_z$"),
    )
    for ax, (img, kw, lab) in zip(axes, panels):
        im = ax.pcolormesh(x, y, img, shading="auto", **kw)
        _disk(ax, D)
        ax.set_aspect("equal")
        ax.set_ylabel(r"$y$")
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label(lab)
    axes[-1].set_xlabel(r"$x$")
    fig.savefig(out / "wake.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.6, 4.0), facecolor="black")
    ax.set_facecolor("black")
    ax.pcolormesh(x, y, omg, shading="auto", cmap="RdBu_r", vmin=-omg_lim, vmax=omg_lim)
    _disk(ax, D)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out / "street.png", dpi=180, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def animate(out: Path) -> Path:
    """Write street.gif from output/snaps.npz (vorticity, fixed color scale)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    snap = out / "snaps.npz"
    if not snap.is_file():
        raise FileNotFoundError(f"missing {snap}; re-run run.py")
    _, meta = load(out) if (out / "summary.json").is_file() else ({}, {})
    data = np.load(snap)
    x, y = data["x"], data["y"]
    vx, vy, times = data["vx"], data["vy"], data["t"]
    D = float(meta.get("D", 1.0))
    omg = np.stack(
        [_vorticity(x, y, vx[k], vy[k]) for k in range(times.size)], axis=0
    )
    omg_lim = float(np.percentile(np.abs(omg), 99))
    fig, ax = plt.subplots(figsize=(10.6, 4.0), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(
        omg[0],
        origin="lower",
        extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
        cmap="RdBu_r",
        vmin=-omg_lim,
        vmax=omg_lim,
        aspect="equal",
        interpolation="nearest",
    )
    _disk(ax, D)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = ax.set_title(f"t = {times[0]:.2f}", color="0.85", fontsize=11)

    def _frame(k: int):
        im.set_data(omg[k])
        title.set_text(f"t = {times[k]:.2f}")
        return im, title

    anim = FuncAnimation(fig, _frame, frames=times.size, interval=80, blit=False)
    dest = out / "street.gif"
    anim.save(dest, writer=PillowWriter(fps=12), dpi=110)
    plt.close(fig)
    return dest


def draw(out: Path, anim: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    code, meta = load(out)
    plot_cyl(code, meta, out)
    print(f"wrote {out / 'wake.png'} and {out / 'street.png'}", flush=True)
    if anim:
        gif = animate(out)
        print(f"wrote {gif}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--anim", action="store_true")
    args = parser.parse_args()
    draw(args.out, anim=args.anim)


if __name__ == "__main__":
    main()
