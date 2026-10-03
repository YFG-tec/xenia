"""Orszag–Tang animation from a frames.npz dump (no xmf step).

    python examples/mhd_orszag_tang/run.py --n 128 --animate
    python examples/mhd_orszag_tang/animate.py --out output/n128_tpi

Two-panel GIF when `magp` is in the dump (density + magnetic pressure);
density-only for older dumps. Colour scales are fixed across frames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def make(out: Path, dest: Path, fps: int = 12) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    raw = np.load(out / "frames.npz")
    t, rho = raw["t"], raw["rho"]
    x, y = raw["x"], raw["y"]
    magp = raw["magp"] if "magp" in raw.files else None
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    n = int(x.size)
    two = magp is not None

    if two:
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), constrained_layout=True)
        lo_r, hi_r = float(rho.min()), float(rho.max())
        lo_b, hi_b = float(magp.min()), float(magp.max())
        im_r = axes[0].imshow(
            rho[0],
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=lo_r,
            vmax=hi_r,
            interpolation="nearest",
            aspect="equal",
        )
        im_b = axes[1].imshow(
            magp[0],
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=lo_b,
            vmax=hi_b,
            interpolation="nearest",
            aspect="equal",
        )
        axes[0].set_title(r"density $\rho$")
        axes[1].set_title(r"magnetic pressure $|\mathbf{B}|^2/2$")
        for ax in axes:
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
        fig.colorbar(im_r, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)
        title = fig.suptitle("")

        def update(k: int):
            im_r.set_data(rho[k])
            im_b.set_data(magp[k])
            title.set_text(f"Orszag–Tang  ·  {n}²  ·  t = {t[k]:.3f}")
            return im_r, im_b, title

    else:
        fig, ax = plt.subplots(figsize=(5.6, 5.2), constrained_layout=True)
        lo, hi = float(rho.min()), float(rho.max())
        im = ax.imshow(
            rho[0],
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        title = ax.set_title("")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho$")

        def update(k: int):
            im.set_data(rho[k])
            title.set_text(f"Orszag–Tang density  ·  {n}²  ·  t = {t[k]:.3f}")
            return im, title

    ani = FuncAnimation(fig, update, frames=t.size, blit=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ani.save(dest, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {dest} ({t.size} frames, {fps} fps)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    dest = args.dest or args.out / "orszag_tang.gif"
    make(args.out, dest, args.fps)


if __name__ == "__main__":
    main()
