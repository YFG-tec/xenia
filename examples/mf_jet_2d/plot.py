"""Redraw mf_jet_2d figures from a saved dump. No xmf step.

    python examples/mf_jet_2d/plot.py
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


def plot_jet(code: dict[str, np.ndarray], meta: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    x, y = code["x"], code["y"]
    rho = code["light_rho"]
    du = np.sqrt(
        (code["light_vx"] - code["heavy_vx"]) ** 2
        + (code["light_vy"] - code["heavy_vy"]) ** 2
    )
    omg = _vorticity(x, y, code["light_vx"], code["light_vy"])
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.6), constrained_layout=True)
    fig.suptitle(
        f"2-D slab jet  ·  {meta.get('riemann', 'hllc')}/"
        f"{meta.get('reconstruction', 'weno3')}  ·  "
        f"{int(meta.get('nx', x.size))}×{int(meta.get('ny', y.size))}  ·  "
        f"t={float(meta.get('t', 0.0)):.2f}  ·  "
        f"M={float(meta.get('Mach', 0.0)):.1f}  ·  "
        f"ν={float(meta.get('nu', 0.0)):.1f}",
        fontsize=11,
        color="0.15",
    )
    panels = (
        (rho, "inferno", r"$\rho$ light"),
        (du, "magma", r"$|\Delta\mathbf{u}|$"),
        (omg, "RdBu_r", r"$\omega_z$ light"),
    )
    for ax, (img, cmap, lab) in zip(axes, panels):
        if cmap == "RdBu_r":
            vmax = float(np.percentile(np.abs(img), 99))
            kw = dict(cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            kw = dict(cmap=cmap)
        im = ax.pcolormesh(x, y, img, shading="auto", **kw)
        ax.set_aspect("equal")
        ax.set_ylabel(r"$y$")
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label(lab)
    axes[-1].set_xlabel(r"$x$")
    fig.savefig(out / "jet.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 3.4), facecolor="black")
    ax.set_facecolor("black")
    logn = np.log10(np.maximum(rho, 1.0e-4))
    ax.pcolormesh(x, y, logn, shading="auto", cmap="inferno")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out / "beam.png", dpi=180, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def draw(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    code, meta = load(out)
    plot_jet(code, meta, out)
    print(f"wrote {out / 'jet.png'} and {out / 'beam.png'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    draw(args.out)


if __name__ == "__main__":
    main()
