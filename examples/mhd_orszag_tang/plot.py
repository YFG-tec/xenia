"""Redraw Orszag–Tang figures from a saved dump. No xmf step.

    python examples/mhd_orszag_tang/plot.py
    python examples/mhd_orszag_tang/plot.py --out examples/mhd_orszag_tang/output
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


def _edges(centers: np.ndarray) -> np.ndarray:
    """Cell-face coordinates from cell centers (uniform grid)."""
    dc = np.diff(centers)
    out = np.empty(centers.size + 1, dtype=np.float64)
    out[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    out[0] = centers[0] - 0.5 * dc[0]
    out[-1] = centers[-1] + 0.5 * dc[-1]
    return out


def _current_z(code: dict[str, np.ndarray]) -> np.ndarray:
    """Out-of-plane current J_z = dB2/dx - dB1/dy (cell-centred differences)."""
    dx = float(code["x"][1] - code["x"][0])
    dy = float(code["y"][1] - code["y"][0])
    db2_dx = np.gradient(code["B2"][0], dx, axis=1)
    db1_dy = np.gradient(code["B1"][0], dy, axis=0)
    return db2_dx - db1_dy


def _title(meta: dict, n: int) -> str:
    return (
        f"Orszag–Tang vortex  ·  hll/plm/rk3  ·  {n}²  ·  "
        f"t={float(meta.get('t', 0.0)):.3f}  ·  "
        f"{int(meta.get('steps', 0))} steps  ·  "
        f"max $\\nabla\\cdot B$={float(meta.get('divb_max', 0.0)):.1e}  ·  "
        f"$\\Delta E/E$={float(meta.get('energy_drift', 0.0)):.1e}"
    )


def draw(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    code, meta = load(out)
    x, y = code["x"], code["y"]
    xe, ye = _edges(x), _edges(y)
    n = x.size

    rho = code["rho"][0]
    prs = code["P"][0]
    mag_p = 0.5 * (code["B1"][0] ** 2 + code["B2"][0] ** 2 + code["B3"][0] ** 2)
    jz = _current_z(code)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 9.2), constrained_layout=True)
    fig.suptitle(_title(meta, n), fontsize=11, color="0.15")

    panels = (
        (rho, "inferno", r"density $\rho$"),
        (prs, "viridis", r"pressure $P$"),
        (mag_p, "magma", r"magnetic pressure $|\mathbf{B}|^2/2$"),
        (jz, "coolwarm", r"current density $J_z = \partial_x B_2 - \partial_y B_1$"),
    )
    for ax, (img, cmap, label) in zip(axes.flat, panels):
        if cmap == "coolwarm":
            vabs = float(np.max(np.abs(img))) or 1.0
            im = ax.pcolormesh(
                xe, ye, img, shading="auto", cmap=cmap, vmin=-vabs, vmax=vabs
            )
        else:
            im = ax.pcolormesh(xe, ye, img, shading="auto", cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out / "orszag_tang.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'orszag_tang.png'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    draw(args.out)


if __name__ == "__main__":
    main()
