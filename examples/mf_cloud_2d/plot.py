"""Plot driver for the two-fluid shock-dense-cloud example (D2).

Per tier: one figure with two panels -- cloud-fluid density (morphology)
and |v_wind - v_cloud| (the drag diagnostic). With three or more tiers
present, also a combined paper panel (rows = tiers, columns = the two
diagnostics) at the same final time.

    python examples/mf_cloud_2d/plot.py --tiers off mid strong
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

ORDER = ("off", "weak", "mid", "strong")
LABEL = {
    "off": r"$K=0$ (no collision)",
    "weak": r"weak ($\tau_c = 10\,t_{\mathrm{cc}}$)",
    "mid": r"mid ($\tau_c = t_{\mathrm{cc}}$)",
    "strong": r"strong ($\tau_c = 0.1\,t_{\mathrm{cc}}$)",
}


def _load(out: Path, tier: str):
    fpath = out / f"fields_{tier}.npz"
    spath = out / f"summary_{tier}.json"
    if not fpath.exists() or not spath.exists():
        return None, None
    data = np.load(fpath)
    meta = json.loads(spath.read_text(encoding="utf-8"))
    return data, meta


def _panels(ax_rho, ax_du, data, meta) -> None:
    x, y = data["x"], data["y"]
    rho_c = data["cloud_rho"]
    du = np.sqrt(
        (data["wind_vx"] - data["cloud_vx"]) ** 2
        + (data["wind_vy"] - data["cloud_vy"]) ** 2
    )
    im0 = ax_rho.pcolormesh(
        x, y, np.log10(rho_c), shading="nearest", cmap="inferno"
    )
    ax_rho.set_title(
        f"{LABEL[meta['tier']]}: cloud density "
        rf"$\log_{{10}}\rho_B$ at $t={meta['t_over_tcc']:.2f}\,t_{{cc}}$"
    )
    im1 = ax_du.pcolormesh(x, y, du, shading="nearest", cmap="viridis")
    ax_du.set_title(r"$|\mathbf{v}_A - \mathbf{v}_B|$")
    for ax in (ax_rho, ax_du):
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax_rho.figure.colorbar(im0, ax=ax_rho, label=r"$\log_{10}\rho_B$")
    ax_du.figure.colorbar(im1, ax=ax_du, label=r"$|\Delta\mathbf{u}|$")


def draw_tier(out: Path, tier: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, meta = _load(out, tier)
    if data is None:
        raise FileNotFoundError(f"no fields_{tier}.npz / summary_{tier}.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _panels(axes[0], axes[1], data, meta)
    fig.tight_layout()
    path = out / f"panels_{tier}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    return [path]


def draw_combined(out: Path, tiers: tuple[str, ...]) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = []
    for tier in ORDER:
        if tier in tiers:
            data, meta = _load(out, tier)
            if data is not None:
                loaded.append((data, meta))
    if len(loaded) < 3:
        return None
    fig, axes = plt.subplots(len(loaded), 2, figsize=(11, 4.0 * len(loaded)))
    for row, (data, meta) in enumerate(loaded):
        _panels(axes[row][0], axes[row][1], data, meta)
    fig.tight_layout()
    path = out / "paper_panel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(ORDER),
        choices=list(ORDER),
    )
    args = parser.parse_args()
    for tier in ORDER:
        if tier in args.tiers and (OUT / f"fields_{tier}.npz").exists():
            draw_tier(OUT, tier)
    draw_combined(OUT, tuple(args.tiers))


if __name__ == "__main__":
    main()
