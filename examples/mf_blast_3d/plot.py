"""Redraw mf_blast_3d figures from a saved dump. No xmf step.

    python examples/mf_blast_3d/plot.py
    python examples/mf_blast_3d/plot.py --out examples/mf_blast_3d/output
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


def _mid(n: int) -> int:
    return n // 2


def _title(meta: dict, x: np.ndarray) -> str:
    return (
        f"3-D two-fluid blast  ·  {meta.get('riemann', 'tvdlf')}  ·  "
        f"n={x.size}³  ·  t={float(meta.get('t', 0.0)):.3f}  ·  "
        f"{int(meta.get('steps', 0))} steps  ·  "
        f"wall {float(meta.get('wall_s', 0.0)):.1f} s"
    )


def plot_slices(code: dict[str, np.ndarray], meta: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    x, y, z = code["x"], code["y"], code["z"]
    ix, iy, iz = _mid(x.size), _mid(y.size), _mid(z.size)
    rho = code["light_rho"]
    du = np.sqrt(
        (code["light_vx"] - code["heavy_vx"]) ** 2
        + (code["light_vy"] - code["heavy_vy"]) ** 2
        + (code["light_vz"] - code["heavy_vz"]) ** 2
    )
    speed = np.sqrt(
        code["light_vx"] ** 2 + code["light_vy"] ** 2 + code["light_vz"] ** 2
    )
    panels = (
        (rho[iz, :, :], x, y, r"$x$–$y$"),
        (rho[:, iy, :], x, z, r"$x$–$z$"),
        (rho[:, :, ix], y, z, r"$y$–$z$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), constrained_layout=True)
    fig.suptitle(_title(meta, x), fontsize=11, color="0.15")
    cmap_r = plt.get_cmap("inferno")
    cmap_u = plt.get_cmap("magma")
    for ax, (img, xa, ya, title) in zip(axes[0], panels):
        im = ax.pcolormesh(xa, ya, img, shading="auto", cmap=cmap_r)
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    du_panels = (
        (du[iz, :, :], x, y, r"$x$–$y$"),
        (du[:, iy, :], x, z, r"$x$–$z$"),
        (speed[iz, :, :], x, y, r"$|u|$ light, $x$–$y$"),
    )
    labels = (r"$|\Delta\mathbf{u}|$", r"$|\Delta\mathbf{u}|$", r"$|u|$")
    for ax, (img, xa, ya, title), lab in zip(axes[1], du_panels, labels):
        im = ax.pcolormesh(xa, ya, img, shading="auto", cmap=cmap_u)
        ax.set_aspect("equal")
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(lab)
    fig.savefig(out / "slices.png", dpi=160)
    plt.close(fig)


def plot_shell(code: dict[str, np.ndarray], out: Path) -> None:
    import matplotlib.pyplot as plt

    x, y = code["x"], code["y"]
    iz = _mid(code["z"].size)
    rho = code["light_rho"]
    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="black")
    ax.set_facecolor("black")
    logn = np.log10(np.maximum(rho[iz, :, :], 1.0e-8))
    ax.pcolormesh(x, y, logn, shading="auto", cmap="inferno")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out / "shell.png", dpi=180, facecolor="black")
    plt.close(fig)


def _edges(centers: np.ndarray) -> np.ndarray:
    """Cell-face coordinates from cell centers (uniform or stretched)."""
    dc = np.diff(centers)
    out = np.empty(centers.size + 1, dtype=np.float64)
    out[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    out[0] = centers[0] - 0.5 * dc[0]
    out[-1] = centers[-1] + 0.5 * dc[-1]
    return out


def _paint(ax, X, Y, Z, C, cmap, norm) -> None:
    """X,Y,Z are vertex meshes (M, N); C is face data (M-1, N-1)."""
    colors = cmap(norm(np.asarray(C, dtype=np.float64)))
    ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=False,
        linewidth=0,
    )


def plot_cutaway(code: dict[str, np.ndarray], meta: dict, out: Path) -> None:
    """Cutaway: only the +x+y+z quarter of each midplane (Earth corner)."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    x, y, z = code["x"], code["y"], code["z"]
    nx, ny, nz = int(x.size), int(y.size), int(z.size)
    ix, iy, iz = _mid(nx), _mid(ny), _mid(nz)
    rho = code["light_rho"]
    xe, ye, ze = _edges(x), _edges(y), _edges(z)
    x0, y0, z0 = float(xe[ix]), float(ye[iy]), float(ze[iz])

    cmap = matplotlib.colormaps["inferno"]
    norm = Normalize(vmin=float(np.min(rho)), vmax=float(np.max(rho)))

    fig = plt.figure(figsize=(8.4, 7.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.view_init(elev=26, azim=45)

    rho_xy = rho[iz, :, :]
    X, Y = np.meshgrid(xe[ix : nx + 1], ye[iy : ny + 1])
    _paint(ax, X, Y, np.full_like(X, z0), rho_xy[iy:ny, ix:nx], cmap, norm)
    X, Z = np.meshgrid(xe[ix : nx + 1], ze[iz : nz + 1])
    _paint(ax, X, np.full_like(X, y0), Z, rho[iz:nz, iy, ix:nx], cmap, norm)
    Y, Z = np.meshgrid(ye[iy : ny + 1], ze[iz : nz + 1])
    _paint(ax, np.full_like(Y, x0), Y, Z, rho[iz:nz, iy:ny, ix], cmap, norm)

    lo = (float(xe[0]), float(ye[0]), float(ze[0]))
    hi = (float(xe[-1]), float(ye[-1]), float(ze[-1]))
    edges = [
        [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2])],
        [(lo[0], hi[1], lo[2]), (hi[0], hi[1], lo[2])],
        [(lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2])],
        [(lo[0], hi[1], hi[2]), (hi[0], hi[1], hi[2])],
        [(lo[0], lo[1], lo[2]), (lo[0], hi[1], lo[2])],
        [(hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2])],
        [(lo[0], lo[1], hi[2]), (lo[0], hi[1], hi[2])],
        [(hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2])],
        [(lo[0], lo[1], lo[2]), (lo[0], lo[1], hi[2])],
        [(hi[0], lo[1], lo[2]), (hi[0], lo[1], hi[2])],
        [(lo[0], hi[1], lo[2]), (lo[0], hi[1], hi[2])],
        [(hi[0], hi[1], lo[2]), (hi[0], hi[1], hi[2])],
    ]
    ax.add_collection3d(
        Line3DCollection(edges, colors="0.55", linewidths=0.6, zorder=0)
    )

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    L = 0.28
    for dx, dy, dz, lab, col in (
        (L, 0.0, 0.0, r"$x$", "#c0392b"),
        (0.0, L, 0.0, r"$y$", "#1e8449"),
        (0.0, 0.0, L, r"$z$", "#2471a3"),
    ):
        ax.quiver(
            x0,
            y0,
            z0,
            dx,
            dy,
            dz,
            color=col,
            arrow_length_ratio=0.18,
            linewidth=1.8,
            normalize=False,
        )
        ax.text(
            x0 + 1.12 * dx,
            y0 + 1.12 * dy,
            z0 + 1.12 * dz,
            lab,
            color=col,
            fontsize=13,
        )
    ax.text(0.38, 0.38, z0, r"$xy$", color="0.2", fontsize=10)
    ax.text(0.38, y0, 0.38, r"$xz$", color="0.2", fontsize=10)
    ax.text(x0, 0.38, 0.38, r"$yz$", color="0.2", fontsize=10)
    ax.set_title(_title(meta, x), fontsize=10, color="0.2", pad=8)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.06, fraction=0.04)
    cb.set_label(r"$\rho$ (light)")
    fig.tight_layout()
    fig.savefig(out / "cutaway.png", dpi=180, facecolor="white")
    plt.close(fig)


def draw(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    code, meta = load(out)
    plot_slices(code, meta, out)
    plot_shell(code, out)
    plot_cutaway(code, meta, out)
    print(
        f"wrote {out / 'slices.png'}, {out / 'shell.png'}, {out / 'cutaway.png'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    draw(args.out)


if __name__ == "__main__":
    main()
