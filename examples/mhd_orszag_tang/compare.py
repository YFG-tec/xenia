"""Resolution comparison for the Orszag-Tang vortex (no xmf step).

Loads per-resolution dumps written by run.py --out output/n<N> and draws
density / magnetic pressure / J_z side by side, plus a horizontal density
cut through the vortex centre. Shared colour scales per row so the
structural differences are honest.

    python examples/mhd_orszag_tang/compare.py                 # 64/128/512
    python examples/mhd_orszag_tang/compare.py --ns 64 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def _load(out: Path) -> tuple[dict[str, np.ndarray], dict]:
    raw = np.load(out / "fields.npz")
    code = {key: raw[key] for key in raw.files}
    summary = out / "summary.json"
    meta = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
    return code, meta


def _current_z(code: dict[str, np.ndarray]) -> np.ndarray:
    dx = float(code["x"][1] - code["x"][0])
    dy = float(code["y"][1] - code["y"][0])
    db2_dx = np.gradient(code["B2"][0], dx, axis=1)
    db1_dy = np.gradient(code["B1"][0], dy, axis=0)
    return db2_dx - db1_dy


def draw(ns: list[int], root: Path, dest: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for n in ns:
        code, meta = _load(root / f"n{n}")
        runs.append((n, code, meta))

    rows = (
        ("rho", "inferno", r"density $\rho$", None),
        ("magp", "magma", r"magnetic pressure $|\mathbf{B}|^2/2$", None),
        ("jz", "coolwarm", r"current density $J_z$", "sym"),
    )
    fields = []
    for _, code, _ in runs:
        fields.append(
            {
                "rho": code["rho"][0],
                "magp": 0.5
                * (code["B1"][0] ** 2 + code["B2"][0] ** 2 + code["B3"][0] ** 2),
                "jz": _current_z(code),
            }
        )

    fig, axes = plt.subplots(
        4, len(runs), figsize=(4.2 * len(runs), 15.2), constrained_layout=True
    )
    for row, (key, cmap, label, sym) in enumerate(rows):
        lo = min(float(f[key].min()) for f in fields)
        hi = max(float(f[key].max()) for f in fields)
        if sym:
            vabs = max(abs(lo), abs(hi)) or 1.0
            lo, hi = -vabs, vabs
        for col, ((n, code, meta), f) in enumerate(zip(runs, fields)):
            ax = axes[row, col]
            im = ax.pcolormesh(
                code["x"], code["y"], f[key],
                shading="auto", cmap=cmap, vmin=lo, vmax=hi,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    f"{n}$^2$  ·  t={float(meta.get('t', 0.0)):.3f}\n"
                    f"max $\\nabla\\cdot B$={float(meta.get('divb_max', 0.0)):.1e}  ·  "
                    f"$\\Delta E/E$={float(meta.get('energy_drift', 0.0)):.1e}",
                    fontsize=10,
                )
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: density cut through the vortex centre (y = pi).
    ax = axes[3, 0]
    for (n, code, meta), f in zip(runs, fields):
        y = code["y"]
        j = int(np.argmin(np.abs(y - np.pi)))
        ax.plot(code["x"], f["rho"][j, :], lw=1.2, label=f"{n}$^2$")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\rho(x,\ y=\pi)$")
    ax.set_title("density cut through the vortex centre", fontsize=10)
    ax.legend()
    for col in range(1, len(runs)):
        axes[3, col].axis("off")

    fig.suptitle(
        "Orszag–Tang vortex — resolution study (hll/plm/rk3, cfl=0.4, CT)",
        fontsize=12,
    )
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    print(f"wrote {dest}", flush=True)

    # Console table.
    hdr = f"{'n':>4} {'steps':>6} {'t':>8} {'wall_s':>8} {'max_divb':>10} {'dE/E':>10} {'L1_rho':>9}"
    print(hdr)
    for n, _, meta in runs:
        print(
            f"{n:>4} {int(meta.get('steps', 0)):>6} {float(meta.get('t', 0.0)):>8.4f} "
            f"{float(meta.get('wall_s', 0.0)):>8.2f} {float(meta.get('divb_max', 0.0)):>10.2e} "
            f"{float(meta.get('energy_drift', 0.0)):>10.2e} {float(meta.get('l1_rho', 0.0)):>9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=[64, 128, 512])
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--dest", type=Path, default=OUT / "compare_resolution.png")
    args = parser.parse_args()
    draw(args.ns, args.root, args.dest)


if __name__ == "__main__":
    main()
