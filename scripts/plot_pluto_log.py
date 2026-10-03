#!/usr/bin/env python3
"""Plot a PLUTO/PLUTO_XMF log file.

This is a convenience script for quick sanity checks after a run.
It tries to parse numeric columns from a PLUTO-style log file and plot
common quantities vs. time/step.

Usage:
  python scripts/plot_pluto_log.py examples/smoke_test/output/pluto.log

It will write a PNG next to the log file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(path: Path):
    lines = path.read_text(errors="ignore").splitlines()

    # Keep lines that contain many numbers.
    data = []
    for ln in lines:
        if ln.strip().startswith("#"):
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", ln)
        if len(nums) >= 3:
            data.append([float(x.replace("D", "E").replace("d", "e")) for x in nums])

    if not data:
        raise SystemExit(f"No numeric table detected in log: {path}")

    # Make rectangular by trimming to min length.
    m = min(len(r) for r in data)
    data = [r[:m] for r in data]
    return data


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip())
        raise SystemExit(2)

    log_path = Path(sys.argv[1]).resolve()
    data = parse_log(log_path)

    import numpy as np

    arr = np.array(data)

    x = arr[:, 0]
    out_png = log_path.with_suffix(".png")

    plt.figure(figsize=(10, 6))
    # Plot up to 4 y-series to avoid clutter.
    for j in range(1, min(arr.shape[1], 5)):
        plt.plot(x, arr[:, j], label=f"col{j+1}")

    plt.xlabel("col1")
    plt.ylabel("value")
    plt.title(log_path.name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
