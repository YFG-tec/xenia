"""Compare output/hhe_omp_bench/nt*.npz against the 1-thread dump."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUT_OMP as OUT  # noqa: E402
THREADS = (1, 2, 8, 16)
SKIP = {"r"}  # mesh is identical by construction; still report it


def max_abs_rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    scale = np.maximum(scale, 1e-300)
    return float(np.max(diff)), float(np.max(diff / scale))


def main() -> None:
    ref_path = OUT / "nt1.npz"
    if not ref_path.is_file():
        raise SystemExit(f"missing {ref_path}")
    ref = np.load(ref_path)
    lines = ["# rad_hhe_pe_1d OpenMP compare (nx=256, stretch=1.015, radiation on)", ""]
    metas = {}
    for nt in THREADS:
        jp = OUT / f"nt{nt}.json"
        if jp.is_file():
            metas[nt] = json.loads(jp.read_text(encoding="utf-8"))
    if metas:
        t1 = metas.get(1, {}).get("wall_one_run_s")
        lines.append("## wall time")
        for nt in THREADS:
            m = metas.get(nt)
            if not m:
                continue
            wall = m["wall_one_run_s"]
            speed = (t1 / wall) if (t1 and wall) else float("nan")
            lines.append(
                f"  nt={nt:2d}  steps={m['steps']}  t_final={m['t_final']:.6e}  "
                f"wall={wall:.1f}s  speedup vs nt=1 = {speed:.3f}"
            )
        lines.append("")

    lines.append("## field compare vs nt=1 (max |Δ|, max |Δ|/scale)")
    for nt in THREADS:
        if nt == 1:
            continue
        path = OUT / f"nt{nt}.npz"
        if not path.is_file():
            lines.append(f"  nt={nt}: MISSING {path.name}")
            continue
        cur = np.load(path)
        lines.append(f"  --- nt={nt} ---")
        worst_rel = 0.0
        worst_key = ""
        bitwise = True
        for key in ref.files:
            a, b = ref[key], cur[key]
            if a.shape != b.shape:
                lines.append(f"    {key}: SHAPE {a.shape} vs {b.shape}")
                bitwise = False
                continue
            if np.array_equal(a, b):
                lines.append(f"    {key}: bitwise")
                continue
            bitwise = False
            ad, rel = max_abs_rel(a, b)
            if rel > worst_rel:
                worst_rel, worst_key = rel, key
            lines.append(f"    {key}: max|d|={ad:.6e}  maxrel={rel:.6e}")
        tag = "BITWISE" if bitwise else f"worst rel {worst_rel:.6e} on {worst_key}"
        lines.append(f"    summary: {tag}")
    text = "\n".join(lines) + "\n"
    (OUT / "compare.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
