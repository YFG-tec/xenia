"""Compare CUDA H/He dump against the C1b OpenMP benches. Do not git add."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import OUT_CUDA, OUT_OMP  # noqa: E402

cuda = json.loads((OUT_CUDA / "cuda.json").read_text(encoding="utf-8"))
print(
    f"CUDA  steps={cuda['steps']}  t={cuda['t_final']:.4e}  "
    f"wall={cuda['wall_one_run_s']:.1f}s  resid={cuda['resid_final']:.4e}"
)
for nt in (1, 2, 8, 16):
    jp = OUT_OMP / f"nt{nt}.json"
    if not jp.is_file():
        continue
    m = json.loads(jp.read_text(encoding="utf-8"))
    ratio = cuda["wall_one_run_s"] / m["wall_one_run_s"]
    print(
        f"OMP nt={nt:2d}  wall={m['wall_one_run_s']:.1f}s  "
        f"CUDA/OMP={ratio:.2f}x"
    )

refp = OUT_OMP / "nt1.npz"
curp = OUT_CUDA / "cuda.npz"
if refp.is_file() and curp.is_file():
    a, b = np.load(refp), np.load(curp)
    n_bit = 0
    worst_rel, worst_key, worst_abs = 0.0, "", 0.0
    for key in a.files:
        if a[key].shape != b[key].shape:
            print(f"SHAPE {key} {a[key].shape} vs {b[key].shape}")
            continue
        if np.array_equal(a[key], b[key]):
            n_bit += 1
            continue
        diff = np.abs(a[key] - b[key])
        scale = np.maximum(np.maximum(np.abs(a[key]), np.abs(b[key])), 1e-300)
        rel = float(np.max(diff / scale))
        ad = float(np.max(diff))
        if rel > worst_rel:
            worst_rel, worst_key, worst_abs = rel, key, ad
    print(
        f"vs OMP nt=1: {n_bit}/{len(a.files)} bitwise; "
        f"worst rel {worst_rel:.3e} on {worst_key} (max|d|={worst_abs:.3e})"
    )
