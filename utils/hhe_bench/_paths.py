"""Locate the rad_hhe_pe_1d example and its gitignored dumps."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "rad_hhe_pe_1d"
OUT_OMP = EXAMPLE / "output" / "hhe_omp_bench"
OUT_CUDA = EXAMPLE / "output" / "hhe_cuda_bench"
