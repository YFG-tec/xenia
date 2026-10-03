"""Sequential OMP_NUM_THREADS=1,2,8,16 radiation-on benches (separate processes)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import ROOT  # noqa: E402

BENCH = HERE / "hhe_omp_bench.py"
COMPARE = HERE / "hhe_omp_bench_compare.py"
THREADS = (1, 2, 8, 16)


def main() -> None:
    extra = sys.argv[1:]
    for nt in THREADS:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(nt)
        env.pop("OMP_PROC_BIND", None)
        env.pop("OMP_PLACES", None)
        cmd = [sys.executable, str(BENCH), "--threads", str(nt), *extra]
        print(f"==== start nt={nt} ====", flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT), env=env)
        print(f"==== done nt={nt} ====", flush=True)
    subprocess.check_call([sys.executable, str(COMPARE)], cwd=str(ROOT))


if __name__ == "__main__":
    main()
