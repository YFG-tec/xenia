"""Does dt_max=5 keep nx=256 / stretch=1.015 bounded past the t~2500 blow-up?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xmf  # noqa: E402
from hhe_config import RUN, hhe_1d  # noqa: E402
from run import DATA, pack_ic  # noqa: E402


def main() -> None:
    nx, stretch, dt_max = 256, 1.015, 5.0
    max_steps = 900
    RUN.nx = nx
    RUN.stretch_factor = stretch
    RUN.dt_max = dt_max
    RUN.plots = False
    cfg = hhe_1d(
        DATA, radiation=True, nx=nx, tstop=2.0e5,
        stretch="log", stretch_factor=stretch, dt_max=dt_max,
    )
    run = xmf.setup(cfg)
    pack_ic(run, cfg)
    worst = {"resid": 0.0, "step": 0, "t": 0.0}
    vx_hi = 0.0
    try:
        for steps in range(1, max_steps + 1):
            run.step()
            if run.source_resid > worst["resid"]:
                worst = {"resid": run.source_resid, "step": steps, "t": run.t}
            if steps % 100 == 0 or steps <= 5:
                fld = run.fields_full()
                vx = fld["HII"]["vx"][0, 0, :]
                vx_hi = float(np.max(np.abs(vx)))
                print(
                    f"step {steps:4d} t={run.t:.4e} dt={run.dt:.3e} "
                    f"resid={run.source_resid:.3e} |vx_HII|_max={vx_hi:.3e}",
                    flush=True,
                )
        print(
            f"OK {max_steps} steps t={run.t:.4e} worst_resid={worst['resid']:.3e} "
            f"at step {worst['step']} t={worst['t']:.4e}",
            flush=True,
        )
    except Exception as exc:
        print(
            f"FAIL at t={run.t:.6e} dt={run.dt:.3e} resid={run.source_resid:.3e}: {exc}",
            flush=True,
        )
        raise
    finally:
        run.close()


if __name__ == "__main__":
    main()
