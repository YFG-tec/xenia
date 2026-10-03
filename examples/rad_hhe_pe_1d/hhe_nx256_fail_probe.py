"""Find when nx=256 / stretch=1.015 radiation-on dies (xmf_step -1)."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xmf  # noqa: E402
from hhe_config import RUN, hhe_1d  # noqa: E402
from run import DATA, pack_ic  # noqa: E402

OUT = HERE / "output" / "hhe_omp_bench"


def mesh_note(run, tag: str) -> str:
    import ctypes

    lib = xmf.api._lib()
    nxt = run._cfg.n[0] + 2 * run._cfg.ng
    dx = np.empty(nxt, dtype=np.float64)
    lib.xmf_mesh_copy_dx(run._h, 0, dx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    interior = dx[run._cfg.ng : run._cfg.ng + run._cfg.n[0]]
    rp = 1.36 * 7.1492e9
    return (
        f"{tag}: nx={run._cfg.n[0]}  dx/Rp first/last="
        f"{interior[0]/rp:.6e}/{interior[-1]/rp:.6e}  "
        f"ratio={interior[-1]/interior[0]:.4f}  "
        f"dt_max={run._cfg.numerics.dt_max}"
    )


def try_run(nx: int, stretch: float, dt_max: float, max_steps: int) -> dict:
    RUN.nx = nx
    RUN.stretch_factor = stretch
    RUN.dt_max = dt_max
    RUN.plots = False
    RUN.log_every = 0
    cfg = hhe_1d(
        DATA,
        radiation=True,
        nx=nx,
        tstop=RUN.tstop,
        stretch="log",
        stretch_factor=stretch,
        dt_max=dt_max,
    )
    run = xmf.setup(cfg)
    pack_ic(run, cfg)
    note = mesh_note(run, f"nx={nx} stretch={stretch} dt_max={dt_max}")
    print(note, flush=True)
    steps = 0
    last = {"t": 0.0, "dt": run.dt, "resid": 0.0}
    err = None
    try:
        while run.t < RUN.tstop and steps < max_steps:
            run.step()
            steps += 1
            last = {"t": run.t, "dt": run.dt, "resid": run.source_resid}
            if steps <= 5 or steps % 50 == 0:
                print(
                    f"  step {steps} t={run.t:.4e} dt={run.dt:.3e} "
                    f"resid={run.source_resid:.3e}",
                    flush=True,
                )
    except Exception as exc:
        err = str(exc)
        print(f"FAIL after {steps} ok steps: {err}", flush=True)
        print(f"  last ok t={last['t']:.6e} dt={last['dt']:.3e} resid={last['resid']:.3e}", flush=True)
        traceback.print_exc()
    finally:
        run.close()
    return {"note": note, "steps_ok": steps, "last": last, "err": err}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    lines = [
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}",
        "",
        "## factory vs requested mesh (setup only)",
    ]
    # mesh-only compares
    for nx, st, dtm in ((128, 1.025, 10.0), (256, 1.015, 10.0)):
        rec = try_run(nx, st, dtm, max_steps=0)
        lines.append(rec["note"])
    print("==== crash hunt nx=256 stretch=1.015 dt_max=10, 400 steps ====", flush=True)
    rec = try_run(256, 1.015, 10.0, max_steps=400)
    lines += [
        "",
        "## crash hunt",
        rec["note"],
        f"steps_ok={rec['steps_ok']} err={rec['err']!r}",
        f"last={rec['last']}",
    ]
    if rec["err"]:
        print("==== retry dt_max=5, 200 steps ====", flush=True)
        rec5 = try_run(256, 1.015, 5.0, max_steps=200)
        lines += [
            "",
            "## dt_max=5 retry",
            rec5["note"],
            f"steps_ok={rec5['steps_ok']} err={rec5['err']!r}",
            f"last={rec5['last']}",
        ]
    (OUT / "fail_probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "fail_probe.txt", flush=True)


if __name__ == "__main__":
    main()
