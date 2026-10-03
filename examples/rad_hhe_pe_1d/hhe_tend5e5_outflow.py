"""Radiation-on rad_hhe_pe_1d: tstop=5e5, nx=256, stretch=1.015, 14 threads.

Same physics table as the example factory. x1beg stays the user
planetary base; x1end is default copy-outflow (drop the old-tree
IEND power-law tapes). Writes output/tend5e5/ (gitignored).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output" / "tend5e5"
sys.path.insert(0, str(HERE))

import xmf  # noqa: E402
from hhe_config import RUN, hhe_1d  # noqa: E402
from run import DATA, FLUIDS, lib, mdot_g_s, pack_ic, snapshot  # noqa: E402
from xmf.config import AMU, KB  # noqa: E402

NX = 256
STRETCH = 1.015
TSTOP = 5.0e5
DT_MAX = 5.0
LOG_EVERY = 1000
HIST_EVERY = 50
R_HILL = 4.5


def apply_factory_knobs() -> None:
    RUN.nx = NX
    RUN.stretch = "log"
    RUN.stretch_factor = STRETCH
    RUN.tstop = TSTOP
    RUN.dt_max = DT_MAX
    RUN.plots = False
    RUN.log_every = LOG_EVERY


def inner_fixed_outer_outflow(cfg: dict) -> dict:
    """Keep x1beg user (fixed n, v=0, P at R_p). Outer face = copy outflow."""
    cfg = dict(cfg)
    user_bc = dict(cfg.get("user_bc") or {})
    user_bc.pop("x1end", None)
    cfg["user_bc"] = user_bc
    boundary = dict(cfg.get("boundary") or {})
    boundary["x1beg"] = "user"
    boundary["x1end"] = "outflow"
    cfg["boundary"] = boundary
    return cfg


def mdot_total(snap: dict, r: np.ndarray, masses_g: dict[str, float]) -> np.ndarray:
    tot = np.zeros_like(r)
    for name in FLUIDS:
        tot += mdot_g_s(snap[name]["n"], snap[name]["vx"], r, masses_g[name])
    return tot


def pack_npz(snap: dict) -> dict[str, np.ndarray]:
    out = {
        "r": np.asarray(snap["r"], dtype=np.float64),
        "P_e": np.asarray(snap["P_e"], dtype=np.float64),
        "T_e": np.asarray(snap["T_e"], dtype=np.float64),
    }
    for name in FLUIDS:
        for key in ("n", "vx", "P", "T"):
            out[f"{name}_{key}"] = np.asarray(snap[name][key], dtype=np.float64)
    return out


def main() -> None:
    apply_factory_knobs()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = inner_fixed_outer_outflow(
        hhe_1d(
            DATA,
            radiation=True,
            nx=NX,
            tstop=TSTOP,
            stretch="log",
            stretch_factor=STRETCH,
            dt_max=DT_MAX,
        )
    )
    assert cfg["boundary"]["x1beg"] == "user"
    assert cfg["boundary"]["x1end"] == "outflow"
    assert "x1beg" in cfg["user_bc"] and "x1end" not in cfg["user_bc"]
    print(
        f"BC x1beg={cfg['boundary']['x1beg']} (user keys={list(cfg['user_bc']['x1beg'])})  "
        f"x1end={cfg['boundary']['x1end']}",
        flush=True,
    )
    run = xmf.setup(cfg)
    pack_ic(run, cfg)
    r_p = cfg["constants"]["R_p"]
    ng = cfg["grid"]["nghost"]
    xc = np.empty(NX + 2 * ng, dtype=np.float64)
    import ctypes

    dptr = ctypes.POINTER(ctypes.c_double)
    lib().xmf_mesh_copy_x(run._h, 0, xc.ctypes.data_as(dptr))
    r = xc[ng : ng + NX]
    ic = snapshot(run, r)
    c = cfg["constants"]
    masses_g = {
        "HI": c["m_HI_amu"] * AMU,
        "HEI": c["m_HEI_amu"] * AMU,
        "HII": c["m_HII_amu"] * AMU,
        "HEII": c["m_HEII_amu"] * AMU,
    }

    hist = {k: [] for k in (
        "t", "dt", "resid",
        "mdot_mid", "mdot_top", "mdot_outer_mean", "mdot_outer_relstd",
        "ion_h_base", "ion_he_base", "ion_h_top", "ion_he_top",
        "vx_hii_max", "vx_heii_max", "n_hi_base", "n_hi_top",
    )}
    t_wall = time.time()
    steps = 0
    failed = None
    try:
        while run.t < TSTOP:
            run.step()
            steps += 1
            take = (steps % HIST_EVERY == 0) or (run.t >= TSTOP)
            if take:
                fn = snapshot(run, r)
                md = mdot_total(fn, r, masses_g)
                rr = r / r_p
                outer = md[rr >= R_HILL]
                mean = float(np.mean(outer)) if outer.size else float("nan")
                std = float(np.std(outer)) if outer.size else float("nan")
                hist["t"].append(run.t)
                hist["dt"].append(run.dt)
                hist["resid"].append(run.source_resid)
                hist["mdot_mid"].append(float(md[len(r) // 2]))
                hist["mdot_top"].append(float(md[-1]))
                hist["mdot_outer_mean"].append(mean)
                hist["mdot_outer_relstd"].append(
                    std / abs(mean) if abs(mean) > 0 else float("nan")
                )
                hist["ion_h_base"].append(fn["HII"]["n"][0] / max(fn["HI"]["n"][0], 1e-300))
                hist["ion_he_base"].append(fn["HEII"]["n"][0] / max(fn["HEI"]["n"][0], 1e-300))
                hist["ion_h_top"].append(fn["HII"]["n"][-1] / max(fn["HI"]["n"][-1], 1e-300))
                hist["ion_he_top"].append(fn["HEII"]["n"][-1] / max(fn["HEI"]["n"][-1], 1e-300))
                hist["vx_hii_max"].append(float(np.max(np.abs(fn["HII"]["vx"]))))
                hist["vx_heii_max"].append(float(np.max(np.abs(fn["HEII"]["vx"]))))
                hist["n_hi_base"].append(float(fn["HI"]["n"][0]))
                hist["n_hi_top"].append(float(fn["HI"]["n"][-1]))
            if LOG_EVERY and (steps % LOG_EVERY == 0 or run.t >= TSTOP):
                print(
                    f"[on] step {steps}  t = {run.t:.4e} s  "
                    f"dt = {run.dt:.3e} s  resid = {run.source_resid:.2e}  "
                    f"Mdot_top = {hist['mdot_top'][-1]:.3e}  "
                    f"outer relstd = {hist['mdot_outer_relstd'][-1]:.3e}",
                    flush=True,
                )
    except Exception as exc:
        failed = str(exc)
        print(f"FAIL at step {steps} t={run.t:.6e} dt={run.dt:.3e}: {exc}", flush=True)

    wall = time.time() - t_wall
    fn = snapshot(run, r)
    md = mdot_total(fn, r, masses_g)
    rr = r / r_p
    outer = md[rr >= R_HILL]
    arrays = {k: np.asarray(v) for k, v in hist.items()}
    # Last 10% of recorded history: relative drift of Mdot_top and base n.
    conv = {}
    if arrays["t"].size >= 4:
        t = arrays["t"]
        tcut = t[0] + 0.9 * (t[-1] - t[0])
        sl = t >= tcut
        for key in ("mdot_top", "mdot_outer_mean", "n_hi_base", "vx_hii_max",
                    "ion_h_top", "ion_he_top"):
            y = arrays[key][sl]
            y0, y1 = float(y[0]), float(y[-1])
            conv[f"{key}_last10_rel"] = (y1 - y0) / max(abs(y0), 1e-300)
            conv[f"{key}_last10_abs"] = y1 - y0
    finite = all(
        np.all(np.isfinite(fn[s][k]))
        for s in FLUIDS
        for k in ("n", "vx", "P", "T")
    ) and np.all(np.isfinite(fn["P_e"]))

    meta = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "nx": NX,
        "stretch_factor": STRETCH,
        "tstop": TSTOP,
        "dt_max": DT_MAX,
        "boundary": cfg["boundary"],
        "steps": steps,
        "t_final": float(run.t),
        "dt_final": float(run.dt),
        "resid_final": float(run.source_resid),
        "wall_s": wall,
        "failed": failed,
        "finite": bool(finite),
        "mdot_mid": float(md[len(r) // 2]),
        "mdot_top": float(md[-1]),
        "mdot_outer_mean": float(np.mean(outer)) if outer.size else None,
        "mdot_outer_relstd": (
            float(np.std(outer) / abs(np.mean(outer)))
            if outer.size and abs(np.mean(outer)) > 0
            else None
        ),
        "n_HI_base_top": [float(fn["HI"]["n"][0]), float(fn["HI"]["n"][-1])],
        "n_HI_ic_base_top": [float(ic["HI"]["n"][0]), float(ic["HI"]["n"][-1])],
        "vx_HII_minmax": [float(np.min(fn["HII"]["vx"])), float(np.max(fn["HII"]["vx"]))],
        "convergence_last10": conv,
    }
    np.savez_compressed(OUT / "final.npz", mdot=md, **pack_npz(fn))
    np.savez_compressed(OUT / "hist.npz", **arrays)
    (OUT / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# rad_hhe_pe_1d tend=5e5 x1beg=user x1end=outflow (14 threads)",
        f"steps={steps} t_final={run.t:.6e} wall={wall:.1f}s failed={failed!r}",
        f"finite={finite} resid_final={run.source_resid:.3e} dt_final={run.dt:.3e}",
        f"Mdot mid/top = {md[len(r)//2]:.6e} / {md[-1]:.6e} g/s",
        (
            f"r>{R_HILL} Rp  mean={np.mean(outer):.6e}  std/|mean|="
            f"{np.std(outer)/abs(np.mean(outer)):.4e}"
            if outer.size and abs(np.mean(outer)) > 0
            else "outer Mdot undefined"
        ),
        f"n_HI base/top IC = {ic['HI']['n'][0]:.6e} / {ic['HI']['n'][-1]:.6e}",
        f"n_HI base/top    = {fn['HI']['n'][0]:.6e} / {fn['HI']['n'][-1]:.6e}",
        "last-10% relative drift:",
    ]
    for k, v in conv.items():
        if k.endswith("_rel"):
            lines.append(f"  {k} = {v:.4e}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    run.close()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
