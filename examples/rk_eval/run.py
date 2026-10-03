"""1D Sod, HLL+PCM: compare rk1/rk2/rk3 while stretching CFL (including >1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import xmf
from xmf.config import KB

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from utils.riemann import exact_hi  # noqa: E402

OUT = HERE / "output"
RKS = ("rk1", "rk2", "rk3")
CFLS = (0.2, 0.4, 0.8, 1.0, 1.2, 1.5, 2.0)
TSTOP = 0.60  # 4 × shock_tube() default 0.15
GAMMA = 5.0 / 3.0
NL, TL = 1.0e13, 1.0e4
NR, TR = 1.25e12, 8.0e3
PL, PR = NL * KB * TL, NR * KB * TR
X0 = 5.5


def _l1(num, exact, scale: float) -> float:
    return float(np.mean(np.abs(num - exact)) / scale)


def _one(rk: str, cfl: float) -> dict:
    cfg = xmf.shock_tube()
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": rk}
    cfg["time"]["cfl"] = cfl
    cfg["time"]["tstop"] = TSTOP
    run = xmf.setup(cfg)
    rec = {"rk": rk, "cfl": cfl, "ok": False}
    try:
        nstep = 0
        dts: list[float] = []
        while run.t < run.tstop:
            run.step()
            nstep += 1
            dts.append(float(run.dt))
            hi = run.fields["HI"]
            if not (
                np.all(np.isfinite(hi.n))
                and np.all(np.isfinite(hi.v))
                and np.all(np.isfinite(hi.p))
            ):
                rec.update(nstep=nstep, t=float(run.t), reason="non-finite")
                return rec
            if np.min(hi.n) <= 0.0 or np.min(hi.p) <= 0.0:
                rec.update(nstep=nstep, t=float(run.t), reason="negative n or P")
                return rec
            if float(np.max(np.abs(hi.v))) > 1.0e12 or float(np.max(hi.p)) > 1.0e20:
                rec.update(nstep=nstep, t=float(run.t), reason="fields exploded")
                return rec
        hi = run.fields["HI"]
        n_ex, v_ex, p_ex = exact_hi(
            hi.x, run.t, X0, NL, NR, 0.0, 0.0, PL, PR, gamma=GAMMA,
        )
        l1n = _l1(hi.n, n_ex, NL)
        l1v = _l1(hi.v, v_ex, max(float(np.max(np.abs(v_ex))), 1.0))
        l1p = _l1(hi.p, p_ex, float(np.max(p_ex)))
        rec.update(
            t=float(run.t),
            nstep=nstep,
            dt_mean=float(np.mean(dts)) if dts else 0.0,
            L1_n=l1n,
            L1_v=l1v,
            L1_p=l1p,
            x=hi.x,
            num=(hi.n, hi.v, hi.p),
            exact=(n_ex, v_ex, p_ex),
        )
        if max(l1n, l1v, l1p) > 0.5:
            rec["reason"] = "L1 exploded"
            return rec
        rec["ok"] = True
        return rec
    except Exception as exc:
        rec.update(reason=type(exc).__name__)
        return rec
    finally:
        run.close()


def _profiles(recs: list[dict], cfl: float, png: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    labels = (r"$n$", r"$v$", r"$P$")
    drawn_exact = False
    nplot = 0
    for rec in recs:
        if not rec.get("ok") or rec["cfl"] != cfl:
            continue
        nplot += 1
        x, (n, v, p) = rec["x"], rec["num"]
        n_ex, v_ex, p_ex = rec["exact"]
        if not drawn_exact:
            axes[0].plot(x, n_ex, "k-", lw=1.0, label="exact")
            axes[1].plot(x, v_ex, "k-", lw=1.0)
            axes[2].plot(x, p_ex, "k-", lw=1.0)
            drawn_exact = True
        axes[0].plot(x, n, lw=1.1, label=rec["rk"])
        axes[1].plot(x, v, lw=1.1)
        axes[2].plot(x, p, lw=1.1)
    for ax, lab in zip(axes, labels):
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(r"$x_1$")
    axes[0].set_title(f"Sod  ·  hll / pcm  ·  CFL = {cfl:g}  ·  $t_\\mathrm{{stop}}={TSTOP:g}$")
    fig.tight_layout()
    if nplot:
        fig.savefig(png, dpi=140)
    plt.close(fig)
    if nplot == 0 and png.exists():
        png.unlink()
    return nplot > 0


def _l1_vs_cfl(rows: list[dict], png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharex=True)
    keys = ("L1_n", "L1_v", "L1_p")
    titles = (r"$L_1(n)/\mathrm{scale}$", r"$L_1(v)/\mathrm{scale}$", r"$L_1(P)/\mathrm{scale}$")
    for ax, key, title in zip(axes, keys, titles):
        for rk in RKS:
            xs, ys = [], []
            for rec in rows:
                if rec["rk"] == rk and rec.get("ok"):
                    xs.append(rec["cfl"])
                    ys.append(rec[key])
            ax.plot(xs, ys, "o-", ms=4, label=rk)
        ax.axvline(1.0, color="0.5", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("CFL")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Sod  ·  hll / pcm  ·  $L_1$ vs CFL  ·  $t_\\mathrm{{stop}}={TSTOP:g}$", y=1.02)
    fig.tight_layout()
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _slim(rec: dict) -> dict:
    skip = {"x", "num", "exact"}
    out = {}
    for k, v in rec.items():
        if k in skip:
            continue
        if isinstance(v, float):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [_one(rk, cfl) for cfl in CFLS for rk in RKS]
    for rec in rows:
        tag = f"{rec['rk']:4s}  CFL={rec['cfl']:.1f}"
        if rec.get("ok"):
            print(
                f"{tag}  nstep={rec['nstep']:4d}  dt={rec['dt_mean']:.3e}  "
                f"L1(n,v,P)={rec['L1_n']:.3e} {rec['L1_v']:.3e} {rec['L1_p']:.3e}"
            )
        else:
            print(f"{tag}  FAIL  {rec.get('reason', '?')}  nstep={rec.get('nstep', 0)}")

    (OUT / "results.json").write_text(
        json.dumps([_slim(r) for r in rows], indent=2), encoding="utf-8"
    )
    _profiles(rows, 0.4, OUT / "profiles_cfl04.png")
    _profiles(rows, 1.2, OUT / "profiles_cfl12.png")
    _profiles(rows, 1.5, OUT / "profiles_cfl15.png")
    _l1_vs_cfl(rows, OUT / "l1_vs_cfl.png")
    print(f"figures → {OUT}")


if __name__ == "__main__":
    main()
