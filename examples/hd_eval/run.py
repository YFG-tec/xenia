"""A4 HD evaluation example: Riemann cross-matrix + three classic 2D tests.

    python examples/hd_eval/run.py

Writes figures and numeric tables under examples/hd_eval/output/ (gitignored).
The narrative report is examples/hd_eval/REPORT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cases import CLASSIC_RUNNERS, GEOM_RUNNERS, RIEMANN_RUNNERS  # noqa: E402
from kernel import OUT, SOLVERS  # noqa: E402
from plotting import (  # noqa: E402
    plot_classic_one,
    plot_classic_panel,
    plot_cyl_rz,
    plot_extruded_2d,
    plot_lineout,
    plot_sedov_radial,
    plot_solver_overlay,
    plot_sph_lineout,
    plot_sph_rth,
)

CASES_1D = ("1D-A", "1D-B")
CASES_2D = ("2D-A", "2D-B")
CLASSICS = ("C-R2D", "C-SEDOV", "C-KH")


def _slim(rec: dict) -> dict:
    skip = {
        "coord", "num", "exact", "field2d", "vx2d", "vy2d",
        "x", "y", "rho", "vx", "vy", "vz", "p",
    }
    out = {}
    for k, v in rec.items():
        if k in skip:
            continue
        if isinstance(v, float):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def run_riemann_matrix() -> list[dict]:
    recs = []
    by_case: dict[str, list[dict]] = {c: [] for c in CASES_1D + CASES_2D}
    for case in CASES_1D + CASES_2D:
        runner = RIEMANN_RUNNERS[case]
        for solver in SOLVERS:
            print(f"[cross] {case} × {solver} ...", flush=True)
            rec = runner(solver)
            recs.append(rec)
            by_case[case].append(rec)
            tag = f"{case.lower()}_{solver}"
            if case in CASES_1D:
                plot_lineout(rec, OUT / f"{tag}_lineout.png")
            else:
                plot_extruded_2d(rec, OUT / tag)
            print(
                f"         L1(n,v,P)/scale = {rec['L1_n']:.3e} {rec['L1_v']:.3e} {rec['L1_p']:.3e}"
                f"  steps={rec['nstep']}",
                flush=True,
            )
        plot_solver_overlay(by_case[case], OUT / f"{case.lower()}_overlay.png")
    return recs


def run_classics() -> list[dict]:
    recs = []
    by_case: dict[str, list[dict]] = {c: [] for c in CLASSICS}
    for case in CLASSICS:
        runner = CLASSIC_RUNNERS[case]
        for solver in SOLVERS:
            print(f"[classic] {case} × {solver} ...", flush=True)
            rec = runner(solver)
            recs.append(rec)
            by_case[case].append(rec)
            plot_classic_one(rec, OUT / f"{case.lower()}_{solver}.png")
            extra = ""
            if "circularity" in rec:
                extra = f"  circ={rec['circularity']:.4f}  R={rec['shock_R']:.3f}"
            if "ke_y" in rec:
                extra = f"  KE_y={rec['ke_y']:.3e}"
            print(f"          t={rec['t']:.3f}  steps={rec['nstep']}{extra}", flush=True)
        plot_classic_panel(by_case[case], OUT / f"{case.lower()}_panel.png")
    plot_sedov_radial(by_case["C-SEDOV"], OUT / "c_sedov_radial.png")
    return recs


def run_geom() -> list[dict]:
    recs = []
    print("[geom] G-CYL × hllc  (cylindrical R,z; φ-symmetric) ...", flush=True)
    cyl = GEOM_RUNNERS["G-CYL"]("hllc")
    recs.append(cyl)
    plot_cyl_rz(cyl, OUT / "g_cyl_hllc.png")
    print(
        f"         t={cyl['t']:.3f}  steps={cyl['nstep']}  "
        f"circ={cyl['circularity']:.4f}  R={cyl['shock_R']:.3f}  "
        f"max|v_φ|={cyl['max_vphi']:.2e}",
        flush=True,
    )
    print("[geom] G-SPH × hllc  (spherical r,θ; φ-symmetric) ...", flush=True)
    sph = GEOM_RUNNERS["G-SPH"]("hllc")
    recs.append(sph)
    plot_sph_rth(sph, OUT / "g_sph_hllc.png")
    plot_sph_lineout(sph, OUT / "g_sph_hllc_lineout.png")
    print(
        f"         t={sph['t']:.3f}  steps={sph['nstep']}  "
        f"θ-dev={sph['theta_dev']:.3e}  max|v_θ|={sph['max_vth']:.2e}  "
        f"max|v_φ|={sph['max_vphi']:.2e}",
        flush=True,
    )
    return recs


def run_conservation() -> list[dict]:
    """B3 closed-box conservation: same IC, reflect everywhere, track mass & energy."""
    recs = []
    print("[conservation] G-CYL closed box (reflect-all) ...", flush=True)
    cyl = GEOM_RUNNERS["G-CYL"]("hllc", conservation=True)
    recs.append(cyl)
    _print_conservation("G-CYL", cyl)

    print("[conservation] G-SPH closed box (reflect-all) ...", flush=True)
    sph = GEOM_RUNNERS["G-SPH"]("hllc", conservation=True)
    recs.append(sph)
    _print_conservation("G-SPH", sph)

    return recs


def _print_conservation(label: str, rec: dict) -> None:
    hist = rec.get("conservation", [])
    if not hist:
        print(f"  {label}: no conservation data", flush=True)
        return
    m0 = hist[0][2]
    e0 = hist[0][3]
    m_final = hist[-1][2]
    e_final = hist[-1][3]
    dm = abs(m_final - m0)
    de_rel = abs(e_final - e0) / abs(e0) if e0 != 0 else float("inf")
    # Check bitwise mass conservation
    mass_exact = all(h[2] == m0 for h in hist)
    print(f"  {label}: t={rec['t']:.4f}  steps={rec['nstep']}", flush=True)
    print(f"    mass  initial={m0:.15e}  final={m_final:.15e}  "
          f"|Δm|={dm:.3e}  bitwise={mass_exact}", flush=True)
    print(f"    energy initial={e0:.15e}  final={e_final:.15e}  "
          f"|dE/E0|={de_rel:.3e}", flush=True)
    if not mass_exact:
        # Find max mass deviation
        max_dm = max(abs(h[2] - m0) for h in hist)
        print(f"    mass max |Δm| over history: {max_dm:.3e}", flush=True)
    # Write history to file
    hist_path = OUT / f"conservation_{label.lower()}.csv"
    with open(hist_path, "w") as f:
        f.write("step,t,mass,energy,dm,de_rel\n")
        for step, t, m, e in hist:
            f.write(f"{step},{t:.10e},{m:.17e},{e:.17e},"
                    f"{abs(m - m0):.3e},{abs(e - e0) / abs(e0) if e0 else 0:.3e}\n")
    print(f"    history → {hist_path}", flush=True)


def write_tables(riemann_recs: list[dict], classic_recs: list[dict],
                 geom_recs: list[dict]) -> Path:
    """Numeric tables only. Narrative lives in REPORT.md (do not overwrite it)."""
    lines = ["# 本次 run.py 数字表", ""]
    if riemann_recs:
        lines += ["## 交叉 $L_1(n,v,P)/scale$", "",
                  "| 题 | hll | hllc | tvdlf |", "|---|---|---|---|"]
        for case in CASES_1D + CASES_2D:
            cells = [case]
            for s in SOLVERS:
                rec = next(r for r in riemann_recs if r["case"] == case and r["riemann"] == s)
                cells.append(f"{rec['L1_n']:.3e}, {rec['L1_v']:.3e}, {rec['L1_p']:.3e}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    if classic_recs:
        lines += ["## Sedov", "", "| solver | R | circ | steps |", "|---|---|---|---|"]
        for s in SOLVERS:
            rec = next(r for r in classic_recs if r["case"] == "C-SEDOV" and r["riemann"] == s)
            lines.append(f"| {s} | {rec['shock_R']:.4f} | {rec['circularity']:.4f} | {rec['nstep']} |")
        lines += ["", "## KH KE_y", "", "| solver | KE_y | steps |", "|---|---|---|"]
        for s in SOLVERS:
            rec = next(r for r in classic_recs if r["case"] == "C-KH" and r["riemann"] == s)
            lines.append(f"| {s} | {rec['ke_y']:.4e} | {rec['nstep']} |")
        lines.append("")
    if geom_recs:
        lines += ["## Axisymmetric geometry", ""]
        for rec in geom_recs:
            extra = ""
            if rec["case"] == "G-CYL":
                extra = f"circ={rec['circularity']:.4f} R={rec['shock_R']:.3f}"
            if rec["case"] == "G-SPH":
                extra = f"theta_dev={rec['theta_dev']:.3e}"
            lines.append(f"- {rec['case']} {rec['riemann']}: t={rec['t']:.3f} {extra}")
        lines.append("")
    path = OUT / "tables.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    only = sys.argv[1:]
    want_all = not only
    want_geom = want_all or "--geom" in only
    want_cross = want_all or "--cross" in only
    want_conservation = "--conservation" in only
    OUT.mkdir(parents=True, exist_ok=True)
    riemann_recs = run_riemann_matrix() if want_cross else []
    classic_recs = run_classics() if want_cross else []
    geom_recs = run_geom() if want_geom else []
    cons_recs = run_conservation() if want_conservation else []
    slim = {
        "riemann": [_slim(r) for r in riemann_recs],
        "classic": [_slim(r) for r in classic_recs],
        "geom": [_slim(r) for r in geom_recs],
        "conservation": [_slim(r) for r in cons_recs],
    }
    (OUT / "results.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    tables = write_tables(riemann_recs, classic_recs, geom_recs)
    print(f"wrote {tables}")
    print(f"narrative: {HERE / 'REPORT.md'}")
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
