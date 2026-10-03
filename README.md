<div align="center">

# ✦ Xenia

### Table-driven multi-fluid hydrodynamics for transonic atmospheric escape

**Declare fluids, chemistry, radiation and source terms in Python —
a lean C / Kokkos kernel carries them through the sonic point.**
[![Paper](https://img.shields.io/badge/Paper%201-arXiv%20forthcoming-blue)](#paper-and-release)

[![License: GPLv2](https://img.shields.io/badge/license-GPLv2-blue.svg)](#license)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](#quick-start)
[![Kokkos](https://img.shields.io/badge/Kokkos-4.7.04-orange)](#license)
[![Backends](https://img.shields.io/badge/backends-Serial%20·%20OpenMP%20·%20CUDA-76B900?logo=nvidia&logoColor=white)](#scope)
[![API](https://img.shields.io/badge/API-setup%20·%20step%20·%20fields%20·%20close-success)](#first-step)
[![Paper](https://img.shields.io/badge/Paper%201-arXiv%20forthcoming-blue)](#paper-and-release)

<!-- BANNER PLACEHOLDER — generated art belongs here.
     Authoring assets live in readme_writer/; final images commit to docs/assets/.
     Example once ready:
     <img src="docs/assets/xenia_banner.png" width="880" alt="Xenia — multi-fluid escape">
-->

```python
import xmf

run = xmf.setup(xmf.shock_tube())        # or your own config dict
while run.t < run.tstop:
    run.step()
hi = run.fields["HI"]
print(run.t, hi.n.shape, float(hi.n.max()))
run.close()
```

**Four calls are the whole public API:** `setup → step → fields → close`.

**Paper and release:** this repository is the public code release for
*Xenia: A Table-Driven Multi-Fluid Hydrodynamics Framework with Implicit
Stiff Sources for Planetary Atmospheric Escape*. The arXiv preprint link will
be added when the paper is posted.

</div>

---

## Seventy years of transonic escape

Almost seventy years ago, Eugene Parker showed that a million-degree corona
cannot sit still: it must expand, accelerate through a critical point, and
blow outward as a **supersonic wind** ([Parker 1958](https://doi.org/10.1086/146579)).
The same transonic mathematics decides the fate of planetary atmospheres.
Stellar XUV irradiation heats the upper atmospheres of close-in planets until
hydrogen flows outward, **crosses the sound barrier**, and escapes to space.

The physics was laid down in stages: the **energy-limited** framework for the
early Earth and Venus ([Watson, Donahue & Walker 1981](https://doi.org/10.1016/0019-1035(81)90101-9));
the first time-dependent **transonic escape models** built for extrasolar
atmospheres ([Tian et al. 2005](https://doi.org/10.1086/427204)); and
photoionization-heated winds of **hot Jupiters** with realistic heating and
cooling, which made the calculation a standard tool
([Murray-Clay, Chiang & Murray 2009](https://doi.org/10.1088/0004-637X/693/1/23)).
The **sonic point** — where the flow passes Mach 1 — became the structure
that mass-loss rates are measured by.

## Then the observations arrived

Escape is no longer a model-only affair:

- A **comet-like hydrogen cloud larger than the host star** around GJ 436b,
  seen in Lyman-α transit at 56% depth
  ([Ehrenreich et al. 2015, *Nature*](https://doi.org/10.1038/nature14501)).
- The prediction of the metastable **He I 10830 Å** line as a transit probe
  of escaping upper atmospheres ([Oklopčič & Hirata 2018](https://doi.org/10.3847/2041-8213/aaada9)).
- The **first helium detection** in an eroding exoplanet atmosphere,
  WASP-107b, losing 10¹⁰–3×10¹¹ g s⁻¹
  ([Spake et al. 2018, *Nature*](https://ui.adsabs.harvard.edu/abs/2018Natur.557...68S/abstract)).

Today helium surveys run from space and from the ground, and Lyman-α campaigns
keep growing. The question has shifted from *whether* atmospheres escape to
**what exactly escapes**: which species, at what rate, with what fractionation.

## Escape sculpts entire populations

What began as single-planet physics now sets the demographics of the whole
close-in population:

- The **radius valley** — a clean gap between ~1.5 and 2.0 R⊕ in the Kepler
  census, splitting rocky super-Earths from enveloped sub-Neptunes
  ([Fulton et al. 2017](https://doi.org/10.3847/1538-3881/aa80eb)).
- **Photoevaporation** as the sculptor: XUV-driven mass loss alone reproduces
  the valley's location and slope ([Owen & Wu 2017](https://doi.org/10.3847/1538-4357/aa890a);
  see the review by [Owen 2019](https://doi.org/10.1146/annurev-earth-053018-060246)).
- The **cosmic shoreline**: across the solar system and beyond, escape —
  insolation against escape velocity — draws the line between worlds that
  hold atmospheres and worlds that do not
  ([Zahnle & Catling 2017](https://doi.org/10.3847/1538-4357/aa7846)).
- Around M dwarfs, a violent pre-main sequence can strip oceans and pile up
  **abiotic O₂**, rewriting the rules of habitability
  ([Luger & Barnes 2015](https://doi.org/10.1089/ast.2014.1231)).

The full zoo of escape processes — Jeans escape, hydrodynamic winds, ion
pick-up, sputtering — is catalogued in the reviews of
[Gronoff et al. (2020)](https://doi.org/10.1029/2019JA027639) and
[Hazra (2025)](https://arxiv.org/abs/2502.18124). Escape is no longer an
exotic footnote; it is a first-class agent of planet evolution, and of the
search for life.

## The multi-fluid frontier — and the gap Xenia fills

Here is the catch: the new observables are **multi-species quantities**.
Helium and hydrogen escape together but not identically — and a single-fluid
model, which forces every species to share one bulk velocity, is structurally
blind to the difference. The frontier is moving composition-aware:

- **He/H mass fractionation falls out naturally** in a multi-fluid wind:
  heavier helium is dragged less efficiently by the hydrogen flow, with no
  helium depletion needed at the lower boundary
  ([Xing, Yan & Guo 2023](https://doi.org/10.3847/1538-4357/ace43f)).
- **Multispecies models with excited H and He** now reproduce the very
  signals observers measure, He 10830 and Hα
  ([Taylor et al. 2025](https://doi.org/10.3847/1538-4357/ade3c9)).
- **Multi-component H₂–H₂O winds** show escape can be far less effective
  than single-species estimates — composition changes the answer again
  ([Yoshida et al. 2022](https://arxiv.org/abs/2207.06570)).
- **Magnetic fields reshape the wind**: dipole magnetospheres confine and
  redirect the escaping plasma, altering the absorption signatures and
  coupling the planet to the stellar wind
  ([Villarreal D'Angelo et al. 2018](https://academic.oup.com/mnras/article/479/3/3115/5035846)).

Yet each of these studies runs on bespoke, single-purpose code. Community
codes leave this niche empty:

| Landscape today | What it covers |
|---|---|
| FARGO3D, Idefix | gas + pressureless dust — not many thermally pressured fluids |
| PLUTO, Athena++ | single fluid, or chemistry forced to one shared velocity |
| BATS-R-US | strong multi-ion MHD — not Python, not CUDA, not single-node friendly |

> **Many thermally pressured fluids + table-driven source terms + a friendly
> Python front end + a GPU-ready kernel.**

**Xenia exists to turn composition-aware escape from bespoke studies into a
tool anyone can configure in Python and run on a laptop or a single GPU.**
Fluids, reactions and sources are **tables you hand in**, not branches
compiled in. The first application is H/He atmospheric escape — but the
kernel does not know that atmosphere, and does not need to.

## Highlights

| | |
|---|---|
| 🧾 **Table-driven setup** | Fluids (mass, charge, kind), reaction stoichiometry, symbolic source terms, SED radiation beams and cross sections — declared in a plain Python config dict. |
| 🔒 **Four-call API** | `setup → step → fields → close`. Nothing else to learn, nothing else to call. |
| 🧬 **Compiled source IR** | Python compiles reactions and sources into a versioned opcode IR; the C kernel only evaluates it. No per-cell Python `eval`, no species strings in C, no hard-coded gravity. |
| ☀️ **`RadiationScan`** | Non-local radiative transfer sweep — SED × σ through the grid, thick-cell photon-conserving absorption — cleanly separated from the hyperbolic update. |
| ⚡ **Pressure-only electrons** | Electron pressure shared by charge into the ion Riemann problem: the cheap, robust closure used in Xing-style escape models. |
| 🌊 **Godunov hydrodynamics** | HLL / HLLC / TVDLF solvers, reconstruction and RK1–3 as configured, six-face boundaries, 1-D / 2-D / 3-D on one set of operators — the same path for every dimensionality. |
| 🚀 **Kokkos backends** | Serial, OpenMP, or one NVIDIA GPU via CUDA. Same kernel, chosen at build time. |
| ✅ **Contract-tested** | Toro / Sod L1 rulers, flux and RK regression suites, ABI/config contract tests; conservation regressions are written into every change card. |

## Quick start

Requires Python ≥ 3.10, a C++17 compiler, CMake and Ninja.
Kokkos 4.7.04 ships as a git submodule — clone with it.

```text
git clone --recurse-submodules https://github.com/YFG-tec/xenia.git
# already cloned without submodules?
git submodule update --init --recursive

python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
.venv\Scripts\python.exe -m pip install .
# Linux / WSL: use .venv/bin/python
```

One NVIDIA GPU (Linux):

```text
CMAKE_ARGS='-DXMF_CUDA=ON' .venv/bin/python -m pip install .
```

After editing C sources during development, the one supported sync command is
`python scripts/sync_xmf.py`.

## First step

`shock_tube()` is a one-fluid Sod-like tube (HLL + PCM + RK1). Replace the
dict to add fluids, `electron: {mode: "pressure-only"}`, radiation beams, or
`sources[]` strings — `setup()` parses names, units and stoichiometry, and
the kernel receives only the packed IR.

## Examples

Each example runs as `python examples/<name>/run.py` from the repo root.

| Group | Paths | What they show |
|---|---|---|
| 🎯 Verification rulers | `examples/hd_eval/`, `examples/flux_eval/`, `examples/rk_eval/` | Toro / Sod L1 accuracy, flux and time-integration regression, 2-D HD classics |
| ☀️ Radiation | `examples/rad_hd_1d/` | 1-D radial photoionization, radiation on/off |
| ⚡ Electrons | `examples/rad_h_pe_1d/` | H + pressure-only electrons |
| 🧪 Xing-style escape | `examples/rad_hhe_pe_1d/`, `examples/HD209458b/` | H/He ions + $P_e$; the first full chemistry table on a real escape setup |
| 🧭 Curvilinear | `examples/cyl_2d/` | cylindrical 2-D geometry |
| 🌌 Multi-fluid drift | `examples/mf_jet_2d/`, `examples/mf_blast_3d/` | species drift in 2-D / 3-D |

## Architecture

```text
              Python  ·  you describe the problem
   ┌──────────────────────────────────────────────────────────┐
   │  import xmf                                              │
   │    fluids[] · reactions[] · sources[] · radiation(SED,σ) │
   │    electron · grid · boundaries · time                   │
   │  setup() → step() → fields() → close()                   │
   └───────────────────────┬──────────────────────────────────┘
                           │  versioned flat struct + source IR
                           │  (no ini files, no species strings)
                           ▼
   ┌──────────────────────────────────────────────────────────┐
   │  C / Kokkos kernel                                       │
   │                                                          │
   │   RadiationScan      hyperbolic update     cell-local    │
   │   non-local, 1–2×    HLL / HLLC / TVDLF    stiff sources │
   │   per step           recon + RK1–3         J + LU solve  │
   │                                                          │
   │   ndim = 1 | 2 | 3   ·   Serial | OpenMP | CUDA          │
   └───────────────────────┬──────────────────────────────────┘
                           ▼
              examples/ · tests/ · science runs
```

The layering rule is absolute: **Python eats the semantics, the kernel only
sees tables.** New physics enters by adding read/write names and `kind`s to
the IR — never by changing opcodes, and never by teaching the C core what
an “H I” is.

## Release channel

This repository is the cleaned public release channel for Xenia Paper 1. It
publishes one reviewed snapshot and its release tag rather than the private
development history. The Paper 1 preprint is the methods manual for this
snapshot; its arXiv link will be added when the preprint is posted.

The source also contains a gated ideal-MHD superset inherited from the
development tree. It is not part of the validated Paper 1 interface and remains
reserved for the later Xenia magnetic-physics paper.

## Scope

**In this release:** multi-fluid Godunov (HLL / HLLC / TVDLF), reconstruction
and RK as configured, six-face boundaries, pressure-only electrons, compiled
source IR, radial `RadiationScan`, Kokkos Serial / OpenMP / single-GPU CUDA,
1-D through 3-D.

**Outside the Paper 1 validation scope:** MPI, ideal MHD / constrained
transport, AMR, viscosity, and resistivity. A gated ideal-MHD source superset
is present in the tree, but magnetic physics is not enabled or validated as
part of this release. The design leaves the doors open — three velocities,
`k,j,i` ordering, rays as cell sequences + $\Delta s$, and six-face boundaries
— without claiming those extensions here.

## The longer view

Escape physics is becoming a bridge between stellar activity, planet
evolution and habitability. Xenia's roadmap grows with that arc:

- **Magnetic fields (Paper 2).** The solar wind taught us that transonic
  outflows are magnetized. Fields confine and collimate the wind, reshape
  transit absorption, and couple the escaping atmosphere to the stellar wind
  and CMEs. The MHD desk brings **ideal MHD with constrained transport** to
  the same four-call API — multi-fluid, magnetized escape that is
  configurable rather than bespoke.
- **Radiation and chemistry as data (Paper 2).** Ray-traced radiation beyond
  the radial scan, and reaction tables compiled into the same IR — so a new
  chemistry network or SED is a config change, not a fork.
- **MPI / multi-node (Paper 2).** The cell-local design is MPI-ready by
  construction.
- **AMR + dissipation (Paper 3).** Resolving the sonic surface, shocks and
  cometary tails where they need it, with $\nu$ and $\eta$ where they matter.
- **Beyond the series — the vision.** Escape models that speak the language
  of observables: forward-predicted He 10830, Hα and Lyman-α transit
  signatures for the JWST-era surveys; escape coupled to demographics —
  carving the radius valley, locating the cosmic shoreline — and to the
  habitability inventory of rocky planets around M dwarfs. The table-driven
  kernel is built so that each of these is **a new table, not a new code**.

## If you want to read deeper

Start with the reviews: [Owen (2019)](https://doi.org/10.1146/annurev-earth-053018-060246)
for escape as the shaper of close-in planets;
[Gronoff et al. (2020)](https://doi.org/10.1029/2019JA027639) for the full
process inventory; [Hazra (2025)](https://arxiv.org/abs/2502.18124) for the
state of observations and models. The bibliography below collects the arc.

## Documentation

| File | Role |
|---|---|
| `docs/README.md` | map of the rest of `docs/` |
| `docs/ROADMAP.md` | product split, framework diagrams, 3-D readiness |
| `docs/KERNEL_AND_CHEMISTRY.md` | source IR, reactions and Jacobians |
| `docs/HARNESS.md` | local `.venv`, `sync_xmf.py`, tests |
| `PLUTO_XMF_USAGE.md` | manual for the historical PLUTO-based tree |

## Lineage

Xenia grew out of **PLUTO_XMF**, a multi-fluid layer on PLUTO developed for
XUV-driven H/He escape — the code behind
[Xing, Yan & Guo (2023)](https://doi.org/10.3847/1538-4357/ace43f).
This release is a standalone kernel: the same science lineage, without
shipping PLUTO. The historical PLUTO-based tree is preserved under
`SSH_220804/` for comparison only.

## References

**Reviews**

- Owen, J. E. 2019, *ARA&A*, 47, 67 — atmospheric escape and the evolution of close-in exoplanets. doi:[10.1146/annurev-earth-053018-060246](https://doi.org/10.1146/annurev-earth-053018-060246)
- Gronoff, G. et al. 2020, *JGR Space Phys.*, 125, e2019JA027639 — escape processes and planetary atmospheric evolution. doi:[10.1029/2019JA027639](https://doi.org/10.1029/2019JA027639)
- Hazra, G. 2025, *Rev. Mod. Plasma Phys.*, 9, 18 — recent observations and theoretical models. [arXiv:2502.18124](https://arxiv.org/abs/2502.18124)

**Transonic foundations**

- Parker, E. N. 1958, *ApJ*, 128, 664 — the transonic wind. doi:[10.1086/146579](https://doi.org/10.1086/146579)
- Watson, A. J., Donahue, T. M., & Walker, J. C. G. 1981, *Icarus*, 48, 150 — energy-limited escape. doi:[10.1016/0019-1035(81)90101-9](https://doi.org/10.1016/0019-1035(81)90101-9)
- Tian, F., Toon, O. B., Pavlov, A. A., & De Sterck, H. 2005, *ApJ*, 621, 1049 — transonic escape for exoplanets. doi:[10.1086/427204](https://doi.org/10.1086/427204)
- Murray-Clay, R. A., Chiang, E. I., & Murray, N. 2009, *ApJ*, 693, 23 — photoionized escape from hot Jupiters. doi:[10.1088/0004-637X/693/1/23](https://doi.org/10.1088/0004-637X/693/1/23)

**Observational milestones**

- Ehrenreich, D. et al. 2015, *Nature*, 522, 459 — a comet-like escaping cloud around GJ 436b. doi:[10.1038/nature14501](https://doi.org/10.1038/nature14501)
- Oklopčič, A., & Hirata, C. M. 2018, *ApJL*, 855, L11 — the He I 10830 Å window. doi:[10.3847/2041-8213/aaada9](https://doi.org/10.3847/2041-8213/aaada9)
- Spake, J. J. et al. 2018, *Nature*, 557, 68 — helium in an eroding atmosphere. [ADS](https://ui.adsabs.harvard.edu/abs/2018Natur.557...68S/abstract)

**Populations, demographics, habitability**

- Fulton, B. J. et al. 2017, *AJ*, 154, 109 — the radius valley. doi:[10.3847/1538-3881/aa80eb](https://doi.org/10.3847/1538-3881/aa80eb)
- Owen, J. E., & Wu, Y. 2017, *ApJ*, 847, 29 — the evaporation valley. doi:[10.3847/1538-4357/aa890a](https://doi.org/10.3847/1538-4357/aa890a)
- Zahnle, K. J., & Catling, D. C. 2017, *ApJ* — the cosmic shoreline. doi:[10.3847/1538-4357/aa7846](https://doi.org/10.3847/1538-4357/aa7846)
- Luger, R., & Barnes, R. 2015, *Astrobiology*, 15, 119 — extreme water loss and abiotic O₂ around M dwarfs. doi:[10.1089/ast.2014.1231](https://doi.org/10.1089/ast.2014.1231)

**Multi-species and multi-component escape**

- Xing, L., Yan, D., & Guo, J. 2023, *ApJ*, 953, 166 — multi-fluid He/H mass fractionation. doi:[10.3847/1538-4357/ace43f](https://doi.org/10.3847/1538-4357/ace43f)
- Taylor, A. R. et al. 2025, *ApJ*, 989, 68 — a multispecies escape model with excited H and He. doi:[10.3847/1538-4357/ade3c9](https://doi.org/10.3847/1538-4357/ade3c9)
- Yoshida, T. et al. 2022, *ApJ*, 934, 137 — multi-component H₂–H₂O escape. [arXiv:2207.06570](https://arxiv.org/abs/2207.06570)

**Magnetized escape**

- Villarreal D'Angelo, C. et al. 2018, *MNRAS*, 479, 3115 — magnetized winds and the escaping upper atmosphere of HD 209458b. [Oxford](https://academic.oup.com/mnras/article/479/3/3115/5035846)

**Methods and runtime**

- Mignone, A. et al. 2007, *ApJS*, 170, 228; Mignone, A. et al. 2012, *ApJS*, 198, 7 — PLUTO.
- Evans, C. R., & Hawley, J. F. 1988, *ApJ*, 332, 659 — constrained transport for ideal MHD.
- Edwards, H. C., Trott, C. R., & Sunderland, D. 2014, *JPDC*, 74, 3202; Trott, C. R. et al. 2022, *IEEE TPDS*, 33, 805 — Kokkos.

## Citing Xenia

If Xenia contributes to your research, please cite the multi-fluid
fractionation paper it grew from:

```bibtex
@article{Xing2023,
  title   = {The Mass Fractionation of Helium in the Escaping Atmosphere of HD 209458b},
  author  = {Xing, Lei and Yan, Dongdong and Guo, Jianheng},
  journal = {The Astrophysical Journal},
  volume  = {953},
  number  = {2},
  pages   = {166},
  year    = {2023},
  doi     = {10.3847/1538-4357/ace43f}
}
```

The methods paper and code manual for this release is *Xenia: A Table-Driven Multi-Fluid Hydrodynamics Framework with Implicit Stiff Sources for Planetary Atmospheric Escape*. It will be cited here, with its arXiv identifier, when the preprint is posted.

## License

Xenia is **GPLv2** (see `LICENSE`), following PLUTO. Kokkos is Apache-2.0
with LLVM exception and is not part of that grant; its license ships in
`third_party/kokkos/`.

---

<div align="center">

**中文导读** — Xenia（`import xmf`）是一个**表驱动的多流体跨声速逃逸工具**：
Python 里声明流体表、反应式、符号源项与 SED 辐射，C / Kokkos 核负责推进，
公开 API 只有 `setup → step → fields → close`。它站在一条七十年的弧线上：
从 Parker 的超声速太阳风，到系外行星的跨声速逃逸模型，再到 *Nature* 上的
氦与氢的观测时代——逃逸如今塑造着半径谷、宇宙岸线与宜居性的讨论。
单流体强制共速，看不见 He/H 分馏；多流体看得见（Xing, Yan & Guo 2023）。
本仓库是 Paper 1 的清洁公开发布快照：单机 HD + \(P_e\)/Xing 表 + 单卡 CUDA。
树中带有门控关闭的理想 MHD 超集，但不属于本 release 的验证接口。
许可证 GPLv2（Kokkos 为 Apache-2.0）。

</div>
