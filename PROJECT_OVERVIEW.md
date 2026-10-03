# PLUTO_XMF (PLUTO-XMF) — Project Overview

**TL;DR**: PLUTO_XMF is a **multi-fluid extension** of the computational astrophysics code **PLUTO** (Mignone et al. 2007, 2012), developed to model **XUV-driven hydrodynamic escape** of hydrogen–helium exoplanet atmospheres and to capture **He/H mass fractionation** self-consistently.

## Motivation

He I 10830 Å transit spectroscopy has become a major probe of extended, escaping atmospheres. A recurring modeling outcome is that several systems appear to require an effectively subsolar He/H ratio in the escaping region.

In **Xing, Yan & Guo (2023, ApJ 953:166)** we demonstrated that, in a **multi-fluid** outflow, helium can be less efficiently dragged by the escaping hydrogen due to its higher mass, producing a naturally low escaping helium-to-hydrogen flux ratio—without requiring helium depletion at the lower boundary.

Because **single-fluid** hydrodynamics enforces a shared bulk velocity, this kind of fractionation cannot be captured in a single-fluid framework.

### Community interest (citations)

As of **2026-03-27** (accessed via NASA ADS), the paper **Xing et al. (2023)** has **11 citations**:
- ADS: https://ui.adsabs.harvard.edu/abs/2023ApJ...953..166X/abstract

## What is in this repository (current)

- `README.md` — short description
- `papers/` — reference paper PDF(s)
- `SSH_220804/` — the main solver tree (PLUTO + XMF multi-fluid layer + input decks)
  - `PLUTO/` — PLUTO source tree (GPLv2)
  - `mmd_*` — multi-fluid framework (states, state machine, Riemann solvers, tools)
  - `p_*.ini`, `pluto.ini`, `definitions.h` — example configurations (e.g., HD 209458b)
  - `j_*.pbs` — example HPC batch scripts

## Key capabilities

- **Multi-fluid formulation**: species-dependent velocities and thermodynamics, enabling drift between H and He and thus **mass fractionation**.
- **1D spherical outflow setups** (typical configuration): suitable for parameter surveys and quasi-steady escape solutions.
- **XUV-driven heating + source terms**: photoionization/recombination and collisional momentum/energy exchange couplings.
- **Built on PLUTO**: reuses PLUTO’s reconstruction/time stepping and optional MPI parallelization.

## Getting started

See the short manual: `PLUTO_XMF_USAGE.md`.

## License

PLUTO is distributed under **GNU GPL v2** (see `SSH_220804/PLUTO/COPYING`). Because PLUTO_XMF is a derivative work distributed together with PLUTO, this repository is released under **GPLv2** as well (see `LICENSE`).

## References

- Mignone, A. et al. 2007, *ApJS*, 170, 228 (PLUTO)
- Mignone, A. et al. 2012, *ApJS*, 198, 7 (PLUTO)
- Xing, L., Yan, D., & Guo, J. 2023, *ApJ*, 953, 166, doi:10.3847/1538-4357/ace43f
