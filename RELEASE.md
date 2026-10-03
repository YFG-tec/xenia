# Xenia v0.1.1-paper.1 release

This is a cleaned public code release for the Paper 1 Xenia methods manuscript:

**Xenia: A Table-Driven Multi-Fluid Hydrodynamics Framework with Implicit
Stiff Sources for Planetary Atmospheric Escape**

The arXiv preprint link will be added to `README.md` and this file when the
paper is posted.

## Release form

This repository publishes **one clean release snapshot**, not the private
development history. The snapshot contains the Xenia implementation, tests,
examples, public documentation, build files, and the Kokkos submodule pointer.

- Public repository: <https://github.com/YFG-tec/xenia>
- Public release tag: `v0.1.1-paper.1`
- Public snapshot branch: `master`
- Source tree used to assemble the snapshot: Xenia HD `feature/hd` commit
  `330a91d491d632f2e29799a980baef1bb160e5f8`
- Canonical numerical-evidence release commit: `5f61b43d339caec1320d074efebe37116aad5c2e`
- Canonical numerical-evidence tag in the development repository: `v0.1`
- E1--E5 evidence commit: `8bb6299`
- Release-state `hd_eval` report refresh: `89607bf`

The public tag is intentionally distinct from the private development `v0.1`
tag. It identifies the cleaned snapshot associated with the Paper 1 preprint
while preserving the canonical commit provenance above.

## What Xenia contains

Xenia is a table-driven multi-fluid hydrodynamics framework for transonic,
radiation-coupled planetary atmospheric escape. It provides:

- a four-step Python API: `setup`, `step`, `fields`, and `close`;
- versioned Python problem declarations and symbolic source tapes;
- a C/C++ Kokkos kernel with Serial, OpenMP, and single-device CUDA builds;
- pressure-only quasi-neutral electron pressure support;
- a radial Beer--Lambert `RadiationScan`;
- chemistry, drag, heating, cooling, and collision source tables;
- analytic forward-AD source Jacobians for cell-local implicit solves;
- validation examples and a pytest suite.

Distributed-domain MPI and ideal-MHD physics are outside this Paper 1 release.

## Build and test path

A concise quick start is provided in `README.md`; the local harness is described
in `docs/HARNESS.md`. The standard supported sequence is:

```text
python -m venv .venv
<activate the selected interpreter>
python -m pip install --upgrade pip
python -m pip install .[dev]
python scripts/sync_xmf.py
python -m pytest tests -q
```

`scripts/sync_xmf.py` is the supported way to place a newly built kernel into
the live Python import tree.

## License

Xenia source code in this repository is released under GPLv2; see `LICENSE`.

Kokkos is included as a Git submodule at:

```text
third_party/kokkos
```

The recorded Kokkos commit is `82799e4577568f9666bde36265ac15d78da3e6c8`.
Kokkos carries its own Apache-2.0 license with LLVM exception; it is not
redistributed under the Xenia GPLv2 grant.

## Provenance boundary

The numerical evidence quoted in Paper 1 was generated with the canonical v0.1
release binary/source state at commit `5f61b43`. Evidence reports were archived
in the development history at `8bb6299`, with the release-state `hd_eval`
refresh at `89607bf`. This cleaned public snapshot preserves the implementation
and example/report surface while omitting private development workflow files
and the legacy reference tree.