# Example: HD 209458b (PLUTO_XMF)

This example provides a self-contained input deck for running a 1D, spherical, multi-fluid hydrodynamic escape simulation for **HD 209458b**.

It is based on the configuration used in:
- Xing, L., Yan, D., & Guo, J. 2023, *ApJ*, 953, 166. doi:10.3847/1538-4357/ace43f

## Files

- `p_HD209458b_example.ini` — example parameter file
- `inputs/SED_lists/hd209458b-1163.dat` — XUV SED table used by the model
- `inputs/IC_files/HD209458b_IC0.dat` — initial condition file

## Build

From the solver directory:

```bash
cd SSH_220804
# IMPORTANT: make sure PLUTO_DIR in makefile is set to ./PLUTO
make clean
make
```

## Run

Still from `SSH_220804/`:

```bash
./pluto -i ../examples/HD209458b/p_HD209458b_example.ini
```

Outputs will be written to:

- `examples/HD209458b/output/run1/`

## Notes

- This example is intended as a minimal, reproducible starting point.
- For publication-grade results and parameter sweeps, you will likely want to adjust grid resolution, output cadence, and irradiation scaling parameters.
