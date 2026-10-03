# 1D HLL × SSP RK × CFL

固定 HLL + PCM + Sod，只换时间积分和 CFL。\(t_\mathrm{stop}=0.60\)（默认管的 4 倍）。看精度，以及 \(\Delta t\) 拉大（含 CFL>1）时解怎么坏。

```text
python examples/rk_eval/run.py
```

图和表在 `output/`（gitignore）。报告：`REPORT.md`。
