# 1D HLL × 面插值（pcm / plm / weno3）

固定 HLL + RK1 + Sod，只换 `numerics.reconstruction`。\(t_\mathrm{stop}=0.60\)（默认管的 4 倍）。看面左右态怎么改激波管，以及加密时 \(L_1\) 怎么降。附录再跑一组 HLL + RK3，同一套重建。

```text
python examples/flux_eval/run.py
```

图和表在 `output/`（gitignore）。报告：`REPORT.md`。
