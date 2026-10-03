# 3D 两流体爆破（D3）

笛卡尔 $n^3$ 盒、空源项、中心一点能量。求解器是 `tvdlf`（Rusanov）：HLLC+PCM 在轴对齐激波上容易把球壳拧成菱形。轻/重两支各自推进，壳层错开，$v_x,v_y,v_z$ 都非零。用来看 `ndim=3` 仍是四步 API，并给 GPU 足够的横向线数。

```text
python examples/mf_blast_3d/run.py --no-plot
python examples/mf_blast_3d/plot.py
```

推进只写 `output/fields.npz` + `summary.json`。改图不用重算。

- 图：`slices.png`、`shell.png`、`cutaway.png`（三正交中截面剖视）
- 报告：`examples/mf_blast_3d/REPORT.md`

默认 $64^3$（每向扫 $4096$ 条线）。再小的网格 CUDA 吃不饱。不是行星大气，不是 Xing 科学图。
