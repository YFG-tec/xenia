# 2D 圆柱绕流

均匀来流打过直径 $D=1$ 的圆盘。核里没有实体、也没有物理 $\nu$（那是 paper 2 的 P 块）；圆盘是算例层每步钉死的无滑移掩膜，尾迹雷诺数是数值的。题型跟 Braza, Chassaing & Ha Minh 1986 的圆柱近尾迹、Williamson 1996 综述同一条线。短时横向扰动用来尽快起涡街。

```text
python examples/cyl_2d/run.py --no-plot
python examples/cyl_2d/plot.py --anim
```

推进写 `output/fields.npz`、`snaps.npz`、`summary.json`。改图不用重算。

- 图：`wake.png`、`street.png`、`street.gif`（涡量动画）
- 报告：`examples/cyl_2d/REPORT.md`
