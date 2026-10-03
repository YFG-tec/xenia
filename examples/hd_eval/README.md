# HD 评估 example（A4）

契约单测仍在 `tests/`。这里是给人看的交叉求解和经典二维题。

```text
python examples/hd_eval/run.py
```

- 图：`examples/hd_eval/output/`（gitignore）
- 报告：`examples/hd_eval/REPORT.md`（跑完按实测 \(L_1\) 重写）

交叉：`hll` / `hllc` / `tvdlf` ×（1D-A Sod、1D-B Toro 2、2D-A 挤出、2D-B 沿 \(x_2\)）。二维题有二维图，不只线出。

经典二维 HD（文献原题，无简单精确解）：

1. Lax & Liu (1998) Configuration 3 四象限 Riemann
2. 二维柱 Sedov–Taylor 爆炸
3. McNally et al. (2012) 型 Kelvin–Helmholtz

环对称二维（$A$/$dV$ 已在网格里，$L_f$ 仍用 $\Delta x$）：

```text
python examples/hd_eval/run.py --geom
```

- 柱坐标 $(R,z)$ 点爆炸，`n_φ=1`
- 球坐标 $(r,\theta)$ 径向球，`n_φ=1`
