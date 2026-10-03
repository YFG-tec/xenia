# 2D 两流体板状喷流（D1）

轻束打进重背景，只推轻的。入口是 tanh 剪切层，横向用小幅正弦拍打，好出 Kelvin–Helmholtz 涡。空源项：涡是轻束自己的剪切，不是碰撞带。题型跟 Ferrari, Trussoni & Massaglia 1982 以及 Bodo 等人的二维板状喷流 KH 同一条线。

```text
python examples/mf_jet_2d/run.py --no-plot
python examples/mf_jet_2d/plot.py
```

推进只写 `output/fields.npz` + `summary.json`。改图不用重算。

- 图：`jet.png`（$\rho$、 $|\Delta\mathbf{u}|$、$\omega_z$）、`beam.png`
- 报告：`examples/mf_jet_2d/REPORT.md`
