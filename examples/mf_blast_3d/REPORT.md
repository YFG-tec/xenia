# 3D 两流体爆破

空源项两流体欧拉方程，笛卡尔盒 $[-1/2,1/2]^3$，copy 六面。中心球
$r < 3.5\,\Delta x$ 里放入能量 $E=1$，环境压 $p=10^{-6}$，$\gamma=1.4$。
轻支 $\rho=1$、重支 $\rho=4$，初速为零。没有辐射、没有碰撞带，两支
只通过各自的 Godunov 推进；轻壳跑得快，重壳落后，中截面上能看见
$|\Delta\mathbf{u}|$。Riemann 用 `tvdlf`（局部 Lax–Friedrichs / Rusanov）：
HLLC+PCM 在笛卡尔轴对齐激波上会把球壳拧成菱形，`tvdlf` 更耗散、各向更匀。

$$
p_{\mathrm{in}} = (\gamma-1)\,E\Big/\Big(\tfrac{4}{3}\pi r_0^3\Big)
$$

三速度都非零，因为球壳沿 $x,y,z$ 同时张开。这是 D3 要证明的：
`ndim=3` 和 1D/2D 同一套 `setup → step → fields → finalize`。

不是行星逃逸，也没有磁场。1D 开门柱只有一条横向线，GPU 闲着；这里
每向 $n^2$ 条线（默认 $64^3$ → $4096$），才谈得上和主机比墙钟。

场写在 `output/fields.npz`，图由 `plot.py` 单独画：`slices.png`
（三中截面密度 + $|\Delta\mathbf{u}|$）、`shell.png`（$z$ 中面 $\log\rho$）、
`cutaway.png`（$xy$、$xz$、$yz$ 中截面交于原点的剖视）。
