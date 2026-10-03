# mhd_orszag_tang — 2D Orszag–Tang 涡（理想 MHD / CT）

标准 MHD 基准（Orszag & Tang 1979, JFM 95:119）。$[0, 2\pi]^2$ 周期域，
$\gamma=5/3$：

$$
\rho=\frac{\gamma^2}{4\pi},\quad P=\frac{5}{12\pi},\quad
\mathbf{u}=(-\sin y,\ \sin x,\ 0),\quad
\mathbf{B}=\frac{(-\sin y,\ \sin 2x,\ 0)}{\sqrt{4\pi}}.
$$

单支离子（$q=1$, $w_s=1$），电子关，磁场开——感应走 M5 的 CT 路径
（棱心 $E=-u_+\times\mathbf{B}$ 更新面心 B）。数值档与
`tests/test_m5_ct.py` 一致：hll / plm / rk3，cfl $= 0.4$。

停时默认 $t=\pi$（文献里常用的「激波与电流片长好了」的快照）。
合同钉仍是 $t=0.5$（短跑，不要拿长积分去对那条 $L_1$ 上界）。

```text
# 说明用：128² 演到 t=π，写出密度+磁压 GIF
python examples/mhd_orszag_tang/run.py --n 128 --animate --out examples/mhd_orszag_tang/output/n128_tpi

# 合同短跑（t=0.5，无动画）
python examples/mhd_orszag_tang/run.py --n 64 --tstop 0.5 --no-plot

# 已有 dump 重画
python examples/mhd_orszag_tang/plot.py --out examples/mhd_orszag_tang/output/n128_tpi
python examples/mhd_orszag_tang/animate.py --out examples/mhd_orszag_tang/output/n128_tpi
```

`--animate` 按时间均匀抓约 60 帧（`--frame-dt = tstop/60`），写入
`frames.npz` 和 `orszag_tang.gif`。只要密度、不要双联时，旧的
`--frames-every K` 仍可用。

128² 演到 $t=\pi$ 大约 740 步、墙钟约 40 s（本机 OpenMP）。展示动画锁名
`orszag_tang_tpi_128_showcase.gif`（`output/n128_tpi/`）。HLL 在长积分上
有耗散，$\Delta E/E$ 会到几个百分点；合同钉仍看 $t=0.5$ 的上界，不要拿
长跑去放宽那条。$\max\nabla\cdot B$ 仍在 $10^{-5}$ 量级。

## 墙钟 / 占用（Port A–D）

默认 $1000^2$、$t=0.1\pi$，大网格自动帽 80 步（占用不用跑满 $0.1\pi$）。
`OMP_NUM_THREADS` 默认吃满逻辑核。步进时旁路采样 CPU% / GPU%。
`run.py` 的 $\nabla\cdot B$ 跟进度打印一起采（每 20 步），避免报告口整场 pull 灌进墙钟。

```text
# Windows OpenMP（本树 .venv）
.venv\Scripts\python.exe examples/mhd_orszag_tang/bench.py --label cpu

# 关磁对照：同一涡、感应关掉，看 OpenMP 流体能不能把核吃满
.venv\Scripts\python.exe examples/mhd_orszag_tang/bench.py --label cpu-hd --no-mhd

# WSL CUDA（先在那棵树上 sync_xmf.py）
python3 examples/mhd_orszag_tang/bench.py --label cuda

# 已有 summary 拼表
.venv\Scripts\python.exe examples/mhd_orszag_tang/bench.py --compare
```

Port D 之后感应在设备核上。Windows OpenMP 对照（同一台 16 逻辑核）：

| | $n$ | 步 | Port C 墙钟 | Port D CPU | Port D CUDA | CPU% | GPU% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 开磁 | 128 | 70 | 4.0 s | 1.65 s | 0.66 s | 51/61 | 42/56 |
| 开磁 | 256 | 140 | 29.8 s | 8.47 s | 3.45 s | 64/84 | 66/77 |
| 开磁 | 1000 | 80 | 257 s | 82.9 s | 22.3 s | 75/98 | 83/99 |
| 关磁 | 1000 | 80 | 33.7 s | — | — | 69/91 | — |

$1000^2$ 开磁：CPU 相对 Port C 快 $3.1\times$，占用 29% → 75%；CUDA / CPU $=3.7\times$。$n=512$ 表里若还是 200 s 那一档，是 Port D 前的旧数，不要拿来比。

产物在 `--out` 目录（默认 `output/`，gitignore）：

- `fields.npz` — 末态 $\rho, \mathbf{u}, P, \mathbf{B}$
- `summary.json` — 步数、$t$、墙钟、全程 $\max\nabla\cdot B$、能量漂移、$L_1$
- `orszag_tang.png` — 末态密度 / 压强 / 磁压 / $J_z$
- `orszag_tang.gif` — 密度 + 磁压双联（`--animate`）
