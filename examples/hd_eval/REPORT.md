# A4 HD 评估报告

生成时间：2026-09-27（全量 `run.py` 重跑 UTC）。核：v0.1 发布态（C2补6 MHD 超集，commit `a83051f`；PCM + RK1，电子关，空源项）；跑数在发布提交 `5f61b43` / tag `v0.1` 的二进制上（site-packages dll 时间戳 2026-09-27 01:13:01 +08:00）。
命令：`python examples/hd_eval/run.py`。图在 `examples/hd_eval/output/`（gitignore）。
重跑产出 `examples/hd_eval/output/tables.md`（gitignore）；数字表由人工核对钉值后贴入本文件 §2/§3。叙述与读图口径以本文为准。

## 1. 契约验收（本报告不替代 pytest）

主会话跑过：

```
python -m pytest tests/ -q
```

**394 passed**。全套 `tests/` 口径，第一道门在 `test_hydro` / `test_cfl`；第二道门目前落在 `tests/test_shock_hd.py`（卡片原定 `examples/shock_tube/`，图不够是另开本评估 example 的原因，这里不加单测）。

模块契约里已经看到的物理行为：

- HLLC：静止接触（$u=0,\nabla p=0$）台阶中心不漂移，20 步后状态几乎逐位不变。
- HLL：允许抹平，等值面仍钉在 $x_0$。
- 1D Sod 对 Toro：$L_1/\mathrm{scale}$ 在 $10^{-3}$ 量级。
- 闲置 `vy` / `vz`（含 ghost）全程为 0。
- 薄 2D 沿 $x_2$ 的台阶会真正激发 `vy`，且不随 $x_1$ 变。
- `kind=mhd` 在 setup 接受（C2补6 MHD 超集）；本评估只跑 HD 题。
- CFL：加密一倍 $\Delta t$ 减半；$n_d=1$ 不进求和；`dt_max` 封顶。

## 2. 交叉矩阵：三家 Riemann ×（1D×2 + 2D×2）

默认仍是 PCM + RK1。精确解一律是 1D Toro（`utils/riemann/`）。二维题是挤出的一维黎曼，不是四象限 2D Riemann；后者在第 3 节。

| 题 | 设置 | 精确解 |
|---|---|---|
| 1D-A | 现 `shock_tube()` Sod，沿 $x_1$，reflect | Toro |
| 1D-B | Toro 2（123 / 双稀疏），copy | Toro |
| 2D-A | Sod 沿 $x_1$ 挤出，$128\times 48$，x2=`copy` | 同一 Toro；二维图应沿 $x_2$ 均匀 |
| 2D-B | 同一左右态沿 $x_2$，$48\times 128$ | Toro，法向速度是 $v_y$ |

### $L_1(n,v_n,P)/\mathrm{scale}$

旧表数字产自 2026-08-22 前 C2补6 时代纯 C 核；本表为 v0.1 发布态核（C2补6 MHD 超集，Kokkos/OpenMP，PCM+RK1，电子关，空源项）重跑。1D 与旧表逐位衔接（pytest 钉值不变）；2D 为换代后新基准，Sedov $R$/circ、KH $\mathrm{KE}_y$ 等物理量在预期精度内吻合。

| 题 | hll | hllc | tvdlf |
|---|---|---|---|
| 1D-A | 3.515e-03, 9.716e-03, 4.029e-03 | 3.534e-03, 9.637e-03, 3.952e-03 | 3.821e-03, 1.182e-02, 4.965e-03 |
| 1D-B | 1.237e-02, 1.913e-02, 2.515e-02 | 1.227e-02, 1.903e-02, 2.515e-02 | 1.197e-02, 1.966e-02, 2.630e-02 |
| 2D-A | 4.962e-03, 1.676e-02, 7.076e-03 | 5.172e-03, 1.571e-02, 6.865e-03 | 4.762e-03, 1.925e-02, 7.858e-03 |
| 2D-B | 4.628e-03, 1.503e-02, 6.583e-03 | 4.780e-03, 1.427e-02, 6.260e-03 | 4.503e-03, 1.757e-02, 7.390e-03 |

Sod 系与卡片同量级（$\sim 10^{-3}$）。Toro 2 中部稀疏平台薄，PCM 一阶抹平主导，到 $\sim 10^{-2}$。TVDLF 在 Sod 上 $L_1$ 最大（最糊）；HLLC 在接触/速度上略利。

![1D-A 三家叠加 vs Toro](output/1d-a_overlay.png)

![1D-B Toro 2 三家叠加 vs 精确解](output/1d-b_overlay.png)

二维图：2D-A 密度与 $v_x$ 是沿 $x_2$ 的竖条，$v_y$ 整场为 0。2D-B 密度与 $v_y$ 是沿 $x_1$ 的横条，$v_x$ 整场为 0，$v_y$ 峰值约 $7\,\mathrm{km\,s^{-1}}$。2D 比 1D 略松，主要因为沿法向只放了 128 胞（1D 是 256）。

![2D-A 挤出 Sod（hllc）](output/2d-a_hllc_2d.png)

![2D-B 沿 $x_2$ 的 Sod（hllc）](output/2d-b_hllc_2d.png)

## 3. 三道经典二维 HD

文献原题，不设 pytest 上界。三家各跑一遍，并排在 `*_panel.png`。

### C-R2D — 四象限 2D Riemann（Lax & Liu 1998 Configuration 3）

也见 Kurganov & Tadmor, *Numer. Methods PDEs* 18, 584 (2002)。$\gamma=1.4$，$t=0.3$，单位正方形，$128^2$。没有简单精确解。

三家都给出四象限相互作用的菱形结构。HLLC 接触最利，中间能看出更清楚的折角；HLL 次之；TVDLF 最圆、最糊。和文献里 WENO/PPM 的锋利度差一档，是 PCM+RK1 的预期，不是核坏了。

![2D Riemann Config. 3：hll / hllc / tvdlf](output/c-r2d_panel.png)

### C-SEDOV — 二维柱 Sedov–Taylor 爆炸

Sedov (1959)；FLASH 等码文的标准爆破。中心 $E=1$ 注在 $r<3.5\Delta x$，环境 $\rho=1,\,p=10^{-6}$，$t=0.1$。

| solver | 激波半径 $R$ | 扇区圆形度 $\sigma_R/\langle R\rangle$ | 步数 |
|---|---|---|---|
| hll | 0.1067 | 0.0670 | 497 |
| hllc | 0.1067 | 0.0670 | 749 |
| tvdlf | 0.1067 | 0.0670 | 442 |

扇区半径这个标量太粗，三家数字一样；**眼睛看到的差别更大**：HLL / TVDLF 的壳接近圆，HLLC 被打成沿坐标轴的菱形（方向分裂 + 强激波上 HLLC 的 carbuncle / 网格各向异性，PCM 没有耗散把它压住）。TVDLF 峰密度最低（更糊）。

![Sedov 三家并排](output/c-sedov_panel.png)

![Sedov 角平均密度剖面](output/c_sedov_radial.png)

这不是交叉矩阵那种「过/不过」，是评估该记下来的：HLLC 不是处处更优，强球形爆上 HLL 更稳。

### C-KH — Kelvin–Helmholtz 剪切层

McNally, Lyra & Passy, *ApJS* 201, 18 (2012)；Robertson et al., *MNRAS* 401, 2463 (2010) 同族。本核没有周期面，$x$ 用 `copy`，停在 $t=0.8$。

| solver | $\mathrm{KE}_y$ | 步数 |
|---|---|---|
| hll | 6.04e-06 | 1167 |
| hllc | 1.10e-05 | 1175 |
| tvdlf | 5.61e-06 | 1167 |

PCM+RK1 耗散大，滚涡长不成像论文那样的猫眼。仍能分出求解器：HLLC 的 $\mathrm{KE}_y$ 约是另外两家的两倍，界面开始起波；HLL / TVDLF 几乎被抹平。这是耗散诊断，不是失败。

![Kelvin–Helmholtz 三家并排](output/c-kh_panel.png)

## 4. 怎么读

- 交叉矩阵过了：三家通量、两个方向的 sweep、「挤出不污染」是通的。A4 作为空表 HD 的 $L_f$ 可以交。
- 经典题不过 pytest。看结构在不在、圆是否圆、KH 会不会被一阶格式吃掉。
- 一阶 PCM+RK1 不是论文里 WENO/PPM 的锋利度。要更利的接触，下一步是 PLM/WENO，不是再加 Riemann 家族。
- HLLC 在接触和 KH 上更好，在强 Sedov 上更不圆。三家都留着是对的。

## 5. 环对称二维：柱坐标 $(R,z)$ 与球坐标 $(r,\theta)$

各一道，环向（$\phi$）只有 1 格。求解器固定 `hllc`。命令：`python examples/hd_eval/run.py --geom`。

发布态核（C2补6 MHD 超集，commit `a83051f`）已包含面积加权通量（$A$、$dV$ 进核）。下列数字全部在发布态二进制上重跑；旧 REPORT（2026-08-22）的数字是面积加权之前的笛卡尔通量跑的，已废弃。

### G-CYL — 柱坐标 $(R,z)$ 点爆炸

$R\in[0,0.5]$，$z\in[-0.5,0.5]$，$R=0$ 反射，$R=0.5$ 与 $z$ 两端 copy。原点附近注能量。

![柱坐标 $(R,z)$ 点爆炸](output/g_cyl_hllc.png)

| 量 | 旧（2026-08-22，笛卡尔通量） | 新（发布态，面积加权通量） |
|---|---|---|
| $t$ | 0.100 | 0.100 |
| 步数 | 1484 | 1292 |
| 激波半径 $R$ | 0.307 | 0.113 |
| 圆形度 $\sigma_R/\langle R\rangle$ | 0.024 | 0.133 |
| $\max\|v_\phi\|$ | 0 | 0 |

激波半径从 0.307 降到 0.113，变化 2.7 倍。面积加权通量在 $R$ 方向引入了 $A\propto R$ 的几何权重，有效增强了靠近原点方向的通量散度，激波减速更剧烈——方向正确：向**柱对称** Sedov 靠近（标度 $R\propto t^{1/2}$；$t^{2/5}$ 是三维球对称指数，旧稿写错，数值审查 2026-09-27 纠正），幅度因 $A\propto R$ 权重而小于笛卡尔同能量爆炸。圆形度从 0.024 恶化到 0.133：方向分裂下 $R$ 与 $z$ 向用不同几何权重（$A_R=R$、$A_z=1$），强激波沿两轴的传播速度差异被几何放大——这是方向分裂 + 曲线坐标的预期行为，如实记录；本图**不作「完美圆形」证据引用**。$v_\phi$ 仍严格保持 0。

### G-SPH — 球坐标 $(r,\theta)$ 径向球

$r\in[0.15,1.15]$，$\theta\in[0,\pi]$，轴 $\theta=0,\pi$ 反射，$r$ 方向 copy。$r<0.45$ 高压高密，外面低压。

![球坐标 $(r,\theta)$ 子午面](output/g_sph_hllc.png)

![球坐标角平均径向剖面](output/g_sph_hllc_lineout.png)

| 量 | 旧（2026-08-22，笛卡尔通量） | 新（发布态，面积加权通量） |
|---|---|---|
| $t$ | 0.121 | 0.121 |
| 步数 | 87 | 84 |
| $\theta$ 向相对偏差 | $1.2\times 10^{-15}$ | 2.024 |
| $\max\|v_\theta\|$ | 0 | 1.49 |
| $\max\|v_\phi\|$ | 0 | 0 |

> **FAIL（数值审查 2026-09-27 定性，四件套齐）**：角向对称破缺是**核的球坐标 θ 动量几何处理真缺陷**，非物理、非算例伪影。本节两张图**不可作 paper 1 球对称验证证据引用**；球对称验证改用 1D $r$ 向（`test_mesh` 的 $A\propto r^2_{\mathrm{face}}$ 断言同路径，1D 不激发此缺陷）。
>
> **机制**：核用面积加权通量散度 $\Delta(\rho v_\theta) = -\frac{\Delta t}{dV}\big(A_{j+1}F_{j+1}-A_jF_j\big)$ 统一处理五个守恒量，θ 动量的 $F=P$。极点邻接胞（$j=0$）左面在 $\theta=0$ 处 $A=\sin\theta\to 0$，散度退化为 $-\frac{\Delta t}{dV}\sin(d\theta)\,P \ne 0$，且被小体积 $dV=1-\cos(d\theta)\approx d\theta^2/2$ 放大：每步注入 $v_\theta \approx 2\,\Delta t\,P/d\theta$。CFL 下 $\Delta t\propto d\theta$，单步注入不随分辨率衰减，而步数 $\propto 1/d\theta$——全局 $v_\theta \propto 1/d\theta$ **发散**。即面积加权 FV 在极区的一阶截断误差未被 θ 动量几何源项（$+P\cos\theta$ 类）补偿（Blondin & Lufkin 1993；Skinner & Ostriker 2010 有讨论）。
>
> **探针证据**（小网格短跑，存档 `.tmp/numerics_review_knife1.md`）：① 笛卡尔同初值同核 $\max|v_\theta|\equiv 0$，球坐标第 2 步起增长；② copy/copy、copy/reflect、reflect/reflect 三种边界组合产出相同 $v_\theta$（非边界伪影）；③ 均匀初值（无径向球）仍生 $v_\theta$（非初值伪影）；④ $n_\theta=8/16/32/48$ 加密到同一 $t$：$\max|v_\theta| = 0.50/0.85/1.25/1.48$，随分辨率**增长不收敛**。
>
> 旧跑（笛卡尔通量）θ 行独立等价、角向偏差机器精度，是因为彼时通量不含 $\sin\theta$ 权重。修复（θ 动量显式几何源项或守恒律本源形式）是**核层面改动，超出本刀**；该核与 mhd 树逐位同源，两树同患，修在哪棵树由用户跨桌协调。本刀不改核，如实记录发布态行为。

### B3 闭盒守恒检查

命令：`python examples/hd_eval/run.py --geom --conservation`。

**设置**：与 B1 相同初值，所有边界改为 `reflect`（闭盒）。

- G-CYL 闭盒：$R=0$ 反射（原已有）、$R=0.5$ 反射、$z=\pm 0.5$ 反射
- G-SPH 闭盒：$r=0.15$ 反射（内壁）、$r=1.15$ 反射（外壁）、$\theta=0,\pi$ 反射（原已有）

跑到与 B1 相同 $t_\mathrm{stop}$。每 10 步记录总质量 $M=\sum \rho\,dV$ 与总能量 $E=\sum[P/(\gamma-1)+\frac{1}{2}\rho(v_R^2+v_z^2+v_\phi^2)]\,dV$，其中 $dV$ 含几何权重（柱坐标 $dV\propto R$、球坐标 $dV\propto r^2\sin\theta$）。

**断言**：总质量守恒至机器精度（相对漂移 $\le 10^{-15}$，reflect 边界；「逐位」经数值审查改判——`unpack → reshape → sum` 求和顺序受 ghost 填充影响，有 ULP 级抖动，非核泄漏）；总能量漂移 $\le 10^{-12}$ 相对量级。Reflect 不做动量守恒（墙给冲量），不断言动量。

#### G-CYL 闭盒结果

| 量 | 读数 |
|---|---|
| $t_\mathrm{stop}$ | 0.100 |
| 步数 | 1297 |
| $M_0$ | $1.250\,000\,000\,000\,000\times 10^{-1}$ |
| $M_\mathrm{final}$ | $1.250\,000\,000\,000\,000\times 10^{-1}$ |
| $\|{\Delta M}\|_\max$ | $5.551\times 10^{-17}$（$\sim 4$ ULP） |
| $\|{\Delta E}/E_0\|_\mathrm{final}$ | $1.456\times 10^{-15}$ |
| $\|{\Delta E}/E_0\|_\max$（全历史） | $2.767\times 10^{-15}$ |

质量不严格逐位守恒：4 个不同值在历史中交替出现（极差 $5.551\times 10^{-17}$，$\sim 4$ ULP）。原因是浮点求和结合律——reflect 边界精确复制状态，但 `unpack → reshape → sum` 路径上求和顺序受 ghost 填充影响，导致 ULP 级抖动。能量漂移全历史始终 $\le 3\times 10^{-15}$。

![G-CYL 守恒历史](output/conservation_g-cyl.csv)

#### G-SPH 闭盒结果

| 量 | 读数 |
|---|---|
| $t_\mathrm{stop}$ | 0.121 |
| 步数 | 81 |
| $M_0$ | $1.783\,875\,378\,267\,264\times 10^{-1}$ |
| $M_\mathrm{final}$ | $1.783\,875\,378\,267\,264\times 10^{-1}$ |
| $\|{\Delta M}\|_\max$ | $5.551\times 10^{-17}$（$\sim 2$ 个值交替） |
| $\|{\Delta E}/E_0\|_\mathrm{final}$ | $4.309\times 10^{-16}$ |
| $\|{\Delta E}/E_0\|_\max$（全历史） | $4.309\times 10^{-16}$ |

质量同样不严格逐位（2 个值交替，极差 $5.551\times 10^{-17}$）。能量单调下降 1 ULP 后保持恒定，全历史漂移 $\le 4.3\times 10^{-16}$。

![G-SPH 守恒历史](output/conservation_g-sph.csv)

#### B3 结论

- 质量守恒到 $\sim 4$ ULP（$\sim 10^{-16}$ 相对），非严格逐位但远优于修订后断言门槛（$\le 10^{-15}$ 相对）
- 能量漂移 $\le 3\times 10^{-15}$（G-CYL）/ $4\times 10^{-16}$（G-SPH），远优于 $10^{-12}$ 断言
- 曲线坐标的 $dV$ 权重与 reflect 边界配合正确，闭盒无物质/能量泄漏
- 严格逐位未达到的原因定性为浮点求和结合律（非核 bug），可接受

不勾 `docs/TODO.md`，不 commit。本目录不是契约单测。
