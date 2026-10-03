# rad_hhe_pe_1d 结果报告（B3：四流体 + 双吸收体 + 电荷加权 $n_e$）

默认：`nx = 128`，`ng = 2`，`n_beam = 53`（整表，无单束回退），
`tstop = 2.0e5 s`，`dt_max = 10.0 s`（稳定帽，见 §2）。
HLL + PLM + RK2，`cfl = 0.4`，球对称 $r \in [R_p, 10 R_p]$，
log 拉伸（factor = 1.025，首末 $\Delta x/R_p = 9.96\times10^{-3}$ /
0.229，比 23.0）。全程 cgs，SED / 电离截面**引用**
`../rad_hd_1d/data/`，不拷贝（$\sigma_{\mathrm{He}}$ 与 $\sigma_{\mathrm{H}}$
同文件 `photoIon_sigma.dat` 的两段）。数字来自首登
`python examples/rad_hhe_pe_1d/run.py`（当时开/关辐射各跑到
$t_{\mathrm{stop}}$，各 20000 步）。**以后默认只推进 radiation ON**；
下文 radiation off 是那次快照，`run.py` 不再重算。工厂仍接受
`radiation=False`，留给契约测。

本算例验证的是 **B3 合同**，不是完整逃逸物理：

- 四组分流体 $\mathrm{HI}/\mathrm{HEI}/\mathrm{HII}/\mathrm{HEII}$
  （质量 $1.00794 / 4.002602 / 1.00794 / 4.002053\,\mathrm{amu}$，
  电荷 $0/0/1/1$，kind neutral/neutral/ion/ion；HEII = He 原子减一个
  电子质量 $5.4858\times10^{-4}\,\mathrm{amu}$）。电子仍是独立压强块：
  `electron = {"mode": "pressure-only", "share": "by-charge",
  "shock_heat": "ions-only"}`，`num_states = 5*4 + 1 = 21`
  （$P_e$ 殿后，不是第五支动量流体）。
- 准中性派生读与电荷加权电子速度（核内只认 $q_s \ne 0$，从不认名字）：

$$
n_e = n_{\mathrm{HII}} + n_{\mathrm{HEII}},\qquad
v_e = \frac{n_{\mathrm{HII}} v_{\mathrm{HII}} + n_{\mathrm{HEII}} v_{\mathrm{HEII}}}{n_e}.
$$

  带子里**没有** `vx_e` 读名。
- **第二吸收体**（B3 追加辅助口 `xmf_rad_load_absorbers`）：
  `radiation.absorbers = ["HI", "HEI"]`，每吸收体一行 $\sigma$，
  逐束光深 $\tau_b = (\sigma_{\mathrm{H},b} n_{\mathrm{HI}}
  + \sigma_{\mathrm{He},b} n_{\mathrm{HEI}})\,\Delta s$；
  单吸收体路径不经过多吸收体口（test_scan 钉逐位不变）。
- 完整 yardstick 源表（`SSH_220804/mmd_model_src.c`），每元素对先求
  **一份化学和**再做 CHE 压强转移 / 速度继承：

$$
\mathrm{ion_H} = \mathrm{photo_H} + \mathrm{CX1} + \mathrm{eimp_H},
\qquad \mathrm{rec_H} = \mathrm{rec_{H,photo}} + \mathrm{CX2},
$$

$$
\mathrm{ion_{He}} = \mathrm{photo_{He}} + \mathrm{CX2} + \mathrm{eimp_{He}},
\qquad \mathrm{rec_{He}} = \mathrm{rec_{He,photo}} + \mathrm{CX1}.
$$

  电荷交换（yardstick :84-94，启动门 $\min(1,e^{t-20})$ 锁 1）：

$$
\mathrm{CX1}:\ \mathrm{HI}+\mathrm{HEII}\to\mathrm{HII}+\mathrm{HEI},
\quad k_1 = 1.25\times10^{-15}\,(300/T)^{-0.25},
$$

$$
\mathrm{CX2}:\ \mathrm{HII}+\mathrm{HEI}\to\mathrm{HI}+\mathrm{HEII},
\quad k_2 = 1.75\times10^{-11}\,(300/T)^{0.75}\,e^{-128000/T},
$$

  $T$ 是配对的质量加权约化温度 $(T_1 m_2 + T_2 m_1)/(m_1+m_2)$。
- 碰撞：HI–HII / HEI–HEII 共振，HI–HEI / HII–HEII 异种对，
  离子–电子库仑 $\nu_e = 0.0299\,(T_e+T_{\mathrm{floor}})^{-3/2}$
  （带内写成 $1/(\sqrt{T_e}\,T_e)$）。库仑压强带是完整 COL_PRS_TRANS
  （B2 单离子时恒 0 的摩擦热 $\tfrac{2}{3} m_e n_{\mathrm{ion}} n_e
  (v_{\mathrm{ion}}-v_e)^2$ 这次必须跟着走）；$dP_e$ 带是两条离子侧
  带的精确反号，库仑不写任何 `dvx_*`。

## 1. 初值与边界

`pack_ic` 在 `setup` 之后按格心半径填四支（A5 的薄指数包络）：

$$
n_s(r) = \max\bigl(n_{\mathrm{beg}}\,f_s\,e^{-(r-R_p)/H_{\mathrm{ic}}},\ n_{\mathrm{ic,min}}\,f_s\bigr),
\quad H_{\mathrm{ic}} = 0.04\,R_p,\quad n_{\mathrm{ic,min}} = 10^{6},
$$

$$
f_{\mathrm{HI}} = 1-Y,\quad f_{\mathrm{HEI}} = Y,\quad
f_{\mathrm{HII}} = 10^{-3}(1-Y),\quad f_{\mathrm{HEII}} = 10^{-3}Y,
$$

$$
n_{\mathrm{beg}} = 10^{13}\,\mathrm{cm^{-3}}\ \text{（H+He 总量）},\quad
Y = \mathrm{HE\_RATIO} = 0.0776441,\quad \mathrm{HE\_SH\_COEF} = 1.0,
$$

$$
T = 1500\,\mathrm{K},\quad v_r = 0,\quad P_s = n_s k_{\mathrm{B}}T,\qquad
P_e = (n_{\mathrm{HII}}+n_{\mathrm{HEII}})\,k_{\mathrm{B}}T\ \text{（准中性）}.
$$

`x1beg=user`：解析带子只给 **ghost** 赋四支流体的 $(n, v=0, P)$
（底层钉 $n_{\mathrm{beg}}$ 配方），$P_e$ 的 ghost 走既有 copy@1。
**x1end 按辐射开关分岔**（沿用 B2 补的形状）：

- 开门：`user`，旧树 `outflow_BC`，四流体全带；
  $g=x_{\mathrm{IEND}}/x_{\mathrm{IEND}-1}$，

$$
n = n_{@1}^{1+g}/n_{@2}^{g},\quad
P = P_{@1}^{1+g}/P_{@2}^{g},\quad
v_r = \max\bigl(0,\,(1+g)v_{@1}-g\,v_{@2}\bigr),
$$

  `max` 写成 $0.5(v+\sqrt{v^2})$。两层 ghost 共用同一对 `@1/@2`。
- 关门：默认 `outflow`（核里 = copy）。

| 量（IC） | 底 | 顶 |
|---|---|---|
| $n_{\mathrm{HI}}$ (cm⁻³) | 8.144e12 | 9.224e5 |
| $n_{\mathrm{HEI}}$ (cm⁻³) | 6.855e11 | 7.764e4 |
| $P_e$ (erg/cm³) | 1.829e-3 | 2.071e-10 |
| $a_{\mathrm{net}}=-GM_p/r^2+3GM_\star r/a^3$ (cm s⁻²) | −902.8 | +113.9 |

（与 B2 的差别只在 $n_{\mathrm{beg}}$ 现在是 H+He 总量：
$n_{\mathrm{HI}}$ 底从 8.829e12 变为 $8.829\times10^{12}\times(1-Y)$；
$P_e$ 底数值不变，因为 $n_e = 10^{-3}\,n_{\mathrm{beg}}$ 与 B2 同。）

## 2. 跑况与稳定帽 `dt_max = 10 s`

| | radiation on | radiation off |
|---|---|---|
| 步数 / 墙钟 | 20000 / 1083 s | 20000 / 997.8 s |
| $t_{\mathrm{final}}$ (s) | 2.0000e5 | 2.0000e5 |
| 末步 dt (s) | 10.0（= dt_max） | 10.0（= dt_max） |
| source_resid 末步 / 历史最大 | 1.025 / 3.985 | 1.031 / 3.932 |

`dt_max = 10 s` 是本刀钉死的**稳定帽**，不是物理参数：源步合同是
「单步线性化后向欧拉，一次牛顿」（yardstick `get_du` 同款），启动段
自由落体弛豫里 CX / 动量装载 / $T$ 激活通道有一个 $\sim 10\,\mathrm{s}$
尺度的瞬态增长模式，单步线性化要求 $\Delta t\,\lambda < 1$。nx = 128
的实测：裸 CFL dt $\approx 22\,\mathrm{s}$ 发散；$\Delta t = 18\,\mathrm{s}$
一步内 $A$ 近奇异、`xmf_step` 返回 $-1$；$\Delta t = 15\,\mathrm{s}$
能跑但瞬态鼓包贴近增长上界；$10\,\mathrm{s}$ 全程场有限并留 $\sim 1.5$
倍裕度。CFL 公式本身不变（`cfl = 0.4`），B3 硬门的电子伪扫描信号
$|u_{e,d}|+c_e$ 已并进 `cell_signal_sum`（带电流体 $\ge 2$ 才启用，
单离子态逐位不变）。

`source_resid` 是**报告量**：归一一步残差，rel ≈ 1 对应硬源项把某分量
一步抽穿、被 $10^{-30}$ 地板接住的钳位事件；历史最大 3.99/3.93 与 B2
的 4.47/5.67 同级，所有场全程有限，不是发散。

每步墙钟 $\approx 54\,\mathrm{ms}$（B2 是 $\approx 9.5\,\mathrm{ms}$）：
21 态稠密牛顿、26 条源带、双吸收体扫描，都是按合同新增的开销。

## 3. 辐射开/关对比

| 量（末态） | on | off |
|---|---|---|
| $n_{\mathrm{HI}}$ 底 / 顶 (cm⁻³) | 6.688e12 / 822.0 | 6.688e12 / 642.1 |
| $n_{\mathrm{HEI}}$ 底 / 顶 (cm⁻³) | 4.553e11 / 1.351 | 4.554e11 / 1.329 |
| $n_{\mathrm{HII}}$ 底 / 顶 (cm⁻³) | 2.985e9 / 4718 | 2.985e9 / 245.4 |
| $n_{\mathrm{HEII}}$ 底 / 顶 (cm⁻³) | 5.783e7 / 317.8 | 5.783e7 / 0.147 |
| $n_{\mathrm{HII}}/n_{\mathrm{HI}}$ 底 / 顶 | 4.46e-4 / **5.74** | 4.46e-4 / 0.382 |
| $n_{\mathrm{HEII}}/n_{\mathrm{HEI}}$ 底 / 顶 | 1.27e-4 / **235** | 1.27e-4 / 0.111 |
| $x_{\mathrm{ion,H}} = n_{\mathrm{HII}}/(n_{\mathrm{HI}}+n_{\mathrm{HII}})$ 顶 | **0.852** | 0.276 |
| $x_{\mathrm{ion,He}} = n_{\mathrm{HEII}}/(n_{\mathrm{HEI}}+n_{\mathrm{HEII}})$ 顶 | **0.996** | 0.0997 |
| $v_{r,\mathrm{HI}}$ 最小 / 最大 (cm/s) | −6.05e4 / 2.959e6 | −2.52e5 / 2.798e6 |
| $v_{r,\mathrm{HII}}$ 最小 / 最大 (cm/s) | −6.09e4 / 2.975e6 | −1.55e5 / 2.813e6 |
| $T_{\mathrm{HI}}$ 底 / 顶 (K) | 1539 / 1733 | 1539 / 298 |
| $T_e$ 底 / 顶 (K) | 1529 / 1963 | 1528 / 299 |
| $P_e$ 底 / 顶 (erg/cm³) | 6.421e-4 / 1.365e-9 | 6.420e-4 / 1.012e-11 |
| $P_e$ 全域最小 / 最大 | 1.365e-9 / 6.421e-4 | 9.860e-12 / 6.420e-4 |

on/off 开关比（取样半径 $r/R_p = 1.005 / 2.561 / 9.885$）：

| 比 | 底 | 中 | 顶 |
|---|---|---|---|
| $n_{\mathrm{HII}}$ on/off | 1.000 | **7.75** | **19.2** |
| $n_{\mathrm{HEII}}$ on/off | 1.000 | **869** | **2160** |

- **辐射推动流体成立**：开门跑顶层 H 仍维持 $4.7\times10^{3}\,\mathrm{cm^{-3}}$
  量级的电离风（$x_{\mathrm{ion,H}}=0.85$），He 顶层几乎全电离
  （$x_{\mathrm{ion,He}}=0.996$）；中层 $n_{\mathrm{HEII}}$ 开关比 869。
  关门跑没有光致电离源，顶层电离只剩热机制（电子碰撞 + CX），
  $x_{\mathrm{ion,H}}$ 掉到 0.28。
- 底层比 ≈ 1 与 B2 同理：两跑 `x1beg` user_bc 都在补四支的底态。
- $P_e$ 读口全程可用（`fields` 可取，上表即 `xmf_state_copy_pe` 读回）。

![密度、速度、温度、压强剖面（含 $P_e$）](output/profiles.png)

![压强与电离分数（含 He 两支）](output/pressure_ionfrac.png)

![53 路向内扫描衰减（双吸收体 $\tau_b = (\sigma_H n_{\mathrm{HI}}+\sigma_{He} n_{\mathrm{HEI}})\Delta s$）](output/flux_attenuation.png)

![时间历史：dt、source_resid、底/顶 $P_e$ 等](output/history.png)

## 4. 与 B2（rad_h_pe_1d，不含 He）的对比

同一套核、同一 SED、同一网格/拉伸/$t_{\mathrm{stop}}$；B2 数字取自
`examples/rad_h_pe_1d/REPORT.md`（已提交，未改动）。

| 量 | B2（HI+HII） | B3（HI/HEI/HII/HEII） |
|---|---|---|
| `num_states` / 源带数 | 11 / 8 | 21 / 26 |
| 吸收体 | ["HI"] | ["HI","HEI"]（σ 行表） |
| 步数（on / off） | 6993 / 6958 | 20000 / 20000 |
| 末步 dt (s) | 25.74 / 26.18（裸 CFL） | 10.0 / 10.0（dt_max 帽） |
| source_resid 末 / max（on） | 1.002 / 4.470 | 1.025 / 3.985 |
| 墙钟（on / off, s） | 66.4 / 61.5 | 1083 / 997.8 |
| $n_{\mathrm{HII}}$ 顶（on, cm⁻³） | 2.820e4 | 4.718e3 |
| $x_{\mathrm{ion}}$ 顶（on） | 0.699（仅 H） | H 0.852 / He 0.996 |
| $T_e$ 顶（on, K） | 3426 | 1963 |
| on/off $n_{\mathrm{HII}}$ 比 中 / 顶 | 1938 / 3.50e5 | 7.75 / 19.2 |
| $\dot M$（on, $r>4.5R_p$, g/s） | 2.38e10 | 3.50e9 |
| 同区 std/\|mean\| | 0.017 | 0.071 |

读法：

- **步数与墙钟的差不是物理**：B3 钉了 `dt_max = 10 s`（§2 的稳定帽），
  B2 用裸 CFL（~26 s）；加上 21 态牛顿与双吸收体扫描的每步开销。
- **顶层更电离、但绝对密度更低**：He 吸收体分走了 XUV 光子（双吸收体
  光深叠加），外区热结构改变（on 顶 $T_e$ 从 3426 K 变为 1963 K），
  质量流从 $2.38\times10^{10}$ 降到 $3.50\times10^{9}\,\mathrm{g/s}$。
  CX 与电子碰撞在四个组分之间重新分配能量与电荷。方向性与「辐射开 =
  电离风、关 = 抽空」的结论不变。
- **关门跑不再接近真空**：B2 off 顶 $x_{\mathrm{ion}}=1.7\times10^{-3}$，
  B3 off 顶 $x_{\mathrm{ion,H}}=0.276$ —— 四流体网络里电子碰撞 / CX
  由热背景维持一部分电离（off 顶 $T$ 也从 22–35 K 变为 ~298 K）。
  开关比因此以中层为准的判读不变，但数值整体变小。
- 本刀**不**用这张表对照科学文献的逃逸速率：这是合同算例
  （完整 Xing+2023 表 + 双吸收体在同一套核里跑得通的证明）。

## 5. 球壳质量流 $\dot{M} = \sum_s 4\pi m_s n_s r^2 v_r$

| | on | off |
|---|---|---|
| $\dot M$ 中层 / 顶 (g/s) | 2.306e9 / 3.933e9 | −6.562e8 / 4.863e8 |
| $r>4.5 R_p$ 均值 (g/s) | **3.499e9** | 2.954e8 |
| 同一区域 std/\|mean\| | **0.071** | 0.390 |

开门跑外区仍是定常风快照（mid/top 同量级，std/|mean| = 0.071，比 B2
的 0.017 宽 —— 四流体化学在 $2.0\times10^{5}\,\mathrm{s}$ 内仍在缓慢
重整外区，未到 B2 那种单组分平台）。关门跑不是定常对照：中层内流、
顶层量级小，外区在抽空。

![球壳质量流](output/mass_flux.png)

## 6. A.2 回归（守恒律管测）

`tests/test_a2_conservation.py`（16 胞笛卡尔均匀盒、20 短步、不叠激波管）：

1. **核数**：CX 真实转换下盒积分 $N_{\mathrm{H}} = \sum(n_{\mathrm{HI}}
   + n_{\mathrm{HII}})$、$N_{\mathrm{He}} = \sum(n_{\mathrm{HEI}}
   + n_{\mathrm{HEII}})$ 相对漂移 $\le 10^{-12}$（reflect 闭盒，
   静止初态，实测逐位 0）。
2. **准中性**：核内派生读 $n_e$ 与 $n_{\mathrm{HII}}+n_{\mathrm{HEII}}$
   之差 $\le 10^{-12}\,n_e$（探针带 `dP_e = n_e-(n_HII+n_HEII)` 直读，
   实测在舍入级）。
3. **总动量**：只开碰撞 / CX / 库仑，无外力无辐射，四流体总动量相对
   漂移 $\le 10^{-9}$（copy 边界均匀态 = 离散周期盒；实测本底
   $1.5\times10^{-11}$，来自速度槽更新的舍入）。

已知固有行为（写进测试文件头）：偏离压强平衡的初态下，单步线性化源步
有一个饱和型伪漂移（核数最高 $\sim 4\times10^{-5}$，不随步数增长），
与 §2 的启动瞬态同源；回归从平衡初态出发，不压这段。

工厂合同在 `tests/test_hhe_factory.py`（18 测：RUN 钉、四组分表、
整表 53 束、双吸收体 σ 行、化学和的反对称与 CX 交叉、CHE/drag 用和
速率、库仑三带精确反号、无 `vx_e`、无 `**2`、开门能 setup 出
$N=21$ / 53 束 / 26 带）。

## 7. 读法与边界

1. **这是 B3 合同算例**：四组分 + 双吸收体 + 电荷加权 $n_e$ /
   $v_e$、完整 yardstick 源表、x1end 分岔在同一套核里跑到
   $2\times10^{5}\,\mathrm{s}$，场有限、辐射开关判别量成立、$P_e$ 可
   读。不声称复现任何文献的完整逃逸结果。
2. **`dt_max = 10 s` 是源步数值稳定帽**（§2），CFL 公式未改；
   论文若引用本算例的时间尺度，须连帽一起说明。
3. `source_resid ≈ 1` 按报告量读（同 B2）；库仑 / 化学的判据在管测
   （`test_coulomb.py`、`test_a2_conservation.py`），不在残差。
4. `shock_heat` 钉 ions-only 是模型选择（Tóth 约定：三档无一为真值）。
5. 摘要层：本刀回归绿了之后才谈「完整 Xing+2023 + 双吸收体」进论文；
   之前只写到合同层。

逐格表：`output/profiles_on.txt`；
数字块：`output/numbers.txt`。

旁路（不进默认 `run.py`）：墙钟校准在 `utils/hhe_bench/`
（dump 仍写本目录 `output/hhe_*_bench/`）；$t_{\mathrm{stop}}=5\times10^{5}\,\mathrm{s}$
开门 `hhe_tend5e5_outflow.py` + `hhe_tend5e5_plot.py`（`output/tend5e5/`）。
`nx=256` 帽证据：`hhe_nx256_hat_notes.md`。
