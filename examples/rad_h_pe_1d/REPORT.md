# rad_h_pe_1d 结果报告（B2 补：离子 + 独立 $P_e$ + HII–电子库仑）

默认：`nx = 128`，`ng = 2`，`n_beam = 53`，`tstop = 2.0e5 s`。
HLL + PLM + RK2，`cfl = 0.4`，球对称 $r \in [R_p, 10 R_p]$，
log 拉伸（factor = 1.025，首末 $\Delta x/R_p = 9.96\times10^{-3}$ /
0.229，比 23.0）。全程 cgs，SED / 电离截面**引用**
`../rad_hd_1d/data/`，不拷贝。
数字来自首登 `python examples/rad_h_pe_1d/run.py`（当时开/关辐射各跑到
$t_{\mathrm{stop}}$）。**以后默认只推进 radiation ON**；下文 radiation
off 是那次快照，`run.py` 不再重算。工厂仍接受 `radiation=False`，留给契约测。

本算例验证的是 **B2 + B2 补合同**，不是完整逃逸物理：

- 电子是独立压强块：`electron = {"mode": "pressure-only", "share":
  "by-charge", "shock_heat": "ions-only"}`，`num_states = 5*2 + 1 = 11`
  （$P_e$ 殿后，不是第五支动量流体）。
- `shock_heat` 三档（ions-only / by-charge / electron）是 Tóth 约定的
  可切换模型选项，**没有一档是物理真值**；本算例显式钉 ions-only，
  Godunov 激波增量全留离子压，光电加热另走 `dP_e` 源带。
- 光电离加热经追加写名 `dP_e` **直写** $P_e$ 块（不写进离子 $P$ 再投影）；
  复合读派生 `n_e`（不是 `n_HII*n_HII`）。
- HI–HII 共振碰撞按 yardstick（`SSH_220804/mmd_model_src.c:155-156`）：
  $\mathrm{coef} = 2.67\times10^{-10}\,T^{1/2}(1-0.083\log_{10}T)^2$，
  $T = \tfrac12(T_{\mathrm{HI}}+T_{\mathrm{HII}})$，净 $\mathrm{d}n=0$，
  只换动量与内能（COL_PRS_TRANS 结构）。
- **B2 补** HII–电子库仑热交换（同一尺子 `:164,:192,:207`）：

$$
\nu = 0.0299\,(T_e+T_{\mathrm{floor}})^{-3/2},\qquad
\Delta = 2(n_{\mathrm{HII}}P_e-n_e P_{\mathrm{HII}}),
$$

$$
\dot P_{\mathrm{HII}} = \frac{\nu}{m_{\mathrm{HII}}+m_e}\Delta,\qquad
\dot P_e = -\dot P_{\mathrm{HII}}.
$$

  单离子 $u_e\equiv u_{\mathrm{HII}}$，摩擦项恒 0，省略；不写 `dvx_*`。
  $m_e = 5.4858\times10^{-4}\,\mathrm{amu}$。IR 不乘 $G_{\Delta t}$（核自己乘 $\Delta t$）。

## 1. 初值与边界

`pack_ic` 在 `setup` 之后按格心半径填两支（A5 的薄指数包络）：

$$
n_{\mathrm{HI}}(r)=\max\bigl(10^{13}\,e^{-(r-R_p)/H_{\mathrm{ic}}},\,10^{6}\bigr),
\quad n_{\mathrm{HII}}=10^{-3}\,n_{\mathrm{HI}},
\quad T=1500\,\mathrm{K},\quad v_r=0,
$$

$$
P_s = n_s k_{\mathrm{B}}T,\qquad
P_e = n_{\mathrm{HII}} k_{\mathrm{B}}T\ \text{（准中性）}.
$$

`x1beg=user`：解析带子只给 **ghost** 赋两支流体的 $(n, v=0, P)$
（A5 形状，底层钉 $n_{\mathrm{beg}}=10^{13}$）；$P_e$ 的 ghost 走既有
copy@1。**x1end 按辐射开关分岔**（以后关门对照都这样）：

- 开门：`user`，旧树 `outflow_BC`（`SSH_220804/mmd_model_IC_BC.c:387-390`）。
  $g=x_{\mathrm{IEND}}/x_{\mathrm{IEND}-1}$，

$$
n = n_{@1}^{1+g}/n_{@2}^{g},\quad
P = P_{@1}^{1+g}/P_{@2}^{g},\quad
v_r = \max\bigl(0,\,(1+g)v_{@1}-g\,v_{@2}\bigr).
$$

  `max` 写成 $0.5(v+\sqrt{v^2})$。两层 ghost 共用同一对 `@1/@2`。
- 关门：默认 `outflow`（核里 = copy）。对数外插对抽空柱不友好，对照跑
  不用那套。

两跑底层 HII 都有 user_bc 补给，底层密度不是辐射开关的判别量（见 §3）。

| 量（IC） | 底 | 顶 |
|---|---|---|
| $n_{\mathrm{HI}}$ (cm⁻³) | 8.829e12 | 1.0e6 |
| $P_e$ (erg/cm³) | 1.829e-3 | 2.071e-10 |
| $a_{\mathrm{net}}=-GM_p/r^2+3GM_\star r/a^3$ (cm s⁻²) | −902.8 | +113.9 |

## 2. 跑况

| | radiation on | radiation off |
|---|---|---|
| 步数 / 墙钟 | 6993 / 66.4 s | 6958 / 61.5 s |
| $t_{\mathrm{final}}$ (s) | 2.0002e5 | 2.0002e5 |
| 末步 dt (s) | 25.74 | 26.18 |
| source_resid 末步 / 历史最大 | 1.002 / 4.470 | 1.039 / 5.669 |

`source_resid` 是**报告量**：源步合同是「单步线性化 $(I-\Delta t J)dU
= \Delta t\,R(U^*)$ 就是这一步」，残差 $|U^*+\Delta t R-U^{\mathrm{new}}|$
按 $|U^*|+\Delta t|R|+10^{-30}$ 归一。rel ≈ 1 对应硬源项（光电离 /
复合 / 库仑）把某分量一步抽穿、被 $10^{-30}$ 地板接住的钳位事件；
$2.0\times10^{5}\,\mathrm{s}$ 内所有场有限、两跑 dt 都锁在 ≈ 26 s，
状态没有发散。管测不把这一残差当库仑平衡判据
（见 `tests/test_coulomb.py`）。

加库仑后步数从 B2 的 ~2e4 / ~1e4 掉到 ~7e3：电子–离子热耦合抬高了
外区声速，CFL 步更长，墙钟反而更短。

## 3. 辐射开/关对比

| 量（末态） | on | off |
|---|---|---|
| $n_{\mathrm{HI}}$ 底 / 顶 (cm⁻³) | 7.513e12 / 1.213e4 | 7.513e12 / 48.2 |
| $n_{\mathrm{HII}}$ 底 / 顶 (cm⁻³) | 3.126e9 / 2.820e4 | 3.127e9 / 0.0806 |
| $n_{\mathrm{HII}}/n_{\mathrm{HI}}$ 底 / 顶 | 4.16e-4 / **2.33** | 4.16e-4 / 1.67e-3 |
| $x_{\mathrm{ion}}=n_{\mathrm{HII}}/(n_{\mathrm{HI}}+n_{\mathrm{HII}})$ 顶 | **0.699** | 0.00167 |
| $v_{r,\mathrm{HI}}$ 最小 / 最大 (cm/s) | −5.20e4 / 3.118e6 | −2.06e5 / 2.740e6 |
| $v_{r,\mathrm{HII}}$ 最小 / 最大 (cm/s) | −5.50e4 / 3.125e6 | −1.25e5 / 2.741e6 |
| $T_{\mathrm{HI}}$ 底 / 顶 (K) | 1514 / 3302 | 1514 / 35.5 |
| $T_{\mathrm{HII}}$ 底 / 顶 (K) | 1514 / 3431 | 1514 / 22.5 |
| $T_e$ 底 / 顶 (K) | 1412 / 3426 | 1408 / 33.4 |
| $P_{\mathrm{HII}}$ 底 / 顶 (erg/cm³) | 6.53e-4 / 1.34e-8 | 6.54e-4 / 2.50e-16 |
| on/off $n_{\mathrm{HII}}$ 比（底/中/顶） | 1.00 / **1938** / **3.50e5** | — |

（取样半径 $r/R_p = 1.005 / 2.561 / 9.885$。）

- **辐射推动流体成立**：中层 $n_{\mathrm{HII}}$ 开关比 1938；on 顶层
  仍维持 $10^{4}\,\mathrm{cm^{-3}}$ 量级的电离风，$x_{\mathrm{ion}}=0.70$。
- 底层比 ≈ 1 不是失效：§1 说过，两跑的 user_bc ghost 都在补 HII。
- 关门对照的顶层用默认 copy-outflow，可以报数；开关比仍以中层为准
  （顶层面条件已经不是同一套）。

![密度、速度、温度、压强剖面（含 $P_e$）](output/profiles.png)

![压强与电离分数（$P_e$ 点线）](output/pressure_ionfrac.png)

![53 路向内扫描衰减](output/flux_attenuation.png)

![时间历史：dt、source_resid、底/顶 $P_e$ 等](output/history.png)

## 4. $P_e$ 与库仑耦合

| 量 | on | off |
|---|---|---|
| $P_e$ 底 / 顶 (erg/cm³) | 6.095e-4 / 1.313e-8 | 6.076e-4 / 3.721e-16 |
| $P_e$ 全域最小 / 最大 | 1.313e-8 / 6.095e-4 | 1.084e-18 / 6.076e-4 |
| $T_e$ 底 / 中 / 顶 (K) | 1412 / 7232 / 3492 | 1408 / 2155 / 33.4 |
| $T_{\mathrm{HII}}$ 底 / 中 / 顶 (K) | 1514 / 7231 / 3482 | 1514 / 2143 / 22.5 |

本刀前（无库仑）：on 底 $T_e\sim 2270\,\mathrm{K}$ / 离子 1514 K；
off 底 $T_e\sim 8\,\mathrm{K}$。本刀后：

- **中层和顶层已经锁在一起**：on 中 $T_e-T_{\mathrm{HII}}\approx 1\,\mathrm{K}$，
  顶 $\approx 10\,\mathrm{K}$。off 中也只差约 12 K。
- **底层温差从 756 K 收到 102 K**（$T_e$ 略低于 $T_{\mathrm{HII}}$）。
  底是 user_bc 钉离子 $(n,v,P)$、$P_e$ ghost 走 copy@1 的地方，光电加热
  与边界补给同时在，不会逐位等温；判别量是「被拉近」，不是底等温。
- off 底 $T_e$ 从 ~8 K 拉到 1408 K：没有光电时，库仑仍把电子压强绑在
  离子底温附近。
- 电子槽全程只压强、无动量：`num_states=11`，库仑带只写 `dP_HII` /
  `dP_e`，不写 `dvx_*`（yardstick 里 HII 动量不跟 ELCI 换）。

## 5. 球壳质量流 $\dot{M}_s = 4\pi m_s n_s r^2 v_r$

| | on | off |
|---|---|---|
| $\dot{M}_{\mathrm{HI+HII}}$ 中层 / 顶 (g/s) | 2.09e10 / 2.45e10 | −8.92e8 / 2.57e7 |
| $r>4.5 R_p$ 均值 (g/s) | **2.38e10** | 1.10e7 |
| 同一区域 std/\|mean\| | **0.017** | 0.65 |

**开门跑仍是定常风快照**：外区 mid/top 同量级，
$r>4.5 R_p$ 的 std/|mean| = 0.016（本刀前无库仑时是 0.0096）。
均值从 $3.16\times10^{10}$ 降到 $2.38\times10^{10}\,\mathrm{g\,s^{-1}}$：
库仑改了热结构（外区 $T\sim 3\times10^{3}\,\mathrm{K}$，本刀前顶层
$T_{\mathrm{HI}}\sim 391\,\mathrm{K}$），质量流跟着变，但仍空间定常。

**关门跑不是定常对照**：中层仍是内流，顶层 $\dot{M}$ 比 on 小三个量级，
外区密度在抽空。关门 x1end 是默认 copy，外区可以报数。

本刀**不**用这张表对照科学文献的逃逸速率（没有 He、没有第二吸收体）。

![球壳质量流](output/mass_flux.png)

逐格表：`output/profiles_on.txt`；
数字块：`output/numbers.txt`。

## 6. 读法与边界

1. **这是 B2 + B2 补合同算例**：独立 $P_e$ 块、`dP_e` 直写、$n_e$ 派生读、
   yardstick 共振碰撞、HII–电子库仑热交换、shock_heat 分档在同一套核里
   跑得通。开门跑到 $2\times10^{5}\,\mathrm{s}$ 后外区 $\dot{M}$ 空间定常，
   $T_e$ 与 $T_{\mathrm{HII}}$ 被拉近。不是、也不声称复现任何文献的完整
   逃逸结果。
2. **shock_heat 钉 ions-only 是模型选择**（Tóth 约定：三档无一为真值）。
3. **He、第二吸收体、H–He 电荷交换不在本刀**（B3 内容）；吸收体只有
   `["HI"]`。
4. **source_resid ≈ 1 按 §2 读**：单步线性化源步的报告量，不是收敛判据；
   库仑管测看的是热流方向和 $|T_e-T_{\mathrm{HII}}|$ 随步下降。
5. `n_beam=1` 只当冒烟，不当物理对照（同 A5）。
6. **x1end 分岔**：开门用旧树对数外插，关门对照只用默认 outflow。
   顶层开关比不当成同一套边界下的物理差。
