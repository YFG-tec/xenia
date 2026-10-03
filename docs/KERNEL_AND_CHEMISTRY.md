# 计算核、配表化学与刚性源项

对照 `docs/ROADMAP.md` §2。本文是设计说明，不是第二份路线图。文档地图见 `docs/README.md`；对照已收集代码见 `docs/APPENDIX.md`。

本文把「C 核 + Python 配表」和「刚性 Jacobian 怎么来」写死，避免做成 HD209458b 专用大气码。源项以符号式进 config；**C 只解析传入的 IR，不带特化反应或外源力。**

现码对照：`SSH_220804/mmd_model_src.c`（`mmd_get_lc`、`get_xuv_flux`、`quasi_neutrality`）、`SSH_220804/mmd_integrate.c`（前向差分 Jacobian + 稠密 LU）。

---

## 1. 结论（先定死）

1. **对流、单元内刚性源项、沿射线辐射留在 C/CUDA。** 组分、符号源项、SED、单位走 config。
2. **跟随网格和流体场的源项以符号式进 config。** setup 绑到场/网格槽，编成 IR。C 只解析这份 IR 求 $\mathcal{L}_c(u)$。核里禁止出现组分名、光致电离公式、$-GM/r^2$、潮汐。每格回调 Python 禁止。
3. **HD209458b 是第一张预设表，不是核。** 激波管 = `sources: []`。换 CO2 或只开碰撞，不改 `step`。
4. **没有「HD209458b 算不动」的物理死墙。** 卡点是工程契约：源项来自 IR、电子闭合显式、辐射不进主循环。
5. **反应式在 setup 解析计量；Jacobian 可以后换。** 先差分（现码），再前向 AD 或对 IR 生成解析 $J$。$N\approx 13$ 时不必上 JFNK。

公开 API 仍只有 `setup` → `step` → `fields` → `finalize`。

---

## 2. 分层是否合理，会不会锁死成大气仿真

合理。三类算子本来就不该和「H 还是 He」绑在一起。现码看起来像大气专用，是因为组分写进了 `enum` 和公式。

| 留在 C 核 | 走 Python config | 写死在 C 就会特化 |
|---|---|---|
| Godunov / HLL、RK、CFL | 流体表：质量、电荷、HD 或 MHD | `HI/HEI/HII/HEII/ELCI` |
| 单元内隐式：Jacobian + 局部 LU | `sources[]` 符号式 + `constants{}` | `mmd_get_lc` 里的 Arrhenius 数字 |
| `RadiationScan`：柱密度前缀积 | 吸收体列表 + 每束截面 + SED | `get_xuv_flux` 只扫 HI、HEI |
| 电子闭合算法（压强-only、电荷加权） | 选哪种闭合；是否准中性 | `quasi_neutrality` 写死 $n_e=n_{\mathrm{H}^+}+n_{\mathrm{He}^+}$ |
| 对 IR 求值（场、$r$、系数） | 体力写成对 $\mathbf{v}$ 的符号式 | `G_COST` / `TIDAL_COST` 当默认物理 |

关源项应是 `sources: []` 或显式 `sources_enabled=false`，不是 `XUV_FACTOR`、`BASE_ION_RATE`、`HEATING_EFFICIENCY` 三个数碰巧为 0。

### 2.1 工程卡点（过得去，不是物理死墙）

| 卡点 | 过得去？ | 为什么 |
|---|---|---|
| 不能把任意 Python 函数送进每格 / CUDA | 过得去 | 符号式在 setup 收成固定 IR（操作码 + 槽）。新物理 = 换 config，不加 C 分支。 |
| 刚性：现码对 $\mathcal{L}_c$ 做有限差分再 LU | 过得去 | 表驱动的 $\mathcal{L}_c$ 仍是纯函数 $u\to$ 源项。组分变多则矩阵 $(3N_f+1)^2$ 变大，是代价不是死墙。 |
| 电子不是第五个动量流体 | 过得去，但是真难点 | $n_e$、$\mathbf{u}_e$ 由离子拼；$P_e$ 按电荷份额进离子 Riemann。这是闭合选项，必须在流体表里写死；激波加热份额要可调。见 `ROADMAP.md` §4。 |
| 辐射沿径向是链，不是逐格 | 过得去 | 现码 `IEND→IBEG` 的 $\exp(-\Delta\tau)$。必须单独 `RadiationScan`。1D 不切径向。见 `ROADMAP.md` §5。 |
| 单位：`TMP` 就是 `PRS` | 过得去 | schema 里 `NDT↔ρ`、`T↔P` 必须写死。 |
| 若继续链 PLUTO | 过得去，但是发布约束 | 对外发布可能整包 GPLv2。 |

违反「源项只来自 IR / 电子闭合显式 / 辐射不进主循环」任一条，CUDA 和通用性会一起坏。

### 2.2 谁已经这么做了

分层思想常见，**没有现成孪生码**（多带压流体 + 表驱动 $\mathcal{L}_c$ + `RadiationScan` + Python API）。

| 代码 | 分层 | 和这里的差别 |
|---|---|---|
| Cantera / PelePhysics | YAML 反应网络，C++ 核算速率和 Jacobian | 燃烧化学，不是多流体 Godunov，也没有 XUV scan |
| KROME | Python 预生成 Fortran 化学网络，嵌进 FLASH/RAMSES | 单流体 + 示踪化学；没有每种成分自己的速度 |
| Dedalus | Python 写方程，编译核跑 | 谱方法、低马赫；证明「Python 配、C 跑」成立 |
| BATS-R-US 多离子 | 多流体 + 电子压强，主要在 Fortran 里写死 | 物理近，入口不是 Python 配表 |
| CHEMKIN / pyJac / TChem | 反应式 → 生成解析 Jacobian | 化学网络工具，不是多流体核 |

Athena++ / 官方 PLUTO 没有这一套。空位是真的，不要指望换一个开源核就自带。多流体对照见 `docs/MULTIFLUID_CODES.md`。

---

## 3. 把现在的 HD209458b 拆成符号式（写进预设表，不写进 C）

`mmd_get_lc` / `get_xuv_flux` / `quasi_neutrality` 里每一项都应变成 config 里的式子或开关。核里不准再出现这些公式。

### 3.1 配表即可（变成 `sources[]` 表达式）

| 现码 | 进 config 的形态（示意） |
|---|---|
| 光致电离 HI/HEI + 加热 | $\sigma_b F_b n_s/h\nu$，加热 $\eta\sigma F n$ |
| e 碰撞电离、辐射复合 | Arrhenius / 幂律写在式子里 |
| H–He⁺、He–H⁺ 电荷交换 | 对参与流体的 $n,v,P$ 交换式 |
| HI–HEI、离子–电子碰撞 | 拖曳 / 库仑式 |
| Lyα 冷却 $7.5\times 10^{-19}\,n_{\mathrm{H}} n_e\exp(-118348/T_e)$ | 同一串写进 `sources[]` |
| `CHE_PRS_TRANS` / `COL_PRS_TRANS` | 由计量 + 相对速度在 IR 里生成，与组分名无关 |

### 3.2 必须写成选项，不能当默认物理

| 现码 | 怎么避免特化 |
|---|---|
| 行星重力 + 潮汐 | `dv = -G*M_p/r**2 + 3*G*M_star*r/a**3`；激波管不写 |
| $n_e = n_{\mathrm{H}^+} + n_{\mathrm{He}^+}$ | 准中性：按电荷求和 |
| $P_e$ 进离子 Riemann | `electron: pressure-only, share-by-charge` |
| 光学深度只扫 HI、HEI | `absorbers: [HI, HEI]` |
| `XUV_FACTOR==0` 当模式开关 | `sources: []` 就是关 |

第一版 IR 要能表达上表那些代数/指数组合（场、网格坐标、常数、已扫描的 $F_b$）。C 不因「这是潮汐」开分支。

---

## 4. 符号式怎么进核：setup 绑定，C 只解析 IR

用户写反应方程式和依赖网格/流体的源项。这是后面要做的功能，不是现在的占位核。

**该做成的：**

```text
用户 config
    reactions[]   "H + e → H+ + 2e"
    sources[]     dn_HI = -k * n_HI * n_e
                  dv    = -G * M_p / r**2 + …
    constants{}   k, G, M_p, …
                ↓  setup（Python）
计量矩阵；符号式 → 操作码 + 操作数槽（场 / r / F_b / 常数）
                ↓
C/CUDA：只解析这份 IR，求值累加 Lc(u)；J 由差分 / AD / 对 IR 生成
```

**禁令：**

- C 里不准有 `if (photoion)`、`planet_tide`、`HI` 下标、写死的 $-GM/r^2$。
- 不准每格 `eval` Python，不准用户塞函数指针当公共 API。
- 非局部辐射仍是 `RadiationScan`；符号式里的 $F_b$ 是 scan 之后的已知场。

反应式负责计量（谁变谁）。速率和体力都是符号式，跟网格坐标、流体场走。

---

## 5. 刚性：Jacobian 从哪里来

现码（`mmd_integrate.c`）：每格对 $\mathcal{L}_c$ 做 $N$ 次前向差分（漂移约 $10^{-7}|u|$），再稠密 LU。HD209458b 的 $N=\mathrm{TOT\_VAR}\approx 13$，每格大约 14 次 `mmd_get_lc`。

难的不是「化学不可微」，而是手写 $\partial\mathcal{L}_c/\partial u$ 太烦。速率律若是 Arrhenius、幂律、$\sigma F/h\nu$、库仑拖曳，$\mathcal{L}_c$ 对状态几乎处处解析可微。**限制「只接受可微速率律」完全合理**，也是 Cantera / KROME / pyJac 的默认假设。

### 5.1 方法比较

| 方法 | 每格代价（相对一次 $\mathcal{L}_c$） | $J$ 精度 | 适合现在 |
|---|---|---|---|
| 现码：前向有限差分 | $1+N\approx 14$ | 截断 $\sim 10^{-7}$，受尺度影响 | 已经能跑；换组分表仍可用 |
| 前向 AD（对偶 / Enzyme） | 约 $2$–$4\times N$，或一次向量前向 | 机器精度 | 好，尤其以后 CUDA 同一套 $\mathcal{L}_c$ |
| 符号 / 代码生成解析 $J$ | 约 $2$–$8$（稀疏） | 机器精度，最快 | 最好的长期形态 |
| 反向 AD | 对 $\mathbb{R}^N\to\mathbb{R}^N$ 仍要约 $N$ 次 | 机器精度 | 不划算：残差不是标量 |
| JFNK（无矩阵 Newton–Krylov） | 每次 Krylov 一次 $Jv\approx\mathrm{FD}$ | 不显式存 $J$ | $N>100$ 才值得；仍要预条件 |

$N$ 小时，形成稠密 $J$ 再 LU 是正确默认。JFNK 省的是存储，**不是刚性**；刚性化学的 Krylov 没有近似 $J$ 当预条件会迭代爆炸，等于把 Jacobian 工作换了个地方。以后若做到上百种成分，再接 CVODE / JFNK + 近似 $J$ 预条件。

辐射通量已算子分裂，当局部源项的外源，**AD 不用跨格子**。现码这条分裂是对的。

### 5.2 分段不可微

左右导数取平均是 Clarke 广义梯度，连续函数上说得通，但 Newton 在折点会抖。要的是**一致的 $J$**，不是最对称的次梯度。能平滑就平滑；必须开关就选一侧并保持连续。

现码里真正的折点很少，而且多数不进状态 Jacobian：

| 现码里的非光滑点 | 对 $\partial\mathcal{L}_c/\partial u$ | 更好的做法 |
|---|---|---|
| `MIN(1, exp(t−20))` 电荷交换斜坡 | 只依赖时间，不进 $J$ | 保持；或改成平滑开关 |
| 密度地板 / `max(n, ε)` | 折点在近真空格 | 对数变量，或 `softplus(n−ε)` |
| 准中性：$n_e$ 由离子代数拼出 | 必须链式求导 | AD 或消元；不要当常数 |
| $T=P/(n k_B)$，$n\to 0$ | 奇点，不是折点 | 温度裁剪写进速率律，并对其求导 |

### 5.3 刚度真正难在哪

不在「怎么得到 $J$」，而在 Newton 稳健：密度不能穿零、电子温度裁剪、辐射当外源。隐式算法（差分或 AD 得到 $J$，再 LU）可以长期不动。

---

## 6. 推荐落地顺序

| 阶段 | 用户看见的 | 核看见的 |
|---|---|---|
| 现在 | `sources: []` | 跳过 $\mathcal{L}_c$ |
| 下一步 | 符号式 + 反应式（HD209458b 当预设表） | C 解析 IR 求 $\mathcal{L}_c$；$J$ 仍可差分 |
| 随后 | 不变 | 对 IR 做前向 AD，或生成解析 $J$；隐式算法不动 |
| 很远 | 上百种成分 | 再考虑 JFNK + 近似 $J$ 预条件 / CVODE |

不冻结「C 通道类型枚举」。冻结的是 IR 操作集（加载场/坐标、加减乘除、幂、exp）。Python 绑定名字。辐射仍是每步一次 scan。电子闭合单独开关。激波管继续当空表回归。

近期待办见 `docs/TODO.md`。文档入口见 `docs/README.md`。
