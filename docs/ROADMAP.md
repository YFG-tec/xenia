# PLUTO_XMF 后续路线（草案）

近期行星大气研究里，多流体一直是重要方向，但业界缺少一个**好用的多流体 MHD 仿真工具**。本仓库原来的 `SSH_220804/` 能算，界面不友好、配置靠宏和行序、并行绑死 MPI。**三篇系列（都走 ApJS）：** paper 1 本篇 = 框架与多流体 HD（单机 Python 入口 + \(P_e\)/Xing 表 + Kokkos Serial/OpenMP + 单卡 CUDA）；paper 2 = MHD 与附属物理（理想 MHD/CT + 追踪辐射 + 二次耗散 \(\nu,\eta\)）；paper 3 = 大规模计算（MPI + AMR）。2/3 的清单只在 MHD 树 `docs/TODO.md`，本篇不要开。近期目标很具体：

1. 改成对用户能用的版本（Python 配问题，不改 C 宏）；
2. 计算核走向 **CUDA 生态**（本机一块 NVIDIA 卡可测，不上多卡 / 多机）；
3. 物理上仍以 Xing, Yan & Guo (2023, *ApJ* 953:166) 的 H/He 逃逸为第一张表，但框架不绑死大气。

文档地图见 `docs/README.md`。框架图见 §2。三维进库留口见 §2.6。化学 / Jacobian 见 `docs/KERNEL_AND_CHEMISTRY.md`。  
对照已收集代码的结论（骨架不改；守恒与 GPU 是可接受劣势）见 `docs/APPENDIX.md`。

---

## 1. 产品定位（把规划算进去）

不是再写一个 Athena++，也不是只做一个 HD209458b 脚本。

**要交付的是：** 表驱动的多流体（HD/MHD）守恒律工具。流体和耦合在 Python 里声明，C/CUDA 核做推进。行星大气是主场和第一批算例，同一套核能换组分表和通道表。

| | 现在（`SSH_220804/`） | 规划 |
|---|---|---|
| 入口 | `definitions.h` + `pluto.ini` 行序 | Python API + YAML，名字对字段 |
| 对象 | 写死的五流体 | 流体表：质量、电荷、HD/MHD、是否准中性 |
| 耦合 | `mmd_get_lc` 里写死 H/He | 通道表：碰撞、电荷交换、光化学、加热 |
| 并行 | MPI + ArrayLib 写进 IC/BC | 主循环单元局部；辐射单独 scan；默认 CUDA |
| 用户 | 改 C、对行号 | 配表、跑 step、取场、画图 |

和开源核的错位仍然成立：FARGO3D/Idefix 是气+无压尘埃；官方 PLUTO/Athena++ 是单流体或同速化学；BATS-R-US 多离子强，但不是 Python/CUDA、也不面向单机易用。空着的是 **多种带热压流体 + 表驱动源项 + 好用入口 + CUDA**。

---

## 2. 框架图（由面到点）

下面几张都是文字图，从上到下收窄。化学 / Jacobian 的论证见 `docs/KERNEL_AND_CHEMISTRY.md`；电子压强与辐射并行见本文 §4、§5。

### 2.1 面：谁配、谁算、谁对照

```text
  用户 / 算例 / 今后的图形界面
           │  dict · YAML · UI   （只收集，不另开进核的路）
           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Python 包  xmf                                         │
  │  厚：流体表 · 反应式解析 · 通道表 · SED · 单位 · schema  │
  │  公开面只有四步                                          │
  │      setup(config) → handle                             │
  │      step(handle)  → t                                  │
  │      fields(handle)→ 名字对缓冲                          │
  │      finalize(handle)                                   │
  └───────────────────────────┬─────────────────────────────┘
                              │  版本化扁平 struct
                              │  无 ini / 无行号 / 无化学字符串
                              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  C/CUDA 核（库，不是 pluto_xmf 可执行文件）                │
  │                                                         │
  │   RadiationScan     双曲推进 L_f      单元内 L_c + J + LU │
  │   非局部 · 每步1–2次  单元/方向并行     单元局部 · 可 CUDA  │
  │   ndim = 1 | 2 | 3   （同一套算子；1D 回归，2D/3D 同路径）  │
  │   backend: serial | host-threads | cuda                 │
  └───────────────────────────┬─────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
     examples/           SSH_220804/        空表回归
     shock_tube          旧树对照            sources=[]
     HD209458b           不在原地堆 CUDA
```

空位仍是：多种带热压流体 + 表驱动源项 + Python 入口 + CUDA。不是再写一个 Athena++，也不是只做一个 HD209458b 脚本。

### 2.2 层：setup 把语义吃掉，核只看见表

```text
  config
    fluids[]     name, mass, charge, kind=neutral|ion
                 MHD 是问题级统一开关（磁场开 → 所有离子走 MHD，中性仍 HD）
                 禁止每支流体自选 hd|mhd
                 一支中性流体热压 / 声速 → 0 = 无压尘埃极限（第一篇方法节提；
                 对照 FARGO3D 2016/2019、Idefix 2023、MPI-AMRVAC 2014）
    electron     off | pressure-only, share-by-charge
                 激波加热份额: electron | by-charge | ions-only
    reactions[]  "H + e → H+ + 2e"   计量（setup 解析）
    sources[]    符号式：依赖格点 (r,…) 与场 (n_s, v_s, P_s, T_s, F_b)
                 系数放 constants{}；空表 = 无源项
    radiation    SED, nbeam, absorbers[], σ[absorber, beam]
    body_force   不再是 C 枚举；写成 sources[] 里对速度的符号式
    grid         geometry, x1/x2/x3 活动界面（绝对位置）, ndim
    time         cfl, tstop, rk
    ic / bc      analytic | array | file ;  reflect | user | …
    sources_enabled   false  当且仅当 sources=[] （禁止三参数==0 哨兵）
    schema_version
           │
           │  Python：校验 · 单位（NDT↔ρ, TMP≡PRS）· 反应式→计量
           ▼
  packed struct / handle
    nfluid, nvar = n_hd*(1+3+1) + n_electron_energy   （1D 闲置 vy,vz=0；现码写死 ≈13）
    stoich[n_rxn, nfluid]
    source_ir[]   绑定到场/网格槽的求值程序（操作码，不是组分名）
    sigma[n_abs, nbeam], sed[nbeam], hmu[nbeam]
    w_s[]  电荷份额（电子压强进离子 Riemann）
           │
           ▼
  C 只解析这份 IR / 表。不认 HI、不写 −GM/r²、不 eval Python
```

源项（化学、碰撞、冷却、体力）**以后以符号式进 config**。Python 做名字→下标、单位、校验；C 只对传入的 IR 求值并累加 \(\mathcal{L}_c\)。核内禁止特化反应或「行星潮汐」分支。每格 `eval` Python 仍禁止。辐射 scan 非局部，不塞进逐格符号式。

### 2.3 一步：`step` 里三种算子，辐射不准混进另外两个

规划顺序（现码 `mmd_integrate.c` 仍是 `mmd_get_lf` → `get_xuv_flux` → 逐格 `mmd_get_lc`，要把 scan 拆出来）：

```text
  step(dt)
       │
       ├─ 1. RadiationScan(n_absorber, σ, 射线)     ← 非局部
       │        Δτ_{i,b} = Σ_a  n_a,i · Δx_i · σ_{a,b}
       │        F_{i,b}  = F_{i+1,b} · exp(−Δτ_{i,b})     （现码 IEND→IBEG）
       │        厚格吸收用 (1−e^{−Δτ}) 保光子
       │        输出：每格每束 F   之后当已知场
       │        实现：沿每条射线做前缀积 / CUB scan
       │        1D = 一条径向；2D/3D = 一束射线，禁止写死 IEND→IBEG
       │        域分解：不切单条射线方向（1D 即不切径向）
       │
       ├─ 2. 各流体双曲 L_f                         ← 单元/方向并行
       │        中性 / 离子：Godunov (HLL/…)
       │        离子有效压强  P_eff,s = P_s + w_s P_e      （§4）
       │        先更 ρ_s, u_s，再更 P_s
       │        电子：无独立动量；P_e 单独平流，只取压强增量
       │
       └─ 3. 逐格刚性源项                           ← 单元局部
                quasi_neutrality:
                    n_e = Σ_ions |q_s| n_s
                    u_e = 电荷加权
                L_c = Σ_通道  template_k(u, F)
                组装 (I − ∂L_c/∂u) Δu = L_c + L_f
                稠密 LU（HD 三维：N = n_hd*(1+3+1)+n_e ；现码 1D 约 13）
                辐射 F 当外源，J 不跨格子
```

并行硬约定：主循环禁止沿射线依赖。`RadiationScan` 点名单独调用。域分解优先切角向 / 切射线。

### 2.4 一格：\(L_c\) 由封闭模板累加，\(J\) 可以后换

```text
  一格状态 u = [ (n, vx,vy,vz, P)_fluid… , P_e ]   ← 1D 也留三速度，闲置分量为 0
       │
       │  sources[] 符号式 → IR（Python 绑定，C 求值）
       ▼
  ┌──────────────┬──────────────────────────────────────────┐
  │ 预设表里的式 │ 现码对应（写在 config，不写进 C）          │
  ├──────────────┼──────────────────────────────────────────┤
  │ σ F n / hν   │ 光致电离 + η 加热                         │
  │ Arrhenius 式 │ e 碰撞电离、辐射复合                      │
  │ CX / 拖曳式  │ H–He⁺；HI–HEI；离子–电子                  │
  │ 冷却式       │ Lyα  7.5e-19 n_H n_e exp(−118348/T_e)    │
  │ −GM/r²+潮汐  │ 体力；激波管不写这条                      │
  │ 计量交换     │ 反应引起的 n,v,P 交换                     │
  └──────────────┴──────────────────────────────────────────┘
       │
       ▼
  L_c(u ; F)     纯函数，可微（只接受解析速率律）
       │
       ├─ 现在   前向差分  drift≈1e-7|u|，每格 1+N 次 L_c
       ├─ 随后   前向 AD 或 setup 生成解析 J
       └─ 很远   N>100 再谈 JFNK + 预条件 / CVODE
                 （JFNK 不消除刚性）
       │
       ▼
  LU: (I − J) Δu = L_c + L_f
```

分段折点：`MIN(1,exp(t−20))` 只依赖时间、不进 \(J\)；密度地板用 softplus / 对数变量；\(n_e\) 必须链式求导。不要默认左右导数平均。

### 2.5 点：两张预设表，同一套核

```text
  shock_tube                         HD209458b（Xing+2023）
  ─────────                          ─────────────────────
  fluids: 一种中性 HD                 fluids: HI, HEI, HII, HEII
  electron: off                      + ELCI (pressure-only)
  sources: []                        符号式：CX / 电离复合 / 拖曳 / Lyα / 重力潮汐
  radiation: off                     absorbers=[HI,HEI] + SED
  BC: reflect                        BC: 行星底 / 外向
  核路径: 只走 L_f                    核路径: Scan + L_f + L_c+LU
```

换组分或只开碰撞 = 换表，不改 `step`。右列那些式子是 **HD209458b 预设 config**，不是核里的 `photoion` / `planet_tide` 分支。

### 2.6 三维框架：进库前必须留的口

1D 回归可以、1D 特例不能写进核。现码 `SSH_220804/` 几乎全是「只有 `VX1` + `IEND→IBEG`」，照搬进库后再升维等于重写 \(\mathcal{L}_c\)。下面几条进库当天就要在，哪怕 `nx2=nx3=1`。

| 口 | 现码怎么写死的 | 进库时怎么留 |
|---|---|---|
| 状态 | 每流体 `(n, VX1, P)`，`TOT_VAR≈13` | 每流体 `(n, vx,vy,vz, P)`；1D 闲置分量为 0。隐式矩阵随 \(N\) 变，不要写死 13 |
| 循环 | 不少地方 `k=0,j=0` 或只扫 `i` | 一律 `k,j,i`；`nx2=nx3=1` 时循环仍在 |
| 相对速度 | `DIFF_VX1`、`CHE/COL_PRS_TRANS` 里只有 \(\Delta v_x^2\) | \(\|\Delta\mathbf{u}\|^2\) 与 \(\Delta\mathbf{u}\) 矢量；拖曳/CX/化学动量三项一起 |
| 体力 | `dv_G` 只加在 `VX1`，半径用 `grid[IDIR].x[i]` | 用格心坐标算 \(\mathbf{r}\)，\(\mathbf{g}\) 投到三个速度槽 |
| 电子 | \(u_e\)、\(P_{\mathrm{eff}}\) 只进 X1 Riemann | 三个 sweep 都加 \(w_s P_e\)；\(u_e\) 三维电荷加权 |
| `RadiationScan` | `for i=IEND…IBEG`，\(\Delta\tau=n\,\mathrm{d}x_1\,\sigma\) | **射线束**：一条射线 = 有序格子 + 路径长 \(\Delta s\)（≠ 一定是 \(\mathrm{d}x_1\)）。第一版 2D/3D 球坐标 = 每个 \((\theta,\phi)\) 一条径向，与 1D 同构。禁止把「\(i\) 递减」当唯一实现 |
| 几何 | 面积/体积常隐含 \(r^2\) | 每向只存算法要用的两类格点：体心 \(x\)、界面 \(x_{i\pm1/2}\)（一根 `xf` 即可），外加度量 `dx`、面上的 \(A\)、格体积因子 \(\mathrm{d}V\)。**Python 交活动界面绝对位置**，C 验单调后垫 ghost（邻格宽）并算中点/`dx`/`A`/`dV`；笛卡尔也填 `A=1`、`dV=dx`，CFL/通量不另读轴标量 `dx1`。配方（均匀、`log`）只在 Python 采样成界面。通量用界面面积（在面上求，球坐标 \(A_r\propto r_{\mathrm{face}}^2\)），源项用格体积；scan 用各向间距拼 \(\Delta s\)。不存顶角、棱心、体积质心（后两者需要时由界面现算；错位 \(B\) 后挂） |
| 边界 | `IBEG_BOX` 只垫 X1；行星底 = `i=IBEG` | 六面 ghost；内边界是「\(r=R_p\) 的面」，不是固定 `i`。解析型 `user`（若开）与源项同一哲学：Python 写封闭式，C 只跑带子、禁止函数指针。**邻格最多两层**（本面内侧第 1、第 2 活动格，与 `ng≤2`、旧 outflow 外推同宽）；再宽的模板、任意 stencil、`max`/`min` op 都不做 |
| 场 API | 1D 向量 | `fields` 形状带 `(…, nz, ny, nx)`；1D 是退化 |
| 分解 | — | 永不切单条射线；2D/3D 切角向或切不同射线 |

第一版辐射**不必**做任意方向短特征或散射。球坐标径向一束就够撑到 3D 行星图，也够码文里的 2D。要防的是把 1D 下标写进 `Lc` 和 scan。

原则：

1. **对外接口薄，包内配置厚。** 化学、体力、相互作用、SED 都在 config 的符号式里；今后的图形界面也只接到这层。C 核只解析这份 config 再 `step`。
2. 配置运行时注入，不用宏决定组分。
3. C 不读带语义的 ini，不带特化反应/外源力；只吃对齐过的 struct / 源项 IR。
4. Python 不重写 Riemann / Jacobian；也不把 HD209458b 公式写进 C。
5. **并行契约：主循环必须单元局部；非局部算子显式点名、单独调用。**
6. 旧树先当对照，不在原地堆 Windows/CUDA。
7. **运行时问题定义走状态机制度**（分发定义、设备施加、惰性维持），见 §2.7；任何约束类特性不另开第二个同步口。

### 2.7 运行时定义：分发定义、设备施加、惰性维持（状态机制度）

`setup` 吃掉的是问题的**初始**语义；运行中会变的定义（外边界 `user` 地板/带子、内部钉死掩膜+钉值、面类型）走状态机制度：核只分发定义，不每步重算、不每步往返。

```text
  驱动（Python / 算例）            C 核（xmf_bc 状态机）
        │  分发定义：面类型 / 地板 / 带子 / 掩膜+钉值
        │  （只在变化时发，可多拍不发）
        ▼
   状态机持有上一拍定义 ── 设备常驻
        │  每步：默认执行空间施加（填皮写鬼格 / 钉死覆写掩膜内区）
        ▼
   热循环自足：不读主机指针、不整场 pull、无每步握手
```

合同（外边界 C2补3、内部钉死 C2补4 同一形状）：

1. **不更新则维持。** 持有的定义持续生效直到被新分发替换；部分分发合法（没送来的项维持上一拍）。维持的是**定义**，不是填值：copy / periodic 等数据依赖面型每步读设备镜子现施加（periodic = 对面源 copy，成对原子分发），仍无往返。
2. **分发与 step 节拍解耦。** 驱动按自己节奏更新，核按自己节奏推进；新加载的定义下一拍生效。
3. **设备侧数据、设备侧施加。** 皮留设备 View，设备核不读主机指针；皮跟活动维数走（1D 两端 / 2D 四边 / 3D 六面，\(n_d=1\) 无该向面）。
4. **核不认几何、不认物理。** 外不认 `HI`、内不认 `cylinder`，掩膜与带子皆是数据。填皮与钉死是两个算子、一个合同。

为什么成立（效率侧）：省钱的是**去同步**——定义设备常驻、热循环无主机握手；状态机是让去同步跨步存活的纪律。收益比值 = 步数 / 定义更新次数，在本问题类里极大（地板/带子近乎不变、钉值偶尔换）。若将来定义需每步随解反馈，正确演进是把更新规则也搬上设备（设备侧闭包），状态机仍是骨架。

三笔义务（清晰性不免费）：

1. **所有权与生命周期显式。** 谁持有、谁释放、换面类型是否为合法转移，写进合同，不留隐状态。
2. **陈旧定义是测试命门。** 驱动以为发了、核没收到 = 用旧定义安静跑错物理；「动态 user 下一拍生效 / 不更新维持」类合同测必备（`tests/test_bc_skin.py`）。
3. **checkpoint 序列化状态机。** 重启只存场不存持有定义 = 偷换边界；checkpoint 开刀时持有定义随场入档。

本制度是「意义放数据侧、核只当解释器」的第三个实例：源项走 IR 带子（§2.2、§4）、几何走活动界面数组（§2.6）、边界与问题定义走状态机。核越笨，越快、越能测、越敢换后端。

---

## 3. 两边各管什么

三层：**外层运行 Python**（config、名字、式子、算例）；**底层计算 C**（网格、Godunov、IR 栈、scan、局部 LU；公开 `extern "C"`）；**并行用 C++ 接 Kokkos**（一份循环，执行空间再选 Serial / OpenMP / CUDA）。MPI 是另一轴（多进程 + halo），不是和 Kokkos 并列的第三种 `backend`。C++ 不是流体对象模型，也不替代 Python 配表。串行路径（A1 补、B \(P_e\)）继续写 C。

| Python | C（底层库） | C++（并行刀） |
|---|---|---|
| 流体列表、每支 HD 或 MHD | 内存布局、打包、Riemann | 同一循环的 host/device 执行 |
| 符号源项、反应式、系数、名字→下标 | 解析 IR，求值 \(\mathcal{L}_c\)，局部 LU | 同上，不改操作码 |
| SED、截面、加热效率 | `RadiationScan` + 逐格加热/电离 | scan 的库前缀积（CUB / Kokkos） |
| 初值、边界类型、输出变量 | ghost；MPI halo 的数据面 | halo 交换的通信后端 |
| 算例、画图、CI | CFL、RK | Kokkos 选执行空间；MPI 另开（域分解） |

关源项应是配置项，不是三个参数碰巧为 0。

---

## 4. 值得保留的物理/数值特色：电子压强

电子**不是**第五个完整动量流体。当前实现（`quasi_neutrality` + `mmd_model_adv.c`）是：

- \(n_e = n_{\mathrm{H}^+} + n_{\mathrm{He}^+}\)，\(\mathbf{u}_e\) 由离子电荷加权；
- 电子只推进**压强/温度**；XUV 加热和辐射冷却进电子能量；
- 推离子动量时，把电子压强按权重加进该离子的有效压强，再走 PLUTO 的 Godunov：  
  \(P_{\mathrm{eff},s} = P_s + w_s P_e\)，先更新 \(\rho_s,\mathbf{u}_s\)，再单独更新 \(P_s\)；
- \(P_e\) 用拼出来的密度和离子加权速度再积一次，只取压强增量。

这样电子压强梯度能进离子 Riemann（双极电场的流体表现），又不必解电子惯性，也不把 \(P_e\) 算进离子内能。

### 文献里有没有同类？

**物理闭合：有，而且空间物理常用，不是孤例。**

- 准中性 + 电子无独立动量 + 单独电子能量/压强方程：Ma et al. (2011, Titan，*JGR*) 已对电子、离子分写压强；电子速度用 Hall 闭合。
- BATS-R-US 多流体带电子压强：Najib et al. (2011)、Rubin et al. (2014)；Dong et al. (2017) 水世界逃逸也写「endowed with the electron pressure equation」。
- Sakai et al. (2019, *JGR*) 明确比较「多流体」和「多流体 + 电子压强方程」：火星上 \(T_e > T_i\)，离子逃逸质量通量高 50–110%，强调双极电场。这是和本处理**物理上最接近**的同行工作。
- Lyon et al. (2025, GAMERA 多流体方程) 写到：\(P_e\) **按电荷份额分给各离子**，并不均分。与这里的 \(w_s P_e\) 同一思想。
- 两温度 MHD / 扩展 MHD（太阳风、聚变）普遍是单速度 + \(T_e\neq T_i\)；多离子时不少码仍把 \(P_e\) 做成 \(P_i\) 的经验比例，或只当动量源项 \(-\nabla P_e\)，并不真推电子能量（见 Tóth 多离子系统的后续离散，如 2024 年 DG 文）。

**数值做法：写得少，这才是可以当成特色说的部分。**

同行多半是：在离子动量里加 \(-\nabla P_e\) 源项，或总压 \(P_i+P_e\) 走单流体 MHD。把 \(P_e\) **按权重拆进每种离子的 Godunov 压强、动量和内能分两次积、\(P_e\) 单独平流**——这是为了复用 PLUTO 单流体积分器想出来的配方，公开方法文里很少按这个拆法写。Tóth 等还指出：激波上非绝热加热如何在 \(T_e\) 与 \(T_i\) 之间分配，取决于格式、不一定收敛；你们的「两次积分」正好踩在这个问题上，重构时应写成明确选项（加热进电子 / 按份额 / 只进离子），并用激波管对照。

结论：电子压强作为**物理**不是独门，火星/Titan/BATS-R-US 已经证明它改变逃逸；作为**接到 Godunov 多离子上的算法**，以及用在 H/He 光化学逃逸，仍然值得保留并写清楚。规划里它是流体表上的一种闭合（`electron: pressure-only, share-by-charge`），不是写死的五流体特例。

---

## 5. 辐射：不挡框架，但必须单独并行

提出辐射，不是因为传递方程难，而是因为它会**干扰并行图**。

流体和碰撞是逐格独立的，CUDA 按格铺线程。现在的 XUV 是

\[
F_i = F_{i+1}\,e^{-\Delta\tau_i},
\]

沿径向（或沿一条射线）的链。若写进和源项同一个 kernel，或径向也做域分解，整套「单元局部」就被扯掉。

**并行契约（先定死，辐射算法可以后换）：**

1. 主循环（通量、碰撞、局部 LU）**禁止**沿射线的串行依赖。
2. 辐射是显式算子 `RadiationScan`：每步一两次，输入各中性柱密度，输出各束 \(F\)。之后 \(F\) 当已知场，源项仍逐格并行。
3. 实现用前缀积（CPU 循环、`MPI_Scan`、CUB / Kokkos scan），不要手写「从 `IEND` 减到 `IBEG`」当唯一路径。
4. 域分解优先切垂直于射线的方向 / 切不同射线；**不切单条射线**（1D 即不切径向）。一卡装下整条径向没有问题。2D/3D 是同一 `RadiationScan`，不是后加的传递模块。
5. 默认物理仍是**瞬时形式解**（\(c\Delta t\gg R\)，Murray-Clay / 现码）。厚格用 \((1-e^{-\Delta\tau})\) 算吸收，保证光子守恒。含时 RT、RSLA、M1 都不挡这条约定，以后若要再加，仍是另一种 `RadiationScan`。

CUDA 开工顺序：先搬局部源项和单流体通量（与辐射无关）；scan 用现成库。不必等「想好辐射」才写并行框架。

---

## 6. 配置上提（清单）

**流体与开关：** 组分表；每支 HD/MHD；电子闭合（压强-only + 电荷份额，见 §4）；几何与维数；Riemann / RK 枚举。

**源项：** 跟随网格与流体场的项（化学、碰撞、冷却、体力）以**符号式**进 `sources[]`。setup 绑到场/网格槽，编成 IR；**C 只解析这份 IR，不带任何特化反应或外源力公式。** 空表 = 关。准中性开/关。Jacobian（差分 → AD / 生成式；\(N\) 小时不上 JFNK）见 `docs/KERNEL_AND_CHEMISTRY.md`。

**光化学 / XUV：** SED、截面、加热效率；吸收体列表；源项总开关（空表 = 关，不用三个数碰巧为 0）。

**运行：** 网格、CFL、初值（解析 / 文件 / Python 数组）、`backend=cuda|host|serial`。

带 schema 版本。禁止行号映射。

---

## 7. 落地顺序

不重写整棵 PLUTO。每步能对照。

| 阶段 | 做什么 | 完成标准 |
|---|---|---|
| **0** | 冻结 `SSH_220804/` 与 HD209458b / smoke_test | 旧可执行文件仍能复现短算例 |
| **1** | Python 流体表 + schema | `shock_tube` 不再依赖 `[Parameters]` 行序 |
| **2** | 按模块拆库 + 单测（`TODO.md` A0–A7）；Python 四步 | `pytest tests/` 绿；空表激波管从 Python 跑通 |
| **3** | \(\mathcal{L}_c\) 已表驱动；**B \(P_e\) 闭合**（还原 Xing+2023） | 关电子 = 现 A5；开 \(P_e\) + He 后原子/电荷/动量回归 |
| **4** | `RadiationScan` 已拆出 | 主循环无沿射线依赖 |
| **5** | **C 并行（本篇）**：Kokkos **Serial → OpenMP → 单卡 CUDA**（后一刀吃前一刀；OpenMP 易、CUDA 难） | 一份循环换执行空间；与 CPU 逐格对照后再谈加速比。不上 MPI |

A1 补之后先 \(P_e\)，再并行。理想 MHD / 追踪辐射 / 二次耗散见 MHD 树 paper 2；MPI / AMR 见 paper 3。不挡本篇 Kokkos 界面。  
核的布局和三种算子从收成库起按 `ndim=1|2|3` 铺，不要先做死 1D 再「升维」。1D 是回归主路径；2D 结果随框架给出（码文会用）；3D 与 2D 同路径。

---

## 8. 刻意后做

- 不在现有 `makefile` + `Linux.mpicc.defs` 上堆 CUDA。
- 第一界面是 Python API + 同一套 schema（dict / YAML）；图形 config 后做，仍走这套处理再进 C。
- 不把 PLUTO 译成 Python。
- **paper 2 / 3，本篇不要开：** 清单只在 MHD 树 `docs/TODO.md`。paper 2 = MHD 与附属物理（理想 MHD/CT + 追踪辐射 + 二次耗散）；paper 3 = 大规模计算（MPI + AMR）。Xing 表本篇继续手写 `sources[]`。图形界面不进这三篇。
- MPI 与 Kokkos 执行空间不是同一轴；不在旧 `makefile` 上堆。本篇并行实现参考 Idefix / AthenaK 的执行空间，不参考旧 `Linux.mpicc.defs`，也不做多 rank。
- 含时辐射 / RSLA / FLD 不是本篇范围。

---

## 9. 风险

对照码里认下来的两处劣势（守恒 / 正定性、GPU 可移植）及管法见 `docs/APPENDIX.md`，不在这里重判骨架。

- **GPLv2**：绑 PLUTO 衍生核则发布形态仍受 GPLv2。
- **单位与闭合**：`NDT↔RHO`、`TMP↔PRS`、电子压强份额必须在 schema 里写死。
- **激波上 \(T_e/T_i\) 加热分配**：见 Tóth 等对 XMHD 激波的警告；两次 Godunov 的配方要有测试和可调份额。
- **辐射若漏回主 kernel**：CUDA 占用和正确性一起坏。拆 `RadiationScan` 是硬约定。
- **两套结果**：对照关掉之前不删旧树。
- **进库的 C 若只能编成 MinGW DLL**：上 CUDA 等于第二遍重写。收成库时按 host / 以后 cuda 可换写。

---

## 10. 一句话

做成好用的多流体工具：繁杂配置收进 Python 包，对外只留 setup/step/fields。三篇都走 ApJS：本篇单机 + \(P_e\)；paper 2 = MHD 与附属物理；paper 3 = 大规模计算（MPI + AMR）。2D 结果随框架给出。守恒和 GPU 起步弱于最强对照，接受。
