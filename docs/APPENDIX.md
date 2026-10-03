# 附录：对照已收集代码

材料：`docs/HD_MHD_CODES.md`、`docs/MULTIFLUID_CODES.md`、本地 `codes/`。  
正文骨架：`docs/ROADMAP.md` §2。化学与 Jacobian：`docs/KERNEL_AND_CHEMISTRY.md`。

本附录只回答一件事：**和那几十个码比，这套骨架要不要改。** 结论是不改。有两处明显劣势，接受，用执行管住，不回头拆分层。

---

## A.1 结论

骨架没有问题。Python 配表、C/CUDA 跑对流 / 局部刚性源项 / 点名的 `RadiationScan`，Dedalus、FARGO3D、Idefix、BOUT++ 都走同一层，不是外行发明。

空位也还在。开源天体码的「多流体」几乎都是气 + 无压尘埃；带电多流体在 BATS-R-US / GAMERA，不是 Athena++ 那种可换发行版。H/He 分速度 + XUV 化学没有第二套现成核可换。

相对那批码，**较大、且认下来的劣势只有两个：数值守恒 / 正定性，以及 GPU 可移植特性。** 其余（GPLv2、1D 观感、Windows 工具链、辐射 scan 的对外叫法）是代价，不是改骨架的理由。

---

## A.2 可接受劣势一：守恒性与正定性

FARGO3D（2019）把气–尘隐式拖曳做到动量守恒到机器精度。BATS-R-US 在多离子正定性上修了很多年。Castro / FLASH 的化学走 CVODE，并盯元素守恒。

我们的 \(\mathcal{L}_c\) 更脏：化学 + 电荷交换 + 库仑 + 电子代数闭合，现码是局部后向 Euler + 有限差分 Jacobian + 稠密 LU。这一档比「专门设计的隐式拖曳」弱，Newton 可以出负密度、电荷不闭合、碰撞不守恒。化学时标远小于 CFL 时，一阶算子分裂误差也可以盖过 Riemann。

**接受。** 不因此改成单流体、也不因此先接一套燃烧码。管法：

- 空表激波管继续只考双曲守恒（已有 Toro）。
- 开源项后加三条回归，不叠在 Riemann 上：原子数、电荷（准中性投影后）、流体间动量交换。学的是 FARGO3D 2019 的测试态度，不是抄气–尘矩阵。
- 密度地板 / 温度裁剪写进速率律；\(n_e\) 必须链式求导，不当常数。
- 源项子循环默认不开：本地 \(\mathcal{L}_c\) + 隐式 Newton 已覆盖刚度。不必第一天接 CVODE。\(N\sim 10\) 仍用稠密 LU，不上 JFNK。单次 Newton 扛不住再谈切段。

电子压强进离子 Riemann（\(P_{\mathrm{eff},s}=P_s+w_s P_e\)）是特色，也是独审点。Tóth / Lyon 已写明激波上 \(T_e/T_i\) 加热不唯一。份额做成选项并做管测，不改闭合类型。

---

## A.3 可接受劣势二：GPU 特性

2023–26 那波文章几乎全是「一份源码、多后端」：Idefix / AthenaK / KHARMA（Kokkos）、gPLUTO（OpenACC）、FARGO3D（C→CUDA）、Castro（AMReX）。我们规划是先串行 C 核，再上 CUDA。相对他们，可移植来得晚，ctypes 占位核若按「只能编 Windows DLL」长，上 GPU 就是第二遍重写。

**接受。** 不第一天上 Kokkos，也不改公开 API。管法：

- 核从收成库的那天起按「同一份 C，host / 以后 cuda 可换」写：主循环单元局部，`RadiationScan` 单独点名。这已经写在 ROADMAP §2.3、§5。
- 不要先做一个只能 MinGW `.dll` + ctypes 的内环，再翻译成 CUDA。占位 HLL demo 可以扔；`mmd_AdvanceStep` 进库时就要能换后端。
- Windows 当开发机和纯流体回归。GPU 对照放 WSL 或 Linux（CUDA 官方链要 MSVC，和 conda MSC Python、MinGW 不是一条链）。
- 阶段 2 之前不上 GPU；上卡后先逐格对照 CPU，再谈加速比。

---

## A.4 不改骨架的其它代价

| 代价 | 相对谁 | 怎么放着 |
|---|---|---|
| 链 PLUTO → GPLv2 | Athena++ BSD；SWMF Apache | 发布形态认下来；借鉴算法 ≠ 复制源码 |
| 第一篇码文不做完整多流体 MHD / AMR | 几乎所有 Godunov 发行版 | 维数（到 3D HD）是框架能力；MHD/AMR 后开。空位在化学多流体，不在「会不会 2D」 |
| `RadiationScan` 只是径向前缀积 | Athena++ / Castro / MURaM 的辐射 MHD | 对外不要写成「带辐射核」；算法以后可换，并行契约不换 |
| MHD 开太早会掉进 BATS/GAMERA 主场 | 空间天气多离子 | HD + 化学表稳了再开 |
| 通用表没有校验器 | FLASH 模块过多的教训 | schema、量纲、计量、空表回归；核不解释字符串 |

---

## A.5 并不弱、不必因为总表而改的

- **Python 配、C 跑**：先例见上。弱的若出现，是核，不是这层。
- **封闭模板 / setup 解析反应式**：Cantera、KROME、CHEMKIN 也是命名速率律，不是核内解释器。
- **自己养带电多流体**：发行版不会替你做 H/He/电子闭合。
- **\(N\approx 13\) 不用 JFNK**：大网络才是 CVODE 的地盘。

---

## A.6 码文叙事（开源是前提）

仓库必然开源。

**卖点（摘要、方法节、验收）：** 多流体 + 辐射这套东西——分速度、通道表 \(\mathcal{L}_c\)、电子闭合、`RadiationScan` 与对流/源项分裂。不要把 HD209458b 再画成第二篇科学图。

**尘埃兼容（方法节提一句，不当本篇科学图、不进验收）：** 某支中性流体把热压强（或声速）收到接近 0，方程回到无压尘埃极限。FARGO3D（Benítez-Llambay & Masset 2016；Benítez-Llambay et al. 2019）、Idefix（Lesur et al. 2023）、MPI-AMRVAC（Porth et al. 2014）已经用同一极限做气–尘多流体。摘要只写「公式兼容」，不写 streaming instability，也不声称隐式拖曳打平 FARGO3D 2019 的机器精度动量守恒。

**包装（现在算基础配置，没有别人不会用）：** `setup/step/fields`、名字对字段的 schema、开源能复现、以及可换后端（串行先通，host / CUDA / 以后 Kokkos 同一份核）。Idefix / AthenaK 已经把「一份源码多后端」做成码文标配；我们摘要不靠这个压过他们，**发布物不能缺。** 开发顺序仍是先单核把流程跑通，但进库当天就要按可换后端写，不能做成只能本机 DLL 的死核。

正文三块，开发也按这三块验收：

| 突出什么 | 论文里要写清的 | 验证上必须有 |
|---|---|---|
| **多流体框架** | 每种成分自己的 \(\mathbf{u},p\)；通道表装配 \(\mathcal{L}_c\)；HD209458b 只是第一张表。方法节补一句：一支流体 \(P\to 0\) 即无压尘埃极限（FARGO3D / Idefix / AMRVAC 已用） | 空表 Toro；开源项：原子数 / 电荷 / 动量交换。尘埃极限不单独立算例 |
| **电子处理** | 无独立动量；准中性；\(P_{\mathrm{eff},s}=P_s+w_s P_e\) 进离子 Godunov；加热份额是选项 | 开关份额的激波管；声明 \(T_e/T_i\) 在激波上不唯一 |
| **算子分裂** | `RadiationScan`（非局部）与 \(L_f\)（双曲）与 \(L_c+J+\)LU（局部隐式）分开；\(F\) 当外源，\(J\) 不跨格 | 光子数 / 厚格 \(1-e^{-\Delta\tau}\)；关 scan 或关源项可单独回归 |

包装放实现节，不当标题贡献，但发布缺包装就不算工具。  
**维数不是补丁：** 核按 1D/2D/3D 多流体设计。1D 是第一条回归，2D 结果是同一套算子的自然输出，码文里会给。进库当天必须留的口（三速度、\(\|\Delta\mathbf{u}\|\)、射线束、六面边界）见 `ROADMAP.md` §2.6，不要把现码的 `VX1` + `IEND→IBEG` 编进库。MHD 仍可后开，和维数无关。  
和 Xing (2023) 的切法：那篇是科学结果，这篇是框架、闭合与分裂——旧树短算例只作对照，不重画逃逸结论。

---

## A.7 一句话

骨架不动。卖点是多流体 + 辐射（框架、电子、分裂）；包装（四步 API、可换后端 / Kokkos）是别人会用的门槛，摘要不靠它，发布不能缺。守恒和 GPU 起步弱于最强对照，接受。不拆 `setup/step/fields`，也不把化学解释器或每格 Python 塞进核。
