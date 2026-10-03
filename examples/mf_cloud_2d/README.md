# mf_cloud_2d — 两流体激波打重云（D2，paper 1 经典功能演示）

Klein, McKee & Colella (1994) 云破碎题的**两流体版**：激波只存在于流体 A
（轻，风），重云是**另一支流体** B；动量只靠 `sources[]` 里的碰撞带传递。
碰撞系数 $K$ 是唯一物理旋钮：

- $K=0$（`--tier off`，空 `sources[]`）：激波穿过流体 B，云纹丝不动——
  分速度的存在证明，单流体码画不出来；
- $K$ 大（`--tier strong`）：激波后高速风拖曳云，形变/剥离。

无磁场、无辐射、无行星参数；类比口径太阳风 / ISM（不写成 C 激波或日球结论）。

## 用法

```text
python examples/mf_cloud_2d/run.py --tier off --no-plot       # K=0 冒烟
python examples/mf_cloud_2d/run.py --tier strong              # 强耦合
python examples/mf_cloud_2d/run.py --tier mid --nx 256 --ny 128
python examples/mf_cloud_2d/plot.py --tiers off mid strong    # 合并面板
```

输出：`output/fields_<tier>.npz` + `summary_<tier>.json` + `panels_<tier>.png`，
三档齐时另有 `paper_panel.png`。

## 设置（码单位）

- 区域 $[0,4]\times[-1,1]$，四边 `copy`；激波走初值（$x<0.5$ 为
  Rankine–Hugoniot 后态），激波后流超声速，左边界储层不受扰动。
- 入射激波 $M_s=10$，$\gamma=5/3$；风 $(\rho_{A0},P_{A0})=(1,1)$；
  云：圆心 $(1.5,0)$、半径 $r_c=0.25$、$\chi=\rho_B/\rho_{A0}=10$，压平衡。
  流体 B 以 $\rho=10^{-2}$ 均匀背景充满区域、$P_B\equiv P_0$：
  各流体自带压平衡（流体间不共享压强），云边缘是稳定接触间断。
- 云破碎时标 $t_{\mathrm{cc}}=\sqrt{\chi}\,r_c/v_{A1}$；默认跑到 $3\,t_{\mathrm{cc}}$。
- 拖曳率定义在激波后风密度上：$\tau_c = 1/(K\rho_{A1})$，
  档位 `off / weak / mid / strong` = $\tau_c/t_{\mathrm{cc}} = \infty / 10 / 1 / 0.1$。
- 两支同质量 → 动量交换逐格严格反对称，带内无除法
  （带子只读 $n$、$v$，物理口不加载，`mass_code=1`，带内 $n_s$ 即密度槽）。
- 求解器：`hllc + plm + rk2`，CFL 0.4。用 plm 而不是 `mf_jet_2d` 的 weno3：
  云摧毁后欠分辨丝状结构上 WENO3 positivity 不足，$t\approx1.2\,t_{\mathrm{cc}}$
  即崩（已记录于 REPORT）。

规划与验收见 `overleaf/PAPER1_COMPUTE_PLAN_zh.md`（本例不开 SESSION 卡片）。
