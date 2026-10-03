# rad_hd_1d — 辐射驱动的双流体（HI/HII）一维球对称逃逸

A5 切片「复现基本辐射–源项驱动」的示例入口。对照参考模型
（SSH_220804 的 HD209458b 热逃逸设定），但走的是本仓自己的四步
API：`setup → step → fields → finalize`，不读 `pluto.ini`。

## 跑法

```bash
pip install .
python examples/rad_hd_1d/run.py
python examples/rad_hd_1d/compare_mesh.py   # 均匀 1000 vs log-stretch 128
```

没有 CLI。网格、停时、拉伸、束数、出图、步进日志都写在
`rad_config.py` 置顶的 `RUN` 里，改完再跑 `run.py`。默认是
`tstop=2000 s`、`nx=128`、均匀网格、53 束、每 100 步打一行
`step / t / dt / resid`（`log_every=0` 关掉）。冒烟可改成
`tstop=2e2`、`n_beam=1`、`plots=False`。

`stretch="log"` 走 A1 补的 Python 界面工厂：宽度按
`stretch_factor` 几何增长，底（行星侧）密、顶疏；C 核只收到界面
数组（严格递增校验 + 邻格宽 ghost）。`nx=128`、`factor=1.025` 时
首格 \(\approx 0.010\,R_p\)（均匀格是 \(0.070\,R_p\)，而初值包络尺度
\(H_{ic}=0.04\,R_p\)），末格 \(\approx 0.23\,R_p\)，宽比约 23；
ghost 宽 = 邻格宽，左侧 ghost 落在 \(R_p-\Delta_0\)、\(R_p-2\Delta_0\)，
远离 \(r=0\) 红线。`stretch="none"`（默认）保持均匀考试基线。

为何值得：`tstop=2000`、53 束下均匀 128 的**辐射开**跑不分辨底部
吸收层（底部 \(\Delta\tau/\)格 \(\approx 4.4\times10^3\)，全部 XUV 灌进
一两个格子），t≈600 s 起 dt 崩到 \(10^{-6}\) s，爆出数值爆震
（\(v_{HI}\to-6\times10^8\) cm/s、\(T_{HII}\to3\times10^9\) K、
9627 步）；同样 128 格换 `stretch="log"`、`stretch_factor=1.025`
与均匀 256 在外区对到 n 2.6%、\(v_r\) 3×10⁵ cm/s 量级，94 步、
0.7 s 跑完（验证对拍脚本在 `.tmp/verify_a1_rad_hd_1d.py`，图
`a1_verify_*.png`）。透射流量衰减与网格无关（柱密度积分），差的是
加热沉积的分辨。

注意：`n_beam=1` 的单束回退只保未衰减边界处的 Γ 与 H；
`σ_eff = H/F_tot ≈ 6.4e-19 cm²` 在本柱密度下光学极厚
（底部 `Δτ/格 ≈ 4.4e3`），有效光束在最外层格内就被吸收，
on/off 会逐位相同——它只是冒烟模式，物理对比请用默认 53 束
（详见 REPORT.md §3）。

产物写在 `output/`（gitignore）：`profiles.png`、`velocity.png`、
`pressure_ionfrac.png`、`flux_attenuation.png`、`history.png`、
`mass_flux.png`、`numbers.txt`。REPORT.md 引用这些图和数字块。

## 物理设定

- 行星：HD209458b 按比例放大，`R_p = 1.36 × 7.1492e9 cm`，
  `M_p = 0.69 × 1.8982e30 g`；球对称网格 `r ∈ [R_p, 10 R_p]`，
  `nx = 128`，`ng = 2`。
- 组分：HI / HII，质量同为 1.00794 amu。初态 `n_HI(R_p) = 1e13 cm⁻³`，
  `n_HII/n_HI = 1e-3`，`T = 1500 K`，`P = n k_B T`，
  `n ∝ exp(-(r-R_p)/H_ic)`（`H_ic = 0.04 R_p`，地板 `1e6 cm⁻³`），
  `v = 0`。底层钉在 \(R_p\)；包络压在末态风之下，避免
  \(r^{-2}\) 那种「全域过密、靠通量把多余质量慢慢扔出去」。
  v=0 初态**不**是稳态，早期内区会被行星重力拉一下；外区靠恒星潮汐解绑。
- 辐射：SED 与截面表在 `data/`（从 SSH 参考表复制），工厂按
  `XUV_RESIZE` 标定后再乘 `XUV_FACTOR / R_Orbit² = 0.25/0.04707`；
  默认 53 束，每束 `hν_b = hc/λ_b × 1e8`；光学厚度按格
  `Δτ = σ_b n_HI Δs`，扫描从外边界向内指数衰减。
- 源项：光电离（HI、HII 各一条）、光致加热（HII，`η = 0.20`）、
  复合（HI、HII，温度幂律，`T_HII+1 K` 地板）、电离–复合拖曳、
  化学加热、重力+潮汐
  （`dvx = -G M_p/r² + 3 G M_star r/a³`），共 10 条磁带，走同一套
  半隐式向后欧拉（每格 10×10 稠密 LU，单次线性化）。
- 数值：HLL + PLM + RK2，`cfl = 0.4`；每步
  `dt = cfl / max_格 Σ_d (s_{d,格}/dx_{d,格})`（A1 补口径：逐格
  `s_{d,格} = max_流体 (|v_d| + c_s)`，只和活动向求和，逐格 `dx`；
  非均匀网格下 dt 由最紧的格定）。通量更新走逐格
  `U -= dt/dV (A₊F₊ − A₋F₋)`（球面 `A=r²`、`dV=|Δr³|/3`）。
- 边界：`x1beg = user`（解析 `user_bc`：ghost 赋底层
  `(n, v=0, P)`，不钉最内活动格），`x1end = outflow`（现实现 = copy）。

## 单位

全 cgs：所有单位因子取 1（`unit_density = unit_velocity = unit_length = 1`），
代码值即 cgs 值。**长度单位是厘米，不是 R_p**（仓库契约禁止
`UNIT_LENGTH = R_p`）。温度、截面等常数用
`xmf.config.KB / AMU` 与 `rad_config.AMU_H`。

## 结构

- `rad_config.py` — 算例私有配置：置顶 `RUN`（跑法旋钮）+
  HD209458b 考试工厂 `hh_rad_1d(...)`、行星常数与 SED/截面读表。
  不在公共池 `xmf.config` 里，`import xmf` 不导出。pytest 仍可往
  工厂传 `nx` / `n_beam` / `tstop` 覆盖单字段。
- `run.py` — 读 `RUN`、建 cfg、pack 双流体初态（`xmf_state_pack`
  全块布局）、on/off 两跑到 `tstop`、记 `(t, dt, source_resid)`
  历史、出图。无 argparse。
- `data/hd209458b-1163.dat`、`data/photoIon_sigma.dat` — SED 波长–通量
  表与 `sigma_H` 截面表。
- `REPORT.md` — 数字与讨论（on/off 对比、隐式残差、厚格说明、
  t_stop 依据）。
- `compare_mesh.py` — 调 `run.one_run`，对拍等距 1000 格与
  log-stretch 128；写出 `report_mesh_compare.md` 和
  `output/mesh_compare/`。
