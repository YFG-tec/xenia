# 2D 两流体板状喷流

笛卡尔盒 $x\in[0,16]$、$y\in[-5,5]$（以束半径 $R=1$ 为单位）。重支
$\rho=4$ 铺满、初速为零。轻支从左边界窄带射入，$\rho=1$，
$M=V_j/c_{s,\mathrm{light}}=2$，压强匹配 $P=0.6$，$\gamma=5/3$。
空源项，只推轻的。

入口用板状喷流常用的 tanh 剪切层（Ferrari, Trussoni & Massaglia 1982,
*MNRAS* 198, 1065；Bodo et al. 1995 反对称板状模），再叠一层横向拍打：

$$
\eta(y)=\tfrac12\bigl(1-\tanh((|y|-R)/a)\bigr),\qquad
v_x=V_j\,\eta,\qquad
v_y=A V_j\sin(2\pi t/\tau)\,e^{-(y/R)^2}
$$

$a=0.12\,R$，$A=0.10$，$\tau=2$。轻束两侧的速度跳是 Kelvin–Helmholtz
源；反对称拍打对应 Bodo 等人的板状反对称模。没有磁场，不是行星大气。

IR 没有 `abs` / 分段，入口钉在算例层（每步覆盖最左几格），不进核。
求解器用 `hllc` + `weno3` + `rk2`：`tvdlf` 太耗散，KH 涡出不来。

图在 `output/jet.png`（轻支密度、$|\Delta\mathbf{u}|$、$\omega_z$）和
`output/beam.png`（$\log\rho$）。

## 碰撞开关（D1 后半）

`--tier on` 打开与 `mf_cloud_2d` 同形的反对称拖曳带
（$\mathrm{d}\mathbf v_{\mathrm{light}}=-K n_{\mathrm{heavy}}\Delta\mathbf v$，
$\mathrm{d}\mathbf v_{\mathrm{heavy}}=+K n_{\mathrm{light}}\Delta\mathbf v$，
等质量逐格动量守恒），$\tau=1/K=2$（束内 $n_{\mathrm{light}}=1$ 的拖曳时标，
束穿越时标 $16/V_j=8$ 的四分之一）。换档只换 `sources[]`，其余一字不改：

| tier | $K$ | $|\Delta\mathbf{u}|_{\max}$ | $v_{y,\mathrm{light},\max}$ | nstep | wall (s) |
|---|---:|---:|---:|---:|---:|
| off | 0 | 4.397 | 1.975 | 3370 | 53.1 |
| on | 0.5 | 1.986 | 0.634 | 2823 | 235.4 |

off：束两侧 KH 卷起、横向拍打发展，$|\Delta\mathbf u|$ 在剪切层全程保持。
on：拖曳把重支拖进束里，剪切层层流化，束头光滑成帽状，
$|\Delta\mathbf u|$ 只剩喷嘴新注入段（重支尚未追上）与波前弧。
同一几何、同一核，「拧碰撞」就是换一条带子。图在 `output/on/jet.png`。

on 档壁钟约为 off 的 4.4 倍：隐式源步每格 $10\times10$ LU 是预期成本，
不是回归（off 无带子，源步不跑）。
