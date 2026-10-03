# 2D 圆柱绕流

笛卡尔盒 $x\in[-4,12]$、$y\in[-5,5]$，圆柱直径 $D=1$ 放在原点。来流
$\rho=1$、$U_\infty=1$，马赫 $M=0.25$（靠近不可压圆柱绕流的常用设定）。
$\gamma=1.4$。空源项，单流体。

核没有固体边界，也没有物理粘性 $\nu$（$\nu,\eta$ 是 paper 2 的 P 块）。圆盘
$r\le D/2$ 在算例层每步钉成无滑移；左边界钉均匀来流。尾迹的有效
雷诺数来自 `hllc` + `weno3` 的数值耗散，不是 $Re=U D/\nu$ 的粘性定义。

题型对照 Braza, Chassaing & Ha Minh 1986, *JFM* 165, 79（二维圆柱近尾迹，
$Re=100,200,1000$）和 Williamson 1996, *Annu. Rev. Fluid Mech.* 28, 477。
Braza 用短时扰动把涡脱落拉起来；这里同样在 $t<2$ 给来流一个
$0.08\,U_\infty$ 的横向踢，避免等自然失稳。

$$
St \approx 0.2 \quad\Rightarrow\quad T \approx D/U_\infty / St \approx 5
$$

默认收到 $t=16$，大约三个脱落周期。二维、无物理 $\nu$ 时，尾迹是
卡门涡街加上剪切层上的小涡，不是三维 $Re=3900$ 的 LES 湍流。

图在 `output/wake.png`（$|u|$ 与 $\omega_z$）、`street.png`，动画在
`output/street.gif`（由 `snaps.npz` 重画，默认 $\Delta t=0.25$）。
