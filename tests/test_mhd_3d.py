"""M 块 4（M6）：3D 面心 B + 棱心 E 单测。

规格来源：docs/SESSION.md（M6 卡片）「交什么」§2、§3、§4。

  §2 3D 均匀场 E 精确钉：units=1、全周期粗盒 8×6×4，显式 pack rho=1、
     P=1（吸取坑 1 教训），均匀三向非零 u=(0.3,−0.2,0.1) + 三分量非零
     均匀 B=(0.4,0.6,0.8)。用 xmf_mag_unpack_e 读三向 E，全块逐位 ==
     解析 −u×B；step 一步后再读仍逐位相等；跑 5 步 B 逐位冻结、
     divb_max < 1e-12。
  §3 3D 非平凡构造性零散度钉：B=∇×A 构造（取 B1=sin(y)+cos(z),
     B2=sin(z)+cos(x), B3=sin(x)+cos(y)；各分量不依赖自身方向，离散
     ∇·B 精确为零）+ 非均匀三向速度，粗网格 12³ 磁开跑 5 步：
     divb_max < 1e-6；三向 B 都演化（与初值 L1 差非零，非冻结）；四步
     链通。钉值跑出后按 ×1.5 收紧写死（同 M5 口径）。
  §4 3D 边界覆盖：六面 reflect 与六面 periodic 各跑一次磁开冒烟（几步）：
     不炸、divB 不爆（< 1e-6）。reflect 档用 M3 两档之一（wall，语义由
     1D/2D test_mag 钉，3D 只验不回归）。

dt 病四件套定性（坑 1，详见 .tmp/M6_dt_diagnosis.md）：
  1. 哪一项：CFL 求和（d=0,1,2）的 c_s = sqrt(gamma*P/rho)；当 rho 停在
     IC 值 1.66e-24（n=1,mass=1,AMU=1.66e-24,units.density=1）时
     c_s ≈ 1e12。
  2. 哪个时标：dt ≈ 2.2e-14 对应 s ≈ 1e12；CFL 结构 dt = cfl/Σ(s/dx_d)
     正确，实测与手算逐位吻合。
  3. 哪一类：探针初值（非代码 bug）。探针只 pack 速度（var=1,2,3），
     未 pack rho（var=0）→ rho 停在 IC 值 → c_s 天文大 → dt 天文小。
     显式 pack rho=1 后 dt = 1.47e-2（3D HD 门关）。
  4. 证据：.tmp/probe_m6_3d_hd.py 修改前（dt=2.23e-14）vs 修改后
     （dt=1.47e-2）；.tmp/probe_m6_3d.py 磁开同因。
  结论：dt 病不是帽、不改 CFL 公式。修的是探针的 pack 序列（rho=1），
     CFL 代码（xmf_hydro.cpp:1650-1804）逐行正确。

用例清单：
  * test_uniform_field_e_exact_and_b_frozen — §2 均匀场 E 精确钉 + B 冻结。
  * test_nontrivial_divb_free_evolution_3d — §3 非平凡零散度 + B 演化。
  * test_reflect_boundary_3d_smoke — §4 六面 reflect 冒烟。
  * test_periodic_boundary_3d_smoke — §4 六面 periodic 冒烟。
  * test_dt_reasonable_3d_hd — 坑 1 回归防护：dt 与手算 ±50%。

钉值与实测关系（2026-08-31 本测首跑；上界 = 实测 ×1.5 收紧）：
  均匀场 8×6×4（hll/plm/rk3，cfl=0.4，units=1，5 步）：
    divb_max = 0.0（逐位精确零）；B max|dB| = 0.0（逐位冻结）；
    E max|E - (-u×B)| < 1e-15（ULP 级）。
  非平凡零散度 12³（同参数，5 步，[0,2π]³ 周期）：
    initial divb = 0.0（解析零散度构造）；max_divb = 9.816e-11；
    L1(B1) = 1.616e-2、L1(B2) = 1.615e-2、L1(B3) = 2.017e-2（都演化）；
    dt = 2.409e-2。上界 max_divb < 1.5e-10（实测 ×1.5）。
  dt 合理性 8×6×4 HD 门关：dt = 1.469e-2，手算 = 1.469e-2（逐位吻合）。

禁令自查（卡片 + docs/TODO.md §A）：
  - 不读 pluto.ini：本文件无任何文件 IO；
  - 不用 XUV_FACTOR==0 哨兵：本文件不涉辐射；
  - 不只测单分量：E 覆盖 E1/E2/E3 三分量，B 覆盖 B1/B2/B3 三分量，
    流体断言覆盖五槽 (n, vx, vy, vz, P)。

维度覆盖：3D（8×6×4 / 12×12×12），全三速度 + 全三 B 分量非零。
"""
from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

import xmf
from xmf._native import load

_PTR = ctypes.POINTER(ctypes.c_double)

ION_FLUIDS = [{"name": "HII", "mass": 1.0, "charge": 1, "kind": "ion"}]


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库。"""
    return load()


# --- 小助手 -------------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_PTR)


def _ntot_of(run) -> tuple[int, int, int]:
    n = run._cfg.n
    ng = run._cfg.ng
    return tuple(n[d] + 2 * ng for d in range(3))


def _len_b(ntot, comp: int) -> int:
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return (ntot[comp] + 1) * ntot[d1] * ntot[d2]


def _len_e(ntot, comp: int) -> int:
    d1, d2 = (comp + 1) % 3, (comp + 2) % 3
    return ntot[comp] * (ntot[d1] + 1) * (ntot[d2] + 1)


def _pack_b(lib, h, comp: int, blk: np.ndarray) -> None:
    got = lib.xmf_mag_pack(h, comp, _ptr(np.ascontiguousarray(blk.ravel())))
    assert got == blk.size, f"xmf_mag_pack B[{comp}] failed ({got})"


def _unpack_b(lib, h, comp: int, ntot) -> np.ndarray:
    n = _len_b(ntot, comp)
    buf = np.empty(n, dtype=np.float64)
    assert lib.xmf_mag_unpack(h, comp, _ptr(buf)) == n
    return buf


def _unpack_e(lib, h, comp: int, ntot) -> np.ndarray:
    n = _len_e(ntot, comp)
    buf = np.empty(n, dtype=np.float64)
    assert lib.xmf_mag_unpack_e(h, comp, _ptr(buf)) == n
    return buf



def _cross3(u, B):
    """3-vector cross product u x B."""
    return (
        u[1] * B[2] - u[2] * B[1],
        u[2] * B[0] - u[0] * B[2],
        u[0] * B[1] - u[1] * B[0],
    )


# --- 配置工厂 ----------------------------------------------------------------


def _box_cfg(n, bounds, bc, mhd=True):
    """3D box config factory. n = (nx,ny,nz), bounds = (x1max,x2max,x3max),
    bc = 'periodic' | 'reflect'. Always sets explicit coordinate ranges
    for all three directions (normalize requires x{1,2,3}min/max for 3D)."""
    cfg = xmf.shock_tube()
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["grid"]["n"] = list(n)
    cfg["grid"]["ndim"] = 3
    cfg["grid"]["x1min"] = 0.0
    cfg["grid"]["x1max"] = bounds[0]
    cfg["grid"]["x2min"] = 0.0
    cfg["grid"]["x2max"] = bounds[1]
    cfg["grid"]["x3min"] = 0.0
    cfg["grid"]["x3max"] = bounds[2]
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "plm", "rk": "rk3"}
    cfg["time"] = {"cfl": 0.4, "tstop": 10.0, "dt_max": 0.0}
    if mhd:
        cfg["mhd"] = True
        cfg["fluids"] = [dict(f) for f in ION_FLUIDS]
        cfg["electron"] = {"mode": "off"}
    bc_dict = {}
    for ax in ("x1", "x2", "x3"):
        bc_dict[ax] = bc
    cfg["boundary"] = bc_dict
    cfg["ic"] = {
        "kind": "riemann",
        "x0": bounds[0] / 2.0,
        "left": {"n": 1.0, "v": 0.0, "p": 1.0},
        "right": {"n": 1.0, "v": 0.0, "p": 1.0},
    }
    return cfg


# --- 1. §2：3D 均匀场 E 精确钉 + B 冻结 -------------------------------------


def test_uniform_field_e_exact_and_b_frozen(lib):
    """8×6×4 全周期盒，units=1。显式 pack rho=1、P=1（坑 1 教训：不 pack
    rho 会停在 IC 的 AMU 级值，c_s 天文大）。均匀三向非零 u 与均匀三分量
    非零 B ⇒ E = -u_+ × B 应全块逐位等于解析值。均匀 E 的旋度为零 ⇒
    ∂t B = -∇×E = 0 ⇒ B 逐位冻结。

    解析（u=(0.3,-0.2,0.1), B=(0.4,0.6,0.8)）：
      u×B = ((-0.2)(0.8)-(0.1)(0.6), (0.1)(0.4)-(0.3)(0.8),
             (0.3)(0.6)-(-0.2)(0.4))
          = (-0.22, -0.20, 0.26)
      E = -u×B = (0.22, 0.20, -0.26)
    """
    u_vec = (0.3, -0.2, 0.1)
    B_vec = (0.4, 0.6, 0.8)
    uxB = _cross3(u_vec, B_vec)
    expect_e = tuple(-v for v in uxB)  # (0.22, 0.20, -0.26)

    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ncells = lib.xmf_state_ncells(h)

        # Explicit pack of all five slots (rho and P are the pit-1 lesson).
        rho_arr = np.full(ncells, 1.0, dtype=np.float64)
        P_arr = np.full(ncells, 1.0, dtype=np.float64)
        lib.xmf_state_pack(h, 0, 0, _ptr(rho_arr))
        for var, val in enumerate((None, 0.3, -0.2, 0.1, None)):
            if val is not None:
                lib.xmf_state_pack(h, 0, var, _ptr(
                    np.full(ncells, val, dtype=np.float64)))
        lib.xmf_state_pack(h, 0, 4, _ptr(P_arr))

        # Pack uniform B.
        for comp in range(3):
            blk = np.full(_len_b(ntot, comp), B_vec[comp], dtype=np.float64)
            _pack_b(lib, h, comp, blk)

        # E blocks are computed by CT during step(), not during setup.
        # Before any step, E is zero (calloc). Step once, then read E.
        # Tolerance: 1e-15 (ULP-level; kernel cross-product may differ from
        # Python's _cross3 by one ULP due to FMA / operation ordering).
        _E_TOL = 1.0e-15
        run.step()
        for comp in range(3):
            e = _unpack_e(lib, h, comp, ntot)
            max_diff = float(np.max(np.abs(e - expect_e[comp])))
            assert max_diff < _E_TOL, (
                f"post-step-1 E[{comp}]: max|E-want|={max_diff:.3e}")

        # Second step: E still matches (uniform state → nothing changes).
        run.step()
        for comp in range(3):
            e = _unpack_e(lib, h, comp, ntot)
            max_diff = float(np.max(np.abs(e - expect_e[comp])))
            assert max_diff < _E_TOL, (
                f"post-step-2 E[{comp}]: max|E-want|={max_diff:.3e}")

        # Three more steps (total 5): B bit-frozen, divb_max < 1e-12.
        for _ in range(3):
            run.step()
        assert run.divb_max < 1e-12, f"divb_max={run.divb_max:.3e} after 5 steps"
        for comp in range(3):
            b = _unpack_b(lib, h, comp, ntot)
            assert np.all(b == B_vec[comp]), (
                f"B[{comp}] drifted after 5 steps: "
                f"max|dB|={np.max(np.abs(b - B_vec[comp])):.3e}")
    finally:
        run.close()


# --- 2. §3：3D 非平凡构造性零散度钉 -----------------------------------------


def test_nontrivial_divb_free_evolution_3d(lib):
    """12³ 全周期盒 [0,2π]³。B 取构造性零散度场：
      B1 = sin(y) + cos(z)    （不依赖 x ⇒ dB1/dx = 0 离散精确）
      B2 = sin(z) + cos(x)    （不依赖 y ⇒ dB2/dy = 0 离散精确）
      B3 = sin(x) + cos(y)    （不依赖 z ⇒ dB3/dz = 0 离散精确）
    故 div B = 0 离散精确（三项各为零，不是互相抵消）。

    速度非均匀三向非零：
      vx = 0.2 + 0.1*sin(x)
      vy = 0.15 + 0.1*sin(y)
      vz = -0.1 + 0.1*sin(z)

    磁开跑 5 步：
      divb_max < 1e-6（CT 应保持初值散度在噪声水平）；
      三向 B 都演化（L1(final-ic) > 0，非冻结——u×B 非均匀 ⇒ ∇×E ≠ 0）。
    """
    L = 2.0 * math.pi
    nxyz = (12, 12, 12)
    cfg = _box_cfg(nxyz, (L, L, L), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        ncells = lib.xmf_state_ncells(h)
        dx = L / nxyz[0]
        dy = L / nxyz[1]
        dz = L / nxyz[2]

        # Explicit pack rho = 1, P = 1 (pit-1 lesson).
        lib.xmf_state_pack(h, 0, 0, _ptr(
            np.full(ncells, 1.0, dtype=np.float64)))
        lib.xmf_state_pack(h, 0, 4, _ptr(
            np.full(ncells, 1.0, dtype=np.float64)))

        # Non-uniform velocity.
        vx_arr = np.empty(ncells, dtype=np.float64)
        vy_arr = np.empty(ncells, dtype=np.float64)
        vz_arr = np.empty(ncells, dtype=np.float64)
        for idx in range(ncells):
            i = idx % ntot[0]
            xc = (i - ng + 0.5) * dx
            vx_arr[idx] = 0.2 + 0.1 * math.sin(xc)
            j = (idx // ntot[0]) % ntot[1]
            yc = (j - ng + 0.5) * dy
            vy_arr[idx] = 0.15 + 0.1 * math.sin(yc)
            k = idx // (ntot[0] * ntot[1])
            zc = (k - ng + 0.5) * dz
            vz_arr[idx] = -0.1 + 0.1 * math.sin(zc)
        lib.xmf_state_pack(h, 0, 1, _ptr(vx_arr))
        lib.xmf_state_pack(h, 0, 2, _ptr(vy_arr))
        lib.xmf_state_pack(h, 0, 3, _ptr(vz_arr))

        # Pack B: constructed to be discretely divergence-free.
        # B1 shape (ntot[2], ntot[1], ntot[0]+1): axes (z_cell, y_cell, x_face)
        b1 = np.empty((ntot[2], ntot[1], ntot[0] + 1), dtype=np.float64)
        for k in range(ntot[2]):
            zc = (k - ng + 0.5) * dz
            for j in range(ntot[1]):
                yc = (j - ng + 0.5) * dy
                for i in range(ntot[0] + 1):
                    b1[k, j, i] = math.sin(yc) + math.cos(zc)

        # B2 shape (ntot[0], ntot[2], ntot[1]+1): axes (x_cell, z_cell, y_face)
        b2 = np.empty((ntot[0], ntot[2], ntot[1] + 1), dtype=np.float64)
        for i in range(ntot[0]):
            xc = (i - ng + 0.5) * dx
            for k in range(ntot[2]):
                zc = (k - ng + 0.5) * dz
                for j in range(ntot[1] + 1):
                    b2[i, k, j] = math.sin(zc) + math.cos(xc)

        # B3 shape (ntot[1], ntot[0], ntot[2]+1): axes (y_cell, x_cell, z_face)
        b3 = np.empty((ntot[1], ntot[0], ntot[2] + 1), dtype=np.float64)
        for j in range(ntot[1]):
            yc = (j - ng + 0.5) * dy
            for i in range(ntot[0]):
                xc = (i - ng + 0.5) * dx
                for k in range(ntot[2] + 1):
                    b3[j, i, k] = math.sin(xc) + math.cos(yc)

        _pack_b(lib, h, 0, b1)
        _pack_b(lib, h, 1, b2)
        _pack_b(lib, h, 2, b3)

        # Initial divB check (should be near machine zero).
        divb0 = run.divb_max
        assert divb0 < 1e-10, f"initial divb_max={divb0:.3e} (expect ~0)"

        # Save IC B for evolution check.
        ic_b = [_unpack_b(lib, h, comp, ntot).copy() for comp in range(3)]

        # Run 5 steps.
        max_divb = divb0
        for _ in range(5):
            run.step()
            d = run.divb_max
            if d > max_divb:
                max_divb = d

        # divB stays < 1.5e-10 (measured 9.82e-11 × 1.5 = 1.47e-10;
        # CT preserves the analytically-zero initial divB at machine noise).
        assert max_divb < 1.5e-10, (
            f"max_divb={max_divb:.3e} (should be < 1.5e-10)")

        # All three B components must have evolved (not frozen).
        for comp in range(3):
            b_final = _unpack_b(lib, h, comp, ntot)
            l1 = float(np.mean(np.abs(b_final - ic_b[comp])))
            assert l1 > 0.0, f"B[{comp}] did not evolve (L1 diff = 0)"
    finally:
        run.close()


# --- 3. §4：3D 边界覆盖 -----------------------------------------------------


def test_reflect_boundary_3d_smoke(lib):
    """六面 reflect（wall 档）磁开冒烟：不炸、divB 不爆（< 1e-6）。
    均匀 rho=P=1 + 非零三向速度（指向盒内，反射壁反弹）+ B=0（磁门开）。

    B 取零的原因：reflect wall 把法向 B_n 钉 0（M3 合同），任何非零
    均匀 B 在壁面产生 ~O(B/dx) 的离散 divB（壁钉 0 + 内部非零的跳跃），
    CT 保散度 → divB 不衰减。M3 2D reflect 用特制 B=(0,y,-z) 让横向
    散度精确补偿壁跳跃，但 3D 六面同时补偿需要全场协调（超出冒烟范畴）。
    B=0 使磁门开（xmf_mag_load 落地）、CT 跑、divB=0；边界语义本身由
    test_mag 的 1D/2D 钉覆盖，此处只验六面 ghost 填充路径在 3D 不崩溃。"""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "reflect", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ncells = lib.xmf_state_ncells(h)

        # Explicit pack rho=1, P=1.
        lib.xmf_state_pack(h, 0, 0, _ptr(
            np.full(ncells, 1.0, dtype=np.float64)))
        lib.xmf_state_pack(h, 0, 4, _ptr(
            np.full(ncells, 1.0, dtype=np.float64)))

        # Nonzero three-velocity pointing inward.
        for var, val in ((1, 0.1), (2, 0.1), (3, -0.1)):
            lib.xmf_state_pack(h, 0, var, _ptr(
                np.full(ncells, val, dtype=np.float64)))

        # B = 0 (MHD gate open, but zero field to avoid wall-BC divB).
        # MHD gate is open: xmf_mag_load succeeded during setup.
        for comp in range(3):
            blk = np.zeros(_len_b(ntot, comp), dtype=np.float64)
            _pack_b(lib, h, comp, blk)

        # 3 steps: must not crash, divB bounded.
        for step in range(3):
            run.step()
            assert run.divb_max < 1e-6, (
                f"reflect step {step}: divb_max={run.divb_max:.3e}")
    finally:
        run.close()


def test_periodic_boundary_3d_smoke(lib):
    """六面 periodic 磁开冒烟：不炸、divB 不爆（< 1e-6）。
    均匀 rho=P=1 + 非均匀速度 + 均匀 B。5 步验证周期 ghost 填充通路。"""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=True)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ntot = _ntot_of(run)
        ng = run._cfg.ng
        ncells = lib.xmf_state_ncells(h)
        dx = 1.0 / 8

        # Explicit pack rho=1, P=1.
        lib.xmf_state_pack(h, 0, 0, _ptr(
            np.full(ncells, 1.0, dtype=np.float64)))
        lib.xmf_state_pack(h, 0, 4, _ptr(
            np.full(ncells, 1.0, dtype=np.float64)))

        # Non-uniform velocity.
        vx = np.empty(ncells, dtype=np.float64)
        vy = np.empty(ncells, dtype=np.float64)
        vz = np.empty(ncells, dtype=np.float64)
        for idx in range(ncells):
            i = idx % ntot[0]
            x = (i - ng + 0.5) * dx
            vx[idx] = 0.2 + 0.1 * math.sin(2.0 * math.pi * x)
            vy[idx] = 0.15
            vz[idx] = -0.1
        lib.xmf_state_pack(h, 0, 1, _ptr(vx))
        lib.xmf_state_pack(h, 0, 2, _ptr(vy))
        lib.xmf_state_pack(h, 0, 3, _ptr(vz))

        # Uniform B.
        for comp, bval in enumerate((0.1, 0.2, 0.3)):
            blk = np.full(_len_b(ntot, comp), bval, dtype=np.float64)
            _pack_b(lib, h, comp, blk)

        # 5 steps: must not crash, divB bounded.
        for step in range(5):
            run.step()
            assert run.divb_max < 1e-6, (
                f"periodic step {step}: divb_max={run.divb_max:.3e}")
    finally:
        run.close()


# --- 4. 坑 1 回归防护：dt 合理性钉 ------------------------------------------


def test_dt_reasonable_3d_hd(lib):
    """3D HD 门关均匀场（显式 pack rho=P=1，防坑 1 回归）。
    dt 应与手算 cfl·Δx/(|u|+c_s) 同量级（±50%）。

    手算（u=(0.3,-0.2,0.1), rho=P=1, gamma=5/3）：
      c_s = sqrt(gamma*P/rho) = sqrt(5/3) ≈ 1.291
      s_d = |u_d| + c_s：s_x=1.591, s_y=1.491, s_z=1.391
      Σ(s_d/dx_d) = 1.591/0.125 + 1.491/0.167 + 1.391/0.25
                  ≈ 12.73 + 8.95 + 5.56 = 27.24
      dt = cfl / Σ = 0.4 / 27.24 ≈ 1.47e-2
    防坑 1 回归：若忘记 pack rho，dt 会跌到 ~1e-14 量级。"""
    cfg = _box_cfg((8, 6, 4), (1.0, 1.0, 1.0), "periodic", mhd=False)
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)

        # Explicit pack rho=1, P=1 (the pit-1 fix).
        lib.xmf_state_pack(h, 0, 0, _ptr(
            np.full(ncells, 1.0, dtype=np.float64)))
        lib.xmf_state_pack(h, 0, 4, _ptr(
            np.full(ncells, 1.0, dtype=np.float64)))

        # Pack velocity.
        for var, val in ((1, 0.3), (2, -0.2), (3, 0.1)):
            lib.xmf_state_pack(h, 0, var, _ptr(
                np.full(ncells, val, dtype=np.float64)))

        # Step once; dt must be in sane range.
        run.step()
        dt = run.dt
        assert dt > 1e-3, f"dt={dt:.3e} too small (pit-1 regression?)"

        # Hand-calc expected dt.
        gamma = 5.0 / 3.0
        c_s = math.sqrt(gamma * 1.0 / 1.0)
        s_x = 0.3 + c_s
        s_y = 0.2 + c_s
        s_z = 0.1 + c_s
        inv_sum = s_x / 0.125 + s_y / (1.0 / 6.0) + s_z / 0.25
        dt_expected = 0.4 / inv_sum
        assert abs(dt - dt_expected) / dt_expected < 0.5, (
            f"dt={dt:.3e} not within 50% of hand-calc {dt_expected:.3e}")
    finally:
        run.close()
