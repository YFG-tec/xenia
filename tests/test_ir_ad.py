"""A4.7 前向 AD 合同单测（test_ir_ad）：同一条带子，栈上对偶数。

合同来自本刀卡片（docs/SESSION.md 第 4 条）与 docs/cmd/PROMPT_A4_AD.md §4：
  1. 乘积式带子（两流体 dn_HI = -k*n_HI*n_HEII）的隐式 J 对解析 1e-12：
     J[0,0] = -k*n_HEII、J[0,5] = -k*n_HI（下标 f*5+slot），恰有两个非零分量；
     顺带钉 num_states == 10。密度取该格读回的真实场值。
  2. 重力带（dvx = -G*M/r**2）隐式 J 全阵逐位 0——r 不是自变量、没有列；
     但 apply 一次后 vx 增量对 -GM/r²（mesh 体心 r）相对差 ≤1e-12，
     且 n、P 不被这条带子改。
  3. POW ∂/∂r 自检走 xmf_ir_eval_tape（不是 jacobian 的列）：ṙ=1 时
     out_v == -GM/r²、out_vdot == 2GM/r³（体心 r，至少抽检 2 格）；
     无种子（-1）时 v̇ == 0。
  4. AD 比前向差分更准：幂式带子 dn_HI = -k3*n_HI**2 的 ∂/∂n_HI 分量，
     |AD-解析| ≤ |差分(δ=1e-7|u|)-解析|，且 |AD-解析| ≤ 1e-12·量级。
  5. 空表：num_states == 5*n_fluid、jacobian 全 0；一步 step 后与
     「无 sources 键」基线逐位相同（钉法同 test_ir_eval 的空表用例）。
  6. 裸 dvx（dest_fluid=-1）写进每支的 vx 行：两流体下 dvx = -k2*n_HI 的 J
     在行 1 与行 6 同时非零，列仅 n_HI 列（裸写 ADDS 语义）。

约定（同 test_ir_eval.py）：
  一次 apply_sources() = 每条带子的瞬时值直接**加**到写目标，不乘 Δt、
  不走 hydro、不求值 ghost、不推进时间。jacobian / eval_tape 是辅助口，
  不是第五个公开动词。

口径（同 test_hydro.py / test_ir_eval.py）：unpack 出的是核内码单位，
块长 = xmf_state_ncells 含 ghost，reshape 顺序 (nztot, nytot, nxtot)，
索引 (k,j,i)，i 最快。jacobian 的 (k,j,i) 是含 ghost 的全格下标。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；不只测 VX1——
n 目标（乘积式 / 幂式）与 vx 目标（重力 / 裸 dvx）都覆盖；不碰本刀仍被
拒载的槽（流体 T、n_e/T_e、FLUX）；r 永远不当 jac 的列（合同 2/3 联保）。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.source_ir import GRID_X1, KIND_GRID, pack_slot

# 槽序 (n, vx, vy, vz, P) == 0..4（A2 契约）
SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

_PTR = ctypes.POINTER(ctypes.c_double)

# 重力测常数：与 test_ir_eval 相同，刻意取非 1 的数避免碰巧重合
G, M = 2.5, 3.75
# 乘积式 / 幂式 / 裸 dvx 的速率常数：同样刻意非 1、互不相同
K_CX = 1.7e-3
K3 = 2.75e-2
K2 = 1.3e-1

# r = x1 的打包槽码（pack_slot 定义 (kind<<16)|(a<<8)|b，按定义为 0）。
# 不硬编码 0：种子身份永远由打包规则构造。
SEED_R = pack_slot(KIND_GRID, GRID_X1)


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（state/mesh/ir 符号已绑定）。"""
    return load()


# --- 只读小助手（口径同 test_ir_eval.py / test_hydro.py） --------------------


def _geom(run):
    """从 run 的已归一化 config 取 (ng, nx, ny, nz, nxtot, nytot, nztot)。"""
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _unpack(lib, run, f: int, var: int) -> np.ndarray:
    """unpack 流体 f 槽 var 的整块（含 ghost），返回 flat 缓冲。"""
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, f, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _centres(lib, run) -> np.ndarray:
    """x1 体心坐标（含 ghost），长度 nx+2ng；与 fields 的 x 同数组。"""
    ntot = lib.xmf_mesh_ncell(run._h, 0)
    buf = np.empty(ntot, dtype=np.float64)
    assert lib.xmf_mesh_copy_x(run._h, 0, buf.ctypes.data_as(_PTR)) == ntot
    return buf


def _rel(got: float, want: float) -> float:
    """相对误差；解析目标按构造非零（各用例里先断言过）。"""
    return abs(float(got) - float(want)) / abs(float(want))


def _eval_tape(lib, run, t: int, k: int, j: int, i: int, seed: int):
    """xmf_ir_eval_tape 的 Python 包装：返回 (err, v, vdot)。"""
    v = ctypes.c_double()
    vdot = ctypes.c_double()
    err = lib.xmf_ir_eval_tape(
        run._h, t, k, j, i, seed, ctypes.byref(v), ctypes.byref(vdot)
    )
    return err, v.value, vdot.value


# --- 用例配置 ----------------------------------------------------------------


def _two_fluids() -> list[dict]:
    """HI + HEII 两支组分（名字↔下标只在 Python；核只认整数流体下标）。"""
    return [
        {"name": "HI", "mass": 1.00794, "charge": 0, "kind": "hd"},
        {"name": "HEII", "mass": 4.0026, "charge": 1, "kind": "hd"},
    ]


def _product_cfg() -> dict:
    """合同 1：两流体，一条乘积式带子 dn_HI = -k_cx*n_HI*n_HEII。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [16, 1, 1]
    cfg["fluids"] = _two_fluids()
    cfg["constants"] = {"k_cx": K_CX}
    cfg["sources"] = [
        {"name": "cx", "expr": {"dn_HI": "-k_cx*n_HI*n_HEII"}},
    ]
    return cfg


def _gravity_cfg() -> dict:
    """合同 2/3：笛卡尔小网格重力用例，x1∈[1,2] 避开 r=0，单流体裸 dvx。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [16, 1, 1]
    cfg["grid"]["x1min"] = 1.0
    cfg["grid"]["x1max"] = 2.0
    cfg["constants"] = {"G": G, "M": M}
    cfg["sources"] = [{"name": "g", "expr": {"dvx": "-G*M/r**2"}}]
    return cfg


def _power_cfg() -> dict:
    """合同 4：幂式带子 dn_HI = -k3*n_HI**2（对 n_HI 的二阶导非零，
    前向差分才有真截断误差——乘积式对单变量是线性的，差分碰巧精确）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [16, 1, 1]
    cfg["constants"] = {"k3": K3}
    cfg["sources"] = [{"name": "p2", "expr": {"dn_HI": "-k3*n_HI**2"}}]
    return cfg


def _bare_dvx_cfg() -> dict:
    """合同 6：两流体，裸 dvx = -k2*n_HI（dest_fluid=-1，写每一支）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [16, 1, 1]
    cfg["fluids"] = _two_fluids()
    cfg["constants"] = {"k2": K2}
    cfg["sources"] = [{"name": "drag", "expr": {"dvx": "-k2*n_HI"}}]
    return cfg


# --- 1. 乘积式 J 对解析 1e-12，恰两个非零（合同第 1 条） ---------------------


def test_product_jac_matches_analytic_exactly_two_nonzeros(lib):
    """两流体乘积式：J[0,0]=-k*n_HEII、J[0,5]=-k*n_HI，其余严格 0。

    setup 只给流体 0 填 Riemann 初值，流体 1（HEII）整块为 0；先用 state
    助手给 HEII 写入非零密度模式，再对若干内部格调 jacobian。解析值一律
    用该格**读回的真实场值**（不假设初值）。行/列下标 = f*5+slot：
    dn_HI 行 = 0，n_HI 列 = 0，n_HEII 列 = 5。顺带钉 num_states == 10。
    """
    run = xmf.setup(_product_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 1, "一条乘积式带子应加载为 1 条 tape"
        assert run.num_states == 10, "两流体 × 5 槽：N 应为 10"

        # 给 HEII（流体 1）写非零密度；模式任意，解析值用读回的真值
        ncells = lib.xmf_state_ncells(run._h)
        pat = 0.25 + 1.0e-3 * np.arange(ncells, dtype=np.float64)
        assert lib.xmf_state_pack(run._h, 1, SLOT_N, pat.ctypes.data_as(_PTR)) == ncells

        n_hi = _unpack(lib, run, 0, SLOT_N).reshape((nzt, nyt, nxt))
        n_he = _unpack(lib, run, 1, SLOT_N).reshape((nzt, nyt, nxt))

        for iq in (0, nx // 2, nx - 1):  # 左侧、跨间断、右侧各一格
            i = ng + iq
            a = float(n_hi[ng, ng, i])
            b = float(n_he[ng, ng, i])
            assert a > 0.0 and b > 0.0, "解析目标非零才有相对误差可言"

            J = run.jacobian(ng, ng, i)
            assert J.shape == (10, 10), "N=10 的稠密方阵"
            assert _rel(J[0, 0], -K_CX * b) <= 1.0e-12, (
                f"J[0,0]=∂(dn_HI)/∂n_HI 应为 -k*n_HEII={-K_CX * b:.6e}，"
                f"实测 {J[0, 0]:.6e}"
            )
            assert _rel(J[0, 5], -K_CX * a) <= 1.0e-12, (
                f"J[0,5]=∂(dn_HI)/∂n_HEII 应为 -k*n_HI={-K_CX * a:.6e}，"
                f"实测 {J[0, 5]:.6e}"
            )
            nz_idx = np.argwhere(J != 0.0)
            assert np.count_nonzero(J) == 2, (
                f"乘积式 J 应恰有两个非零分量（行 0 列 0/5），实测非零位置 {nz_idx.tolist()}"
            )
    finally:
        run.close()


# --- 2. 重力带隐式 J 全 0（合同第 2 条上半） ---------------------------------


def test_gravity_implicit_jac_bitwise_zero(lib):
    """dvx=-G*M/r**2：r 不是自变量 ⇒ N×N 的 J 逐位全 0。

    对偶数在「种子是流体槽而带子只读 r」时每个 .d 都精确为 0
    （0 的加减乘除仍是 0），故断言逐位 == 0，不留容差。∂/∂r 走
    eval_tape 自检（下一用例），永远不进这张 J——两条合同联保。
    """
    run = xmf.setup(_gravity_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 1
        assert run.num_states == 5, "单流体 N=5，且 r 不得占列"

        for iq in (0, nx // 2, nx - 1):
            J = run.jacobian(ng, ng, ng + iq)
            assert J.shape == (5, 5)
            assert np.all(J == 0.0), (
                f"重力带隐式 J 应逐位全 0（r 无列），实测 max|J|="
                f"{float(np.max(np.abs(J))):.3e}（内部格 iq={iq}）"
            )
    finally:
        run.close()


# --- 2b. apply 增量仍对 -GM/r²，n、P 不被改（合同第 2 条下半） ---------------


def test_gravity_apply_increment_and_n_p_untouched(lib):
    """同一重力带 apply 一次：Δvx 对 -GM/r²（体心 r）≤1e-12；n、P 逐位不变。

    r 钉法同 test_ir_eval：解析式里的 r 用 xmf_mesh_copy_x 读回的 x1 体心
    坐标。n、P 的比对是整块含 ghost 的逐位相等（带子只写内部格 vx）。
    """
    run = xmf.setup(_gravity_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        xc = _centres(lib, run)[ng : ng + nx]

        before_n = _unpack(lib, run, 0, SLOT_N)
        before_p = _unpack(lib, run, 0, SLOT_P)
        before_vx = _unpack(lib, run, 0, SLOT_VX).reshape((nzt, nyt, nxt))

        run.apply_sources()

        after_vx = _unpack(lib, run, 0, SLOT_VX).reshape((nzt, nyt, nxt))
        inc = after_vx[ng, ng, ng : ng + nx] - before_vx[ng, ng, ng : ng + nx]
        exact = -G * M / xc**2
        rel = float(np.max(np.abs((inc - exact) / exact)))
        assert rel <= 1.0e-12, (
            f"apply 增量与解析 -G*M/r² 的最大相对差 {rel:.3e} 超过 1e-12"
        )

        after_n = _unpack(lib, run, 0, SLOT_N)
        after_p = _unpack(lib, run, 0, SLOT_P)
        assert np.array_equal(before_n, after_n), (
            "重力带不得改动 n（整块含 ghost 逐位比对）"
        )
        assert np.array_equal(before_p, after_p), (
            "重力带不得改动 P（整块含 ghost 逐位比对）"
        )
        assert run.t == 0.0, "apply_sources 不得推进模拟时间"
    finally:
        run.close()


# --- 3. POW ∂/∂r 自检：eval_tape 播 ṙ=1（合同第 3 条） ----------------------


def test_pow_dr_selfcheck_via_eval_tape(lib):
    """同一重力带，临时给 r 槽 ṙ=1：v=-GM/r²、v̇=2GM/r³（体心 r，≤1e-12）。

    走 xmf_ir_eval_tape（自检缝），不写成 jacobian 的某一列——用例末尾再
    断言一次 jac 全 0，锁死「∂/∂r 可达但 r 不是自变量」。无种子（-1）时
    v̇ 严格 0、v 仍是 -GM/r²。至少抽检 2 格：这里取首、中、尾 3 格。
    """
    run = xmf.setup(_gravity_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 1
        xc = _centres(lib, run)[ng : ng + nx]

        for iq in (0, nx // 2, nx - 1):
            i = ng + iq
            r = float(xc[iq])
            assert r > 0.0, "x1∈[1,2]：体心 r 必为正（POW 导数要求底 >0）"

            # ṙ = 1：值 + 方向导数
            err, v, vdot = _eval_tape(lib, run, 0, ng, ng, i, SEED_R)
            assert err == 0, f"eval_tape 带种子 r 应成功（err={err}）"
            assert _rel(v, -G * M / r**2) <= 1.0e-12, (
                f"种子下 .v 仍应为 -GM/r²：r={r:.6e} 实测 {v:.6e}"
            )
            assert _rel(vdot, 2.0 * G * M / r**3) <= 1.0e-12, (
                f"∂(-GM/r²)/∂r 应为 2GM/r³={2.0 * G * M / r**3:.6e}，"
                f"实测 {vdot:.6e}（r={r:.6e}）"
            )

            # 无种子：v̇ 严格 0，v 不变
            err0, v0, vdot0 = _eval_tape(lib, run, 0, ng, ng, i, -1)
            assert err0 == 0
            assert vdot0 == 0.0, f"无种子时 v̇ 应严格 0，实测 {vdot0!r}"
            assert _rel(v0, -G * M / r**2) <= 1.0e-12

        # 自检缝不是隐式列：尽管 ∂/∂r ≠ 0，jac 仍全 0
        J = run.jacobian(ng, ng, ng + nx // 2)
        assert np.all(J == 0.0), "∂/∂r 不得漏进 jacobian 的任何一列"
    finally:
        run.close()


# --- 4. AD 比前向差分更准（合同第 4 条） -------------------------------------


def test_ad_more_accurate_than_forward_difference(lib):
    """幂式 dn_HI=-k3*n_HI**2：AD 的 ∂/∂n_HI 比 δ=1e-7|u| 的前向差分更准。

    差分走**同一核同一条带子**的 apply 路径（一次 apply = 瞬时值加到写
    目标，不乘 Δt）：f(u) = apply 后 n 的增量。先取 AD（状态未受扰），
    再扰动该格 n_HI → 再 apply。解析导数 = -2*k3*u，u 为该格读回真值。
    断言：|AD-解析| ≤ |差分-解析|，且 |AD-解析| ≤ 1e-12·|解析|。
    （乘积式对单变量是线性的、差分碰巧精确，故合同选幂式做这条。）
    """
    run = xmf.setup(_power_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 1
        assert run.num_states == 5

        iq = 3  # x≈2.97 < x0=5.5：左侧高密态，u>0 保证 POW 可导
        i = ng + iq
        n_orig = _unpack(lib, run, 0, SLOT_N)
        n3 = n_orig.reshape((nzt, nyt, nxt))
        u = float(n3[ng, ng, i])
        assert u > 0.0

        # AD：状态未受扰时读 J[0,0]（行=dn_HI，列=n_HI）
        ad = float(run.jacobian(ng, ng, i)[0, 0])
        exact = -2.0 * K3 * u

        # 前向差分：同一带子，核内 apply 路径
        delta = 1.0e-7 * abs(u)
        run.apply_sources()
        n_after = _unpack(lib, run, 0, SLOT_N).reshape((nzt, nyt, nxt))
        f_u = float(n_after[ng, ng, i]) - u

        # 复原原场，再只扰动该格：n_HI -> u + δ
        assert lib.xmf_state_pack(run._h, 0, SLOT_N, n_orig.ctypes.data_as(_PTR)) == n_orig.size
        n_pert = n_orig.copy()
        n_pert.reshape((nzt, nyt, nxt))[ng, ng, i] = u + delta
        assert lib.xmf_state_pack(run._h, 0, SLOT_N, n_pert.ctypes.data_as(_PTR)) == n_orig.size
        run.apply_sources()
        n_after2 = _unpack(lib, run, 0, SLOT_N).reshape((nzt, nyt, nxt))
        f_ud = float(n_after2[ng, ng, i]) - (u + delta)
        fd = (f_ud - f_u) / delta

        scale = abs(exact)
        ad_err = abs(ad - exact)
        fd_err = abs(fd - exact)
        assert fd_err > 0.0, "δ=1e-7|u| 的差分应有非零截断误差（防命题空转）"
        assert ad_err <= fd_err, (
            f"AD 应比差分更接近解析：|AD-解析|={ad_err:.3e} > |差分-解析|={fd_err:.3e}"
        )
        assert ad_err <= 1.0e-12 * scale, (
            f"AD 误差 {ad_err:.3e} 超过 1e-12×量级（{1.0e-12 * scale:.3e}）"
        )
        # 差分自身量级合理：截断 ~O(δ)，不该退化也不该爆掉
        assert fd_err <= 1.0e-4 * scale, f"差分误差 {fd_err:.3e} 异常偏大"
    finally:
        run.close()


# --- 5. 空表：N=5*n_fluid、J 全 0、step 逐位同纯 hydro（合同第 5 条） --------


def test_empty_table_jac_zero_and_step_bitwise_pure_hydro(lib):
    """shock_tube()（sources=[]）：num_states==5*n_fluid、jac 全 0；
    一步 step 后五槽 + t + dt 与「无 sources 键」基线逐位相同。

    钉法与 test_ir_eval 的空表用例一致：基线取同一初值、删去 sources 键
    的 setup——两条路径本应同为纯 hydro，任何逐位差都说明空表路径动了
    hydro。
    """
    cfg_empty = xmf.shock_tube()  # 自带 sources: []
    cfg_none = xmf.shock_tube()
    del cfg_none["sources"]

    run_empty = xmf.setup(cfg_empty)
    run_pure = xmf.setup(cfg_none)
    try:
        assert run_empty.num_tapes == 0, "sources=[] 不得加载任何带子"
        assert run_pure.num_tapes == 0
        assert run_empty.num_states == 5 * len(run_empty._cfg.fluids), (
            "空表也有自变量个数：N = 5*n_fluid"
        )

        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run_empty)
        J = run_empty.jacobian(ng, ng, ng + nx // 2)
        assert J.shape == (run_empty.num_states, run_empty.num_states)
        assert np.all(J == 0.0), "空表 jacobian 应全 0"

        run_empty.step()
        run_pure.step()
        assert run_empty.t > 0.0, "step 应真正推进时间"
        for s, name in enumerate("n vx vy vz P".split()):
            a = _unpack(lib, run_empty, 0, s)
            b = _unpack(lib, run_pure, 0, s)
            assert np.array_equal(a, b), (
                f"空表一步 step 后槽 {name} 与纯 hydro 逐位不同（L∞ 差 "
                f"{float(np.max(np.abs(a - b))):.3e}，码单位）"
            )
        assert run_empty.t == run_pure.t, "空表与纯 hydro 的 t 应逐位相同"
        assert run_empty.dt == run_pure.dt, "空表与纯 hydro 的 dt 应逐位相同"
    finally:
        run_empty.close()
        run_pure.close()


# --- 6. 裸 dvx 写每一支的 vx 行（合同第 6 条） -------------------------------


def test_bare_dvx_writes_every_fluid_vx_row(lib):
    """两流体裸带子 dvx=-k2*n_HI：J 的 vx 行（行 1 与行 6）同时有 -k2，
    列仅 n_HI 列——裸写按 apply 的 ADDS 语义加进每一支。

    交叉核对：apply 后两支 vx 同时加同一增量 -k2*n_HI（真值逐格）。
    """
    run = xmf.setup(_bare_dvx_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 1
        assert run.num_states == 10

        iq = nx // 3
        J = run.jacobian(ng, ng, ng + iq)
        assert J.shape == (10, 10)
        # 行 = f*5 + SLOT_VX：流体 0 → 1，流体 1 → 6；列 = n_HI → 0
        assert _rel(J[1, 0], -K2) <= 1.0e-12, (
            f"流体 0 的 vx 行应为 -k2={-K2:.6e}，实测 {J[1, 0]:.6e}"
        )
        assert _rel(J[6, 0], -K2) <= 1.0e-12, (
            f"流体 1 的 vx 行应为 -k2={-K2:.6e}，实测 {J[6, 0]:.6e}"
        )
        assert J[1, 0] != 0.0 and J[6, 0] != 0.0, "两支 vx 行应同时非零"
        nz_idx = np.argwhere(J != 0.0)
        assert np.count_nonzero(J) == 2, (
            f"裸 dvx 的 J 应恰有两个非零（行 1/6 的 n_HI 列），实测 {nz_idx.tolist()}"
        )
        assert np.all(J[:, 1:] == 0.0), "非零列只能在 n_HI 列（列 0）"

        # 交叉核对 apply 的 ADDS 语义：两支 vx 同时加同一增量
        n_hi = _unpack(lib, run, 0, SLOT_N).reshape((nzt, nyt, nxt))
        for f in (0, 1):
            assert np.all(_unpack(lib, run, f, SLOT_VX) == 0.0), (
                f"流体 {f} 的 vx 初值应为 0（apply 前）"
            )
        run.apply_sources()
        expect = -K2 * n_hi[ng, ng, ng : ng + nx]
        assert np.all(expect < 0.0)
        for f in (0, 1):
            vx_int = _unpack(lib, run, f, SLOT_VX).reshape((nzt, nyt, nxt))[
                ng, ng, ng : ng + nx
            ]
            rel = float(np.max(np.abs((vx_int - expect) / expect)))
            assert rel <= 1.0e-12, (
                f"流体 {f} 的 vx 增量应为 -k2*n_HI，最大相对差 {rel:.3e}"
            )
    finally:
        run.close()
