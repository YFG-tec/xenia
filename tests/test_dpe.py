"""B2 IR 追加写名 dP_e 合同单测：直写电子压强块，关门即拒。

合同来自本刀卡片（docs/SESSION.md B2 §交什么 2/5）：
  1. dP_e 是追加写名（WRITE_DPE = 5，操作码 0..9 不动）：写入目标是可选
     P_e 块本身——「直接写」，禁止「写进离子 P 再投影」。dest_fluid 恒为
     -1，永不占流体五槽。
  2. 开门（electron.mode=pressure-only）：
     - apply：dP_e = const ⇒ 内区 P_e += c、ghost 不动、流体五槽不动；
     - jac：dP_e = k*P_e ⇒ J[N-1, N-1] = k、其余逐位 0；
             dP_e = n_e  ⇒ 末行各离子 rho 列 = q/mass_code、中性列 0、
             P_e 列 0（n_e 不含 P_e）；
     - N = 5*nfluid + 1（P_e 殿后）；
     - 源步（半隐式一步）：均匀静态上常速率 ⇒ pe1 = pe0 + dt*c（1e-12）；
       线性速率 dP_e = k*P_e ⇒ pe1 = pe0/(1 - dt*k)（线性 ⇒ 一步精确，
       1e-12），且 source_resid 归零（<= 1e-10，报告量）。
  3. 关门（electron.mode=off）：Python 编译期拒（消息含 "electron"）；
     C 级 xmf_ir_load 对 dest_kind=5 同样拒（-1）；开门但 dest_fluid
     不是 -1（如 0）也拒——dP_e 永不取流体下标。关门 N 仍 5*nfluid。

口径（同 test_pe / test_ir_ad）：unpack/write_pe 块长 = xmf_state_ncells
含 ghost，reshape (nztot, nytot, nxtot)，(k,j,i) i 最快；apply 只写内区、
ghost 永不求值；默认单位下 unit_P = 1。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；钉的是 P_e 块、流体
五槽与整个牛顿维度，不只测 VX1。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import AMU, KB

_PTR = ctypes.POINTER(ctypes.c_double)

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

MASS_ION, MASS_NEUTRAL = 1.00794, 4.0026
WRITE_DPE = 5  # 与 xmf.source_ir.WRITE_DPE / core/xmf_ir.h 的枚举一致
WRITE_DP = 4


@pytest.fixture(scope="module")
def lib():
    return load()


# --- 小助手（同 test_pe 口径） -------------------------------------------------


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _pack_block(lib, run, f: int, var: int, block: np.ndarray) -> None:
    ncells = lib.xmf_state_ncells(run._h)
    data = np.ascontiguousarray(block, dtype=np.float64).ravel()
    assert data.size == ncells
    assert lib.xmf_state_pack(run._h, f, var, data.ctypes.data_as(_PTR)) == ncells


def _write_pe(lib, run, block: np.ndarray) -> None:
    ncells = lib.xmf_state_ncells(run._h)
    data = np.ascontiguousarray(block, dtype=np.float64).ravel()
    assert data.size == ncells
    assert lib.xmf_state_write_pe(run._h, data.ctypes.data_as(_PTR)) == ncells


def _unpack_block(lib, run, f: int, var: int) -> np.ndarray:
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, f, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _copy_pe(lib, run) -> np.ndarray:
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_copy_pe(run._h, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _dpe_cfg(sources, *, electron_mode="pressure-only", fluids=None,
             constants=None):
    """16 胞 1D：HI(中性) + HII(q=1)；空跑辐射束表只为触发 phys 口
    （真实质量 + 单位，n_e / T 的换算才有判别力），扫描不跑、FLUX 读 0。"""
    c = xmf.shock_tube()
    c["grid"]["n"] = [16, 1, 1]
    c["fluids"] = fluids if fluids is not None else [
        {"name": "HI", "mass": MASS_NEUTRAL, "charge": 0, "kind": "neutral"},
        {"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"},
    ]
    c["electron"] = {"mode": electron_mode}
    c["sources"] = sources
    c["constants"] = constants or {}
    c["radiation"] = {
        "enabled": False,
        "absorbers": ["HI"],
        "bins": {"n": 1, "sigma": [1.0e-18], "hnu": [2.0e-11], "flux_inf": [0.0]},
    }
    c["ic"] = {"kind": "none"}
    return c


def _load_raw_tape(lib, handle, dest_kind, dest_fluid, const=1.0):
    """绕过 Python 编译，直打 xmf_ir_load：一条 LOAD_CONST 带。"""
    header = np.array([dest_kind, dest_fluid, 1, 1], dtype=np.int32)
    ops = np.array([0, 0], dtype=np.int32)  # OP_LOAD_CONST, arg=0
    consts = np.array([const], dtype=np.float64)
    return lib.xmf_ir_load(
        handle,
        1,
        header.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ops.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        consts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )


# --- 1. 开门 apply：直写 P_e 块，流体五槽不动 -----------------------------------


def test_dpe_apply_writes_pe_block_directly(lib):
    """dP_e = q_heat（常数）：apply 后内区 P_e += q_heat、ghost 逐位不动、
    两支流体五槽逐位不动（没写进离子 P 再投影）。N = 5*2 + 1 = 11。"""
    q_heat = 0.375
    cfg = _dpe_cfg(
        [{"name": "photoheat", "expr": {"dP_e": "q_heat"}}],
        constants={"q_heat": q_heat},
    )
    run = xmf.setup(cfg)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        assert run.num_states == 11, "开门：N = 5*nfluid + 1（P_e 殿后）"
        assert run.num_tapes == 1

        idx = np.arange(nzt * nyt * nxt, dtype=np.float64).reshape(shape)
        pe0 = 0.25 + 0.01 * idx
        _pack_block(lib, run, 0, SLOT_N, np.full(shape, 0.8))
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, 0.5))
        _pack_block(lib, run, 1, SLOT_N, np.full(shape, 1.2))
        _pack_block(lib, run, 1, SLOT_P, np.full(shape, 0.6))
        _write_pe(lib, run, pe0)
        snap = {
            (f, var): _unpack_block(lib, run, f, var).copy()
            for f in range(2)
            for var in range(5)
        }

        run.apply_sources()

        pe1 = _copy_pe(lib, run).reshape(shape)
        inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
        d = pe1 - pe0
        assert float(np.max(np.abs(d[inner] - q_heat))) <= 1.0e-12 * q_heat, (
            "dP_e 常数带：内区 P_e 增量应逐格 == q_heat"
        )
        ghost = np.ones(shape, dtype=bool)
        ghost[inner] = False
        assert np.all(d[ghost] == 0.0), "apply 不得触碰 ghost 区的 P_e"
        for key, before in snap.items():
            after = _unpack_block(lib, run, key[0], key[1])
            assert np.array_equal(after, before), (
                f"dP_e 带不得触碰流体槽 {key}（直写电子块，不借道离子 P）"
            )
    finally:
        run.close()


# --- 2. 开门 jac：P_e 殿后未知量 ------------------------------------------------


def test_dpe_jac_pe_dependent_rate(lib):
    """dP_e = k_pe*P_e（单离子流体）：N = 6，J[5,5] == k_pe，其余逐位 0。"""
    k_pe = 0.125
    c = xmf.shock_tube()
    c["grid"]["n"] = [16, 1, 1]
    c["fluids"] = [{"name": "HII", "mass": MASS_ION, "charge": 1, "kind": "ion"}]
    c["electron"] = {"mode": "pressure-only"}
    c["sources"] = [{"name": "pe_rate", "expr": {"dP_e": "k_pe * P_e"}}]
    c["constants"] = {"k_pe": k_pe}
    c["ic"] = {"kind": "none"}
    run = xmf.setup(c)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        assert run.num_states == 6, "开门单流体：N = 5 + 1"
        _pack_block(lib, run, 0, SLOT_P, np.full(shape, 0.7))
        _write_pe(lib, run, np.full(shape, 0.3))
        J = run.jacobian(ng, ng, ng + nx // 2)
        assert J.shape == (6, 6)
        assert J[5, 5] == k_pe, (
            f"d(dP_e)/d(P_e) 应为 k_pe={k_pe}，实测 {J[5, 5]!r}"
        )
        Jc = J.copy()
        Jc[5, 5] = 0.0
        assert np.all(Jc == 0.0), (
            "k*P_e 带子的 jac 除 (P_e 行, P_e 列) 外应逐位全 0"
        )
    finally:
        run.close()


def test_dpe_jac_ne_rate_columns(lib):
    """dP_e = n_e（双流体 + phys 口）：末行各离子 rho 列 = q/mass_code，
    中性列 0，P_e 列 0（n_e = Σ q_s n_s 不含 P_e）。"""
    c = _dpe_cfg([{"name": "ne_to_pe", "expr": {"dP_e": "n_e"}}])
    run = xmf.setup(c)
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        shape = (nzt, nyt, nxt)
        assert run.num_states == 11
        rho_n, rho_i, pe_c = 0.8, 1.2, 0.4
        _pack_block(lib, run, 0, SLOT_N, np.full(shape, rho_n))
        _pack_block(lib, run, 1, SLOT_N, np.full(shape, rho_i))
        _write_pe(lib, run, np.full(shape, pe_c))
        mc_i = MASS_ION * AMU / run._cfg.units.density
        J = run.jacobian(ng, ng, ng + nx // 2)
        assert J.shape == (11, 11)
        assert J[10, 0] == 0.0, "中性支不进 n_e"
        assert abs(J[10, 5] - 1.0 / mc_i) <= 1.0e-12 * abs(1.0 / mc_i), (
            "d(n_e)/d(rho_HII) 应为 q/mass_code"
        )
        assert J[10, 10] == 0.0, "n_e 不含 P_e：P_e 列应为 0"
        Jc = J.copy()
        Jc[10, 5] = 0.0
        assert np.all(Jc == 0.0), "n_e 带子的 jac 只许中性外的离子 rho 列非零"
    finally:
        run.close()


# --- 3. 开门源步：半隐式一步钉 --------------------------------------------------


def _static_run(cfg, lib, rho_i=1.0, ps_i=1.0, pe_i=0.5):
    """均匀静态（v=0）：hydro 扫对均匀态是 no-op，步内变化只来自源步。"""
    run = xmf.setup(cfg)
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    shape = (nzt, nyt, nxt)
    for f in range(len(run._cfg.fluids)):
        _pack_block(lib, run, f, SLOT_N, np.full(shape, rho_i))
        _pack_block(lib, run, f, SLOT_P, np.full(shape, ps_i))
    _write_pe(lib, run, np.full(shape, pe_i))
    return run


def test_source_step_constant_dpe_rate_exact(lib):
    """均匀静态 + 常速率 dP_e = q_heat：源步一行线性（J 全 0 ⇒ A = I），
    pe1 = pe0 + dt*q_heat 对 1e-12。"""
    q_heat = 0.05
    cfg = _dpe_cfg(
        [{"name": "heat", "expr": {"dP_e": "q_heat"}}],
        constants={"q_heat": q_heat},
    )
    pe0 = 0.5
    run = _static_run(cfg, lib, pe_i=pe0)
    try:
        run.step()
        dt = run.dt
        assert dt > 0.0
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        pe1 = _copy_pe(lib, run).reshape((nzt, nyt, nxt))
        inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
        expect = pe0 + dt * q_heat
        rel = float(np.max(np.abs((pe1[inner] - expect) / expect)))
        assert rel <= 1.0e-12, (
            f"常速率源步应精确：pe1 = pe0 + dt*q_heat，实测最大相对差 {rel:.3e}"
        )
    finally:
        run.close()


def test_source_step_linear_dpe_rate_exact(lib):
    """线性速率 dP_e = k_pe*P_e：线性化后向欧拉对线性源一步精确，
    pe1 = pe0/(1 - dt*k_pe) 对 1e-12；残差报告量 source_resid <= 1e-10。"""
    k_pe = 1.0e-3
    cfg = _dpe_cfg(
        [{"name": "pe_rate", "expr": {"dP_e": "k_pe * P_e"}}],
        constants={"k_pe": k_pe},
    )
    pe0 = 0.5
    run = _static_run(cfg, lib, pe_i=pe0)
    try:
        run.step()
        dt = run.dt
        assert dt > 0.0 and dt * k_pe < 1.0, "钉在稳定区：1 - dt*k > 0"
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        pe1 = _copy_pe(lib, run).reshape((nzt, nyt, nxt))
        inner = (slice(ng, ng + nz), slice(ng, ng + ny), slice(ng, ng + nx))
        expect = pe0 / (1.0 - dt * k_pe)
        rel = float(np.max(np.abs((pe1[inner] - expect) / expect)))
        assert rel <= 1.0e-12, (
            f"线性速率源步应一步精确：pe1 = pe0/(1-dt*k)，"
            f"实测最大相对差 {rel:.3e}"
        )
        assert run.source_resid <= 1.0e-10, (
            f"线性源的一步线性化残差应归零，实测 {run.source_resid:.3e}"
        )
    finally:
        run.close()


# --- 4. 关门：编译拒（Python）+ 加载拒（C） -------------------------------------


def test_dpe_compile_refused_when_electrons_off():
    cfg = _dpe_cfg(
        [{"name": "heat", "expr": {"dP_e": "0.1"}}], electron_mode="off"
    )
    with pytest.raises(ValueError, match="electron"):
        xmf.setup(cfg)


def test_dpe_c_load_refusals(lib):
    """C 级 xmf_ir_load 与前端同契约：关电子拒 dest_kind=5；开门但
    dest_fluid != -1（dP_e 永不取流体下标）也拒；裸 dP（kind=4,
    fl=-1）同样拒。开门 + fl=-1 放行（返回 1），随后 ntape=0 清空。"""
    run_off = xmf.setup(xmf.shock_tube())
    try:
        assert _load_raw_tape(lib, run_off._h, WRITE_DPE, -1) == -1, (
            "关电子：dest_kind=dP_e 必须被 xmf_ir_load 拒"
        )
        assert lib.xmf_ir_ntape(run_off._h) == 0, "拒绝后不得残留带子"
    finally:
        run_off.close()

    run_on = xmf.setup(_dpe_cfg([]))  # 开电子、空源表
    try:
        assert _load_raw_tape(lib, run_on._h, WRITE_DPE, 0) == -1, (
            "开门：dP_e 的 dest_fluid 必须是 -1，给流体下标 0 也要拒"
        )
        assert _load_raw_tape(lib, run_on._h, WRITE_DP, -1) == -1, (
            "裸 dP（无流体后缀）仍要拒"
        )
        assert _load_raw_tape(lib, run_on._h, WRITE_DPE, -1) == 1, (
            "开门 + fl=-1 应放行一条带"
        )
        assert lib.xmf_ir_ntape(run_on._h) == 1
        assert lib.xmf_ir_load(run_on._h, 0, None, None, None) == 0, (
            "ntape=0 应清空带集"
        )
    finally:
        run_on.close()


# --- 5. N 钉：开门 +1、关门仍 5*nfluid ------------------------------------------


def test_nstate_open_plus_one_closed_five_per_fluid(lib):
    """同一双流体配置：开门 N = 5*2+1 = 11；关门 N = 5*2 = 10（dP_e 的
    追加不改变关门维度）。"""
    run_on = xmf.setup(_dpe_cfg([]))
    try:
        assert run_on.num_states == 11
    finally:
        run_on.close()
    run_off = xmf.setup(_dpe_cfg([], electron_mode="off"))
    try:
        assert run_off.num_states == 10
    finally:
        run_off.close()
