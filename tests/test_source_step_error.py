"""C1b 源步错误传播合同：源步里任一格失败 → 整步必须失败，不得静默通过。

合同来自本刀卡片（docs/SESSION.md §交什么 3）与 docs/cmd/PROMPT_C1B.md：
C1b 把 xmf_source.cpp 的 resid/err 从堆标量裸写改成真正的 Kokkos 归约
（Kokkos::Max<double>(resid) + Kokkos::Sum<int>(err)）。归约改写最容易
悄悄丢的就是错误传播：`err_upd = 1` 漏写、`Sum<int>` 换错形状、或
`if (err) ret = -1` 被吞，都会让「有格失败」退化成「静默通过」。本文件
用现有 config/IR 机制构造一个**必炸**的源项步来钉这条链：

  * 带子 dn_HI = krate*(vx_HI)**0.5。IR 的 POW 操作码在**带种子**求值时
    要求底 > 0（xmf_ir.c：导数要 log a 与 b/a）；无种子的值路保持
    pow(a,b) 不失败。打包均匀初值 vx = -1 后：
      - hydro 对均匀态是逐位 no-op（x1 outflow、pcm/hll/rk1）；
      - source_cell 的 (c) 段对 vx_HI 列做带种子求值 → 底 -1 ≤ 0 →
        xmf_ir_eval 返回 -1 → 本格在写回 (f) 之前失败；
      - 每格的失败经 Sum<int> 归约份额传出 → xmf_source_step 返回 -1 →
        xmf_step 中止（t 不进）→ Python step() raise RuntimeError。
  * 对照（同一卷带、同一配置、只改初值符号）：vx = +1 时同一带子的
    种子求值合法，整步必须成功——证明失败确实来自负底 POW 的定义域，
    而不是配置/装载问题。

失败步的可观察合同（全部写死在本文件，不读 .tmp/）：
  1. step() 抛 RuntimeError（消息含 "xmf_step failed"），不是静默返回；
  2. t 逐位保持 0.0（失败步不推进时间，xmf_step 的 fail 支路）；
  3. source_resid 归 0.0——残差只在成功步上报（xmf_source_step 入口清零、
     成功才写回），失败步不得残留上一成功步或半截归约的值；
  4. 失败格不得留下半截状态：所有格的失败都发生在写回之前，失败后的场
     必须与「同初值、空 sources 表、成功一步」的纯 hydro 参考态**逐位**
     相等，且全有限（若失败被吞，值路 pow(-1,0.5)=NaN 会漏进场里）。
  5. 失败步之后错误不得粘滞：同一 run 重新打包健康初值（vx=+1）再步，
     必须成功推进且源项真的作用了（密度改变）——钉每步归约份额都是
     新鲜的，上一失败步的 err 不得污染下一步。

数值口径：units 全 1、单流体 HI、笛卡尔 1D（8×1×1）、x1 outflow、
hll/pcm/rk1（均匀态走步的最简组合）。带子常数 krate = 1e-2 与指数 0.5
写死在 _fail_pow_cfg 里。参考态的期望不靠手算数值而靠同核对照（空表
配置与带病配置同初值同步数），任何核版本下都只隔离「源步是否留痕」。

线程无关性：断言不依赖线程数；OMP_NUM_THREADS=1 / 2 / 默认多线程下
错误传播必须同样成立（本文件随全套在分线程验收里各跑一遍）。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；断言覆盖五槽全体
（逐位对照 + 全有限 + 密度变化），不只测 VX1。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load

_PTR = ctypes.POINTER(ctypes.c_double)

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

# 必炸带子的常数（刻意非 1）：rate = krate*(vx_HI)**0.5
KRATE = 1.0e-2

# 均匀初值：密度/压强为 1，速度符号是失败/成功的唯一开关
RHO0, VX_BAD, VX_GOOD, P0 = 1.0, -1.0, 1.0, 1.0


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（setup 时 Kokkos 幂等守卫就绪）。"""
    return load()


# --- 小助手（口径同 test_source_step） ---------------------------------------


def _geom(run):
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _pack_uniform(lib, run, rho: float, vx: float, p: float) -> None:
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    ncells = lib.xmf_state_ncells(run._h)
    for var, val in ((SLOT_N, rho), (SLOT_VX, vx), (SLOT_P, p)):
        data = np.full((nzt, nyt, nxt), val, dtype=np.float64).ravel()
        assert data.size == ncells
        assert lib.xmf_state_pack(run._h, 0, var, data.ctypes.data_as(_PTR)) == ncells


def _unpack_all(lib, run) -> dict[int, np.ndarray]:
    """单流体五槽的整块（含 ghost）拷贝。"""
    ncells = lib.xmf_state_ncells(run._h)
    out = {}
    for var in range(5):
        buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(run._h, 0, var, buf.ctypes.data_as(_PTR)) == ncells
        out[var] = buf
    return out


def _fail_pow_cfg(with_tape: bool = True) -> dict:
    """笛卡尔 1D 单流体空题 + （可选）必炸带子 dn_HI = krate*(vx_HI)**0.5。

    units 全 1（码值即 cgs）、hll/pcm/rk1、x1 outflow（均匀态无墙）、
    ic none（初值全靠 pack）。with_tape=False 是空 sources 表的纯 hydro
    参考配置——与带病配置逐字段同构，只差 sources 一项。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [8, 1, 1]
    cfg["grid"]["x1min"] = 1.0
    cfg["grid"]["x1max"] = 2.0
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["fluids"] = [{"name": "HI", "mass": 1.00794, "charge": 0, "kind": "hd"}]
    cfg["ic"] = {"kind": "none"}
    cfg["boundary"] = {"x1": "outflow"}
    if with_tape:
        cfg["constants"] = {"krate": KRATE}
        cfg["sources"] = [
            {"name": "badpow", "expr": {"dn_HI": "krate*(vx_HI)**0.5"}}
        ]
    return cfg


# --- 1. 必炸步：整步失败、时间冻结、状态对纯 hydro 参考逐位无痕 ---------------


def test_failed_source_cell_fails_whole_step(lib):
    """vx 全 -1：每格的带种子 POW 求值都失败（底 ≤ 0）→ Sum<int> 归约把
    错误传出 → step() 必须抛 RuntimeError，t 逐位保持 0，source_resid
    归 0，且场与空表参考一步后的纯 hydro 态逐位相等、全有限。

    对照物是同核同初值的空 sources 配置（不是手算数值）：两跑同一步
    数、同一位打包，唯一差别是带病带子——于是「失败步不得留痕」被隔离
    成逐位相等断言。若错误传播被吞（静默通过），t 会推进、异常不抛，
    本测先红；若失败格的半截写回漏出，逐位对照与全有限断言红。"""
    # 参考跑：空 sources 表，同初值，成功一步（纯 hydro）
    ref = xmf.setup(_fail_pow_cfg(with_tape=False))
    try:
        assert ref.num_tapes == 0, "参考配置不得加载带子"
        _pack_uniform(lib, ref, RHO0, VX_BAD, P0)
        ref.step()
        assert ref.t > 0.0, "参考跑的一步必须成功（空表源步是 no-op）"
        ref_state = _unpack_all(lib, ref)
        ref_dt = ref.dt
    finally:
        ref.close()

    run = xmf.setup(_fail_pow_cfg(with_tape=True))
    try:
        assert run.num_tapes == 1, "带病配置应编出 1 条带子"
        _pack_uniform(lib, run, RHO0, VX_BAD, P0)
        assert run.t == 0.0

        with pytest.raises(RuntimeError, match=r"xmf_step failed"):
            run.step()

        # 时间冻结：失败步不推进（xmf_step 的 fail 支路）
        assert run.t == 0.0, (
            f"源步有格失败时整步必须失败且时间不动，实测 t={run.t!r}"
        )
        # dt 在步首已算好：与参考跑同初值同口径（哨兵：两跑确实同构）
        assert run.dt == ref_dt, (
            f"失败步的步首 dt={run.dt!r} 与参考 {ref_dt!r} 不同（初值不同构？）"
        )
        # 残差只在成功步上报：失败步必须归 0，不得残留半截归约值
        assert run.source_resid == 0.0, (
            f"失败步后 source_resid 应为 0（只报成功步），实测 "
            f"{run.source_resid!r}"
        )

        # 无痕 + 无 NaN：逐位等于纯 hydro 参考态，全槽全有限。
        # （若失败被吞成静默通过，值路 pow(-1,0.5)=NaN 会经 R 漏进
        #   写回；若失败格留半截状态，逐位对照红。）
        after = _unpack_all(lib, run)
        for var, name in enumerate(("n", "vx", "vy", "vz", "P")):
            assert np.all(np.isfinite(after[var])), (
                f"失败步后槽 {name} 出现非有限值——失败的格把脏值写进了场"
            )
            assert np.array_equal(after[var], ref_state[var]), (
                f"失败步后槽 {name} 与同初值纯 hydro 参考态逐位不同"
                "——失败格留下了半截状态"
            )
    finally:
        run.close()


# --- 2. 失败不粘滞 + 正底对照：同卷带、健康初值必须成功 -----------------------


def test_failed_step_does_not_stick_and_positive_base_succeeds(lib):
    """同一 run：先用 vx=-1 触发必炸步（确认抛错），再重新打包成
    vx=+1 的健康初值——同一条带子的种子 POW 求值此时合法，下一步
    必须成功：

    1. 错误归约份额是每步新鲜的：上一失败步的 err 不得粘到下一步
       （若 Sum<int> 的载体被错成持久量、或失败后内核状态残留，
       本步会误失败）；
    2. 正底对照证明失败确实来自负底 POW 定义域而非配置/装载问题：
       带子装载合法、LU 可解、步正常推进；
    3. 源项真的作用了：密度槽必须改变（防拿「成功但空转」冒充对照）。
    """
    run = xmf.setup(_fail_pow_cfg(with_tape=True))
    try:
        assert run.num_tapes == 1

        # 先触发必炸步：错误传播必须成立（与测 1 同口径，防本测自洽假象）
        _pack_uniform(lib, run, RHO0, VX_BAD, P0)
        with pytest.raises(RuntimeError, match=r"xmf_step failed"):
            run.step()
        assert run.t == 0.0

        # 重新打包健康初值（含 ghost 全块），同一条带子、同一 run
        _pack_uniform(lib, run, RHO0, VX_GOOD, P0)
        rho_before = _unpack_all(lib, run)[SLOT_N].copy()

        t1 = run.step()  # 不得再失败：错误不得粘滞
        assert t1 > 0.0 and run.t > 0.0, (
            f"失败步后健康初值的步必须成功推进，实测 t={run.t!r}"
        )
        assert np.isfinite(run.source_resid) and run.source_resid >= 0.0, (
            f"成功步的残差必须可报，实测 {run.source_resid!r}"
        )

        # 源项真的作用了：dn 带子改密度槽（线性率下增量 ≈ dt·mass_code·krate，
        # 非零即可——本测只钉「有增量」，数值精度由 test_source_step 持有）
        rho_after = _unpack_all(lib, run)[SLOT_N]
        assert not np.array_equal(rho_after, rho_before), (
            "健康初值下源步成功但密度未变——带子没有真正作用（对照失效）"
        )
        assert np.all(np.isfinite(rho_after)), "成功步后密度必须全有限"
        # 横速度仍为 0（1D、dn 带子不给横向动量；防失败残留的脏值复活）
        for var in (SLOT_VY, SLOT_VZ):
            assert np.all(_unpack_all(lib, run)[var] == 0.0)
    finally:
        run.close()
