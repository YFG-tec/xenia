"""A4.6 核内求值合同单测（test_ir_eval）：IR 带子在 C 核里的 apply 行为。

文件头约定（与 src/xmf/core/xmf_core.h / xmf_ir.c 的锁一致，本刀写死）：
  一次 apply_sources() = 每条带子的**瞬时值直接加为增量**
  （重力测：Δv_x = -G*M/r²，即 vx += expr）——不乘 Δt、不走 hydro、
  不求值 ghost、不推进时间。时间语义（速率还是增量、Δt 怎么进）是
  A4.7/A5 的事，本刀不发明。空表不加载任何带子，step 保持纯 hydro。

口径（同 test_hydro.py）：unpack 出的是核内码单位，块长 =
xmf_state_ncells 含 ghost，reshape 顺序 (nztot, nytot, nxtot)，索引
(k,j,i)，i 最快。重力比较全在码单位里做：带子写进 vx 槽的就是 -G*M/r²
的数值，r 取核读回的 x1 体心坐标（不是格子索引）。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；不只测 VX1——
同一重力带子作用后必须同时钉住「n、P 逐位不变」「vy、vz 全块仍为 0」
「ghost 区逐位不变」。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load

# 槽序 (n, vx, vy, vz, P) == 0..4（A2 契约）
SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4

_PTR = ctypes.POINTER(ctypes.c_double)

# 重力测常数：刻意取非 1 的数，避免 -G*M/r² 碰巧与任何单位换算重合
G, M = 2.5, 3.75


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（state/mesh/ir 符号已绑定）。"""
    return load()


# --- 只读小助手（口径同 test_hydro.py） --------------------------------------


def _geom(run):
    """从 run 的已归一化 config 取 (ng, nx, ny, nz, nxtot, nytot, nztot)。"""
    p = run._cfg
    ng = p.ng
    nx, ny, nz = p.n
    return ng, nx, ny, nz, nx + 2 * ng, ny + 2 * ng, nz + 2 * ng


def _unpack(lib, run, var):
    """unpack 流体 0 槽 var 的整块（含 ghost），返回 flat 缓冲。"""
    ncells = lib.xmf_state_ncells(run._h)
    buf = np.empty(ncells, dtype=np.float64)
    assert lib.xmf_state_unpack(run._h, 0, var, buf.ctypes.data_as(_PTR)) == ncells
    return buf


def _centres(lib, run):
    """x1 体心坐标（含 ghost），长度 nx+2ng；与 xmf_fields 读到的 x 同数组。"""
    ntot = lib.xmf_mesh_ncell(run._h, 0)
    buf = np.empty(ntot, dtype=np.float64)
    assert lib.xmf_mesh_copy_x(run._h, 0, buf.ctypes.data_as(_PTR)) == ntot
    return buf


def _gravity_cfg() -> dict:
    """笛卡尔小网格重力用例：x1∈[1,2] 避开 r=0，单流体，一条裸 dvx 带子。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [16, 1, 1]
    cfg["grid"]["x1min"] = 1.0
    cfg["grid"]["x1max"] = 2.0
    cfg["constants"] = {"G": G, "M": M}
    cfg["sources"] = [{"name": "g", "expr": {"dvx": "-G*M/r**2"}}]
    return cfg


def _interior_mask(run) -> np.ndarray:
    """(nztot, nytot, nxtot) 布尔块：内部格为 True（1D 即单行内部段）。"""
    ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
    mask = np.zeros((nzt, nyt, nxt), dtype=bool)
    mask[ng : ng + nz, ng : ng + ny, ng : ng + nx] = True
    return mask


# --- 1. 重力带对解析值（卡片第 4 条主断言） ----------------------------------


def test_gravity_apply_matches_analytic_value(lib):
    """一次 apply 后，逐内部格 vx == -G*M/r²（码单位），r 钉体心坐标。

    钉 r 的方式：解析式里的 r 用核读回的 x1 体心坐标（xmf_mesh_copy_x，
    与 fields 的 x 同数组）代入，不许用索引冒充坐标；另与配置网格算出的
    体心交叉核对。G、M 取非 1 常数（2.5、3.75）避免碰巧。
    """
    run = xmf.setup(_gravity_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        assert run.num_tapes == 1, "一条重力带子应加载为 1 条 tape"

        xc = _centres(lib, run)[ng : ng + nx]
        # fields 读到的格心坐标与 mesh 体心同数组（钉 r 的两条读数路径一致）
        assert np.array_equal(run.fields["HI"].x, xc), (
            "fields 的 x 应与核内 x1 体心数组一致"
        )
        # 与配置网格算出的体心交叉核对（均匀网格，容差只防浮点重排）
        dx = (run._cfg.x1max - run._cfg.x1min) / nx
        xc_cfg = run._cfg.x1min + (np.arange(nx) + 0.5) * dx
        assert np.allclose(xc, xc_cfg, rtol=1e-14, atol=0), (
            "核内体心应与配置网格 [1,2]/16 的均匀体心一致"
        )

        vx0 = _unpack(lib, run, SLOT_VX)
        assert np.all(vx0 == 0.0), "IC 两态 v=0，apply 前 vx 应全块为 0"

        run.apply_sources()

        vx = _unpack(lib, run, SLOT_VX).reshape((nzt, nyt, nxt))
        vx_int = vx[ng, ng, ng : ng + nx]
        exact = -G * M / xc**2
        rel = float(np.max(np.abs((vx_int - exact) / exact)))
        assert rel <= 1.0e-12, (
            f"核 apply 与解析 -G*M/r² 的最大相对差 {rel:.3e} 超过 1e-12"
        )
        # 相对差达标后再钉符号与量级：全部内部格应为负（向心），量级 ~G*M
        assert np.all(vx_int < 0.0), "dvx=-G*M/r² 应全为负"
        assert 0.5 * G * M <= float(np.max(np.abs(vx_int))) <= G * M
    finally:
        run.close()


def test_gravity_apply_only_touches_vx_interior(lib):
    """同一条带子作用后：n、P 逐位不变，vy、vz 全块仍 0，ghost 逐位不变。

    禁止只测 VX1 的落点：重力带只写 vx 增量，其余槽与 ghost 区必须
    整块（含 ghost）逐位保持；apply 也不得推进时间。
    """
    run = xmf.setup(_gravity_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        before = {s: _unpack(lib, run, s) for s in range(5)}
        t0 = run.t

        run.apply_sources()

        after = {s: _unpack(lib, run, s) for s in range(5)}

        # n 与 P：整块（含 ghost）逐位不变
        assert np.array_equal(before[SLOT_N], after[SLOT_N]), (
            "重力带不得改动 n（整块含 ghost 逐位比对）"
        )
        assert np.array_equal(before[SLOT_P], after[SLOT_P]), (
            "重力带不得改动 P（整块含 ghost 逐位比对）"
        )
        # 闲置速度分量：全块仍严格为 0
        for s, name in ((SLOT_VY, "vy"), (SLOT_VZ, "vz")):
            assert np.all(after[s] == 0.0), (
                f"重力带（只写 dvx）不得生出 {name}，实测 max|"
                f"{name}|={float(np.max(np.abs(after[s]))):.3e}"
            )
        # vx 的 ghost 段逐位不变（apply 只写内部格）
        interior = _interior_mask(run).ravel()
        assert np.array_equal(before[SLOT_VX][~interior], after[SLOT_VX][~interior]), (
            "apply 不得触碰 ghost 区的 vx"
        )
        # apply 不是第五个动词：不推进时间
        assert run.t == t0 == 0.0, "apply_sources 不得推进模拟时间"
    finally:
        run.close()


def test_second_apply_adds_increment_again(lib):
    """两次 apply ⇒ vx = 2·(-G*M/r²)：写目标是「加增量」而非覆写。

    初值 vx=0 时一次 apply 分辨不出「加」与「赋值」；第二次 apply 把两者
    分开（加 ⇒ 2 倍，赋值 ⇒ 仍是 1 倍），锁死 xmf_ir_apply 的 ADDS 语义。
    """
    run = xmf.setup(_gravity_cfg())
    try:
        ng, nx, ny, nz, nxt, nyt, nzt = _geom(run)
        xc = _centres(lib, run)[ng : ng + nx]
        exact = -G * M / xc**2

        run.apply_sources()
        run.apply_sources()

        vx = _unpack(lib, run, SLOT_VX).reshape((nzt, nyt, nxt))
        vx_int = vx[ng, ng, ng : ng + nx]
        rel = float(np.max(np.abs((vx_int - 2.0 * exact) / exact)))
        assert rel <= 1.0e-12, (
            f"两次 apply 应为 2 倍增量（加而非覆写），最大相对差 {rel:.3e}"
        )
    finally:
        run.close()


# --- 2. 空表 ⇒ L_c=0：一步 step 与纯 hydro 逐位相同 --------------------------


def test_empty_sources_step_bitwise_pure_hydro(lib):
    """同一网格 sources=[]：num_tapes==0；一步 step 后与无 sources 键的
    纯 hydro 逐格逐位相同（五槽全量 + t + dt）。

    基线取「同一初值、无 sources 键」的 setup：两条路径本应同为纯 hydro；
    任何逐位差都说明空表路径动了 hydro。
    """
    cfg_empty = _gravity_cfg()
    cfg_empty["sources"] = []
    cfg_none = _gravity_cfg()
    del cfg_none["sources"]

    run_empty = xmf.setup(cfg_empty)
    run_pure = xmf.setup(cfg_none)
    try:
        assert run_empty.num_tapes == 0, "sources=[] 不得加载任何带子"
        assert run_pure.num_tapes == 0, "无 sources 键不得加载任何带子"

        run_empty.step()
        run_pure.step()
        assert run_empty.t > 0.0, "step 应真正推进时间"

        for s, name in enumerate("n vx vy vz P".split()):
            a = _unpack(lib, run_empty, s)
            b = _unpack(lib, run_pure, s)
            assert np.array_equal(a, b), (
                f"空表一步 step 后槽 {name} 与纯 hydro 逐位不同（L∞ 差 "
                f"{float(np.max(np.abs(a - b))):.3e}，码单位）"
            )
        assert run_empty.t == run_pure.t, "空表与纯 hydro 的 t 应逐位相同"
        assert run_empty.dt == run_pure.dt, "空表与纯 hydro 的 dt 应逐位相同"
    finally:
        run_empty.close()
        run_pure.close()
