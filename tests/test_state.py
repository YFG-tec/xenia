"""A2 state 契约单测（对照 docs/TODO.md → A2 第三行原文）。

第三行原文：
  单测 test_state：pack/unpack 往返；写 `vx` 不影响 `n`；`vy` 在 1D 保持 0；
  `nfluid=2` 时两套场互不覆盖。

卡片（docs/SESSION.md）第 4 条补充：可加 `off` 无 Pe；`pressure-only` 有 Pe
且长度 = ncells。

约定（docs/TODO.md §A）：一块一文件；断言只打 state 模块契约，不靠整条激波管
「看起来像」；维数覆盖 1D（nx2=nx3=1）与 nx2>1 的薄 2D。不读 pluto.ini、
不用 XUV_FACTOR==0 哨兵、不只测单速度分量（槽 0..4 全测）。除契约要求的两步
step（闲置分量 step 后不变、Pe step 后不变）外不跑物理。

被测符号（builder 在 _native.load() 绑定，ctypes 直接用）：
  xmf_state_ncells(h)            -> 含 ghost 的 ncells；非法返回 0
  xmf_state_nfluid(h)            -> 流体支数；非法返回 0
  xmf_state_pack(h,f,var,in)     -> 调用方缓冲→核内一整块；成功返回 ncells，
                                    非法/越界返回 -1 且不写
  xmf_state_unpack(h,f,var,out)  -> 核内→调用方缓冲，同上
  xmf_state_copy_pe(h,out) / xmf_state_write_pe(h,in)
                                 -> pressure-only 返回 ncells；off 或非法返回 -1
                                    且不碰缓冲
布局 [nfluid][5][ncells]，槽序 (n, vx, vy, vz, P) == 槽号 0..4；流体只用整数下标。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load, problem_to_c
from xmf.config import normalize

# --- 用例参数（取自 shock_tube() 默认值，防硬编码漂移） ----------------------
_BASE_GRID = xmf.shock_tube()["grid"]
_N = _BASE_GRID["n"]             # [256, 1, 1]
_NG = _BASE_GRID["nghost"]       # 2
NCELLS_1D = (_N[0] + 2 * _NG) * (_N[1] + 2 * _NG) * (_N[2] + 2 * _NG)  # 6500
NX1_2D, NX2_2D = 8, 4            # 薄 2D；核 sanity 下限 n[0]>=8，同 test_mesh

_PTR = ctypes.POINTER(ctypes.c_double)


def _uniform_faces(blob):
    """A1 补：直调 C 的用例按新签名补三向等距活动界面（n_d+1 个）。
    机械适配：界面由 blob 自身的 xdmin/xdmax/n 生成，断言语义不变。"""
    mins = (blob.x1min, blob.x2min, blob.x3min)
    maxs = (blob.x1max, blob.x2max, blob.x3max)
    arrs = []
    for d in range(3):
        nd = int(blob.n[d]) if int(blob.n[d]) >= 1 else 1
        dx = (maxs[d] - mins[d]) / nd
        arrs.append(
            np.array([mins[d] + k * dx for k in range(nd + 1)], dtype=np.float64)
        )
    return arrs


def _setup_c(lib, blob):
    """按新 xmf_setup(cfg, xf1, xf2, xf3) 签名直调 C。"""
    arrs = _uniform_faces(blob)
    return lib.xmf_setup(
        ctypes.byref(blob),
        arrs[0].ctypes.data_as(_PTR),
        arrs[1].ctypes.data_as(_PTR),
        arrs[2].ctypes.data_as(_PTR),
    )


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（state 查询符号已绑定）。"""
    return load()


# --- 用例构造 ----------------------------------------------------------------


@pytest.fixture
def run_1d():
    """默认激波管：1D、单流体、electron off。"""
    run = xmf.setup(xmf.shock_tube())
    yield run
    run.close()


@pytest.fixture
def run_pe():
    """1D、单流体、electron = pressure-only（键名以 config.normalize 为准）。"""
    cfg = xmf.shock_tube()
    cfg["electron"] = {"mode": "pressure-only"}
    run = xmf.setup(cfg)
    yield run
    run.close()


@pytest.fixture
def run_2d():
    """薄 2D（8×4×1，ndim=2）：§A 维数覆盖约定。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    run = xmf.setup(cfg)
    yield run
    run.close()


@pytest.fixture
def run_nfluid2(lib):
    """nfluid=2：builder 冒烟路线——改 ctypes blob 绕过 Python 前端。

    名字↔下标只在 Python；C 只按整数流体下标，故此处直接给 blob.nfluid=2。
    """
    blob = problem_to_c(normalize(xmf.shock_tube()))
    blob.nfluid = 2
    h = _setup_c(lib, blob)
    assert h, "nfluid=2 的 xmf_setup 应成功（卡片 builder 冒烟路线）"
    yield h
    lib.xmf_finalize(h)


# --- 只读小助手 --------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_PTR)


def _pattern(ncells: int, f: int, var: int, salt: float = 0.0) -> np.ndarray:
    """确定、非常值、逐位可复现的数据模式，且随 (f, var, salt) 互不相同。

    全为小整数加 0.25 的倍数，float64 下逐位精确，往返可用 assert_array_equal。
    """
    return np.arange(ncells, dtype=np.float64) + 0.25 * (var + 1) + 10.0 * f + salt


def _unpack(lib, h, f: int, var: int, ncells: int):
    """unpack 到新缓冲；返回 (ret, buf)。"""
    buf = np.empty(ncells, dtype=np.float64)
    ret = lib.xmf_state_unpack(h, f, var, _ptr(buf))
    return ret, buf


# --- 1. 块长自洽（含 ghost 的证据） ------------------------------------------


def test_block_length_is_ncells_with_ghosts(lib, run_1d):
    """state_ncells == pack/unpack 实际块长 == (n+2ng) 三向之积，且 > 内区 nx。"""
    h = run_1d._h
    ncells = lib.xmf_state_ncells(h)
    assert ncells == NCELLS_1D, (
        f"state 块长应为含 ghost 的 nxtot*nytot*nztot={NCELLS_1D}，实际 {ncells}"
    )
    nx_inner = lib.xmf_ncell(h)
    assert ncells > nx_inner, (
        f"state 块长 {ncells} 必须大于内区 x1 格数 {nx_inner}"
        "（不得复用 xmf_ncell 的内区 nx 冒充）"
    )
    assert lib.xmf_state_nfluid(h) == 1, "默认激波管应只有 1 支流体"
    data = _pattern(ncells, 0, 0)
    assert lib.xmf_state_pack(h, 0, 0, _ptr(data)) == ncells, "pack 成功应返回 ncells"
    ret, _ = _unpack(lib, h, 0, 0, ncells)
    assert ret == ncells, "unpack 成功应返回 ncells"


# --- 2. pack/unpack 往返：流体 0 的每个槽 0..4（第三行原文第 1 条） ----------


def test_pack_unpack_roundtrip_all_slots(lib, run_1d):
    """流体 0 槽 0..4（n,vx,vy,vz,P 全测，不只测 vx/n）：pack 再 unpack 逐位相等。"""
    h = run_1d._h
    ncells = lib.xmf_state_ncells(h)
    for var in range(5):
        data = _pattern(ncells, 0, var)
        got = lib.xmf_state_pack(h, 0, var, _ptr(data))
        assert got == ncells, f"槽 {var}：pack 应返回 ncells={ncells}，实际 {got}"
        ret, out = _unpack(lib, h, 0, var, ncells)
        assert ret == ncells, f"槽 {var}：unpack 应返回 ncells={ncells}，实际 {ret}"
        np.testing.assert_array_equal(
            out, data, err_msg=f"槽 {var}：pack→unpack 应逐位往返相等"
        )


# --- 3. 写 vx 不影响 n（第三行原文第 2 条） ----------------------------------


def test_write_vx_does_not_touch_other_slots(lib, run_1d):
    """pack 槽 1（vx）后：槽 0（n）逐位不变，槽 2/3/4 也不串。"""
    h = run_1d._h
    ncells = lib.xmf_state_ncells(h)
    before = {}
    for var in range(5):
        ret, buf = _unpack(lib, h, 0, var, ncells)
        assert ret == ncells
        before[var] = buf

    new_vx = _pattern(ncells, 0, 1, salt=777.0)
    assert np.any(new_vx != before[1]), "新数据应确实改变槽 1 内容（防空对空断言）"
    got = lib.xmf_state_pack(h, 0, 1, _ptr(new_vx))
    assert got == ncells, f"pack 槽 1 应返回 ncells，实际 {got}"

    ret, out1 = _unpack(lib, h, 0, 1, ncells)
    assert ret == ncells
    np.testing.assert_array_equal(out1, new_vx, err_msg="槽 1 应为新写入的数据")
    for var in (0, 2, 3, 4):
        ret, out = _unpack(lib, h, 0, var, ncells)
        assert ret == ncells
        np.testing.assert_array_equal(
            out, before[var], err_msg=f"写槽 1（vx）不得改动槽 {var}"
        )


# --- 4. 1D 的 vy/vz 保持 0（第三行原文第 3 条） ------------------------------


def test_1d_idle_vy_vz_zero_before_and_after_step(lib, run_1d):
    """1D 用例：setup 后槽 2/3 全 0；跑一步 step 后仍全 0（占位 HLL 不碰闲置分量）。"""
    h = run_1d._h
    ncells = lib.xmf_state_ncells(h)
    for var in (2, 3):
        ret, out = _unpack(lib, h, 0, var, ncells)
        assert ret == ncells
        assert np.all(out == 0.0), f"1D setup 后槽 {var}（vy/vz）应全为 0"

    run_1d.step()
    assert run_1d.t > 0.0, "step 应真正推进时间（防 step 后断言落空）"
    for var in (2, 3):
        ret, out = _unpack(lib, h, 0, var, ncells)
        assert ret == ncells
        assert np.all(out == 0.0), (
            f"step 后 1D 闲置槽 {var}（vy/vz）仍应全为 0（含 ghost 段）"
        )


# --- 5. nfluid=2 两套场互不覆盖（第三行原文第 4 条） -------------------------


def test_nfluid2_fields_do_not_overlap(lib, run_nfluid2):
    """写流体 1 不渗进流体 0 的同槽；反向写流体 0 也不渗进流体 1。"""
    h = run_nfluid2
    assert lib.xmf_state_nfluid(h) == 2, "blob.nfluid=2 后 C 应报 2 支流体"
    ncells = lib.xmf_state_ncells(h)

    # 流体 0 槽 0 是激波管 Riemann 初值，非零；先存快照
    ret, f0_before = _unpack(lib, h, 0, 0, ncells)
    assert ret == ncells
    assert np.any(f0_before != 0.0), (
        "流体 0 的 n 应为非零初值（防止拿双方的 0 冒充『互不覆盖』）"
    )

    # 方向一：写流体 1 槽 0 → 流体 0 槽 0 不变
    pat1 = _pattern(ncells, 1, 0)
    assert lib.xmf_state_pack(h, 1, 0, _ptr(pat1)) == ncells
    ret, out = _unpack(lib, h, 1, 0, ncells)
    assert ret == ncells
    np.testing.assert_array_equal(out, pat1, err_msg="流体 1 槽 0 应为写入的数据")
    ret, f0_after = _unpack(lib, h, 0, 0, ncells)
    assert ret == ncells
    np.testing.assert_array_equal(
        f0_after, f0_before, err_msg="写流体 1 槽 0 不得渗进流体 0 槽 0"
    )

    # 方向二：写流体 0 槽 2 → 流体 1 槽 2 不变（calloc 的 0）
    pat0 = _pattern(ncells, 0, 2)
    assert lib.xmf_state_pack(h, 0, 2, _ptr(pat0)) == ncells
    ret, out0 = _unpack(lib, h, 0, 2, ncells)
    assert ret == ncells
    np.testing.assert_array_equal(out0, pat0, err_msg="流体 0 槽 2 应为写入的数据")
    ret, out1 = _unpack(lib, h, 1, 2, ncells)
    assert ret == ncells
    assert np.all(out1 == 0.0), "写流体 0 槽 2 不得渗进流体 1 槽 2"

    # 方向二补强：写流体 0 槽 4（P）→ 流体 1 槽 4 不变；
    # 且早先写入的流体 1 槽 0 不被后续对流体 0 的写操作覆盖
    patp = _pattern(ncells, 0, 4)
    assert lib.xmf_state_pack(h, 0, 4, _ptr(patp)) == ncells
    ret, out1p = _unpack(lib, h, 1, 4, ncells)
    assert ret == ncells
    assert np.all(out1p == 0.0), "写流体 0 槽 4 不得渗进流体 1 槽 4"
    ret, out1n = _unpack(lib, h, 1, 0, ncells)
    assert ret == ncells
    np.testing.assert_array_equal(
        out1n, pat1, err_msg="流体 1 槽 0 不应被随后对流体 0 的写操作覆盖"
    )


# --- 6. off 无 Pe（卡片第 4 条补充） ------------------------------------------


def test_electron_off_pe_refused(lib, run_1d):
    """默认 shock_tube（electron off）：copy_pe/write_pe 返回 -1 且不碰缓冲。"""
    h = run_1d._h
    ncells = lib.xmf_state_ncells(h)

    out = np.full(ncells, -123.0, dtype=np.float64)  # 哨兵填充，探测越权写入
    assert lib.xmf_state_copy_pe(h, _ptr(out)) == -1, "electron off 时 copy_pe 应返回 -1"
    assert np.all(out == -123.0), "copy_pe 返回 -1 后不得写输出缓冲"

    src = _pattern(ncells, 0, 0)
    assert lib.xmf_state_write_pe(h, _ptr(src)) == -1, "electron off 时 write_pe 应返回 -1"
    assert lib.xmf_state_copy_pe(h, _ptr(out)) == -1, (
        "write_pe 被拒后 Pe 仍不应存在（copy_pe 仍返回 -1）"
    )
    assert np.all(out == -123.0), "缓冲在两次被拒调用后应保持哨兵值"

    # NULL 缓冲同拒
    assert lib.xmf_state_copy_pe(h, None) == -1, "copy_pe(NULL 缓冲) 应返回 -1"
    assert lib.xmf_state_write_pe(h, None) == -1, "write_pe(NULL 缓冲) 应返回 -1"


# --- 7. pressure-only 有 Pe 且长度 = ncells（卡片第 4 条补充） ---------------


def test_pressure_only_pe_block(lib, run_pe):
    """pressure-only：Pe 长 ncells；初值全 0；write/copy 逐位往返；step 不碰 Pe。"""
    h = run_pe._h
    ncells = lib.xmf_state_ncells(h)
    assert lib.xmf_state_nfluid(h) == 1, (
        "电子不是流体：pressure-only 时 nfluid 仍应为 1（Pe 不计进 nfluid）"
    )

    out = np.empty(ncells, dtype=np.float64)
    got = lib.xmf_state_copy_pe(h, _ptr(out))
    assert got == ncells, f"pressure-only 时 copy_pe 应返回 ncells={ncells}，实际 {got}"
    assert np.all(out == 0.0), "Pe 初值应 calloc 为全 0"

    # Pe 与流体槽同长、互不串：写入 Pe 前先存流体 0 槽 4（P）快照
    ret, p_before = _unpack(lib, h, 0, 4, ncells)
    assert ret == ncells

    pe = _pattern(ncells, 9, 4, salt=0.5)
    got = lib.xmf_state_write_pe(h, _ptr(pe))
    assert got == ncells, f"write_pe 应返回 ncells={ncells}，实际 {got}"
    ret, out2 = _unpack_pe(lib, h, ncells)
    assert ret == ncells
    np.testing.assert_array_equal(out2, pe, err_msg="write_pe 后 copy_pe 应逐位相等")

    ret, p_after = _unpack(lib, h, 0, 4, ncells)
    assert ret == ncells
    np.testing.assert_array_equal(
        p_after, p_before, err_msg="写 Pe 不得渗进流体 0 的 P 槽（Pe 是单独能量槽）"
    )

    # 占位 step 不碰 Pe
    run_pe.step()
    assert run_pe.t > 0.0, "step 应真正推进时间"
    ret, out3 = _unpack_pe(lib, h, ncells)
    assert ret == ncells
    np.testing.assert_array_equal(out3, pe, err_msg="step 不应改动 P_e")


def _unpack_pe(lib, h, ncells: int):
    buf = np.empty(ncells, dtype=np.float64)
    ret = lib.xmf_state_copy_pe(h, _ptr(buf))
    return ret, buf


# --- 8. 越界哨兵 --------------------------------------------------------------


def test_out_of_range_sentinels(lib, run_1d, run_nfluid2):
    """f 越界、var 越界（-1、5）→ pack/unpack 返回 -1 且不写缓冲；NULL 句柄同拒。"""
    ncells = lib.xmf_state_ncells(run_1d._h)
    sentinel = -123.0

    for h, nf, tag in ((run_1d._h, 1, "1D"), (run_nfluid2, 2, "nfluid=2")):
        for f in (-1, nf):  # nf == nfluid，越上界
            for var in (0, 2):
                buf = np.full(ncells, sentinel, dtype=np.float64)
                assert lib.xmf_state_pack(h, f, var, _ptr(buf)) == -1, (
                    f"{tag}：f={f} 越界 pack 应返回 -1"
                )
                assert lib.xmf_state_unpack(h, f, var, _ptr(buf)) == -1, (
                    f"{tag}：f={f} 越界 unpack 应返回 -1"
                )
                assert np.all(buf == sentinel), (
                    f"{tag}：f={f} 越界返回 -1 后不得写缓冲"
                )
        for var in (-1, 5):  # 槽只有 0..4
            buf = np.full(ncells, sentinel, dtype=np.float64)
            assert lib.xmf_state_pack(h, 0, var, _ptr(buf)) == -1, (
                f"{tag}：var={var} 越界 pack 应返回 -1"
            )
            assert lib.xmf_state_unpack(h, 0, var, _ptr(buf)) == -1, (
                f"{tag}：var={var} 越界 unpack 应返回 -1"
            )
            assert np.all(buf == sentinel), (
                f"{tag}：var={var} 越界返回 -1 后不得写缓冲"
            )

    assert lib.xmf_state_ncells(None) == 0, "xmf_state_ncells(NULL 句柄) 应返回 0"
    assert lib.xmf_state_nfluid(None) == 0, "xmf_state_nfluid(NULL 句柄) 应返回 0"
    buf = np.full(4, sentinel, dtype=np.float64)
    assert lib.xmf_state_pack(None, 0, 0, _ptr(buf)) == -1, "pack(NULL 句柄) 应返回 -1"
    assert lib.xmf_state_unpack(None, 0, 0, _ptr(buf)) == -1, "unpack(NULL 句柄) 应返回 -1"
    assert lib.xmf_state_copy_pe(None, _ptr(buf)) == -1, "copy_pe(NULL 句柄) 应返回 -1"
    assert lib.xmf_state_write_pe(None, _ptr(buf)) == -1, "write_pe(NULL 句柄) 应返回 -1"
    assert np.all(buf == sentinel), "NULL 句柄调用后不得写缓冲"


# --- 9. 薄 2D 维数覆盖（§A 约定：nx2>1） -------------------------------------


def test_thin2d_block_length_and_roundtrip(lib, run_2d):
    """薄 2D（8×4×1）：块长 = (nx1+2ng)*(nx2+2ng)*(1+2ng) 含 ghost；各槽往返。"""
    h = run_2d._h
    ncells = lib.xmf_state_ncells(h)
    expect = (NX1_2D + 2 * _NG) * (NX2_2D + 2 * _NG) * (1 + 2 * _NG)
    assert ncells == expect, (
        f"薄 2D 块长应为 {expect}（含三向 ghost），实际 {ncells}"
    )
    assert ncells > lib.xmf_ncell(h), "薄 2D 块长仍应大于内区 x1 格数"
    for var in range(5):
        data = _pattern(ncells, 0, var, salt=31.0)
        got = lib.xmf_state_pack(h, 0, var, _ptr(data))
        assert got == ncells, f"薄 2D 槽 {var}：pack 应返回 ncells={ncells}"
        ret, out = _unpack(lib, h, 0, var, ncells)
        assert ret == ncells
        np.testing.assert_array_equal(
            out, data, err_msg=f"薄 2D 槽 {var}：pack→unpack 应逐位往返相等"
        )
