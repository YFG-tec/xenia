"""A3 bc 契约单测（对照 docs/TODO.md → A3 第二行原文）。

第二行原文：
  单测 test_bc：1D 反射后 ghost 密度偶、法向速度奇；2D 只垫 j 面不影响 i 内部。

卡片（docs/SESSION.md）写死的坑：激波管 IC 左右 v=0，只看 ghost 速度是否
为 0 时 copy 与 reflect 无法区分——本文件所有奇偶断言都先 ``xmf_state_pack``
一段**非零**内区数据（五槽全非零），再 ``xmf_fill_bc`` 垫 ghost，最后
``xmf_state_unpack`` 读回比对。禁止只断言「ghost 速度为 0」。

约定（docs/TODO.md §A）：一块一文件；断言只打 bc 模块契约，不靠整条激波管
「看起来像」；维数覆盖 1D（nx2=nx3=1）与薄 2D（8×4）。不读 pluto.ini、
不用 XUV_FACTOR==0 哨兵、不只测单速度分量：槽 0..4 全涉及，x1 面法向槽是
vx（槽 1），2D 的 x2 面法向槽是 vy（槽 2），j 向必须被碰到。除契约要求外
不跑 step 物理。

被测符号（builder 在 _native.load() 绑定，ctypes 直接用）：
  xmf_bc_type(h, face) -> 该面枚举整数（reflect=0, copy=1, outflow=2,
                          user=3, none=4；与 config.BC_TYPES 顺序一致）；
                          face∈0..5 == x1beg,x1end,x2beg,x2end,x3beg,x3end；
                          坏参返回 -1
  xmf_fill_bc(h)       -> 按 run 当前六面类型立刻垫一次 ghost；成功 0，
                          坏参 -1。与 xmf_state_pack 同级的单测工具，
                          不是公开第五动词。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import BC_ENUM, FACES

# --- 用例参数（取自 shock_tube() 默认值，防硬编码漂移） ----------------------
_BASE_GRID = xmf.shock_tube()["grid"]
_NX1 = _BASE_GRID["n"][0]        # 256
_NG = _BASE_GRID["nghost"]       # 2
_NXTOT = _NX1 + 2 * _NG          # 260
_NYTOT = 1 + 2 * _NG             # 5（1D 横方向退化，仍整面写）
_NZTOT = 1 + 2 * _NG             # 5

NX1_2D, NX2_2D = 8, 4            # 薄 2D；核 sanity 下限 n[0]>=8，同 test_state
_NXTOT2 = NX1_2D + 2 * _NG       # 12
_NYTOT2 = NX2_2D + 2 * _NG       # 8
_NZTOT2 = 1 + 2 * _NG            # 5
NCELLS_2D = _NXTOT2 * _NYTOT2 * _NZTOT2

# 槽序 (n, vx, vy, vz, P) == 0..4（A2 契约）
SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4
# 面序写死：0=x1beg .. 5=x3end（与 config.FACES 一致）
FACE_X1BEG, FACE_X1END, FACE_X2BEG, FACE_X2END, FACE_X3BEG, FACE_X3END = range(6)

_PTR = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope="module")
def lib():
    """已安装包内的共享库（bc 查询符号已绑定）。"""
    return load()


# --- 用例构造 ----------------------------------------------------------------


@pytest.fixture
def run_1d():
    """默认激波管：1D、单流体、x1 两端 reflect（boundary={"x1": "reflect"}）。"""
    run = xmf.setup(xmf.shock_tube())
    yield run
    run.close()


@pytest.fixture
def run_copy():
    """1D，x1 两端 copy（轴短写，_faces() 实际接受的键写法）。"""
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1": "copy"}
    run = xmf.setup(cfg)
    yield run
    run.close()


@pytest.fixture
def run_user():
    """1D，x1beg=user、x1end=reflect（六面全名键；同 run 内对照两种面）。"""
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1beg": "user", "x1end": "reflect"}
    run = xmf.setup(cfg)
    yield run
    run.close()


@pytest.fixture
def run_2d():
    """薄 2D（8×4×1，ndim=2）：x1 两端 copy、x2 两端 reflect、x3 为 none。

    x1 面故意设成与 x2 面**不同类型**：同一次 fill 里两面行为必须按各自
    类型分开，才证明是「按面垫」而不是三向一律 reflect。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    run = xmf.setup(cfg)
    yield run
    run.close()


# --- 只读小助手 --------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_PTR)


def _pattern(ncells: int, var: int, salt: float = 0.0) -> np.ndarray:
    """处处非零、逐位可复现、随 var/salt 互不相同的模式。

    arange+常数在 float64 下逐位精确，取负号也精确，奇偶断言可直接用 ==。
    """
    return np.arange(ncells, dtype=np.float64) + 0.25 * (var + 1) + salt


def _pack_all(lib, h, ncells: int, salt: float = 0.0) -> dict:
    """五槽都 pack 非零模式（含 ghost 段）；返回 {var: pattern}。"""
    pats = {}
    for var in range(5):
        pat = _pattern(ncells, var, salt)
        assert lib.xmf_state_pack(h, 0, var, _ptr(pat)) == ncells, (
            f"槽 {var}：pack 应返回 ncells={ncells}"
        )
        pats[var] = pat
    return pats


def _unpack_all(lib, h, ncells: int) -> dict:
    """五槽 unpack 到新缓冲；返回 {var: buf}。"""
    out = {}
    for var in range(5):
        buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(h, 0, var, _ptr(buf)) == ncells, (
            f"槽 {var}：unpack 应返回 ncells={ncells}"
        )
        out[var] = buf
    return out


# --- 1. shock_tube 六面类型经 C 查询（卡片第 4 条） --------------------------


def test_shock_tube_face_types_via_c_query(lib, run_1d):
    """x1beg/x1end 真是 reflect：经 xmf_bc_type 查 C，不是只看 Python dict。

    其余面（x2/x3）应为 none=4，全部与 config 展开后的 boundary 一致。
    """
    h = run_1d._h
    got = [lib.xmf_bc_type(h, face) for face in range(6)]
    expect = [BC_ENUM[run_1d._cfg.boundary[f]] for f in FACES]
    assert got == expect, (
        f"C 查询的六面类型 {got} 应与 config.boundary 编码 {expect} 一致"
        f"（boundary={run_1d._cfg.boundary}）"
    )
    assert got[FACE_X1BEG] == BC_ENUM["reflect"] == 0, "x1beg 应为 reflect=0"
    assert got[FACE_X1END] == BC_ENUM["reflect"] == 0, "x1end 应为 reflect=0"
    for face in (FACE_X2BEG, FACE_X2END, FACE_X3BEG, FACE_X3END):
        assert got[face] == BC_ENUM["none"] == 4, f"面 {face} 应为 none=4"


# --- 2. 1D 反射：密度偶、法向速度奇（第二行原文第 1 条） ----------------------


def test_1d_reflect_density_even_normal_velocity_odd(lib, run_1d):
    """pack 非零五槽 → fill → x1 两端 ghost：vx 逐位镜像翻号，其余槽镜像同号。

    - 密度（槽 0）偶、法向速度 vx（槽 1）奇，是第二行原文主断言；
    - vy/vz（槽 2/3）也 pack 了非零，按偶（不翻号）——反射只翻该面法向槽；
    - P（槽 4）同偶；内区逐位不被 fill 触碰。

    1D 横方向各只有一层内区（j0=k0=ng）：内区只有 (k0,j0) 这一行 i 线；
    其余 (k,j) 行是横方向 ghost 行，按卡片约定由 x2/x3 面整面写（none 本刀
    按 copy 垫、不得未初始化），故横平面断言用「行内奇偶自洽」，主断言在
    中心行上锚定到 pack 原值。
    """
    h = run_1d._h
    ncells = lib.xmf_state_ncells(h)
    pats = _pack_all(lib, h, ncells)
    assert np.all(pats[SLOT_VX] != 0.0), "内区法向速度必须非零（禁 v=0 碰巧）"

    assert lib.xmf_fill_bc(h) == 0, "fill_bc 成功应返回 0"
    out = _unpack_all(lib, h, ncells)

    shape = (_NZTOT, _NYTOT, _NXTOT)
    j0 = k0 = _NG  # 1D 横方向唯一内区层
    for var in range(5):
        got = out[var].reshape(shape)
        pat = pats[var].reshape(shape)
        flip = var == SLOT_VX  # x1 面的法向槽只有 vx
        # 内区（中心行）逐位不被 fill 改动
        np.testing.assert_array_equal(
            got[k0, j0, _NG : _NG + _NX1],
            pat[k0, j0, _NG : _NG + _NX1],
            err_msg=f"槽 {var}：fill 不得触碰内区",
        )
        # 主断言（第二行原文）：中心行 ghost == ± pack 的内区镜像值，
        # 密度等偶（同号）、vx 奇（翻号），全部对非零数据断言
        for m in range(_NG):
            ig, isrc = _NG - 1 - m, _NG + m
            want = -pat[k0, j0, isrc] if flip else pat[k0, j0, isrc]
            assert got[k0, j0, ig] == want, (
                f"槽 {var} x1beg m={m}：ghost 应为内区镜像"
                f"{'翻号（奇）' if flip else '同号（偶）'}"
            )
            ig, isrc = _NG + _NX1 + m, _NG + _NX1 - 1 - m
            want = -pat[k0, j0, isrc] if flip else pat[k0, j0, isrc]
            assert got[k0, j0, ig] == want, (
                f"槽 {var} x1end m={m}：ghost 应为内区镜像"
                f"{'翻号（奇）' if flip else '同号（偶）'}"
            )
        # 横平面整面写：每一 (k,j) 行（含横方向 ghost 行）的 x1 ghost
        # 都与该行自己的内区值奇偶自洽
        for k in range(_NZTOT):
            for j in range(_NYTOT):
                for m in range(_NG):
                    ig, isrc = _NG - 1 - m, _NG + m
                    want = -got[k, j, isrc] if flip else got[k, j, isrc]
                    assert got[k, j, ig] == want, (
                        f"槽 {var} x1beg m={m} (k={k},j={j})：横平面奇偶应自洽"
                    )
                    ig, isrc = _NG + _NX1 + m, _NG + _NX1 - 1 - m
                    want = -got[k, j, isrc] if flip else got[k, j, isrc]
                    assert got[k, j, ig] == want, (
                        f"槽 {var} x1end m={m} (k={k},j={j})：横平面奇偶应自洽"
                    )

    # ghost 法向速度必须非零：内区非零 → 镜像翻号后仍非零，
    # 直接排除「靠 ghost 速度为 0 装反射」
    vxg = out[SLOT_VX].reshape(shape)
    assert np.any(vxg[k0, j0, :_NG] != 0.0), "x1beg ghost vx 应非零"
    assert np.any(vxg[k0, j0, _NG + _NX1 :] != 0.0), "x1end ghost vx 应非零"


# --- 3. fill_bc 幂等（工具语义：只依赖内区数据） ------------------------------


def test_fill_bc_is_idempotent(lib, run_1d):
    """同一内区连垫两次结果与垫一次逐位相同（fill 不改内区、只读内区）。"""
    h = run_1d._h
    ncells = lib.xmf_state_ncells(h)
    _pack_all(lib, h, ncells)
    assert lib.xmf_fill_bc(h) == 0
    first = _unpack_all(lib, h, ncells)
    assert lib.xmf_fill_bc(h) == 0
    second = _unpack_all(lib, h, ncells)
    for var in range(5):
        np.testing.assert_array_equal(
            second[var], first[var], err_msg=f"槽 {var}：二次 fill 应与一次相同"
        )


# --- 4. copy 不翻号：ghost == 最外层内区（卡片断言清单第 3 条） ---------------


def test_copy_face_replicates_outermost_layer_no_flip(lib, run_copy):
    """x1 两端 copy：各槽 ghost 逐位 == 该向最外层内区，法向速度符号不变。"""
    h = run_copy._h
    assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["copy"] == 1
    assert lib.xmf_bc_type(h, FACE_X1END) == BC_ENUM["copy"] == 1

    ncells = lib.xmf_state_ncells(h)
    pats = _pack_all(lib, h, ncells)
    assert lib.xmf_fill_bc(h) == 0
    out = _unpack_all(lib, h, ncells)

    # 同 1D 反射用例：内区只有中心行 (k0,j0)，横方向 ghost 行由 x2/x3 面
    # （none 按 copy 垫）整面写，故横平面断言用行内零梯度自洽，中心行锚定 pack。
    shape = (_NZTOT, _NYTOT, _NXTOT)
    j0 = k0 = _NG
    for var in range(5):
        got = out[var].reshape(shape)
        pat = pats[var].reshape(shape)
        # 内区（中心行）不被 fill 改动
        np.testing.assert_array_equal(
            got[k0, j0, _NG : _NG + _NX1],
            pat[k0, j0, _NG : _NG + _NX1],
            err_msg=f"槽 {var}：fill 不得触碰内区",
        )
        # 中心行 ghost == pack 的最外层内区值（零梯度、不翻号）
        for m in range(_NG):
            assert got[k0, j0, _NG - 1 - m] == pat[k0, j0, _NG], (
                f"槽 {var} x1beg m={m}：copy 应复刻最外层内区"
            )
            assert got[k0, j0, _NG + _NX1 + m] == pat[k0, j0, _NG + _NX1 - 1], (
                f"槽 {var} x1end m={m}：copy 应复刻最外层内区"
            )
        # 横平面整面写：每一 (k,j) 行 ghost == 该行自己的最外层内区层
        for k in range(_NZTOT):
            for j in range(_NYTOT):
                for m in range(_NG):
                    assert got[k, j, _NG - 1 - m] == got[k, j, _NG], (
                        f"槽 {var} x1beg m={m} (k={k},j={j})：copy 应零梯度"
                    )
                    assert got[k, j, _NG + _NX1 + m] == got[k, j, _NG + _NX1 - 1], (
                        f"槽 {var} x1end m={m} (k={k},j={j})：copy 应零梯度"
                    )

    # 非零法向速度不翻号：模式全正，copy 后中心行 ghost 仍全正（reflect 会全负）
    vxg = out[SLOT_VX].reshape(shape)
    assert np.all(vxg[k0, j0, :_NG] > 0.0), "copy 的 x1beg ghost vx 不得翻号"
    assert np.all(vxg[k0, j0, _NG + _NX1 :] > 0.0), "copy 的 x1end ghost vx 不得翻号"


# --- 5. user 可查询且 ghost 有定义（卡片断言清单第 5 条） ---------------------


def test_user_face_queryable_and_ghost_defined(lib, run_user):
    """x1beg=user：C 查询返回 3；fill 后 ghost 有限且按 copy 垫（本刀约定）。

    同一 run 的 x1end 仍是 reflect：user 面不翻号、reflect 面翻号，互为对照，
    证明 user 没有被当成 reflect，也没有留下未初始化垃圾。
    """
    h = run_user._h
    assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["user"] == 3, (
        "user 面经 C 查询应返回 3"
    )
    assert lib.xmf_bc_type(h, FACE_X1END) == BC_ENUM["reflect"] == 0

    ncells = lib.xmf_state_ncells(h)
    pats = _pack_all(lib, h, ncells)
    assert lib.xmf_fill_bc(h) == 0
    out = _unpack_all(lib, h, ncells)

    shape = (_NZTOT, _NYTOT, _NXTOT)
    j0 = k0 = _NG  # 1D 横方向唯一内区层
    for var in range(5):
        got = out[var].reshape(shape)
        pat = pats[var].reshape(shape)
        assert np.all(np.isfinite(got)), f"槽 {var}：fill 后不得有 NaN/Inf"
        # 内区（中心行）不被 fill 改动
        np.testing.assert_array_equal(
            got[k0, j0, _NG : _NG + _NX1],
            pat[k0, j0, _NG : _NG + _NX1],
            err_msg=f"槽 {var}：fill 不得触碰内区",
        )
        # user 面（x1beg）本刀按 copy 垫：中心行 ghost 有定义 == pack 最外层内区
        for m in range(_NG):
            assert got[k0, j0, _NG - 1 - m] == pat[k0, j0, _NG], (
                f"槽 {var}：user 面 ghost 应按 copy 垫（== 最外层内区）"
            )
        # 横平面整面写：每一 (k,j) 行 user 面 ghost 与该行最外层内区相等
        for k in range(_NZTOT):
            for j in range(_NYTOT):
                for m in range(_NG):
                    assert got[k, j, _NG - 1 - m] == got[k, j, _NG], (
                        f"槽 {var} user 面 (k={k},j={j})：ghost 应按 copy 垫"
                    )
        # 对照面 x1end（reflect）：中心行 vx 镜像翻号，与 user 面行为不同
        if var == SLOT_VX:
            for m in range(_NG):
                assert (
                    got[k0, j0, _NG + _NX1 + m]
                    == -pat[k0, j0, _NG + _NX1 - 1 - m]
                ), "同一 run 的 reflect 面仍应镜像翻号（与 user 面区分）"


# --- 6. 薄 2D：只垫 j 面不影响 i 内部（第二行原文第 2 条） --------------------


def test_2d_j_faces_filled_i_interior_untouched(lib, run_2d):
    """x2 两端 reflect、x1 两端 copy：x2 ghost 按 j 向镜像，内区逐位不动。

    证明「垫的确实是 j 面、且按面类型分开」的证据链：
    (a) 内区（x1 与 x2 方向）五槽逐位不变；
    (b) x2beg/x2end ghost 是 x2 方向镜像：密度/切向 vx,vz/P 偶，法向 vy 奇；
    (c) 同一 run 的 x1 面是 copy（复刻最外层、不翻号），不是三向一律 reflect；
    (d) x2 ghost 不是 i 向镜像（显式反证）。
    """
    h = run_2d._h
    ncells = lib.xmf_state_ncells(h)
    assert ncells == NCELLS_2D, f"薄 2D 块长应为 {NCELLS_2D}，实际 {ncells}"

    # 六面类型经 C 查询：x1 两端 copy、x2 两端 reflect、x3 两端 none
    assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["copy"]
    assert lib.xmf_bc_type(h, FACE_X1END) == BC_ENUM["copy"]
    assert lib.xmf_bc_type(h, FACE_X2BEG) == BC_ENUM["reflect"]
    assert lib.xmf_bc_type(h, FACE_X2END) == BC_ENUM["reflect"]
    assert lib.xmf_bc_type(h, FACE_X3BEG) == BC_ENUM["none"]
    assert lib.xmf_bc_type(h, FACE_X3END) == BC_ENUM["none"]

    pats = _pack_all(lib, h, ncells)
    assert np.all(pats[SLOT_VY] != 0.0), "j 向法向速度必须非零（禁 v=0 碰巧）"
    assert lib.xmf_fill_bc(h) == 0
    out = _unpack_all(lib, h, ncells)

    shape = (_NZTOT2, _NYTOT2, _NXTOT2)
    k0 = _NG  # z 方向只有一层内区
    jin = slice(_NG, _NG + NX2_2D)   # j 内区
    iin = slice(_NG, _NG + NX1_2D)   # i 内区

    # (a) 内区逐位不动：x1 内部与 x2 内部都不被 fill 改动
    for var in range(5):
        got = out[var].reshape(shape)
        pat = pats[var].reshape(shape)
        np.testing.assert_array_equal(
            got[k0, jin, iin], pat[k0, jin, iin],
            err_msg=f"槽 {var}：fill 只写 ghost，内区（i 与 j 方向）不得改动",
        )

    # (b) x2beg/x2end ghost 沿 j 镜像：法向槽 vy 奇，其余槽偶。
    #     整行（含 i ghost 列）按 j 向同奇偶——横平面整面写；
    #     内区 i 段再与 pack 原值比对，锚定到实际写入的数据。
    for var in range(5):
        got = out[var].reshape(shape)
        pat = pats[var].reshape(shape)
        flip = var == SLOT_VY  # x2 面的法向槽是 vy（j 向）
        for m in range(_NG):
            jb, js = _NG - 1 - m, _NG + m              # beg：ghost ← 内区镜像
            je, jse = _NG + NX2_2D + m, _NG + NX2_2D - 1 - m  # end
            sign = -1.0 if flip else 1.0
            np.testing.assert_array_equal(
                got[k0, jb, :], sign * got[k0, js, :],
                err_msg=f"槽 {var} x2beg m={m}：整行应为 j 向镜像"
                        f"{'翻号（奇）' if flip else '同号（偶）'}",
            )
            np.testing.assert_array_equal(
                got[k0, je, :], sign * got[k0, jse, :],
                err_msg=f"槽 {var} x2end m={m}：整行应为 j 向镜像"
                        f"{'翻号（奇）' if flip else '同号（偶）'}",
            )
            np.testing.assert_array_equal(
                got[k0, jb, iin], sign * pat[k0, js, iin],
                err_msg=f"槽 {var} x2beg m={m}：内区 i 段应等于 pack 的 j 镜像",
            )
            np.testing.assert_array_equal(
                got[k0, je, iin], sign * pat[k0, jse, iin],
                err_msg=f"槽 {var} x2end m={m}：内区 i 段应等于 pack 的 j 镜像",
            )

    # (c) x1 两端是 copy：ghost i 层逐位 == 最外层内区 i 层（全 j 行，不翻号）。
    #     同一 run 内 x1 面与 x2 面行为不同 ⇒ 不是三向一律 reflect。
    for var in range(5):
        got = out[var].reshape(shape)
        pat = pats[var].reshape(shape)
        for m in range(_NG):
            ib = _NG - 1 - m
            ie = _NG + NX1_2D + m
            np.testing.assert_array_equal(
                got[k0, :, ib], got[k0, :, _NG],
                err_msg=f"槽 {var} x1beg m={m}：copy 应复刻最外层内区 i 层",
            )
            np.testing.assert_array_equal(
                got[k0, :, ie], got[k0, :, _NG + NX1_2D - 1],
                err_msg=f"槽 {var} x1end m={m}：copy 应复刻最外层内区 i 层",
            )
            np.testing.assert_array_equal(
                got[k0, jin, ib], pat[k0, jin, _NG],
                err_msg=f"槽 {var} x1beg：内区 j 段的 ghost 应等于 pack 原值",
            )

    # (d) 显式反证：x2beg ghost 不是 i 向镜像（模式沿 i 单调，i 反序必不同）。
    n_got = out[SLOT_N].reshape(shape)
    n_pat = pats[SLOT_N].reshape(shape)
    for m in range(_NG):
        jb, js = _NG - 1 - m, _NG + m
        assert np.any(n_got[k0, jb, iin] != n_pat[k0, js, iin][::-1]), (
            f"x2beg m={m}：ghost 不得是 i 向镜像（垫的必须是 j 面）"
        )

    # 符号证据：x2 ghost（j ghost 行、内区 i 段）vy 全负（奇），
    # x1 ghost（i ghost 列）vx 全正（copy 不翻号）——两面对照，均非零。
    vy = out[SLOT_VY].reshape(shape)
    assert np.all(vy[k0, :_NG, iin] < 0.0), "x2beg ghost vy 应翻号为负"
    assert np.all(vy[k0, _NG + NX2_2D :, iin] < 0.0), "x2end ghost vy 应翻号为负"
    vx = out[SLOT_VX].reshape(shape)
    assert np.all(vx[k0, :, :_NG] > 0.0), "x1beg copy ghost vx 应保号为正"
    assert np.all(vx[k0, :, _NG + NX1_2D :] > 0.0), "x1end copy ghost vx 应保号为正"


# --- 7. 哨兵（卡片断言清单第 6 条） ------------------------------------------


def test_bc_sentinels(lib, run_1d):
    """face 越界（-1/6）与 NULL 句柄：xmf_bc_type 返回 -1；fill(NULL) 返回 -1。

    坏参不得损坏 run：之后六面仍可查、fill 仍可成功。
    """
    h = run_1d._h
    assert lib.xmf_bc_type(h, -1) == -1, "face=-1 应返回 -1"
    assert lib.xmf_bc_type(h, 6) == -1, "face=6 应返回 -1"
    assert lib.xmf_bc_type(None, 0) == -1, "NULL 句柄 bc_type 应返回 -1"
    assert lib.xmf_bc_type(None, 3) == -1, "NULL 句柄 bc_type 应返回 -1"
    assert lib.xmf_fill_bc(None) == -1, "NULL 句柄 fill_bc 应返回 -1"

    # 坏参调用后 run 状态不受影响
    assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["reflect"], (
        "坏参调用不应损坏 run 的六面类型"
    )
    assert lib.xmf_fill_bc(h) == 0, "坏参调用后 fill_bc 仍应成功"
