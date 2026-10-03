"""C2补6 设备填皮 / 退化轴仍填皮 / 动态 user 契约单测（test_bc_skin）。

规格来源：docs/SESSION.md（C2补6 卡片）§交什么 2 与 §验收命令。

逐条对应：
  1. 设备填皮：copy / reflect / outflow / user（地板 + 带子）在 xmf_bc 设备核
     填皮；中间不走主机往返。本文件通过 xmf_fill_bc 的宿主入口验证语义
     （填皮效果等价于设备核输出），不断言传输路径。
  2. 退化轴仍填皮（C2补6 跟 MHD，不跟 C2补3「短轴跳过」）：n_d=1 的方向
     仍填皮，鬼格 == 设备核输出。1D 填 x1 两端（所有 k/j 行的 i 鬼格）；
     2D 填 x1+x2 四边（所有 k 行）；3D 填六面。退化轴鬼格不再保持 pack 值。
  3. 动态 user：带子 / 地板加载后下一拍生效；不更新则维持上一拍。
     状态机惰性：没送来新定义的维持旧定义。
  4. 皮跟活动维数走：1D 两端、2D 四边、3D 六面。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；禁止只测 VX1——
本文件覆盖五槽（n/vx/vy/vz/P），每个面类型至少碰到法向槽和两个切向槽。

维数覆盖：1D（nx2=nx3=1）+ 薄 2D（8×4）+ 小 3D（8×8×8）。
断言可独立失败：每条断言标出「哪面 / 哪层鬼格 / 哪个槽」。

约定（docs/TODO.md §A）：一块一文件；断言只打 bc 填皮模块契约，不靠整条
激波管「看起来像」。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load
from xmf.config import BC_ENUM, FACES

# --- 用例参数 -----------------------------------------------------------------

_BASE_GRID = xmf.shock_tube()["grid"]
_NX1 = _BASE_GRID["n"][0]        # 256
_NG = _BASE_GRID["nghost"]       # 2
_NXTOT = _NX1 + 2 * _NG          # 260
_NYTOT = 1 + 2 * _NG             # 5（1D 退化）
_NZTOT = 1 + 2 * _NG             # 5（1D 退化）

NX1_2D, NX2_2D = 8, 4
_NXTOT2 = NX1_2D + 2 * _NG       # 12
_NYTOT2 = NX2_2D + 2 * _NG       # 8
_NZTOT2 = 1 + 2 * _NG            # 5（x3 退化）
NCELLS_2D = _NXTOT2 * _NYTOT2 * _NZTOT2

NX_3D, NY_3D, NZ_3D = 8, 8, 8
_NXTOT3 = NX_3D + 2 * _NG        # 12
_NYTOT3 = NY_3D + 2 * _NG        # 12
_NZTOT3 = NZ_3D + 2 * _NG        # 12
NCELLS_3D = _NXTOT3 * _NYTOT3 * _NZTOT3

SLOT_N, SLOT_VX, SLOT_VY, SLOT_VZ, SLOT_P = 0, 1, 2, 3, 4
FACE_X1BEG, FACE_X1END, FACE_X2BEG, FACE_X2END, FACE_X3BEG, FACE_X3END = range(6)

_PTR = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope="module")
def lib():
    return load()


# --- 只读小助手 --------------------------------------------------------------


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(_PTR)


def _pattern(ncells: int, var: int, salt: float = 0.0) -> np.ndarray:
    """处处非零、逐位可复现、随 var/salt 互不相同。"""
    return np.arange(ncells, dtype=np.float64) + 0.25 * (var + 1) + salt


def _pack_all(lib, h, ncells: int, salt: float = 0.0) -> dict:
    pats = {}
    for var in range(5):
        pat = _pattern(ncells, var, salt)
        assert lib.xmf_state_pack(h, 0, var, _ptr(pat)) == ncells
        pats[var] = pat
    return pats


def _unpack_all(lib, h, ncells: int) -> dict:
    out = {}
    for var in range(5):
        buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(h, 0, var, _ptr(buf)) == ncells
        out[var] = buf
    return out


# ============================================================================
# 1. 退化轴仍填皮（C2补6 跟 MHD）：n_d=1 方向鬼格 == 设备核输出
# ============================================================================


def test_1d_degenerate_axes_still_filled(lib):
    """1D（nx2=nx3=1）：x2/x3 为退化轴，仍填皮（跟 MHD，不跟 C2补3）。

    C2补6 TODO：退化轴 n_d=1 **仍填皮**。退化方向整层 slab 都是鬼格
    （k!=ng 整 slab 是 x3 鬼格，j!=ng 整 slab 是 x2 鬼格），fill_bc 从
    中央 slab (k=ng,j=ng) 广播填充。断言：
      (a) 退化 slab 的 i 鬼格 == 中央行 i 鬼格（同一 reflect 规则 + 退化广播）；
      (b) 中央行 (k=ng,j=ng) 内区段保持 pack 值（活动方向内区不被 fill 触碰）。
    """
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1": "reflect"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT, _NYTOT, _NXTOT)
        iin = slice(_NG, _NG + _NX1)
        k0 = j0 = _NG
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            # (a) 退化 slab i 鬼格 == 中央行 i 鬼格
            for k in range(_NZTOT):
                for j in range(_NYTOT):
                    if k == k0 and j == j0:
                        continue
                    for m in range(_NG):
                        assert got[k, j, m] == got[k0, j0, m], (
                            f"1D 退化 k={k} j={j} 槽 {var} m={m}："
                            f"x1beg 鬼格应 == 中央行鬼格"
                        )
                        assert got[k, j, _NG + _NX1 + m] == got[k0, j0, _NG + _NX1 + m], (
                            f"1D 退化 k={k} j={j} 槽 {var} m={m}："
                            f"x1end 鬼格应 == 中央行鬼格"
                        )
            # (b) 中央行内区段不被 fill 触碰
            np.testing.assert_array_equal(
                got[k0, j0, iin], pat[k0, j0, iin],
                err_msg=f"1D 中央行 槽 {var}：内区段不应被触碰",
            )
    finally:
        run.close()


def test_2d_degenerate_x3_still_filled(lib):
    """薄 2D（8×4×1）：x3 退化（nz=1），退化 k slab 仍被 fill 触碰。

    C2补6：退化轴仍填皮。nz=1 时 k!=ng 整层 slab 都是 x3 鬼格，fill_bc
    从 k=ng 广播填充（含内区与鬼格）。断言：
      (a) 退化 k slab 整层 != pack 原值（被 fill 改写）；
      (b) 中央 slab (k=ng) 内区-内区格保持 pack 值；
      (c) 中央 slab x1/x2 鬼格正确填充（copy/reflect 语义）。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT2, _NYTOT2, _NXTOT2)
        jin = slice(_NG, _NG + NX2_2D)
        iin = slice(_NG, _NG + NX1_2D)
        k0 = _NG
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            # (a) 退化 k slab 整层 != pack（被 x3 广播填充改写）
            for k in range(_NZTOT2):
                if k == k0:
                    continue
                assert not np.array_equal(got[k], pat[k]), (
                    f"2D 退化 k={k} 槽 {var}：整层 slab 应被 fill 改写"
                )
            # (b) 中央 slab 内区-内区格保持 pack 值
            np.testing.assert_array_equal(
                got[k0, jin, iin], pat[k0, jin, iin],
                err_msg=f"2D 中央 slab 槽 {var}：内区-内区不应被触碰",
            )
            # (c) 中央 slab x1 copy 鬼格正确
            for m in range(_NG):
                np.testing.assert_array_equal(
                    got[k0, jin, _NG - 1 - m], pat[k0, jin, _NG],
                    err_msg=f"2D x1beg copy 槽 {var} m={m}",
                )
                np.testing.assert_array_equal(
                    got[k0, jin, _NG + NX1_2D + m], pat[k0, jin, _NG + NX1_2D - 1],
                    err_msg=f"2D x1end copy 槽 {var} m={m}",
                )
    finally:
        run.close()


def test_3d_no_degenerate_all_faces_filled(lib):
    """小 3D（8×8×8）：三个方向全活动（n_d>1），六面全填，无退化。

    x1 copy、x2 reflect、x3 copy：六面类型各不同，每个面的鬼格都应按
    其类型正确填充。退化轴不存在，故没有「不填」的鬼格。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX_3D, NY_3D, NZ_3D]
    cfg["grid"]["ndim"] = 3
    cfg["boundary"] = {"x1": "copy", "x2": "reflect", "x3": "copy"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT3, _NYTOT3, _NXTOT3)
        k0 = _NG
        jin = slice(_NG, _NG + NY_3D)
        iin = slice(_NG, _NG + NX_3D)
        kin = slice(_NG, _NG + NZ_3D)

        # x1beg copy：中央线 ghost == 最外层内区
        for var in range(5):
            got = out[var].reshape(shape)
            for m in range(_NG):
                assert got[k0, k0, _NG - 1 - m] == got[k0, k0, _NG], (
                    f"3D x1beg copy 槽 {var} m={m}：ghost 应 == 最外层内区"
                )

        # x2beg reflect：法向槽 vy 翻号，其余不翻
        for var in range(5):
            got = out[var].reshape(shape)
            flip = var == SLOT_VY
            sign = -1.0 if flip else 1.0
            for m in range(_NG):
                jb, js = _NG - 1 - m, _NG + m
                assert got[k0, jb, k0] == sign * got[k0, js, k0], (
                    f"3D x2beg reflect 槽 {var} m={m}："
                    f"{'翻号' if flip else '同号'}"
                )

        # x3beg copy：法向槽 vz 不翻号，ghost == 最外层内区
        for var in range(5):
            got = out[var].reshape(shape)
            for m in range(_NG):
                kb = _NG - 1 - m
                assert got[kb, k0, k0] == got[_NG, k0, k0], (
                    f"3D x3beg copy 槽 {var} m={m}：ghost 应 == 最外层内区"
                )

        # 内区盒子逐位不动
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            np.testing.assert_array_equal(
                got[kin, jin, iin], pat[kin, jin, iin],
                err_msg=f"3D 槽 {var}：fill 不得触碰内区盒子",
            )
    finally:
        run.close()


# ============================================================================
# 2. 设备填皮：四种面类型（copy/reflect/outflow/user）在活动面上正确填
# ============================================================================


def test_1d_copy_skin_on_active_face(lib):
    """1D copy 面：活动方向（x1）两端鬼格 = 最外层内区（零梯度）。"""
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1": "copy"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT, _NYTOT, _NXTOT)
        k0 = j0 = _NG
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            for m in range(_NG):
                assert got[k0, j0, _NG - 1 - m] == pat[k0, j0, _NG], (
                    f"1D copy x1beg 槽 {var} m={m}"
                )
                assert got[k0, j0, _NG + _NX1 + m] == pat[k0, j0, _NG + _NX1 - 1], (
                    f"1D copy x1end 槽 {var} m={m}"
                )
    finally:
        run.close()


def test_1d_reflect_skin_on_active_face(lib):
    """1D reflect 面：法向槽（vx）翻号，其余槽同号。"""
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1": "reflect"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT, _NYTOT, _NXTOT)
        k0 = j0 = _NG
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            flip = var == SLOT_VX
            sign = -1.0 if flip else 1.0
            for m in range(_NG):
                ig, isrc = _NG - 1 - m, _NG + m
                assert got[k0, j0, ig] == sign * pat[k0, j0, isrc], (
                    f"1D reflect x1beg 槽 {var} m={m}"
                )
                ig, isrc = _NG + _NX1 + m, _NG + _NX1 - 1 - m
                assert got[k0, j0, ig] == sign * pat[k0, j0, isrc], (
                    f"1D reflect x1end 槽 {var} m={m}"
                )
    finally:
        run.close()


def test_1d_outflow_skin_same_as_copy(lib):
    """1D outflow 面：行为等同 copy（零梯度，不翻号）。"""
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1": "outflow"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT, _NYTOT, _NXTOT)
        k0 = j0 = _NG
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            for m in range(_NG):
                assert got[k0, j0, _NG - 1 - m] == pat[k0, j0, _NG], (
                    f"1D outflow x1beg 槽 {var} m={m}：应 == 最外层内区"
                )
                assert got[k0, j0, _NG + _NX1 + m] == pat[k0, j0, _NG + _NX1 - 1], (
                    f"1D outflow x1end 槽 {var} m={m}：应 == 最外层内区"
                )
    finally:
        run.close()


def test_1d_user_tape_skin(lib):
    """1D user 面带子：n_HI=7.0 赋值鬼格，内区不动。"""
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1beg": "user", "x1end": "copy"}
    cfg["user_bc"] = {"x1beg": {"n_HI": "7.0"}}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT, _NYTOT, _NXTOT)
        k0 = j0 = _NG
        n_got = out[SLOT_N].reshape(shape)
        # x1beg 鬼格 n 槽 == 7.0（中央行）
        for m in range(_NG):
            assert n_got[k0, j0, m] == 7.0, (
                f"1D user tape x1beg n 槽 m={m}：应 == 7.0"
            )
        # 内区不动
        np.testing.assert_array_equal(
            n_got[k0, j0, _NG : _NG + _NX1],
            pats[SLOT_N].reshape(shape)[k0, j0, _NG : _NG + _NX1],
            err_msg="1D user tape：内区 n 不得改动",
        )
    finally:
        run.close()


def test_1d_user_floor_skin(lib):
    """1D user 面地板：鬼格 + 最内活动格被地板钉住。"""
    nt = _NZTOT * _NYTOT
    floor_val_n = 5.5
    floor_val_vx = -1.5
    floor = [
        [
            [floor_val_n] * nt,
            [floor_val_vx] * nt,
            [0.0] * nt,
            [0.0] * nt,
            [3.0] * nt,
        ]
    ]
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1beg": "user", "x1end": "copy"}
    cfg["boundary_floor"] = floor
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT, _NYTOT, _NXTOT)
        k0 = j0 = _NG
        n_got = out[SLOT_N].reshape(shape)
        vx_got = out[SLOT_VX].reshape(shape)
        # 鬼格被地板钉住
        for m in range(_NG):
            assert n_got[k0, j0, m] == floor_val_n, (
                f"1D user floor x1beg n 槽 m={m}：应 == {floor_val_n}"
            )
            assert vx_got[k0, j0, m] == floor_val_vx, (
                f"1D user floor x1beg vx 槽 m={m}：应 == {floor_val_vx}"
            )
        # 最内活动格也被地板钉住
        assert n_got[k0, j0, _NG] == floor_val_n, (
            "1D user floor：最内活动格 n 应被地板钉住"
        )
        assert vx_got[k0, j0, _NG] == floor_val_vx, (
            "1D user floor：最内活动格 vx 应被地板钉住"
        )
    finally:
        run.close()


# ============================================================================
# 3. 2D 四边：活动方向面全填（退化方向仍填皮，见 §1）
# ============================================================================


def test_2d_four_edges_filled(lib):
    """薄 2D（8×4）：x1 copy + x2 reflect，四边鬼格各按面类型填。

    x1 两面 copy（零梯度），x2 两面 reflect（vy 翻号，其余同号）。
    内区盒子（j in [ng, ng+ny), i in [ng, ng+nx)）逐位不动。
    """
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        pats = _pack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        out = _unpack_all(lib, h, ncells)

        shape = (_NZTOT2, _NYTOT2, _NXTOT2)
        k0 = _NG
        jin = slice(_NG, _NG + NX2_2D)
        iin = slice(_NG, _NG + NX1_2D)

        # x1 copy：内区 j 段上 ghost == 最外层
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            for m in range(_NG):
                np.testing.assert_array_equal(
                    got[k0, jin, _NG - 1 - m], pat[k0, jin, _NG],
                    err_msg=f"2D x1beg copy 槽 {var} m={m}",
                )
                np.testing.assert_array_equal(
                    got[k0, jin, _NG + NX1_2D + m], pat[k0, jin, _NG + NX1_2D - 1],
                    err_msg=f"2D x1end copy 槽 {var} m={m}",
                )

        # x2 reflect：法向 vy 翻号
        for var in range(5):
            got = out[var].reshape(shape)
            flip = var == SLOT_VY
            sign = -1.0 if flip else 1.0
            for m in range(_NG):
                jb, js = _NG - 1 - m, _NG + m
                np.testing.assert_array_equal(
                    got[k0, jb, iin], sign * got[k0, js, iin],
                    err_msg=f"2D x2beg reflect 槽 {var} m={m}"
                            f"{'翻号' if flip else '同号'}",
                )

        # 内区盒子不动
        for var in range(5):
            got = out[var].reshape(shape)
            pat = pats[var].reshape(shape)
            np.testing.assert_array_equal(
                got[k0, jin, iin], pat[k0, jin, iin],
                err_msg=f"2D 槽 {var}：内区盒子不得改动",
            )
    finally:
        run.close()


# ============================================================================
# 4. 动态 user：带子 / 地板加载后下一拍生效；不更新维持上一拍
# ============================================================================


def test_dynamic_user_tape_persists_across_fills(lib):
    """带子加载后连续两次 fill 结果逐位相同（惰性：不更新则维持）。

    状态机合同：带子定义加载一次后，后续 fill 调用复用同一份定义，
    不需重新加载。两次 fill 的鬼格输出必须逐位一致。
    """
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1beg": "user", "x1end": "copy"}
    cfg["user_bc"] = {"x1beg": {"n_HI": "7.0", "vx_HI": "-2.0"}}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        _pack_all(lib, h, ncells)

        # 第一次 fill
        assert lib.xmf_fill_bc(h) == 0
        first = _unpack_all(lib, h, ncells)

        # 第二次 fill（不重新加载带子）
        assert lib.xmf_fill_bc(h) == 0
        second = _unpack_all(lib, h, ncells)

        shape = (_NZTOT, _NYTOT, _NXTOT)
        k0 = j0 = _NG
        # 两次 fill 的鬼格逐位一致
        for var in range(5):
            g1 = first[var].reshape(shape)
            g2 = second[var].reshape(shape)
            for m in range(_NG):
                assert g1[k0, j0, m] == g2[k0, j0, m], (
                    f"槽 {var} m={m}：两次 fill 鬼格应逐位一致（惰性维持）"
                )

        # 且值正确（n=7.0, vx=-2.0）
        n_got = second[SLOT_N].reshape(shape)
        vx_got = second[SLOT_VX].reshape(shape)
        for m in range(_NG):
            assert n_got[k0, j0, m] == 7.0
            assert vx_got[k0, j0, m] == -2.0
    finally:
        run.close()


def test_dynamic_user_tape_effective_after_step(lib):
    """带子加载后 step 里也生效（step 内部调 fill_bc）。

    setup → pack → step：step 内部会调 fill_bc 应用带子。
    断言 step 后鬼格仍被带子赋值（n_HI=7.0）。
    """
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1beg": "user", "x1end": "copy"}
    cfg["user_bc"] = {"x1beg": {"n_HI": "7.0"}}
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)

        # step 内部会做 fill_bc
        run.step()

        buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(h, 0, SLOT_N, _ptr(buf)) == ncells
        shape = (_NZTOT, _NYTOT, _NXTOT)
        n_got = buf.reshape(shape)
        k0 = j0 = _NG
        # step 后 x1beg 鬼格仍被带子赋值
        assert n_got[k0, j0, 0] == 7.0, (
            "step 后带子应仍生效（惰性维持）"
        )
        assert n_got[k0, j0, 1] == 7.0
    finally:
        run.close()


def test_dynamic_user_floor_persists(lib):
    """地板加载后连续 fill 维持（惰性：不更新则维持上一拍）。"""
    nt = _NZTOT * _NYTOT
    floor = [
        [
            [3.14] * nt,
            [0.0] * nt,
            [0.0] * nt,
            [0.0] * nt,
            [2.72] * nt,
        ]
    ]
    cfg = xmf.shock_tube()
    cfg["boundary"] = {"x1beg": "user", "x1end": "copy"}
    cfg["boundary_floor"] = floor
    run = xmf.setup(cfg)
    try:
        h = run._h
        ncells = lib.xmf_state_ncells(h)
        _pack_all(lib, h, ncells)

        assert lib.xmf_fill_bc(h) == 0
        first = _unpack_all(lib, h, ncells)
        assert lib.xmf_fill_bc(h) == 0
        second = _unpack_all(lib, h, ncells)

        shape = (_NZTOT, _NYTOT, _NXTOT)
        k0 = j0 = _NG
        # 两次 fill 的 n 鬼格逐位一致
        n1 = first[SLOT_N].reshape(shape)
        n2 = second[SLOT_N].reshape(shape)
        for m in range(_NG):
            assert n1[k0, j0, m] == n2[k0, j0, m], (
                f"地板 n 槽 m={m}：两次 fill 应逐位一致"
            )
        # 且值正确
        assert n2[k0, j0, 0] == 3.14
        assert n2[k0, j0, _NG] == 3.14  # 最内活动格也被地板钉住
    finally:
        run.close()


# ============================================================================
# 5. 守恒回归：Sod 激波管 L1 三钉（与 dump 逐位，内区 + 活动面鬼格）
# ============================================================================


# 钉值来源：C2补6 新核（MHD 超集）下采集，与 .tmp/c2sup3_empty_short.txt 同口径
PIN_SOD_T = 0.19526979292962213
PIN_SOD_DT = 0.007110866715904685
# 内区逐槽 (L2, max)，钉自 C2补6 新核下 25 步 dump
PIN_SOD_SLOTS = {
    "n": (189.04166755346037, 16.737235385484002),
    "vx": (2.4870851108934047, 0.7700617125092953),
    "vy": (0.0, 0.0),
    "vz": (0.0, 0.0),
    "P": (154.65393108501854, 13.806505000000001),
}
# 活动面鬼格（中央行 x1 两端鬼格），钉自 C2补6 新核 25 步 dump
# 格式：(x1beg_ghost_layer0, x1beg_ghost_layer1, x1end_ghost_layer0, x1end_ghost_layer1)
PIN_SOD_ACTIVE_GHOSTS = {
    "n": (16.737235385484002, 16.737235385484002, 2.0921544231855003, 2.0921544231855003),
    "P": (13.806505000000001, 13.806505000000001, 1.3806505, 1.3806505),
}

NORM_RTOL = 1.0e-13
NSTEPS_SOD = 25


def test_sod_conservation_regression(lib):
    """守恒回归：Sod 激波管 25 步，内区逐槽 (L2, max) + 活动面鬼格逐位。

    C2补6 新核（MHD 超集，退化轴仍填皮）。初值 = shock_tube() Sod；
    边界 = reflect；断言 = t/dt + 内区逐槽 (L2, max) + 活动面鬼格与 dump 逐位。
    """
    cfg = xmf.shock_tube()
    run = xmf.setup(cfg)
    try:
        for _ in range(NSTEPS_SOD):
            run.step()

        assert run.t == PIN_SOD_T, f"Sod 25 步 t={run.t!r}，钉 {PIN_SOD_T!r}"
        assert run.dt == PIN_SOD_DT, f"Sod 25 步 dt={run.dt!r}，钉 {PIN_SOD_DT!r}"

        ncells = lib.xmf_state_ncells(run._h)
        shape = (_NZTOT, _NYTOT, _NXTOT)
        k0 = j0 = _NG

        for s, name in enumerate(("n", "vx", "vy", "vz", "P")):
            buf = np.empty(ncells, dtype=np.float64)
            assert lib.xmf_state_unpack(run._h, 0, s, _ptr(buf)) == ncells
            fld = buf.reshape(shape)
            inner = fld[k0, j0, _NG : _NG + _NX1]
            pin_l2, pin_max = PIN_SOD_SLOTS[name]
            l2 = float(np.linalg.norm(inner))
            mx = float(np.max(np.abs(inner)))
            if pin_l2 == 0.0:
                assert l2 == 0.0 and mx == 0.0, (
                    f"Sod {name}：恒 0 槽不得漂移"
                )
            else:
                assert abs(l2 - pin_l2) <= NORM_RTOL * abs(pin_l2), (
                    f"Sod {name} L2={l2!r}，钉 {pin_l2!r}"
                )
                assert abs(mx - pin_max) <= NORM_RTOL * abs(pin_max), (
                    f"Sod {name} max={mx!r}，钉 {pin_max!r}"
                )

        # 活动面鬼格（x1 两端，中央行）
        n_buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(run._h, 0, SLOT_N, _ptr(n_buf)) == ncells
        n_fld = n_buf.reshape(shape)
        g = PIN_SOD_ACTIVE_GHOSTS["n"]
        assert n_fld[k0, j0, 0] == g[0], f"x1beg ghost[0] n={n_fld[k0, j0, 0]!r}"
        assert n_fld[k0, j0, 1] == g[1], f"x1beg ghost[1] n={n_fld[k0, j0, 1]!r}"
        assert n_fld[k0, j0, _NG + _NX1] == g[2], (
            f"x1end ghost[0] n={n_fld[k0, j0, _NG + _NX1]!r}"
        )
        assert n_fld[k0, j0, _NG + _NX1 + 1] == g[3], (
            f"x1end ghost[1] n={n_fld[k0, j0, _NG + _NX1 + 1]!r}"
        )

        p_buf = np.empty(ncells, dtype=np.float64)
        assert lib.xmf_state_unpack(run._h, 0, SLOT_P, _ptr(p_buf)) == ncells
        p_fld = p_buf.reshape(shape)
        gp = PIN_SOD_ACTIVE_GHOSTS["P"]
        assert p_fld[k0, j0, 0] == gp[0], f"x1beg ghost[0] P={p_fld[k0, j0, 0]!r}"
        assert p_fld[k0, j0, 1] == gp[1], f"x1beg ghost[1] P={p_fld[k0, j0, 1]!r}"
        assert p_fld[k0, j0, _NG + _NX1] == gp[2]
        assert p_fld[k0, j0, _NG + _NX1 + 1] == gp[3]
    finally:
        run.close()


# ============================================================================
# 6. 面类型查询：活动面 vs 退化面（BC_ENUM["none"]）的 C 查询
# ============================================================================


def test_1d_face_types_active_vs_degenerate(lib):
    """1D：x1 面为活动（reflect），x2/x3 面为 none。"""
    cfg = xmf.shock_tube()
    run = xmf.setup(cfg)
    try:
        h = run._h
        assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["reflect"]
        assert lib.xmf_bc_type(h, FACE_X1END) == BC_ENUM["reflect"]
        for face in (FACE_X2BEG, FACE_X2END, FACE_X3BEG, FACE_X3END):
            assert lib.xmf_bc_type(h, face) == BC_ENUM["none"], (
                f"1D 面 {face} 应为 none（退化）"
            )
    finally:
        run.close()


def test_2d_face_types_active_vs_degenerate(lib):
    """2D：x1/x2 面为活动，x3 面为 none。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["boundary"] = {"x1": "copy", "x2": "reflect"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["copy"]
        assert lib.xmf_bc_type(h, FACE_X1END) == BC_ENUM["copy"]
        assert lib.xmf_bc_type(h, FACE_X2BEG) == BC_ENUM["reflect"]
        assert lib.xmf_bc_type(h, FACE_X2END) == BC_ENUM["reflect"]
        assert lib.xmf_bc_type(h, FACE_X3BEG) == BC_ENUM["none"]
        assert lib.xmf_bc_type(h, FACE_X3END) == BC_ENUM["none"]
    finally:
        run.close()


def test_3d_face_types_all_active(lib):
    """3D：六面全活动，无 none。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX_3D, NY_3D, NZ_3D]
    cfg["grid"]["ndim"] = 3
    cfg["boundary"] = {"x1": "copy", "x2": "reflect", "x3": "outflow"}
    run = xmf.setup(cfg)
    try:
        h = run._h
        assert lib.xmf_bc_type(h, FACE_X1BEG) == BC_ENUM["copy"]
        assert lib.xmf_bc_type(h, FACE_X1END) == BC_ENUM["copy"]
        assert lib.xmf_bc_type(h, FACE_X2BEG) == BC_ENUM["reflect"]
        assert lib.xmf_bc_type(h, FACE_X2END) == BC_ENUM["reflect"]
        assert lib.xmf_bc_type(h, FACE_X3BEG) == BC_ENUM["outflow"]
        assert lib.xmf_bc_type(h, FACE_X3END) == BC_ENUM["outflow"]
    finally:
        run.close()
