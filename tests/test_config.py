"""A0 frontend: named fields, no PLUTO parameter-row indices."""
from __future__ import annotations

import math

import pytest

import xmf
from xmf.config import normalize, shock_tube, full_schema_example
from xmf._native import XmfConfigC, problem_to_c


def test_shock_tube_gamma_is_a_number():
    p = normalize(shock_tube())
    assert p.gamma == pytest.approx(5.0 / 3.0)
    assert p.sources_enabled is False
    assert p.n == (256, 1, 1)
    assert p.electron.mode == "off"


def test_reject_pluto_GAMMA_label():
    cfg = shock_tube()
    cfg["GAMMA"] = 7
    with pytest.raises(ValueError, match="gamma"):
        normalize(cfg)


def test_reject_gamma_equals_table_slot():
    cfg = shock_tube()
    cfg["gamma"] = 1.0
    with pytest.raises(ValueError, match="adiabatic"):
        normalize(cfg)


def test_n0_must_be_positive():
    cfg = shock_tube()
    cfg["grid"]["n"] = [0, 1, 1]
    with pytest.raises(ValueError, match="n\\[0\\]"):
        normalize(cfg)


def test_full_schema_keeps_channels_on_python_side():
    p = normalize(full_schema_example())
    assert p.sources_enabled is True
    assert len(p.fluids) == 4
    assert p.electron.mode == "pressure-only"
    assert p.radiation["absorbers"] == ["HI", "HEI"]
    assert "expr" in p.sources[0]
    assert "G" in p.constants
    c = problem_to_c(p)
    assert c.nfluid == 4
    assert c.sources_enabled == 1
    assert c.electron_mode == 1
    assert c.schema_version == 1


def test_c_struct_size_is_stable():
    assert XmfConfigC._fields_[-1][0] == "p_r"
    assert XmfConfigC.schema_version.size == 4
    # A3：六面边界枚举 bc 进入 int 段（原 reserved0 槽、B2 改名
    # shock_heat 之后），int 段口径为 22×int32（88 字节）再接 double；
    # schema_version 保持 1。B2 卡片锁死：槽位改名可以，字节位置不动。
    names = [name for name, _ in XmfConfigC._fields_]
    assert "bc" in names, "六面边界枚举 bc 应在 XmfConfig 的 int 段"
    assert names.index("bc") == names.index("shock_heat") + 1
    assert "reserved0" not in names, "reserved0 已被 B2 改名为 shock_heat"
    assert XmfConfigC.bc.size == 6 * 4
    assert XmfConfigC.x1min.offset == 22 * 4, "int 段应为 22×int32 再接 double"


def test_numerics_and_grid_are_named():
    p = normalize(shock_tube())
    assert p.numerics.riemann == "hll"
    assert p.numerics.reconstruction == "pcm"
    assert p.numerics.rk == "rk1"
    assert p.stretch == "none"
    assert p.boundary["x1beg"] == "reflect"
    assert p.ic_kind == "riemann"
    c = problem_to_c(p)
    assert c.riemann == 0
    assert c.recon == 0
    assert c.rk == 0
    assert c.stretch == 0


def test_unknown_riemann_rejected():
    # A4 起合法 Riemann 为 hll/hllc/tvdlf（enum XmfRiemann）；roe 已移除，
    # 作为非法名拒掉。
    cfg = shock_tube()
    cfg["numerics"]["riemann"] = "roe"
    with pytest.raises(ValueError, match="riemann"):
        normalize(cfg)


def test_full_schema_numerics_pass_through():
    p = normalize(full_schema_example())
    assert p.geometry == "spherical"
    assert p.stretch == "x1"
    assert p.numerics.rk == "rk3"
    assert p.boundary["x1beg"] == "user"
    c = problem_to_c(p)
    assert c.geometry == 1
    assert c.stretch == 1
    assert c.rk == 2


def test_public_pool_free_of_example_factory():
    """A5 补充：考试工厂已搬进 examples/rad_hd_1d/rad_config.py。

    公共 config 池不得残留考试定制符号（工厂、读表器、行星常数，
    卡片 §现状「考试定制（搬走）」全名单）；公共池该有的（自然常数
    KB/AMU、前端三件）仍在；`import xmf` 顶层与 __all__ 都不再导出
    hh_rad_1d（不留 deprecated 别名）。"""
    import xmf.config as cfg

    gone = (
        "hh_rad_1d", "_read_sed", "_read_sigma",
        "AMU_H", "R_JUP", "M_JUP", "M_SUN", "AU", "H_PLANET", "C_LIGHT",
    )
    for name in gone:
        assert not hasattr(cfg, name), f"xmf.config 不应残留考试符号 {name}"
    for name in ("KB", "AMU", "normalize", "shock_tube", "full_schema_example"):
        assert hasattr(cfg, name), f"公共池丢了 {name}"
    assert not hasattr(xmf, "hh_rad_1d"), "import xmf 顶层不得再有工厂"
    assert "hh_rad_1d" not in xmf.__all__


# --- A1 补：每向活动界面数组（grid.x1/x2/x3，别名 x/y/z；stretch 工厂） --------
#
# 卡片（docs/SESSION.md）「交什么」§1 与 §4：界面长度必须 n_d+1，两端必须
# 等于 xdmin/xdmax（不静默改写）；别名与工厂的互斥在配置层拒绝（ValueError）；
# 单调性不在此验——权威在 C（见 tests/test_mesh.py 的拒绝用例）。

_FACES_8 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]  # n=8 → 9 个，两端=1/10


def _grid_cfg(**grid_extra):
    cfg = shock_tube()
    cfg["grid"]["n"] = [8, 1, 1]
    cfg["grid"].update(grid_extra)
    return cfg


def test_grid_faces_wrong_length_rejected():
    """长度不是 n+1 → ValueError（n=8 要 9 个界面）。"""
    with pytest.raises(ValueError, match=r"n\+1=9"):
        normalize(_grid_cfg(x1=_FACES_8[:-1]))
    with pytest.raises(ValueError, match=r"n\+1=9"):
        normalize(_grid_cfg(x1=_FACES_8 + [11.0]))


def test_grid_faces_endpoint_mismatch_rejected():
    """两端必须与 xdmin/xdmax 一致；对不上拒绝，不静默改写。"""
    bad_left = list(_FACES_8)
    bad_left[0] = 1.5
    with pytest.raises(ValueError, match="must equal"):
        normalize(_grid_cfg(x1=bad_left))
    bad_right = list(_FACES_8)
    bad_right[-1] = 9.0
    with pytest.raises(ValueError, match="must equal"):
        normalize(_grid_cfg(x1=bad_right))


def test_grid_faces_nonfinite_entry_rejected():
    """非实数 / 非有限界面 → ValueError，信息点名与下标。"""
    bad = list(_FACES_8)
    bad[4] = math.nan
    with pytest.raises(ValueError, match=r"grid\.x1\[4\]"):
        normalize(_grid_cfg(x1=bad))
    bad[4] = "oops"
    with pytest.raises(ValueError, match=r"grid\.x1\[4\]"):
        normalize(_grid_cfg(x1=bad))


def test_grid_alias_x_and_x1_duplicate_rejected():
    """x1 与别名 x 同给一个方向 → ValueError（同一个槽）。"""
    with pytest.raises(ValueError, match="duplicates"):
        normalize(_grid_cfg(x1=list(_FACES_8), x=list(_FACES_8)))


def test_grid_aliases_x_y_z_replace_x1_x2_x3():
    """x/y/z 键能替代 x1/x2/x3 生效：normalize 后 Problem.xf1/2/3 即所给数组。"""
    cfg = shock_tube()
    cfg["grid"]["n"] = [8, 2, 2]
    cfg["grid"]["x2min"], cfg["grid"]["x2max"] = 0.0, 1.0
    cfg["grid"]["x3min"], cfg["grid"]["x3max"] = 0.0, 2.0
    cfg["grid"]["x"] = list(_FACES_8)
    cfg["grid"]["y"] = [0.0, 0.25, 1.0]
    cfg["grid"]["z"] = [0.0, 1.5, 2.0]
    p = normalize(cfg)
    assert p.xf1 == tuple(_FACES_8), "别名 x 应落进 x1 的界面槽"
    assert p.xf2 == (0.0, 0.25, 1.0), "别名 y 应落进 x2 的界面槽"
    assert p.xf3 == (0.0, 1.5, 2.0), "别名 z 应落进 x3 的界面槽"


def test_stretch_log_and_explicit_x1_mutually_exclusive():
    """stretch='log' 工厂与显式 x1 数组同给 → ValueError（二选一）。"""
    with pytest.raises(ValueError, match="mutually exclusive"):
        normalize(_grid_cfg(stretch="log", stretch_factor=1.5, x1=list(_FACES_8)))


def test_stretch_log_factory_faces_contract():
    """log 工厂端到端契约：严格递增、两端=xdmin/xdmax、相邻宽度比近似常数
    （对数等距）；旧名 'x1' 与 'log' 同义；x1 以外各向不受影响。"""
    p_log = normalize(_grid_cfg(stretch="log", stretch_factor=2.0))
    assert p_log.stretch == "log"
    xf = p_log.xf1
    assert len(xf) == 9, "工厂界面长度必须是 n+1"
    assert xf[0] == 1.0 and xf[-1] == 10.0, "工厂界面两端必须钉在 xdmin/xdmax"
    assert all(b > a for a, b in zip(xf[:-1], xf[1:])), "工厂界面必须严格递增"
    widths = [b - a for a, b in zip(xf[:-1], xf[1:])]
    for k, (w_next, w_prev) in enumerate(zip(widths[1:], widths[:-1])):
        assert w_next / w_prev == pytest.approx(2.0, rel=1e-12), (
            f"对数等距：相邻宽度比应为常数 stretch_factor=2.0，第 {k} 对比值 "
            f"{w_next / w_prev!r}"
        )
    # 旧名 'x1' 是 'log' 的别名：同一工厂、同一数组
    p_alias = normalize(_grid_cfg(stretch="x1", stretch_factor=2.0))
    assert p_alias.stretch == "x1"
    assert p_alias.xf1 == xf, "stretch='x1' 必须与 'log' 吐出同一根界面"
    # x1 以外各向不受影响：仍是各自端点的等距界面
    assert p_log.xf2 == (0.0, 1.0), "x2 未被拉伸工厂触碰"
    assert p_log.xf3 == (0.0, 1.0), "x3 未被拉伸工厂触碰"
