"""A1 mesh 契约单测（对照 docs/TODO.md → A1 第二行原文）。

第二行原文：
  单测 test_mesh：均匀笛卡尔 1D 的体心 x、ΔV、界面坐标；球坐标 1D 的面面积
  A∝r_face²（不用格心 r）；nx2=4 时 j 方向长度对；ghost 层数 = ng。

约定（docs/TODO.md §A）：一块一文件；断言只打 mesh 模块契约，不靠整条激波管
「看起来像」；维数覆盖 1D（nx2=nx3=1）与 nx2>1 的薄 2D。不读 pluto.ini、
不用 XUV_FACTOR==0 哨兵、不只测单速度分量；本文件不调 step，不碰 step 后的
物理场，只读网格几何。

查询符号（builder 在 _native.load() 绑定，ctypes 直接用；d: 0=x1, 1=x2, 2=x3）：
  xmf_mesh_ncell(h,d)  -> n_d + 2*ng（含 ghost）；非法返回 0
  xmf_mesh_nface(h,d)  -> ncell + 1；非法返回 0
  xmf_mesh_copy_x/xf/dx/A/dV(h,d,out) -> 返回写入的 double 数；非法返回 -1
A/dV 是无量纲几何因子，A 不含 4π；三维格体积 = 三向 dV 因子之积。
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xmf
from xmf._native import load, problem_to_c
from xmf.config import normalize

# --- 用例参数 ----------------------------------------------------------------
NG = 2                              # ghost 层数（shock_tube 默认 nghost=2，显式写出）
NX1, X1MIN, X1MAX = 16, 1.0, 10.0   # 1D 用例（笛卡尔/球共用）；x1min > 0
NX1_2D, NX2_2D = 8, 4               # 薄 2D：nx2=4；核 sanity 下限 n[0]>=8，故 nx1 取 8
X2MIN, X2MAX = 0.25, 1.75           # 取非特殊值，防凑数

_PTR = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope="module")
def lib():
    """已安装包内共享库（mesh 查询符号已绑定）。"""
    return load()


# --- 用例构造（在 shock_tube() 返回的 dict 上改字段） ------------------------


def _cfg_cart_1d():
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1, 1, 1]
    cfg["grid"]["nghost"] = NG
    cfg["grid"]["x1min"] = X1MIN
    cfg["grid"]["x1max"] = X1MAX
    return cfg


def _cfg_sph_1d():
    cfg = _cfg_cart_1d()
    cfg["grid"]["geometry"] = "spherical"  # 球 1D：A 必须在界面 r 上取值
    # A1 补：球面 x1 任一界面（含 ghost）r<=0 → setup 失败。X1MIN=1.0 配
    # nx=16、ng=2 时左 ghost 面外推到 1-2*dx=-0.125，落在新契约的拒绝区，
    # 故球面夹具单独抬内半径（2.0 - 2*(10-2)/16 = 1.0 > 0）。断言全部参数化
    # （只读 mesh 数组与 NG/NX1），语义不变。
    cfg["grid"]["x1min"] = 2.0
    return cfg


def _cfg_thin_2d():
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NX1_2D, NX2_2D, 1]
    cfg["grid"]["ndim"] = 2
    cfg["grid"]["nghost"] = NG
    cfg["grid"]["x2min"] = X2MIN
    cfg["grid"]["x2max"] = X2MAX
    return cfg


@pytest.fixture
def run_cart():
    run = xmf.setup(_cfg_cart_1d())
    yield run
    run.close()


@pytest.fixture
def run_sph():
    run = xmf.setup(_cfg_sph_1d())
    yield run
    run.close()


@pytest.fixture
def run_2d():
    run = xmf.setup(_cfg_thin_2d())
    yield run
    run.close()


# --- 只读助手 ----------------------------------------------------------------


def _copy(lib, name, h, d, expect):
    """调 xmf_mesh_copy_<name>：返回值必须等于写入的 double 数。"""
    fn = getattr(lib, f"xmf_mesh_copy_{name}")
    buf = np.empty(expect, dtype=np.float64)
    got = fn(h, d, buf.ctypes.data_as(_PTR))
    assert got == expect, (
        f"xmf_mesh_copy_{name}(d={d}) 应写入 {expect} 个 double，实际返回 {got}"
    )
    return buf


def _axis(lib, h, d, ncell, nface):
    """取方向 d 的全套只读网格数组：x/xf/dx 长 ncell，xf/A 长 nface。"""
    return {
        "x": _copy(lib, "x", h, d, ncell),
        "xf": _copy(lib, "xf", h, d, nface),
        "dx": _copy(lib, "dx", h, d, ncell),
        "A": _copy(lib, "A", h, d, nface),
        "dV": _copy(lib, "dV", h, d, ncell),
    }


# --- 1. 均匀笛卡尔 1D：体心 x、ΔV、界面坐标（第二行原文第 1 条） ------------


def test_cartesian1d_lengths_and_lattice(lib, run_cart):
    """长度（含 ghost）、dx 均匀、xf/x 公式、体心=相邻界面中点。"""
    h = run_cart._h
    ncell = lib.xmf_mesh_ncell(h, 0)
    nface = lib.xmf_mesh_nface(h, 0)
    assert ncell == NX1 + 2 * NG, (
        f"x1 含 ghost 格数应为 nx+2ng={NX1 + 2 * NG}，实际 {ncell}"
    )
    assert nface == ncell + 1, f"面数应为格数+1，实际 {nface}"

    m = _axis(lib, h, 0, ncell, nface)
    dx = (X1MAX - X1MIN) / NX1
    i = np.arange(ncell)
    f = np.arange(nface)
    atol = 1e-12 * X1MAX  # 远小于任何公式性错误（~dx/2），宽于末位舍入

    # dx 均匀且 = (xmax-xmin)/nx
    assert np.ptp(m["dx"]) == 0.0, "本刀是均匀网格，dx 必须逐格相等"
    assert m["dx"][0] == pytest.approx(dx, rel=1e-13), "dx 应等于 (x1max-x1min)/nx"

    # 界面坐标 xf[f] = xmin + (f-ng)*dx，长度 nface
    np.testing.assert_allclose(
        m["xf"], X1MIN + (f - NG) * dx, rtol=0.0, atol=atol,
        err_msg="界面坐标应为 xf[f] = x1min + (f-ng)*dx",
    )
    # 体心 x[i] = xmin + (i-ng+0.5)*dx（含 ghost 段）
    np.testing.assert_allclose(
        m["x"], X1MIN + (i - NG + 0.5) * dx, rtol=0.0, atol=atol,
        err_msg="体心坐标应为 x[i] = x1min + (i-ng+0.5)*dx（含 ghost 段）",
    )
    # 体心是相邻界面中点：x[i] == 0.5*(xf[i]+xf[i+1])
    np.testing.assert_allclose(
        m["x"], 0.5 * (m["xf"][:-1] + m["xf"][1:]), rtol=0.0, atol=atol,
        err_msg="x[i] 应等于 0.5*(xf[i]+xf[i+1])",
    )


def test_cartesian1d_metrics(lib, run_cart):
    """笛卡尔度量：dV == dx；A 全为 1（nface 个，含最左面 A[0]）。"""
    h = run_cart._h
    ncell = lib.xmf_mesh_ncell(h, 0)
    nface = lib.xmf_mesh_nface(h, 0)
    m = _axis(lib, h, 0, ncell, nface)
    np.testing.assert_allclose(
        m["dV"], m["dx"], rtol=0.0, atol=0.0, err_msg="笛卡尔应 dV == dx"
    )
    assert m["A"].shape[0] == nface, "A 的长度应与 xf 同为 nface"
    assert np.all(m["A"] == 1.0), (
        "笛卡尔面面积因子应全为 1（含最左面 A[0]），"
        f"实际首尾为 A[0]={m['A'][0]}, A[-1]={m['A'][-1]}"
    )


# --- 2. 球坐标 1D：A ∝ r_face²，不用格心 r（第二行原文第 2 条） --------------


def test_spherical1d_area_at_face_not_centre(lib, run_sph):
    """A[f] == xf[f]² 逐面成立；内区格右面 A 明显不等于格心 r²；dV=|xr³-xl³|/3。"""
    h = run_sph._h
    ncell = lib.xmf_mesh_ncell(h, 0)
    nface = lib.xmf_mesh_nface(h, 0)
    m = _axis(lib, h, 0, ncell, nface)
    xf, x, A = m["xf"], m["x"], m["A"]

    # 面面积在界面坐标上取值：A[f] == xf[f]^2，逐面（含最左面 f=0）
    np.testing.assert_allclose(
        A, xf * xf, rtol=1e-13, atol=0.0,
        err_msg="球坐标 x1 的面面积应为 A[f] = xf[f]^2（界面 r，不是格心 r）",
    )

    # 显式排除「用格心 r 冒充 A」的实现：任取内区格 i，A[i+1] 等于 xf[i+1]^2
    # 且不等于 x[i]^2（xf 与 x 相差 dx/2，两种取值差异显著，可用量级判不等）
    for i in (NG, NG + NX1 // 2, NG + NX1 - 1):
        assert A[i + 1] == pytest.approx(xf[i + 1] ** 2, rel=1e-12), (
            f"内区格 {i} 的右面面积应取界面坐标 xf[{i + 1}]^2"
        )
        assert A[i + 1] != pytest.approx(x[i] ** 2, rel=1e-6), (
            f"面面积不得在格心 r 上取值：A[{i + 1}] 与 x[{i}]^2 相等"
        )
        # 界面取值与格心取值之差为 (x+dx/2)^2 - x^2 = x*dx + dx^2/4，
        # 下界取 0.5*x*dx 已显著低于真值、又远高于浮点噪声
        gap = abs(A[i + 1] - x[i] ** 2)
        assert gap > 0.5 * x[i] * m["dx"][i], (
            f"界面取值与格心取值差异应显著（> 0.5*x*dx），实际差 {gap}"
        )

    # 体积因子 dV[i] = |xf[i+1]^3 - xf[i]^3| / 3
    np.testing.assert_allclose(
        m["dV"], np.abs(xf[1:] ** 3 - xf[:-1] ** 3) / 3.0, rtol=1e-12, atol=0.0,
        err_msg="球坐标 x1 应 dV[i] = |xf[i+1]^3 - xf[i]^3| / 3",
    )


# --- 3. nx2=4 薄 2D：j 方向长度对（第二行原文第 3 条） -----------------------


def test_thin2d_j_direction(lib, run_2d):
    """nx2=4：j 向 ncell/nface 长度对、x2 体心/界面公式对；x1 向不受影响。"""
    h = run_2d._h
    ncell2 = lib.xmf_mesh_ncell(h, 1)
    nface2 = lib.xmf_mesh_nface(h, 1)
    assert ncell2 == NX2_2D + 2 * NG, (
        f"nx2=4 时 j 向含 ghost 格数应为 4+2ng={NX2_2D + 2 * NG}，实际 {ncell2}"
    )
    assert nface2 == ncell2 + 1, f"j 向面数应为格数+1，实际 {nface2}"

    m2 = _axis(lib, h, 1, ncell2, nface2)
    dx2 = (X2MAX - X2MIN) / NX2_2D
    j = np.arange(ncell2)
    g = np.arange(nface2)
    atol2 = 1e-12 * max(1.0, abs(X2MAX))
    np.testing.assert_allclose(
        m2["xf"], X2MIN + (g - NG) * dx2, rtol=0.0, atol=atol2,
        err_msg="x2 界面坐标应为 xf[f] = x2min + (f-ng)*dx2",
    )
    np.testing.assert_allclose(
        m2["x"], X2MIN + (j - NG + 0.5) * dx2, rtol=0.0, atol=atol2,
        err_msg="x2 体心坐标应为 x[j] = x2min + (j-ng+0.5)*dx2",
    )
    assert np.ptp(m2["dx"]) == 0.0, "x2 方向 dx 应均匀"
    assert m2["dx"][0] == pytest.approx(dx2, rel=1e-13), "dx2 应等于 (x2max-x2min)/nx2"

    # x1 方向不受 nx2>1 影响
    ncell1 = lib.xmf_mesh_ncell(h, 0)
    assert ncell1 == NX1_2D + 2 * NG, (
        f"薄 2D 的 x1 向含 ghost 格数应为 nx1+2ng={NX1_2D + 2 * NG}，实际 {ncell1}"
    )
    m1 = _axis(lib, h, 0, ncell1, ncell1 + 1)
    dx1 = (X1MAX - X1MIN) / NX1_2D
    i = np.arange(ncell1)
    np.testing.assert_allclose(
        m1["x"], X1MIN + (i - NG + 0.5) * dx1, rtol=0.0, atol=1e-12 * X1MAX,
        err_msg="nx2=4 不应影响 x1 方向的体心坐标",
    )


# --- 4. ghost 层数 = ng（第二行原文第 4 条） ---------------------------------


def test_ghost_count_equals_ng(lib, run_cart, run_2d):
    """每个方向 d：(ncell - n_d)/2 == ng；1D 退化向 n_d=1 也成立。"""
    for d, nd in enumerate((NX1, 1, 1)):  # 1D 用例：退化向 n_d=1
        ncell = lib.xmf_mesh_ncell(run_cart._h, d)
        assert ncell - nd == 2 * NG, (
            f"1D 用例 d={d}：ghost 格数应为 2ng={2 * NG}，"
            f"实际 ncell-n_d={ncell - nd}（ncell={ncell}, n_d={nd}）"
        )
        assert (ncell - nd) // 2 == NG, f"1D 用例 d={d}：单侧 ghost 层数应为 ng={NG}"
    for d, nd in enumerate((NX1_2D, NX2_2D, 1)):  # 薄 2D 用例
        ncell = lib.xmf_mesh_ncell(run_2d._h, d)
        assert ncell - nd == 2 * NG, (
            f"薄 2D 用例 d={d}：ghost 格数应为 2ng={2 * NG}，实际 ncell-n_d={ncell - nd}"
        )


def test_ghost_coordinates_extrapolate_in_1d(lib, run_cart):
    """1D 用例 ghost 段坐标按均匀间距外推：x[0] 比 x[ng] 左移 ng 个 dx。"""
    for d, nd in enumerate((NX1, 1, 1)):
        ncell = lib.xmf_mesh_ncell(run_cart._h, d)
        m = _axis(lib, run_cart._h, d, ncell, ncell + 1)
        dx = m["dx"][0]
        assert m["x"][0] == pytest.approx(m["x"][NG] - NG * dx, rel=1e-12), (
            f"d={d}：最左 ghost 体心 x[0] 应比内区首格 x[ng] 左移 ng 个 dx"
        )
        assert m["x"][ncell - 1] == pytest.approx(
            m["x"][NG + nd - 1] + NG * dx, rel=1e-12
        ), f"d={d}：最右 ghost 体心应比内区末格右移 ng 个 dx"


# --- 5. 非法入参按签名表返回哨兵 ---------------------------------------------


def test_invalid_direction_sentinels(lib, run_cart):
    """方向非法：ncell/nface 返回 0，copy_* 返回 -1 且不写缓冲；NULL 句柄同拒。"""
    h = run_cart._h
    buf = np.full(4, -123.0, dtype=np.float64)  # 哨兵填充，探测越权写入
    ptr = buf.ctypes.data_as(_PTR)
    for d in (-1, 3):
        assert lib.xmf_mesh_ncell(h, d) == 0, f"xmf_mesh_ncell(d={d}) 非法应返回 0"
        assert lib.xmf_mesh_nface(h, d) == 0, f"xmf_mesh_nface(d={d}) 非法应返回 0"
        for name in ("x", "xf", "dx", "A", "dV"):
            got = getattr(lib, f"xmf_mesh_copy_{name}")(h, d, ptr)
            assert got == -1, (
                f"xmf_mesh_copy_{name}(d={d}) 非法应返回 -1，实际 {got}"
            )
            assert np.all(buf == -123.0), (
                f"xmf_mesh_copy_{name}(d={d}) 返回 -1 后仍写了输出缓冲"
            )
    assert lib.xmf_mesh_ncell(None, 0) == 0, "xmf_mesh_ncell(NULL 句柄) 应返回 0"
    assert lib.xmf_mesh_copy_x(None, 0, ptr) == -1, (
        "xmf_mesh_copy_x(NULL 句柄) 应返回 -1"
    )


# --- 6. A1 补：非均匀界面（Python 交绝对位置，C 验单调 + 垫 ghost + 整套度量） --
#
# 卡片（docs/SESSION.md）「交什么」§1-§4：配置每向交 n_d+1 个活动界面绝对
# 位置；C 验严格递增后左/右 ghost 全用首/末活动格宽，再自洽算
# x=0.5*(xl+xr)、dx、A、dV。单调性权威在 C；球面 x1 任一界面（含 ghost）
# r<=0 → setup 失败。stretch 只是 Python 工厂，C 只认界面。

# 内密外疏活动界面（宽度 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25）；
# 全部为二进制有理数，逐位断言不受舍入干扰。核 sanity 下限 n[0]>=8。
NU_FACES = (1.0, 1.5, 2.25, 3.25, 4.5, 6.0, 7.75, 9.75, 12.0)
NU_N = len(NU_FACES) - 1
NU_DL = NU_FACES[1] - NU_FACES[0]      # 首活动格宽 0.5
NU_DR = NU_FACES[-1] - NU_FACES[-2]    # 末活动格宽 2.25


def _cfg_nonuniform_1d(faces=NU_FACES):
    """笛卡尔 1D：用户经 grid.x1 交非均匀活动界面（两端与 xdmin/xdmax 一致）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NU_N, 1, 1]
    cfg["grid"]["nghost"] = NG
    cfg["grid"]["x1min"] = faces[0]
    cfg["grid"]["x1max"] = faces[-1]
    cfg["grid"]["x1"] = list(faces)
    return cfg


def _cfg_sph_r(x1min, x1max, n):
    """球面 1D 夹具：只给标量端点，界面走均匀工厂（n+1 个）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["geometry"] = "spherical"
    cfg["grid"]["n"] = [n, 1, 1]
    cfg["grid"]["nghost"] = NG
    cfg["grid"]["x1min"] = x1min
    cfg["grid"]["x1max"] = x1max
    return cfg


def _uniform_faces_from_blob(blob):
    """直调 C 的助手：按 blob 自身 xdmin/xdmax/n 生成三向等距活动界面。"""
    mins = (blob.x1min, blob.x2min, blob.x3min)
    maxs = (blob.x1max, blob.x2max, blob.x3max)
    out = []
    for d in range(3):
        nd = max(int(blob.n[d]), 1)
        dx = (maxs[d] - mins[d]) / nd
        out.append(
            np.array([mins[d] + k * dx for k in range(nd + 1)], dtype=np.float64)
        )
    return out


def _setup_c_faces(lib, blob, x1_faces):
    """按 xmf_setup(cfg, xf1, xf2, xf3) 直调 C：x1 用指定界面，x2/x3 用
    blob 端点的等距界面。返回句柄（NULL 为 None）。"""
    arrs = _uniform_faces_from_blob(blob)
    arrs[0] = np.ascontiguousarray(x1_faces, dtype=np.float64)
    return lib.xmf_setup(
        ctypes.byref(blob),
        arrs[0].ctypes.data_as(_PTR),
        arrs[1].ctypes.data_as(_PTR),
        arrs[2].ctypes.data_as(_PTR),
    )


@pytest.fixture
def run_nonuniform():
    run = xmf.setup(_cfg_nonuniform_1d())
    yield run
    run.close()


def test_nonuniform_x1_active_faces_verbatim_and_increasing(lib, run_nonuniform):
    """用户数组逐位拷进活动段；整条格子（含 ghost 段）严格递增。"""
    h = run_nonuniform._h
    ncell = lib.xmf_mesh_ncell(h, 0)
    nface = lib.xmf_mesh_nface(h, 0)
    assert ncell == NU_N + 2 * NG, (
        f"非均匀 1D 含 ghost 格数应为 {NU_N}+2ng={NU_N + 2 * NG}，实际 {ncell}"
    )
    assert nface == ncell + 1
    m = _axis(lib, h, 0, ncell, nface)
    np.testing.assert_array_equal(
        m["xf"][NG : NG + NU_N + 1], np.array(NU_FACES),
        err_msg="活动界面必须逐位拷自用户数组（含两端，不静默改写）",
    )
    assert np.all(np.diff(m["xf"]) > 0.0), "整条界面序列（含 ghost 段）必须严格递增"


def test_nonuniform_x1_dx_and_ghost_widths(lib, run_nonuniform):
    """活动区 dx 逐格随数组；左/右每个 ghost 格宽 = 首/末活动格宽（邻格宽，
    不做几何外推）；ghost 界面位置按邻格宽线性外推。"""
    h = run_nonuniform._h
    ncell = lib.xmf_mesh_ncell(h, 0)
    m = _axis(lib, h, 0, ncell, ncell + 1)

    widths = np.diff(NU_FACES)
    np.testing.assert_array_equal(
        m["dx"][NG : NG + NU_N], widths,
        err_msg="活动区 dx[i] 必须逐格等于用户界面数组的相邻差",
    )
    assert np.ptp(m["dx"]) > 0.0, "非均匀用例的 dx 不得退化为全等（核没吃数组）"
    # 左 ghost：ng 格全用首活动格宽；右 ghost：全用末活动格宽
    np.testing.assert_array_equal(
        m["dx"][:NG], np.full(NG, NU_DL),
        err_msg="每个左 ghost 格宽都应等于首活动格宽",
    )
    np.testing.assert_array_equal(
        m["dx"][NG + NU_N :], np.full(NG, NU_DR),
        err_msg="每个右 ghost 格宽都应等于末活动格宽",
    )
    # 界面位置：xf_active[0] - (ng-i)*ΔL 与 xf_active[n] + i*ΔR
    left = np.array([NU_FACES[0] - (NG - i) * NU_DL for i in range(NG)])
    right = np.array([NU_FACES[-1] + i * NU_DR for i in range(1, NG + 1)])
    np.testing.assert_allclose(m["xf"][:NG], left, rtol=0.0, atol=1e-14,
                               err_msg="左 ghost 界面应按首活动格宽线性外推")
    np.testing.assert_allclose(m["xf"][NG + NU_N + 1 :], right, rtol=0.0, atol=1e-14,
                               err_msg="右 ghost 界面应按末活动格宽线性外推")


def test_nonuniform_x1_centres_and_cartesian_metrics(lib, run_nonuniform):
    """中点 x=0.5*(xf[i]+xf[i+1]) 全程成立；笛卡尔 A=1、dV=dx 不因非均匀减配。"""
    h = run_nonuniform._h
    ncell = lib.xmf_mesh_ncell(h, 0)
    m = _axis(lib, h, 0, ncell, ncell + 1)
    np.testing.assert_array_equal(
        m["x"], 0.5 * (m["xf"][:-1] + m["xf"][1:]),
        err_msg="x[i] 应等于 0.5*(xf[i]+xf[i+1])（含 ghost 段）",
    )
    assert np.all(m["A"] == 1.0), "笛卡尔面面积因子应全为 1（非均匀也不变）"
    np.testing.assert_array_equal(
        m["dV"], m["dx"], err_msg="笛卡尔应 dV == dx（非均匀逐格成立）"
    )


def test_nonuniform_x1_step_and_fields(lib, run_nonuniform):
    """四步 API 在非均匀格上仍通：dt>0、t 真推进、fields 有限。"""
    for _ in range(3):
        run_nonuniform.step()
    assert run_nonuniform.t > 0.0, "非均匀格上 step 应真正推进时间"
    assert run_nonuniform.dt > 0.0, "非均匀格的 CFL dt 应为正"
    hi = run_nonuniform.fields["HI"]
    assert len(hi.x) == NU_N
    for arr in (hi.n, hi.v, hi.p):
        assert np.isfinite(arr).all(), "非均匀格 step 后 fields 应全有限"


# --- 7. A1 补：拒绝契约（非递增界面 / 球面 r<=0） ----------------------------


def test_setup_rejects_nonincreasing_faces():
    """非递增界面 → setup 失败（前端路径）：相邻相等、递减各一条，
    RuntimeError 信息含 xmf_setup failed。"""
    bad_cases = (
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 12.0],   # 相邻相等
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 7.5, 12.0],   # 递减
    )
    for bad in bad_cases:
        cfg = _cfg_nonuniform_1d(faces=bad)
        with pytest.raises(RuntimeError, match="xmf_setup failed"):
            xmf.setup(cfg)


def test_setup_rejects_nonincreasing_faces_at_c(lib):
    """绕前端直调 C：非递增界面必须返回 NULL（单调性权威在 C）；
    同一 blob 换合法界面必须成功（对照：失败确实来自 lattice）。"""
    bad = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 7.5, 12.0]
    p = normalize(_cfg_nonuniform_1d(faces=bad))  # Python 层不验单调，放行
    blob = problem_to_c(p)
    handle = _setup_c_faces(lib, blob, bad)
    assert not handle, "xmf_setup 对非递增界面必须返回 NULL"
    good = _setup_c_faces(lib, blob, list(NU_FACES))
    assert good, "同一 blob 配合法界面应 setup 成功（对照用例）"
    lib.xmf_finalize(good)


def test_spherical_x1_nonpositive_radius_rejected():
    """球面 x1 任一界面（含 ghost）r<=0 → setup 失败。三条拒绝 + 一条正向对照：
    (1) 活动界面含 r=0；(2) 活动界面全 >0 但左 ghost 外推到 r<0；
    (3) ghost 面恰好触 0；(4) 含 ghost 全 >0 必须成功。"""
    # (1) 活动界面首格即 r=0
    with pytest.raises(RuntimeError, match="xmf_setup failed"):
        xmf.setup(_cfg_sph_r(x1min=0.0, x1max=1.0, n=8))
    # (2) 活动界面全 >0，左 ghost 外推到负半径：dx=9/16 → 1-2dx=-0.125
    with pytest.raises(RuntimeError, match="xmf_setup failed"):
        xmf.setup(_cfg_sph_r(x1min=1.0, x1max=10.0, n=16))
    # (3) ghost 面恰好触到 0：dx=8/16=0.5 → 最左 ghost 面 1-2*0.5=0
    with pytest.raises(RuntimeError, match="xmf_setup failed"):
        xmf.setup(_cfg_sph_r(x1min=1.0, x1max=9.0, n=16))
    # (4) 正向对照：含 ghost 全 >0（2-2*0.5=1.0）必须成功
    run = xmf.setup(_cfg_sph_r(x1min=2.0, x1max=10.0, n=16))
    run.close()


def test_spherical_ghost_radius_rejected_at_c(lib):
    """ghost 触线的拒绝只能出自 C（Python 不知道 ghost 宽）：直调返回 NULL；
    抬内半径后同一路径必须成功。"""
    p = normalize(_cfg_sph_r(x1min=1.0, x1max=10.0, n=16))  # 活动界面全 >0
    blob = problem_to_c(p)
    handle = _setup_c_faces(lib, blob, list(p.xf1))
    assert not handle, "球面左 ghost 外推到 r<=0，xmf_setup 必须返回 NULL"

    p2 = normalize(_cfg_sph_r(x1min=2.0, x1max=10.0, n=16))
    blob2 = problem_to_c(p2)
    good = _setup_c_faces(lib, blob2, list(p2.xf1))
    assert good, "含 ghost 全 r>0 的球面格应 setup 成功（对照用例）"
    lib.xmf_finalize(good)


# --- 8. A1 补：stretch=log 工厂的界面原样进核，只拉 x1 -----------------------


def test_stretch_log_faces_land_verbatim_only_x1(lib):
    """stretch='log' 只是 Python 工厂：吐出的界面原样进 C（活动段逐位相等），
    x1 以外各向不受影响（仍是 xdmin/xdmax 的等距界面）。"""
    cfg = xmf.shock_tube()
    cfg["grid"]["n"] = [NU_N, 1, 1]
    cfg["grid"]["nghost"] = NG
    cfg["grid"]["stretch"] = "log"
    cfg["grid"]["stretch_factor"] = 2.0
    run = xmf.setup(cfg)
    try:
        h = run._h
        xf1 = np.array(run._cfg.xf1, dtype=np.float64)
        widths = np.diff(xf1)
        assert np.all(widths[1:] > widths[:-1]), "factor>1 应内密外疏"

        ncell = lib.xmf_mesh_ncell(h, 0)
        m = _axis(lib, h, 0, ncell, ncell + 1)
        np.testing.assert_array_equal(
            m["xf"][NG : NG + NU_N + 1], xf1,
            err_msg="log 工厂的界面必须原样进核（C 不认配方，只认界面）",
        )
        assert m["dx"][0] == widths[0], "左 ghost 宽仍应等于（最小的）首活动格宽"

        # x2/x3 不受 x1 拉伸影响：等距、端点仍是各自 xdmin/xdmax
        for d, lo, hi in ((1, 0.0, 1.0), (2, 0.0, 1.0)):
            ncd = lib.xmf_mesh_ncell(h, d)
            md = _axis(lib, h, d, ncd, ncd + 1)
            assert np.ptp(md["dx"]) == 0.0, f"d={d} 方向必须保持等距（只拉 x1）"
            np.testing.assert_array_equal(
                md["xf"][NG : NG + 2], np.array([lo, hi]),
                err_msg=f"d={d} 活动界面应仍是 xdmin/xdmax 的等距工厂",
            )
    finally:
        run.close()
