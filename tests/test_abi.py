"""A0 壳契约：setup/finalize 重复 100 次、坏 config 在 C 级被拒、step 前可读初值、薄 2D。

约定（docs/TODO.md §A）：一块一文件；1D（nx2=nx3=1）加 nx2>1 的薄 2D；
不读 pluto.ini、不用 XUV_FACTOR==0 哨兵、不只测单速度分量。
"""
from __future__ import annotations

import ctypes

import numpy as np

import xmf
from xmf._native import load, problem_to_c
from xmf.config import AMU, KB, normalize, shock_tube

REPEAT = 100  # TODO A0：setup→finalize 重复 100 次


def _tiny():
    cfg = shock_tube()
    cfg["grid"]["n"] = [32, 1, 1]
    cfg["time"]["tstop"] = 0.01
    return cfg


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
    ptr = ctypes.POINTER(ctypes.c_double)
    return lib.xmf_setup(
        ctypes.byref(blob),
        arrs[0].ctypes.data_as(ptr),
        arrs[1].ctypes.data_as(ptr),
        arrs[2].ctypes.data_as(ptr),
    )


def _snapshot(run):
    """拷出 HI 组分的初值数组（np.array 拷贝保存，防缓冲复用假象）。"""
    hi = run.fields["HI"]
    return {
        "x": np.array(hi.x),
        "n": np.array(hi.n),
        "v": np.array(hi.v),
        "p": np.array(hi.p),
    }


def test_setup_finalize_repeat():
    """setup→finalize 重复 100 次：每轮功能确定，首尾初值逐元素相等（无残留状态污染）。

    「无泄漏」的可操作口径 = 确定性功能 + 内容一致性断言；不断言 RSS /
    tracemalloc 等内存阈值（Windows pytest 下易抖，tracemalloc 也追不到 C 的 malloc）。
    """
    cfg = _tiny()
    first = None
    for i in range(REPEAT):
        run = xmf.setup(cfg)  # 失败会 raise RuntimeError；不抛即 setup 成功、handle 非 NULL
        assert run.t == 0.0
        assert "HI" in run.fields
        snap = _snapshot(run)
        assert snap["n"][0] > snap["n"][-1]  # 密度初值左 > 右
        if first is None:
            first = snap  # 第一轮：拷贝保存初值
        last = snap
        run.close()  # close() 正常（不抛异常）
    # 最后一轮的初值数组与第一轮逐元素相等
    for key in ("x", "n", "v", "p"):
        assert np.array_equal(last[key], first[key])


def _rad_source_cfg():
    """C1a 生命周期验收用的非空配置：2 流体 + 2 条光化带 + 单束辐射表。

    每一轮 setup→close 都带源项与束表：循环已上 Kokkos 界面后，
    「每周期重复初始化必须幂等、close 不得退出 Kokkos」的守卫必须
    在真正跑过循环（含步内 scan/源步）的路径上验收，而不只是空表。
    """
    cfg = shock_tube()
    cfg["grid"]["n"] = [32, 1, 1]
    cfg["grid"]["x1min"] = 1.0e10
    cfg["grid"]["x1max"] = 2.0e10
    cfg["time"]["tstop"] = 1.0e6  # 只走定步，不受 tstop 截断影响
    cfg["numerics"] = {"riemann": "hll", "reconstruction": "pcm", "rk": "rk1"}
    cfg["units"] = {"density": 1.0, "velocity": 1.0, "length": 1.0}
    cfg["fluids"] = [
        {"name": "HI", "mass": 1.00794, "charge": 0, "kind": "hd"},
        {"name": "HII", "mass": 1.00794, "charge": 1, "kind": "hd"},
    ]
    cfg["electron"] = {"mode": "off"}
    cfg["ic"] = {"kind": "none"}
    cfg["boundary"] = {"x1": "outflow"}
    cfg["constants"] = {"sigma_0": 2.5e-18, "hnu_0": 1.0e-13}
    gamma_ion = "n_HI * (sigma_0 * F_0 / hnu_0)"
    cfg["sources"] = [
        {"name": "photoion_HI", "expr": {"dn_HI": f"-({gamma_ion})"}},
        {"name": "photoion_HII", "expr": {"dn_HII": gamma_ion}},
    ]
    cfg["radiation"] = {
        "enabled": True,
        "absorbers": ["HI"],
        "scan": {"mode": "grid_line", "dilution": "none", "sense": "inward"},
        "bins": {"n": 1, "sigma": [2.5e-18], "hnu": [1.0e-13],
                 "flux_inf": [4.716]},
    }
    return cfg


def test_setup_step_finalize_repeat_with_sources_and_radiation():
    """C1a 生命周期验收（卡片点名的幂等守卫，非空路径）：
    setup→均匀初值→一步 step→close 重复 100 次，每轮都真正跑过
    hydro/scan/源步循环。断言两条：

    1. 每一轮都成功（任何一轮在重复初始化后崩/挂即红）——若 close
       误退了进程级基底（如 Kokkos::finalize），第二轮起必然失败；
    2. 跨轮确定性：每轮一步后的 t / dt / source_resid / 双流体五槽
       与第一轮逐元素（逐位）相等——同一初值同一步数不得因「第几轮」
       而漂出任何残差。
    """
    lib = load()
    cfg = _rad_source_cfg()
    n0, t0 = 1.0e6, 1500.0
    first = None
    for i in range(REPEAT):
        run = xmf.setup(cfg)  # 失败即 raise；幂等守卫在这里兑现
        try:
            assert run.num_tapes == 2, "光化对应编出 2 条带子"
            assert run.nbeam == 1, "单束辐射表"
            # 均匀初值（含 ghost 全块）：units 全 1 → mass_code = mass_amu*AMU
            ng = run._cfg.ng
            nx, ny, nz = run._cfg.n
            nxt, nyt, nzt = nx + 2 * ng, ny + 2 * ng, nz + 2 * ng
            m = 1.00794 * AMU
            ncells = lib.xmf_state_ncells(run._h)
            assert ncells == nzt * nyt * nxt
            ptr = ctypes.POINTER(ctypes.c_double)
            for f in range(2):
                for var, val in ((0, n0 * m), (4, n0 * KB * t0)):
                    data = np.full((nzt, nyt, nxt), val, dtype=np.float64).ravel()
                    assert lib.xmf_state_pack(
                        run._h, f, var, data.ctypes.data_as(ptr)) == ncells

            t1 = run.step()
            assert t1 > 0.0, "每轮一步都应真正推进时间"
            snap = {
                "t": run.t,
                "dt": run.dt,
                "resid": run.source_resid,
                "fld": run.fields_full(),
            }
        finally:
            run.close()  # close 必须正常，且不退进程级基底

        if first is None:
            first = snap
        else:
            assert snap["t"] == first["t"], (
                f"第 {i + 1} 轮一步后 t={snap['t']!r} 与首轮 {first['t']!r}"
                " 不一致（重复初始化引入状态残差）"
            )
            assert snap["dt"] == first["dt"]
            assert snap["resid"] == first["resid"]
            assert set(snap["fld"]) == {"HI", "HII"}
            for name, slots in snap["fld"].items():
                for key, arr in slots.items():
                    assert np.array_equal(arr, first["fld"][name][key]), (
                        f"第 {i + 1} 轮 {name}.{key} 与首轮逐位不同"
                        "（生命周期守卫失效：跨轮残差进了物理态）"
                    )


def test_fields_readable_before_step():
    """setup 后不调 step：初值可读，长度 == xmf_ncell，全有限，密度左>右，v 全 0。"""
    run = xmf.setup(_tiny())
    hi = run.fields["HI"]
    ncell = int(load().xmf_ncell(run._h))
    for arr in (hi.x, hi.n, hi.v, hi.p):
        assert len(arr) == ncell
        assert np.isfinite(arr).all()
    assert hi.n[0] > hi.n[-1]  # 初值间断可见：密度左 > 右
    assert np.all(hi.v == 0.0)  # shock_tube 左右两侧 v=0
    run.close()


def test_bad_schema_rejected_in_c():
    # 改 ctypes blob 绕过 Python 前端，直接打 C 契约
    p = normalize(_tiny())
    blob = problem_to_c(p)
    blob.schema_version = 99
    lib = load()
    handle = _setup_c(lib, blob)
    assert handle is None or handle == 0


def test_n0_rejected_in_c():
    p = normalize(_tiny())
    blob = problem_to_c(p)
    blob.n[0] = 0
    lib = load()
    handle = _setup_c(lib, blob)
    assert handle is None or handle == 0


def test_negative_n0_rejected_in_c():
    """负 n[0] 也在 C 级被拒。normalize 会先行 raise，那是前端契约；必须改 blob 才打得到 C。"""
    p = normalize(_tiny())
    blob = problem_to_c(p)
    blob.n[0] = -8
    lib = load()
    handle = _setup_c(lib, blob)
    assert handle is None or handle == 0


def test_thin2d_setup_and_fields():
    """nx2>1 薄 2D（8×4×1，ndim=2）。只测 setup 与 step 前读初值，不调 step。

    形状取舍：核 sanity 下限 n[0]>=8，最小精神的 (4,4,1) 会被拒成 NULL，故取 (8,4,1)。
    """
    cfg = shock_tube()
    cfg["grid"]["n"] = [8, 4, 1]
    cfg["grid"]["ndim"] = 2  # 声明维数与形状一致，防打架
    assert normalize(cfg).ndim == 2
    run = xmf.setup(cfg)  # 不 raise => handle 非 NULL
    assert run.t == 0.0
    assert int(load().xmf_ncell(run._h)) == 8
    hi = run.fields["HI"]  # step 前可读
    for arr in (hi.x, hi.n, hi.v, hi.p):
        assert len(arr) == 8
        assert np.isfinite(arr).all()
    assert hi.n[0] > hi.n[-1]  # 密度左 > 右
    run.close()


def test_legacy_grid_nx_still_works():
    cfg = shock_tube()
    del cfg["grid"]["n"]
    cfg["grid"]["nx"] = 32
    p = normalize(cfg)
    assert p.n[0] == 32
    run = xmf.setup(cfg)
    run.close()
