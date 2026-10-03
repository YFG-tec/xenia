"""A4.5 拆分逐位同值钉（refactor — 切开 xmf_core.c：只搬文件，不改公式）。

本文件只钉两件事（docs/SESSION.md「交什么」第 5 条：不新开物理题）：

1. 逐位同值。默认 xmf.shock_tube()（hll + pcm + rk1）跑到 tstop，对
   utils/riemann/exact_hi 的 L1(n,v,P)/scale 必须与拆前基线全精度一致。
   常数来自拆前基线（builder 用 .tmp/check_split_bitwise.py 拆前存档比对过），
   口径照抄 tests/test_hydro.py::test_sod_1d_l1_against_toro_bounded：
     scale = (n_l, max|v_ex|, max p_ex)，L1 = mean|num-exact|/scale，
     精确解取在 run.advance() 实际到达的 t（含末步 CFL 过冲）。
   断言用全精度常数直接 ==：这是逐位同值钉，不得写成宽松上界——宽松上界
   （≈2× 实测）已由 test_hydro / test_shock_hd 管，这里若放宽就抓不到
   「仍在上界内、但已不逐位」的小回退。同理把最终 t 也钉住：时间推进
   序列变了，L1 必然跟着变，t 是最便宜的哨兵。

2. 四步仍通。setup → step → fields 至少一步：拆完公开 API 不变。拆出的
   辅助符号 xmf_mesh_* / xmf_state_* / xmf_bc_* 仍从同一份共享库链到，
   已由 tests/test_mesh.py / test_state.py / test_bc.py 覆盖（都经
   xmf._native.load() 绑同一库），本文件不重复深测，只在此注明。

禁令自检：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；钉的是全分量（n,v,P）
L1 与整条四步链，不只测 VX1。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import xmf
from xmf.config import KB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.riemann import exact_hi  # noqa: E402

# 拆前基线（全精度；A4.5 逐位同值钉，禁止改成宽松上界）
PIN_T = 0.15224840316529667
PIN_L1_N = 0.0035152402455663314
PIN_L1_V = 0.0097162878019579242
PIN_L1_P = 0.0040293309314733058


def test_sod_pin_bitwise_against_pre_split_baseline():
    """hll+pcm+rk1 Sod 跑到 tstop：t 与 L1(n,v,P)/scale 逐位等于拆前基线。"""
    cfg = xmf.shock_tube()
    # 钉住数值路径：本钉只对 hll + pcm + rk1 有效
    assert cfg["numerics"]["riemann"] == "hll"
    assert cfg["numerics"]["reconstruction"] == "pcm"
    assert cfg["numerics"]["rk"] == "rk1"

    run = xmf.setup(cfg)
    try:
        run.advance()
        assert run.t >= run.tstop, "advance 应跑到 tstop（可略过冲）"
        assert run.t == PIN_T, (
            f"最终 t 与拆前基线不一致：实测 {run.t!r}，钉值 {PIN_T!r}"
            "（时间推进序列变了，拆分动了不该动的东西）"
        )

        # 口径与 test_hydro.py::test_sod_1d_l1_against_toro_bounded 逐字相同
        hi = run.fields["HI"]
        ic = cfg["ic"]
        p_l = ic["left"]["n"] * KB * ic["left"]["T"]
        p_r = ic["right"]["n"] * KB * ic["right"]["T"]
        n_ex, v_ex, p_ex = exact_hi(
            hi.x, run.t, ic["x0"], ic["left"]["n"], ic["right"]["n"],
            0.0, 0.0, p_l, p_r, gamma=run._cfg.gamma,
        )
        scale_n = ic["left"]["n"]
        scale_v = max(float(np.max(np.abs(v_ex))), 1.0)
        scale_p = float(np.max(p_ex))
        err_n = float(np.mean(np.abs(hi.n - n_ex)) / scale_n)
        err_v = float(np.mean(np.abs(hi.v - v_ex)) / scale_v)
        err_p = float(np.mean(np.abs(hi.p - p_ex)) / scale_p)

        assert err_n == PIN_L1_N, (
            f"L1(n)/scale 与拆前基线逐位不同：实测 {err_n!r}，钉值 {PIN_L1_N!r}"
        )
        assert err_v == PIN_L1_V, (
            f"L1(v)/scale 与拆前基线逐位不同：实测 {err_v!r}，钉值 {PIN_L1_V!r}"
        )
        assert err_p == PIN_L1_P, (
            f"L1(P)/scale 与拆前基线逐位不同：实测 {err_p!r}，钉值 {PIN_L1_P!r}"
        )
    finally:
        run.close()


def test_four_step_api_still_wired():
    """拆后四步仍通：setup → 一步 step → fields 可取，末态物理上合理。"""
    run = xmf.setup(xmf.shock_tube())
    try:
        t1 = run.step()
        assert t1 > 0.0, "一步 step 应真正推进时间"
        assert run.dt > 0.0, "step 后 dt 应仍为正（CFL 仍活着）"
        hi = run.fields["HI"]
        nx = int(run._cfg.n[0])
        assert hi.x.shape == (nx,)
        assert hi.n.shape == hi.v.shape == hi.p.shape == (nx,)
        assert np.all(np.isfinite(hi.n)) and np.all(hi.n > 0.0)
        assert np.all(np.isfinite(hi.v)) and np.all(np.isfinite(hi.p))
        # 一步之后仍是 Sod 管：密度两端值还在量级上（没被拆散的量纲错配洗掉）
        assert float(np.max(hi.n)) / float(np.min(hi.n)) > 2.0, (
            "一步之后左右态密度台阶应仍在（Sod 初值未被破坏）"
        )
    finally:
        run.close()
