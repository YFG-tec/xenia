"""A5 空表步合同单测（test_step_empty）：空 sources / 无辐射的 step 钉法。

合同来自本刀卡片（docs/SESSION.md §交什么 1/6）：
  1. 空表（默认 shock_tube()：sources=[]、无辐射束表）的 pcm Sod L1 与
     最终 t 逐位钉死：(3.5152402455663314e-3, 9.7162878019579242e-3,
     4.0293309314733058e-3) 与 t = 0.15224840316529667。钉法、口径与
     tests/test_core_split.py 完全一致（scale = (n_l, max|v_ex|, max p_ex)，
     精确解取 run.advance() 实际到达的 t，断言用全精度 ==）。本刀加了
     scan / source 两个新模块，空路径必须逐位不动 hydro——这条钉住
     「空表不进 L_c、不扫 F」的底线。
  2. 空表身份：num_tapes == 0、nbeam == 0（A5 的两条空路径哨兵）。
  3. 四步仍通：setup → 一步 step → fields 可取，末态物理上合理。

禁令自查：不读 pluto.ini；不用 XUV_FACTOR==0 哨兵；钉的是全分量
(n, v, P) 的 L1 与整条四步链，不只测 VX1。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import xmf
from xmf.config import KB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.riemann import exact_hi  # noqa: E402

# 空表基线（全精度；逐位同值钉，禁止改成宽松上界）
PIN_T = 0.15224840316529667
PIN_L1_N = 0.0035152402455663314
PIN_L1_V = 0.0097162878019579242
PIN_L1_P = 0.0040293309314733058


def test_empty_tables_sod_pin_bitwise():
    """默认 shock_tube()（空 sources、无辐射）：t 与 L1(n,v,P)/scale 逐位
    等于空表基线——scan / source 新模块不得动纯 hydro 空路径。"""
    cfg = xmf.shock_tube()
    assert cfg["numerics"]["riemann"] == "hll"
    assert cfg["numerics"]["reconstruction"] == "pcm"
    assert cfg["numerics"]["rk"] == "rk1"
    assert cfg["sources"] == [], "本钉只对空 sources 表有效"

    run = xmf.setup(cfg)
    try:
        assert run.num_tapes == 0, "sources=[] 不得加载任何带子"
        assert run.nbeam == 0, "无辐射配置不得有束表"

        run.advance()
        assert run.t >= run.tstop, "advance 应跑到 tstop（可略过冲）"
        assert run.t == PIN_T, (
            f"空表最终 t 与基线不一致：实测 {run.t!r}，钉值 {PIN_T!r}"
            "（A5 的 scan/source 空路径动了时间推进）"
        )

        # 口径与 test_core_split / test_hydro 逐字相同
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
            f"空表 L1(n)/scale 与基线逐位不同：实测 {err_n!r}，钉值 {PIN_L1_N!r}"
        )
        assert err_v == PIN_L1_V, (
            f"空表 L1(v)/scale 与基线逐位不同：实测 {err_v!r}，钉值 {PIN_L1_V!r}"
        )
        assert err_p == PIN_L1_P, (
            f"空表 L1(P)/scale 与基线逐位不同：实测 {err_p!r}，钉值 {PIN_L1_P!r}"
        )
    finally:
        run.close()


def test_empty_tables_four_step_api_still_wired():
    """空表四步仍通：setup → 一步 step → fields 可取；dt/t 为正、状态有限。"""
    run = xmf.setup(xmf.shock_tube())
    try:
        t1 = run.step()
        assert t1 > 0.0, "一步 step 应真正推进时间"
        assert run.dt > 0.0, "step 后 dt 应仍为正（CFL 仍活着）"
        assert run.source_resid == 0.0, (
            "空表没有源步：source_resid 应保持 0，不得被新模块污染"
        )
        hi = run.fields["HI"]
        nx = int(run._cfg.n[0])
        assert hi.x.shape == (nx,)
        assert hi.n.shape == hi.v.shape == hi.p.shape == (nx,)
        assert np.all(np.isfinite(hi.n)) and np.all(hi.n > 0.0)
        assert np.all(np.isfinite(hi.v)) and np.all(np.isfinite(hi.p))
        assert float(np.max(hi.n)) / float(np.min(hi.n)) > 2.0, (
            "一步之后 Sod 台阶应仍在（空路径未破坏初值）"
        )
    finally:
        run.close()
