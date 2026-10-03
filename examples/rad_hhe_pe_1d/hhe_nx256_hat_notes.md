# nx=256 / stretch=1.015 开门炸步定性

- **哪一项：** 源步失败（`xmf_step=-1`），伴随 `source_resid` 升到 1e25–1e27、CFL `dt` 塌到 1e-34。
- **哪个时标：** 工厂化学瞬态仍是 ~10 s；本网格上 `dt=10` 在 t≈2.5×10³ s 把增长模养爆（step ~300）。`dt_max=5` 过同一窗口，残差最坏仍在第 1 步（4.2）。
- **哪一类：** 帽。不是守恒泄漏、不是边界冲量陈述。内区 `dx/R_p` 从 9.96e-3（128/1.025）收到 3.05e-3。
- **证据：** `output/hhe_omp_bench/fail_probe.txt`，`hhe_nx256_hat_probe.py` 输出。CFL 公式未改；帽只写在 `utils/hhe_bench/hhe_omp_bench.py`，工厂 `hhe_config.RUN.dt_max=10` 不动。
