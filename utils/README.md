# utils

仓库侧尺子，不是公开 `xmf` API。算例 `run.py` / 科学长跑仍放 `examples/`。

| 目录 | 干什么 |
|---|---|
| `riemann/` | Toro 精确解。单测和 `hd_eval` / `flux_eval` / `rk_eval` 共用。 |
| `hhe_bench/` | `rad_hhe_pe_1d` 的 OpenMP / CUDA 墙钟与逐位对照。dump 写回该算例的 `output/`。 |

一次性探针写 `.tmp/`。只属于某一个算例、还要再跑的长跑 / 出图，留在那个算例目录。
