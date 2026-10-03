# Claude Code harness

**在本仓库根启动 Claude Code**（`hd/`），不要在父目录 `Xenia_Multi_Fluid/` 启动——那边没有本树的 `CLAUDE.md`，代理会丢树身份。本树是 HD 投稿 / 发布桌。

调度器就是 Claude Code 主会话。状态落在 `docs/TODO.md` 和 `docs/SESSION.md`，不靠聊天记录。

```text
CLAUDE.md                          每会话锁
AGENTS.md                          同一份锁（短）
.claude/agents/spec-guard.md       只写本刀范围
.claude/agents/builder.md          只改这一块实现
.claude/agents/contract-tester.md  只写/改本模块单测
.claude/agents/numerics-review.md  只审 hydro/scan/源项的数
.claude/skills/write-cmd/SKILL.md  写指令卡：PROMPT + SESSION，停
.claude/skills/one-slice/SKILL.md  固定顺序
docs/cmd/PROMPT_*.md               可贴的一刀指令
docs/SESSION.md                    本刀卡片（每次覆盖）
.tmp/                              草稿；只跟踪 README.md，其余 gitignore
```

不要开实验性 Agent Teams，不要四个窗口同时改同一份核。A4.5 已切开：禁止把 `mesh` / `state` / `bc` / `hydro` 并回 `xmf_core.c`；新模块各自一份 `.c`。A4.6 之后：IR 只追加名字/`kind`，操作码不动。

## 第一次：整段贴进 Claude Code

```text
按 .claude/skills/one-slice/SKILL.md 做 docs/TODO.md A0 这一行：
`xmf_setup` / `finalize`：按 XmfConfig 分配/释放网格与场；坏版本号、n[0]<1 返回失败。

先派 spec-guard 覆盖 docs/SESSION.md，写完停下来，等我看一眼再派 builder。
不要改 SSH_220804/，不要把化学或体力写进 C，不要 commit，不要勾 TODO。
对用户用简体中文。
```

以后换一行：把上面的「A0 这一行」换成 `TODO.md` 里下一句未勾原文，其余不动。

## 你每天只做的

1. 看 `docs/SESSION.md`（2 分钟）
2. 看 pytest 红绿
3. 决定勾不勾 `TODO.md`
4. 拒超售（GPU、逃逸科学结论、核里写死化学）

## 调试内环

散了或不守恒时，主会话先分类，再决定动哪一层。细则锁在 `CLAUDE.md`「排障」。

| 残差形态 | 含义 | 动哪里 |
|---|---|---|
| 一步 `A` 近奇异 / `xmf_step=-1`，时标不随 `nx` 变 | 源步单步线性化的增长模 | 算例 `dt_max` 帽 |
| 饱和、不随步涨 | 非平衡初值伪漂移 | 换等温压平衡初值 |
| reflect 墙上 `dp` 很大 | 壁给格子冲量 | 换测试陈述（copy 盒 / 通量 0） |
| 随步涨、反对称一拆就爆 | 真泄漏 | 核记账（才许改 `src/xmf/core/`） |

探针只写 `.tmp/`。看 \(\Delta t\) 用有界短跑（十几步到几千步）。三条逐位老路（单吸收体 scan、单离子 CFL、空表 Sod）是**改核之后**的免疫税，不是扫步长的内环。

测出最慢模 \(\tau\) 后帽取 \(\tau\) 以内留余量，不要再往 2 s / 18 s 两端探。`dt_max` 现钉 10 s 是 B3 的帽，不是下一刀要修掉的数。

**无加热对照放弃调**（A5 `rad_hd_1d` 除外）：抽空柱难定常，关门长跑不进内环、不当「流守恒了没」的尺子。`rad_h_pe_1d` / `rad_hhe_pe_1d` 默认只开门。工厂 `radiation=False` 留给契约测。B3 本身没在关门上烧墙钟；这句锁的是以后的逃逸算例。

## 本机 Python：仓库根 `.venv`

不要用 miniforge **base**，也不要另开一份和仓库无关的全局环境。
解释器：`.venv/Scripts/python.exe`（Windows）。WSL CUDA 是另一套 Linux Python，不和这份 `.venv` 共用。

第一次（用现成的 3.12 解释器建，再装本包）：

```text
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install .
```

`requirements.txt` 是**要维护的**环境清单（直接 import 的包，不要 `pip freeze` 整树）。
新加第三方包：先改这一份，再对一下 `pyproject.toml`。

已有构建树、只想同步进这个环境：`.venv\Scripts\python.exe scripts/sync_xmf.py --pip`。

## 本机 `import xmf` 不是 `build/`

Windows 上 `src/xmf/_native.py` 在**自己旁边**找 `xmf_core.dll`。`python -m pytest` 若 import 到的是
别的环境的 `site-packages/xmf/`，跑的就是上次 `pip install .` 或手工拷进去的副本。
`cmake --build build/cp312-...` 只更新构建树。不同步则：改了 C 仍红（假红）或昨天的 dll 仍绿（假绿）。

改 C 后验收顺序（标准流程，不要发明 `cp`）：

```text
.venv\Scripts\python.exe scripts/sync_xmf.py
.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_abi.py -v
```

`--check` 只打印 DEST / ART / LIVE 时间戳。无构建树：`python scripts/sync_xmf.py --pip`。守恒卡片写法见 `CLAUDE.md`「排障」。
