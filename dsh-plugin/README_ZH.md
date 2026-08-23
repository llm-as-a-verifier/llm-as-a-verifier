# dsh-plugin — LLM-as-a-Verifier 多路径插件（DeepSeek Harness）

让 DeepSeek Harness 的 Agent 具备「**生成多条执行路径 → 自行验证并选择最佳路径**」的能力：

1. **`run_paths`**：为任务并行生成 N 条独立的候选执行路径（每条路径由一个新的 subagent 会话独立完成）；
2. **`submit_path`**：Agent 把自己产出的方案也登记为候选路径；
3. **`list_paths`**：查看某次运行已登记的候选路径；
4. **`select_best_path`**：用 **LLM-as-a-Judge 两两比较锦标赛**（环行扫描 + 枢轴轮次，A/B 槽位轮换消除位置偏差，win-mass 聚合，对应 `llm-as-a-verifier` 项目的 Probabilistic Pivot Tournament）对候选路径排序，返回最优路径。

插件同时向系统提示注入一条多路径验证工作流说明（order 150），引导模型在开放性或正确性难以判断的任务上先多路径生成、再验证选择。四个工具通过 `tools` 注入钩子注册，插件加载后自动出现在模型可用的工具列表里——即使不显式指示，模型在开放性任务上也可能主动使用，但显式指示最稳妥（见「工具注册与判题模型」）。

插件为**纯 Node.js（TypeScript）实现**——验证器逻辑（锦标赛、评分、聚合）已用 TypeScript 完整重写，运行在 Harness 进程内，运行时**完全不依赖 Python**（详见「环境要求」）。

## 环境要求

- Node.js ≥ 22.19（或 ≥ 24）
- pnpm（`npm i -g pnpm`）
- DeepSeek API Key（`dsh web` 中配置，或 `DEEPSEEK_API_KEY` 环境变量）

> **无需 Python 环境。** 本插件是纯 TypeScript 实现：`run_paths` 与 `select_best_path` 完全运行在 Harness（Node.js）进程内——候选路径由进程内 subagent 会话生成，判题锦标赛通过 Harness 的 `ctx.llm` 服务直接调用验证模型。插件从不调用 Python 子进程，也**不需要** `pip install llm-verifier`：Python 侧的 `llm-as-a-verifier` 项目只是锦标赛算法的参考实现，并非运行时依赖。全新环境只需 Node.js、pnpm 和 DeepSeek API Key 即可。

## 从源码安装

在项目根目录（`dsh-plugin/`）执行：

```bash
# 1. 解析依赖（二选一）
pnpm install                        # registry 安装（npm 上的 rc 包依赖完整时可用）
scripts/link_dev_deps.sh <deepseek-harness 源码仓库路径>   # 或链接到源码仓库（推荐，实测可用）

# 2. 构建
pnpm build

# 3. 安装到 web profile
dsh plugin --profile web add .

# 4. 重启 deepseek harness
# dsh web
```

说明：

| 步骤 | 作用 |
|---|---|
| 依赖解析 | 插件的运行时依赖（`@deepseek-ai/cordis`、`dsh-*`）由宿主 Harness 提供（peer 依赖），本目录只需构建工具链。npm 上部分 rc 包依赖未发布的内部包（如 `@deepseek-ai/dsh-type-meta`），直接 `pnpm install` 会 404 —— 此时改用 `scripts/link_dev_deps.*` 把 `node_modules` 链接到 deepseek-harness 源码仓库的 pnpm 虚拟仓库（与本机运行版本完全一致，本机实测通过） |
| `pnpm build` | 用 `tsc` 把 `src/` 编译到 `lib/`（`lib/index.js` 是插件的入口）；`pnpm-workspace.yaml` 已配置 `verifyDepsBeforeRun: false` 兼容 dev-link 模式 |
| `dsh plugin --profile web add .` | 在 `web` profile 中安装本插件：`dsh` 会把 `pnpm add <本目录绝对路径>` 转发到 `~/.dsh/profiles/web` 下执行；安装完成后，由于 `package.json` 声明了 `dsh.bundle.patch`，插件会自动加入该 profile 的 `dsh.profile.bundles` 层 |
| 重启 `dsh web` | 重启后 `cordis.patch.yml` 的挂载行（`name: dsh-plugin`）生效，四个工具可用 |

安装是否成功可以这样验证：

```bash
dsh --profile web --dump-config | grep -A2 llm-verifier-multipath
# - id: llm-verifier-multipath
#   name: dsh-plugin
```

启动后（`dsh web`，访问 http://127.0.0.1:3080），新建会话并让模型执行多路径验证，例如：

> 实现一个 `reverse_string` 函数并充分验证。先 `run_paths` 生成 3 条独立实现路径，再用 `select_best_path` 选出最优方案，最后说明胜出原因。

## 使用流程

```
run_paths(task, n) ──► submit_path(content) ──► list_paths(runId) ──► select_best_path(runId)
      │                                          │                         │
      └── 并行 subagent 路径 ────────────────────┴──► 两两比较锦标赛 ──────┴──► 最优路径
```

- `run_paths`：`{ task, n? }`，返回 `{ runId, paths: [{ id, content, stopReason }] }`。
- `submit_path`：`{ content, runId?, task? }`，返回 `{ runId, pathId, nPaths }`。
- `list_paths`：`{ runId }`，返回全部候选（内容截断显示）。
- `select_best_path`：`{ runId, criteria?, nEvaluations?, pivots? }`，返回 `{ winner, ranking, nComparisons, criteria, usage }`。

比较提示词与 `llm-as-a-verifier` 项目的 pairwise 模板一致（`<score_A>` / `<score_B>`，1–20 整数评分，归一化后按 win-mass 聚合）；`nEvaluations > 1` 时重复比较会交替 A/B 槽位，抵消位置偏差。单次比较失败按平局（0.5/0.5）处理，与 Python 侧 `on_error="tie"` 一致。

## 工具注册与判题模型

### 工具自动注册

四个工具（`run_paths` / `submit_path` / `list_paths` / `select_best_path`）通过 `tools` 注入钩子注册，插件加载后会自动出现在模型可用的工具列表里，无需手动开启或逐会话配置。由于插件还向系统提示注入了多路径验证工作流说明（order 150），即使不显式指示，模型在开放性任务或正确性难以判断的任务上也可能主动使用这些工具。但**显式指示仍然是最稳妥的触发方式**（见「使用示例」）。

### 更换判题模型

验证器（判题）模型默认使用 `deepseek-v4-flash`（provider `deepseek-official`）。想换模型时，修改 `dsh-plugin/cordis.patch.yml` 中 `config` 块的 `model`（必要时还有 `provider`），然后重启 `dsh web` 即可生效：

```yaml
- insert:
    - id: llm-verifier-multipath
      name: dsh-plugin
      config:
        # ... 其他配置 ...
        provider: deepseek-official
        model: deepseek-v4-flash   # <- 在此修改判题模型
```

重启后，后续所有 `select_best_path` 调用都会使用新的判题模型。

## 使用示例

以「实现一个 Python 函数 `reverse_string(s: str) -> str`，正确处理空字符串和 Unicode 字符」为任务，完整的多路径验证会话如下——**不要跳过任何工具调用**：

**第 1 步 — 调用 `run_paths`，生成 3 条独立候选路径：**

```json
run_paths({
  "task": "实现一个 Python 函数 reverse_string(s: str) -> str，正确处理空字符串和 Unicode 字符。",
  "n": 3
})
```

返回 `runId` 与 3 条候选路径（如切片反转、`reversed()` + join、`reduce` 累加等实现）。

**第 2 步 — 调用 `select_best_path`，`runId` 使用第 1 步返回的值：**

```json
select_best_path({
  "runId": "<run_paths 返回的 runId>"
})
```

**第 3 步 — 总结胜出的方案和它胜出的理由**，例如：

> 胜出的是 #0「切片反转实现」：`return s[::-1]`。它以码位为单位整体切片逆序，不会像先 `encode('utf-8')` 再反转字节的实现那样产生无效的 UTF-8 序列；空字符串直接返回 `''`，无需任何边界分支，实现最简洁且完全满足「空字符串 + Unicode」两项要求。

**预期结果**（路径 id 省略）：

```
winner: path #0 (切片反转实现)
ranking:
  #0 score=0.9167   #2 score=0.8083   #1 score=0.7250
comparisons: 9
```

`comparisons: 9` 来自 3 个候选的锦标赛：环行扫描 3 对 + 枢轴轮次 6 对（`pivots=2`、N=3 时全部 6 个有向对都涉及枢轴）= 9 次有向判题比较；#0 的 win-mass 均值（0.9167 ≈ 18.33/20）最高，因此胜出。

## 配置

挂载行 `cordis.patch.yml` 中的 `config` 字段可调整默认行为：

| 键 | 默认值 | 说明 |
|---|---|---|
| `nPaths` | `3` | `run_paths` 默认生成的候选路径数（1–16） |
| `pivots` | `2` | 锦标赛枢轴数 k（成本 O(Nk)） |
| `nEvaluations` | `1` | 每个有向比较的重复验证次数（>1 时自动交替 A/B 槽位） |
| `provider` | `deepseek-official` | 验证器调用的 provider 路由 |
| `model` | `deepseek-v4-flash` | 验证器模型 id |
| `criteria` | `Does the candidate solve the task correctly and completely?` | 默认评估准则 |
| `maxCandidateChars` | `12000` | 判题提示词与工具结果中的候选内容截断长度 |

想更换判题模型时，修改 `cordis.patch.yml` 中 `config` 块的 `model`（必要时还有 `provider`）并重启 `dsh web` 即可——见「工具注册与判题模型」。

## 开发

- `pnpm typecheck`：仅类型检查（`tsc --noEmit`）。
- `pnpm build`：编译到 `lib/`。
- 针对 **DeepSeek Harness 源码仓库**（而非 npm 发布的 rc 包）开发时，让本项目的依赖解析指向该仓库的 pnpm 虚拟仓库，保证与运行时使用完全相同的包版本：

  ```bash
  # Windows
  powershell -ExecutionPolicy Bypass -File scripts/link_dev_deps.ps1 -RepoRoot C:\path\to\deepseek-harness
  # macOS / Linux
  scripts/link_dev_deps.sh /path/to/deepseek-harness
  ```

  这会把 `node_modules` 链接为指向仓库 `node_modules/.pnpm/node_modules` 的 junction/symlink。之后想回到 registry 安装时，先删除该链接再执行 `pnpm install`。

## 项目结构

```
dsh-plugin/
├── package.json          # 标准 Node 包；dsh.bundle.patch 使 `dsh plugin add .` 自动激活
├── tsconfig.json         # 类型检查
├── tsconfig.build.json   # pnpm build（输出 lib/）
├── cordis.patch.yml      # 插件挂载行（bundle patch，name: dsh-plugin）
├── src/
│   ├── index.ts          # 插件入口：inject ['tools','llm','subagents','systemPrompt','agents']
│   └── verifier.ts       # 判题提示词 + 环行/枢轴锦标赛 + win-mass 聚合
└── scripts/
    ├── link_dev_deps.ps1 / .sh   # 源码仓库开发模式依赖链接
```

## 工作原理

- **多路径生成**：`run_paths` 通过 `ctx.subagents.start()`（优先 `spawn` provider）并行启动 N 个一次性 subagent，每个 subagent 独立完成任务，其最终回复作为一条候选路径存入进程内注册表。
- **自行选择**：`select_best_path` 通过 `ctx.llm.stream()` 调用判题模型，对候选做两两比较（环行扫描保证每个候选分别出现在 A/B 槽位各一次，枢轴轮次把预算集中到领先候选，总成本 O(Nk)），按 win-mass 均值排序并返回最优路径。
