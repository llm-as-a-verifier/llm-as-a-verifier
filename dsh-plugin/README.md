# dsh-plugin — LLM-as-a-Verifier Multipath Plugin (DeepSeek Harness)

> **English** | [**中文版 (Chinese README)**](README_ZH.md)

This plugin gives DeepSeek Harness agents the ability to **"generate multiple execution paths → self-verify and select the best path"**:

1. **`run_paths`**: Generate N independent candidate execution paths for a task in parallel (each path is completed independently by a fresh subagent session);
2. **`submit_path`**: Register a solution produced by the agent itself as a candidate path;
3. **`list_paths`**: Inspect the candidate paths registered for a given run;
4. **`select_best_path`**: Rank the candidate paths with an **LLM-as-a-Judge pairwise comparison tournament** (ring scan + pivot rounds, A/B slot alternation to cancel position bias, win-mass aggregation — corresponding to the Probabilistic Pivot Tournament of the `llm-as-a-verifier` project) and return the best path.

The plugin also injects a multipath verification workflow note into the system prompt (order 150), guiding the model to first generate multiple paths and then verify and select for tasks that are open-ended or whose correctness is hard to judge. The four tools are registered through the `tools` inject hook, so they automatically appear in the model's available tool list once the plugin is loaded — the model may therefore use them proactively even without an explicit instruction, though an explicit instruction is the most reliable trigger (see [Tool Registration & Judge Model](#tool-registration--judge-model)).

The plugin is a **pure Node.js (TypeScript) implementation** — the verifier logic (tournament, scoring, aggregation) is fully rewritten in TypeScript and runs inside the Harness process, with **zero Python dependency** at runtime (see [Requirements](#requirements)).

## Requirements

- Node.js ≥ 22.19 (or ≥ 24)
- pnpm (`npm i -g pnpm`)
- DeepSeek API Key (configured in `dsh web`, or via the `DEEPSEEK_API_KEY` environment variable)

> **No Python environment needed.** This plugin is a self-contained TypeScript implementation: `run_paths` and `select_best_path` run entirely inside the Harness (Node.js) process — candidate paths are produced by in-process subagent sessions, and the judge tournament calls the verifier LLM directly through the Harness's `ctx.llm` service. The plugin never shells out to Python and does **not** require `pip install llm-verifier`: the Python-side `llm-as-a-verifier` project is only the reference implementation of the tournament algorithm, not a runtime dependency. A fresh environment needs nothing more than Node.js, pnpm, and a DeepSeek API Key.

## Installing from Source

Run the following in the project root (`dsh-plugin/`):

```bash
# 1. Resolve dependencies (choose one)
pnpm install                        # registry install (works when the rc packages on npm have complete dependencies)
scripts/link_dev_deps.sh <path-to-deepseek-harness-source>   # or link to the source repo (recommended, verified to work)

# 2. Build
pnpm build

# 3. Install into the web profile
dsh plugin --profile web add .

# 4. Restart deepseek harness
# dsh web
```

Notes:

| Step | Purpose |
|---|---|
| Dependency resolution | The plugin's runtime dependencies (`@deepseek-ai/cordis`, `dsh-*`) are provided by the host Harness (peer dependencies); this directory only needs the build toolchain. Some rc packages on npm depend on unpublished internal packages (e.g. `@deepseek-ai/dsh-type-meta`), so a plain `pnpm install` fails with 404 — in that case use `scripts/link_dev_deps.*` to link `node_modules` to the pnpm virtual store of the deepseek-harness source repo (exactly matching the version you run locally; verified on this machine) |
| `pnpm build` | Compiles `src/` to `lib/` with `tsc` (`lib/index.js` is the plugin entry); `pnpm-workspace.yaml` sets `verifyDepsBeforeRun: false` for compatibility with dev-link mode |
| `dsh plugin --profile web add .` | Installs this plugin in the `web` profile: `dsh` forwards `pnpm add <absolute path to this dir>` to be run under `~/.dsh/profiles/web`; after installation, since `package.json` declares `dsh.bundle.patch`, the plugin is automatically added to that profile's `dsh.profile.bundles` layer |
| Restart `dsh web` | After restart, the mount line in `cordis.patch.yml` (`name: dsh-plugin`) takes effect and all four tools become available |

You can verify a successful installation like this:

```bash
dsh --profile web --dump-config | grep -A2 llm-verifier-multipath
# - id: llm-verifier-multipath
#   name: dsh-plugin
```

After startup (`dsh web`, visit http://127.0.0.1:3080), open a new session and ask the model to do multipath verification, for example:

> Implement a `reverse_string` function and verify it thoroughly. First use `run_paths` to generate 3 independent implementation paths, then use `select_best_path` to pick the best solution, and finally explain why it won.

## Usage Flow

```
run_paths(task, n) ──► submit_path(content) ──► list_paths(runId) ──► select_best_path(runId)
      │                                          │                         │
      └── parallel subagent paths ───────────────┴──► pairwise tournament ──┴──► best path
```

- `run_paths`: `{ task, n? }`, returns `{ runId, paths: [{ id, content, stopReason }] }`.
- `submit_path`: `{ content, runId?, task? }`, returns `{ runId, pathId, nPaths }`.
- `list_paths`: `{ runId }`, returns all candidates (content truncated for display).
- `select_best_path`: `{ runId, criteria?, nEvaluations?, pivots? }`, returns `{ winner, ranking, nComparisons, criteria, usage }`.

The comparison prompt matches the pairwise template of the `llm-as-a-verifier` project (`<score_A>` / `<score_B>`, integer scores 1–20, normalized and aggregated by win-mass); with `nEvaluations > 1`, repeated comparisons alternate the A/B slots to cancel position bias. A single failed comparison is treated as a tie (0.5/0.5), consistent with `on_error="tie"` on the Python side.

## Tool Registration & Judge Model

### Automatic tool registration

The four tools (`run_paths` / `submit_path` / `list_paths` / `select_best_path`) are registered through the `tools` inject hook, so once the plugin is loaded they automatically appear in the model's available tool list — no manual enabling or per-session configuration is required. Since the plugin also injects the multipath verification workflow note into the system prompt (order 150), the model may proactively use these tools on open-ended tasks or tasks whose correctness is hard to judge, **even without an explicit instruction**. Still, an explicit instruction is the most reliable way to trigger the workflow (see the [Worked Example](#worked-example)).

### Changing the judge model

The verifier (judge) model defaults to `deepseek-v4-flash` (provider `deepseek-official`). To switch to another model, edit the `model` (and, if needed, `provider`) keys in the `config` block of `dsh-plugin/cordis.patch.yml`, then restart `dsh web`:

```yaml
- insert:
    - id: llm-verifier-multipath
      name: dsh-plugin
      config:
        # ... other settings ...
        provider: deepseek-official
        model: deepseek-v4-flash   # <- change the judge model here
```

After the restart, every subsequent `select_best_path` call uses the new judge model.

## Worked Example

Take "implement a Python function `reverse_string(s: str) -> str` that correctly handles empty strings and Unicode characters" as the task. A complete multipath verification session looks like this — do not skip any tool call:

**Step 1 — call `run_paths` to generate 3 independent candidate paths:**

```json
run_paths({
  "task": "Implement a Python function reverse_string(s: str) -> str that correctly handles empty strings and Unicode characters.",
  "n": 3
})
```

It returns a `runId` plus 3 candidate paths (e.g. slice reversal, `reversed()` + join, `reduce` accumulation).

**Step 2 — call `select_best_path` with the `runId` returned in step 1:**

```json
select_best_path({
  "runId": "<the runId returned by run_paths>"
})
```

**Step 3 — summarize the winner and why it won**, for example:

> The winner is path #0, the slice-reversal implementation: `return s[::-1]`. It reverses by slicing over code points as a whole, so — unlike implementations that `encode('utf-8')` first and then reverse bytes — it can never produce invalid UTF-8 sequences; the empty string returns `''` directly with no edge-case branches. It is the most concise implementation and fully satisfies both requirements (empty string + Unicode).

**Expected result** (path ids omitted for readability):

```
winner: path #0 (slice-reversal implementation)
ranking:
  #0 score=0.9167   #2 score=0.8083   #1 score=0.7250
comparisons: 9
```

`comparisons: 9` comes from the tournament over 3 candidates: 3 ring-scan pairs + 6 pivot-round pairs (with `pivots=2` and N=3, every ordered pair involves a pivot) = 9 directed judge comparisons. Path #0 has the highest win-mass mean (0.9167 ≈ 18.33/20) and therefore wins.

## Configuration

The `config` field of the mount line in `cordis.patch.yml` adjusts default behavior:

| Key | Default | Description |
|---|---|---|
| `nPaths` | `3` | Default number of candidate paths generated by `run_paths` (1–16) |
| `pivots` | `2` | Number of tournament pivots k (cost O(Nk)) |
| `nEvaluations` | `1` | Repeat verifications per directed comparison (>1 alternates A/B slots automatically) |
| `provider` | `deepseek-official` | Provider route used by the verifier |
| `model` | `deepseek-v4-flash` | Verifier model id |
| `criteria` | `Does the candidate solve the task correctly and completely?` | Default evaluation criteria |
| `maxCandidateChars` | `12000` | Truncation length for candidate content in judge prompts and tool results |

To change the judge model, edit the `model` (and, if needed, `provider`) keys in the `config` block of `cordis.patch.yml` and restart `dsh web` — see [Tool Registration & Judge Model](#tool-registration--judge-model).

## Development

- `pnpm typecheck`: type-check only (`tsc --noEmit`).
- `pnpm build`: compile to `lib/`.
- When developing against the **DeepSeek Harness source repo** (instead of the rc packages published on npm), point this project's dependency resolution at that repo's pnpm virtual store so the exact same package versions as the runtime are used:

  ```bash
  # Windows
  powershell -ExecutionPolicy Bypass -File scripts/link_dev_deps.ps1 -RepoRoot C:\path\to\deepseek-harness
  # macOS / Linux
  scripts/link_dev_deps.sh /path/to/deepseek-harness
  ```

  This links `node_modules` as a junction/symlink pointing to the repo's `node_modules/.pnpm/node_modules`. To return to registry installs later, delete the link first and then run `pnpm install`.

## Project Structure

```
dsh-plugin/
├── package.json          # Standard Node package; dsh.bundle.patch makes `dsh plugin add .` activate automatically
├── tsconfig.json         # Type checking
├── tsconfig.build.json   # pnpm build (outputs lib/)
├── cordis.patch.yml      # Plugin mount line (bundle patch, name: dsh-plugin)
├── src/
│   ├── index.ts          # Plugin entry: inject ['tools','llm','subagents','systemPrompt','agents']
│   └── verifier.ts       # Judge prompt + ring/pivot tournament + win-mass aggregation
└── scripts/
    ├── link_dev_deps.ps1 / .sh   # Dependency linking for source-repo development mode
```

## How It Works

- **Multipath generation**: `run_paths` starts N one-shot subagents in parallel via `ctx.subagents.start()` (preferring the `spawn` provider); each subagent completes the task independently, and its final reply is stored as a candidate path in an in-process registry.
- **Self-selection**: `select_best_path` calls the judge model via `ctx.llm.stream()` and compares candidates pairwise (ring scan ensures each candidate appears once in each of the A/B slots; pivot rounds concentrate the budget on the leading candidates, for a total cost of O(Nk)); it ranks by win-mass mean and returns the best path.
