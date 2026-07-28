<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <img alt="LLM-as-a-Verifier" src="assets/logo.png" width=78%>
  </picture>
</p>

<h3 align="center">
Any modality, Many Applications, One Unified Verification Framework
</h3>

<p align="center">
| <a href="https://llm-as-a-verifier.com/docs/"><b>Documentation</b></a> | <a href="https://llm-as-a-verifier.com"><b> Website</b></a> | <a href="https://arxiv.org/abs/2607.05391"><b>Paper</b></a> | <a href="https://github.com/llm-as-a-verifier/TurboAgent"><b>Claude Code Plugin</b></a> | <a href="https://x.com/jackyk02/status/2042347578139033628"><b>Twitter/X</b></a> | <a href="https://join.slack.com/t/llm-as-a-verifier/shared_invite/zt-3utx6oe8m-86ACBqtPGfsOnpOoMJQwng"><b>Slack</b></a> |
</p>

🔥 LLM-as-a-Verifier achieves SOTA performance across agentic benchmarks, including Terminal-Bench V2, SWE-Bench Verified, MedAgentBench, RoboRewardBench and more. We invite the community to contribute more use cases!


---

## Installation

```bash
pip install llm-verifier
```

To install the latest from a clone:

```bash
pip install -e .
```

---

## About

LLM-as-a-Verifier is a general-purpose framework that provides **fine-grained
feedback** for any agent. The key idea is simple: 1) use fine-grained scoring
granularity, 2) take the expectation over the full logprob distribution of LLM
score tokens, and 3) scale repeated evaluation and criteria decomposition. The
resulting fine-grained feedback can be used for test-time scaling, progress
tracking, and reinforcement learning.

<p align="center">
  <img src="assets/llmoverview.png" alt="LLM-as-a-Verifier overview" width="100%">
</p>

---

## Quickstart

### Simple Best-of-N Selection

Run a first end-to-end selection (requires
`VERTEX_API_KEY` in `.env`, or an OpenAI-compatible server that returns
logprobs — e.g. `vllm serve Qwen/Qwen3.5-9B` with
`OPENAI_BASE_URL=http://localhost:8000/v1`; the served model is
auto-detected):

```python
import llm_verifier

problem = "Write a function that reverses a string."
candidates = [
    "def rev(s): return s[::-1]", "def rev(s): return s", "def rev(s): return ''.join(sorted(s))",
]

result = llm_verifier.select(
    problem=problem,
    candidates=candidates,
    criteria={"Correctness": "Does the code actually reverse the string?"},
)
print(result.index)   # index of the best candidate: 0
print(result.scores)  # candidate scores: [0.73104, 0.38446, 0.38449]
```

### Score a pair of candidates directly

`select` is built on a pairwise reward model. For the raw fine-grained rewards
of a single comparison, call `compare`:

```python
reward_a, reward_b = llm_verifier.compare(
    problem, candidates[0], candidates[1],
    criteria={"Overall": "Does the code solve the problem?"},
)
print(reward_a, reward_b)   # fine-grained rewards in [0, 1]: 0.99994 0
```

### Fine-grained Progress Tracking

The same fine-grained reward can also score an agent's progress after each
step with `track`:

```python
steps = [
    'Read the problem statement',
    'Wrote def rev(s): return s ',
    'Tested: rev("abc") returned "abc"',
    'Changed to def rev(s): return s[::-1]',
    'Tested: rev("abc") returned "cba"',
]

result = llm_verifier.track(problem=problem, steps=steps,
                            checkpoint_steps=[1, 2, 3, 4, 5], n_evaluations=4)
print(result.scores)  # progress after each step: [0.00106, 0.02417, 0.03143, 0.62004, 0.99978]
```

---

## Test-Time Scaling for Agentic Benchmarks

Each benchmark ships with its agent trajectories (`data/`). We use Gemini 2.5
Flash (`gemini-2.5-flash`, the default model) as the verifier for all
benchmark below. Expected results:

| Benchmark | Base Model | Harness | Pass@1 | LLM-as-a-Verifier | Oracle |
|---|---|---|---|---|---|
| Terminal-Bench V2 | GPT-5.5 (Best-of-5) | Capy | 83.1% | **86.5%** | 92.1% |
| SWE-Bench Verified | Opus 4.5 / Opus 4.6 / Gemini 3 Flash (Best-of-3) | mini-swe-agent | 76.1% | **78.2%** | 84.4% |
| MedAgentBench | Claude Opus 4.8 (Best-of-5) | AgentBench | 70.2% | **73.3%** | 75.0% |

### Related web-agent benchmark

For teams evaluating agents on live websites and multi-step browser workflows,
[ClawBench](https://github.com/TIGER-AI-Lab/ClawBench) provides an independent,
reproducible benchmark suite with task-level traces and evaluation artifacts.
It is an external evaluation target and is not included in the result table
above. See the [project site](https://claw-bench.com/) for the benchmark
overview and corpus details.

### Reproduce Results

Run a benchmark by name (`python scripts/run.py` with no argument lists them):

```bash
python scripts/run.py terminal_bench
python scripts/run.py swe_bench
python scripts/run.py medagentbench
```

The tournament defaults can be overridden on the command line:

```bash
python scripts/run.py swe_bench --pivots 2 --n-evaluations 8 --seed 0 --max-workers 50
```

Benchmarks are defined in `llm_verifier/benchmarks.py` — add or tweak one there.

### Select Best of N agent trajectories

Given a task and a pool of agent trajectories, pick the best one in a few
lines of code.

```python
import llm_verifier

problem = "Fix the failing test in utils.py."
candidates = [traj_1, traj_2, traj_3, traj_4, traj_5]

result = llm_verifier.select(
    problem=problem,
    candidates=candidates,
    criteria={"Root cause": "Did the agent fix the real cause?",
              "Verification": "Did the agent confirm the fix?"},
    model="gemini-2.5-flash",          # verifier model
    n_evaluations=4,                 # repeated evaluations per criterion
    pivots=2,                          # pivots < N; reduced verification cost
)

print("Best candidate:", result.index)            
print("Ranking:", result.ranking)                
```

Under the hood, `select` runs the
[Probabilistic Pivot Tournament](#probabilistic-pivot-tournament) to rank all
`N` trajectories using `O(Nk)` pairwise verifications instead of a full
`O(N²)` round-robin. `pivots` trades cost for accuracy: more pivots = more
comparisons = higher accuracy.

### Adapt LLM-as-a-Verifier for your own use case

Use the verifier for your own task in three steps — Claude Code does the rest
(generates the criteria, writes a runner, and selects the best-of-N for you):

1. **Add your data.** Copy your agent trajectories into `data/task_name_trajs/`.
2. **Update naming.** Replace every `task_name` in
   [`add_new_benchmark.md`](add_new_benchmark.md) with the name of your task.
3. **Spin up Claude Code in this repo** (or Codex, or whatever you like — with
   permissions disabled) and paste the contents of `add_new_benchmark.md` to let
   it run.

---

## Progress Tracking for Coding Agents

The same fine-grained reward can score a trajectory *at every step* (see
[`track` in the Quickstart](#fine-grained-progress-tracking)). Below, we track two Terminus-2 runs of the Terminal-Bench task `pytorch-model-cli`. The successful trajectory exhibits consistently increasing verifier scores, whereas the failed trajectory is characterized by erroneous behaviors, resulting in lower scores throughout the execution. Reproduce it with:

```bash
python scripts/terminal_bench_progress.py    # scores both runs then plots
```

<p align="center">
  <img src="assets/progress_pytorch_model_cli.png" alt="Progress curves for two pytorch-model-cli runs" width="100%">
</p>


### Online progress tracking

`track` scores a **finished** trajectory. To monitor an agent **while it
runs**, use `ProgressTracker`: feed it each step as it happens and get a live
progress score back — e.g. to stop a hopeless rollout early or decide when to
resample. Since the verifier only ever sees the steps so far, it cannot peek
at the future.

```python
tracker = llm_verifier.ProgressTracker(problem, n_evaluations=4)

score = tracker.update('Read the problem statement')            # 0.00002
score = tracker.update('Wrote def rev(s): return s')            # 0.00013
score = tracker.update('Changed to def rev(s): return s[::-1]') # 0.73938
score = tracker.update('Tested: rev("abc") returned "cba"')     # 0.98604

if score < 0.05:      # after any step: abandon a hopeless rollout early
    ...
```

Replay the two Terminal-Bench trajectories step-by-step through
`ProgressTracker` — printing a live score bar after every step, as an agent
harness would see it:

```bash
python scripts/terminal_bench_progress.py --online
```

---

## Multi-Modal Support

With a multimodal verifier model (e.g. Gemini 2.5 Flash or
`vllm serve Qwen/Qwen3.5-9B`), every
entry point accepts `images` — a single image (`images="frame.png"`) or a
list of images, each a local file path, an http(s) URL, or raw bytes:

```python
result = llm_verifier.select(problem, candidates, criteria=criteria,
                             images=["before.png", "after.png"])

tracker = llm_verifier.ProgressTracker(problem)
score = tracker.update(step, images="camera_frame.png")  # per-step frame
```

Per-step frames stay part of the trajectory for all later updates, so the
verifier always sees the full visual history — e.g. camera frames while
tracking a robot rollout. See the
[multimodal documentation](https://llm-as-a-verifier.com/docs/multimodal/image_inputs.html) for accepted
input forms, backend notes, and verified examples.

---

## Claude Code Plugin

[TurboAgent](https://github.com/llm-as-a-verifier/TurboAgent) brings
LLM-as-a-Verifier to [Claude Code](https://claude.com/claude-code) as a drop-in
LLM API proxy. It sits between your client and the model provider, generating
multiple candidate responses in parallel and selecting the best one with a
[Probabilistic Pivot Tournament](#probabilistic-pivot-tournament).

```bash
pip install git+https://github.com/llm-as-a-verifier/TurboAgent
```

Point Claude Code at the proxy and run as usual:

```bash
turbo-agent                                        # starts on port 8888
ANTHROPIC_BASE_URL=http://localhost:8888 claude
```

It ships a built-in visualizer at
`http://localhost:8888/visualizer` that shows the pipeline DAG, progress scores, candidate
responses, and the final selection. See the
[TurboAgent repository](https://github.com/llm-as-a-verifier/TurboAgent) for
configuration and setup details.

---

## Directory Structure

```
.
├── scripts/                     # command-line entry points
│   ├── run.py                   #   registry-driven benchmark launcher
│   └── terminal_bench_progress.py  # re-score + plot the progress-tracking example
├── criteria/                    # verifier criteria + ground-truth notes
│   ├── TEMPLATE.md              #   copy this to write your own
│   ├── terminal_bench.md
│   ├── swe_bench.md
│   └── medagentbench.md
├── llm_verifier/                  # the reusable framework (import llm_verifier)
│   ├── __init__.py              #   llm_verifier.select(...) / .compare(...)
│   ├── __main__.py              #   python -m llm_verifier <file.md>: preview criteria
│   ├── benchmarks.py            #   BENCHMARKS registry (one Benchmark / launch)
│   ├── fine_grained_reward.py   #   R(x,τ): Gemini logprob scoring + cache
│   ├── progress.py              #   llm_verifier.track(...): per-step progress curve
│   ├── pivot_tournament.py      #   PPT: O(Nk) selection (Bradley-Terry)
│   ├── prompts.py               #   load criteria/*.md + normalize criteria args
│   └── loaders.py               #   per-benchmark trajectory loaders
├── data/                        # agent trajectories per benchmark
├── cache/                       # verifier score caches (written per run)
└── results/                     # result tables (written after each run)
```

---

## How it works

### Fine-grained Reward Estimation

Rather than reducing each distribution into a single discrete score (as in
LLM-as-a-Judge), LLM-as-a-Verifier approximates the reward of a trajectory
$\tau$ on task $x$ as:

$$
R(x, \tau)
= \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K}
\sum_{g=1}^{G} p_{\theta}(v_g \mid x, c, \tau)\,\phi(v_g)
$$

- $C$ = number of evaluation criteria
- $K$ = number of repeated verifications
- $G$ = number of score tokens (granularity level)
- $p_{\theta}(v_g \mid x, c, \tau)$ = probability assigned by model $\theta$ to score token $v_g$
- $\phi(v_g)$ = maps each scoring token to a scalar value
- $V_{\text{score}} = \{v_1, \ldots, v_G\}$ = ordered set of discrete score tokens

This lives in `llm_verifier/fine_grained_reward.py`.

### Probabilistic Pivot Tournament

<p align="center">
  <img src="assets/pivot_tournament.png" alt="Probabilistic Pivot Tournament" width="100%">
</p>

To pick the best of `N` candidate trajectories, a round-robin tournament scores
all $\binom{N}{2}$ pairs — `O(N²)`. Probabilistic Pivot Tournament (PPT) is a
cost efficient ranking algorithm in which every candidate is compared only
against a small set of pivots, reducing the budget from $\mathcal{O}(N^2)$ to
$\mathcal{O}(Nk)$.

1. **Candidates:** the pool $\{\tau_1,\dots,\tau_N\}$ to be ranked.
2. **Ring pass:** a random Hamiltonian cycle scores the $N$ adjacent pairs so
   every candidate appears once in the "A" slot and once in "B", canceling
   the model's positional bias.
3. **Pivot selection:** candidates are ranked by their ring-pass scores
   $w_{(i)}$, and the top-$k$ candidates form the pivot set $\mathcal{P}$.
4. **Pivot tournament:** every *non-pivot–vs–pivot* and *pivot–vs–pivot* pair
   is scored via the pairwise preference
   $p(a \succ b) = \sigma(R_a - R_b)$, concentrating the budget on uncertain
   top candidates and cutting cost from $\mathcal{O}(N^2)$ to
   $\mathcal{O}(Nk)$.
5. **Selection:** comparisons are aggregated into win mass $w_i$ and count
   $c_i$, and the candidate with the highest normalized $w_i/c_i$ is
   returned.

This lives in `llm_verifier/pivot_tournament.py`.

---

## Prompt Templates

### Pairwise Comparison Prompt

```text
You are an expert [domain] reviewer. You will see a task description and two
trajectories.

Evaluation Criteria: [domain specific criteria]

Task: {task prompt}
Trajectory A: {A}
Trajectory B: {B}

Carefully analyze each trajectory, then provide your final scores:
<score_A> INTEGER_1_TO_20 </score_A>
<score_B> INTEGER_1_TO_20 </score_B>

Rating Rules: Rate correctness on a 1-20 scale based on evaluation criteria
(1 = incorrect, 10 = borderline, 20 = correct)
```

### Progress Tracking Prompt

```text
You are an evaluator of [domain] agent attempts. Trust observed output — NOT the agent's narration.

Task: {task prompt}
Agent trajectory ({N} steps): {trajectory}

You will score the trajectory at {N} checkpoints. Given everything the agent has done up to and including this step, would the agent's CURRENT state already complete the task?

Score each checkpoint INDEPENDENTLY, then output exactly N lines:
<c1> INTEGER_1_TO_20 </c1>
...
<cN> INTEGER_1_TO_20 </cN>

Rating Rules: Rate completion on a 1-20 scale (1 = certainly not complete,
10 = uncertain, 20 = verified complete)
```

> Note: we use a letter-based scale (A-T) instead of digits in the actual
> implementation to enable logprob extraction for granularity scaling.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{kwok2026llmasaverifiergeneralpurposeverificationframework,
      title={LLM-as-a-Verifier: A General-Purpose Verification Framework}, 
      author={Jacky Kwok and Shulu Li and Pranav Atreya and Yuejiang Liu and Yixing Jiang and Chelsea Finn and Marco Pavone and Ion Stoica and Azalia Mirhoseini},
      year={2026},
      eprint={2607.05391},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.05391}, 
}
```
