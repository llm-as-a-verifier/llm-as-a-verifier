# LLM-as-a-Verifier: A General-Purpose Verification Framework

LLM-as-a-Verifier is a general-purpose verification framework that provides fine-grained feedback by scaling scoring granularity, repeated verification, and criteria decompositions. It achieves state-of-the-art performance on Terminal-Bench 2 (86.4%) and SWE-Bench Verified (77.8%) when used as a trajectory reward model for test-time scaling. This repository provides the accompanying code for reproducing the results.

## Setup

```bash
pip install google-genai tqdm
```

Create a `.env` file with your Vertex AI API key (required for logprob extraction):

```bash
echo "VERTEX_API_KEY=your_key_here" > .env
```

## Directory Structure

```
.
  README.md
  .env                          # API key (create this)
  scripts/
    verifier_core.py            # Gemini setup + scoring
    run_terminal_bench.py       # Terminal-Bench Evaluation
    run_swe_bench.py            # SWE-bench Verified Evaluation
  data/
    terminal_trajs/             # 5 trajectories x 89 tasks each for Terminal-Bench 2.0
    swebench_verified_trajs/    # 3 trajectories x 500 tasks each for SWE-bench Verified
  cache/                        # Cached API results (created on first run)
    cache_terminal_<agent>.json
    cache_swebench.json
  results/                      # Final result tables (written after each run)
    terminal_<agent>.txt
    swebench_verified.txt
```

## Trajectories

`data/terminal_trajs/forge_gpt54/` contains the Forge + GPT-5.4
submission downloaded from the
[Terminal-Bench 2 Leaderboard](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard/tree/main/submissions/terminal-bench/2.0),
with 5 trajectories per task across 89 tasks:

| Scaffold | Base Model | Pass@1 |
|---|---|---|
| Forge | GPT-5.4 | 81.8% |

`data/swebench_verified_trajs/` contains 3 runs for SWE-bench
Verified (500 instances each) downloaded from the
[SWE-bench Leaderboard](https://github.com/swe-bench/experiments?tab=readme-ov-file#):

| Scaffold | Base Model | Pass@1 |
|---|---|---|
| mini-swe-agent | Claude-Opus-4.5 (high reasoning) | 76.8% |
| mini-swe-agent | Claude-Opus-4.6 | 75.6% |
| mini-swe-agent | Gemini-3-Flash (high reasoning) | 75.8% |

## Evaluating LLM-as-a-Verifier 

### Terminal-Bench

```bash
python scripts/run_terminal_bench.py --granularity 20 --n-verifications 4 --criteria 3
```

Expected:

| Method | Score | Rate |
|---|---|---|
| Pass@1 | 72.8/89 | 81.8% |
| LLM-as-a-Verifier | 76.9±0.3/89 | **86.4%** |
| Oracle (Bo5) | 80/89 | 89.9% |

### SWE-bench Verified

```bash
python scripts/run_swe_bench.py --granularity 20 --n-verifications 4 --criteria 3
```

Expected:

| Method | Score | Rate |
|---|---|---|
| Pass@1 | 380.3/500 | 76.1% |
| LLM-as-a-Verifier | 389.0±0.4/500 | **77.8%** |
| Oracle (Bo3) | 422/500 | 84.4% |

## How it works

Rather than reducing each distribution into a single discrete score (as in LLM-as-a-Judge), **LLM-as-a-Verifier** approximate the reward of
a trajectory $\tau$ on task $t$ as:

$$
R(t, \tau)
= \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K}
\sum_{g=1}^{G} p_{\theta}(v_g \mid t, c, \tau)\,\phi(v_g)
$$

**Where:**

- $C$ = number of evaluation criteria
- $K$ = number of repeated verifications
- $G$ = number of score tokens (granularity level)
- $p_{\theta}(v_g \mid t, c, \tau)$ = probability assigned by model $\theta$ to score token $v_g$
- $\phi(v_g)$ = maps each scoring token to a scalar value
- $V_{\text{score}} = \{v_1, \ldots, v_G\}$ = ordered set of discrete score tokens

To pick the best trajectory among $N$
candidates for a given task, we run a round-robin tournament. For every
pair $(i, j)$ the verifier produces $R(t, \tau_i)$ and $R(t, \tau_j)$
using the formula above. The trajectory with the higher reward
receives a win and the trajectory with the most
wins across all $\binom{N}{2}$ pairs is selected.
