#!/usr/bin/env python3
"""
Run LLM-as-a-Verifier on a benchmark from llm_verifier/benchmarks.py.

Loads the trajectories and criteria, runs a Probabilistic Pivot Tournament
on every "swing" task (where the N trials disagree), and reports Pass@1 vs
LLM-as-a-Verifier vs Oracle. Scoring is two-phase because the pivots depend
on the ring-pass results: score all ring pairs first, then the pivot rounds.

Usage:
    python scripts/run.py                  # list available benchmarks
    python scripts/run.py terminal_bench
    python scripts/run.py swe_bench --pivots 2 --n-evaluations 8
"""

import argparse
import os
import random
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from llm_verifier.benchmarks import BENCHMARKS
from llm_verifier.fine_grained_reward import (
    GRANULARITY,
    USAGE,
    LazyClient,
    MissingAPIKeyError,
    default_max_workers,
    directed_reward,
    format_usage,
    load_dotenv,
    score_directed_pairs,
)
from llm_verifier.loaders import LOADERS
from llm_verifier.prompts import load_prompts, select_criteria
from llm_verifier import pivot_tournament as ppt

load_dotenv(ROOT_DIR)


def classify(tasks):
    """Split tasks into all-pass (every trial succeeds) and swing (trials
    disagree). All-fail tasks are unwinnable and need no verification."""
    all_pass, swing = [], []
    for name, trials in sorted(tasks.items()):
        rewards = [t["reward"] for t in trials]
        if all(r == 1 for r in rewards):
            all_pass.append(name)
        elif not all(r == 0 for r in rewards):
            swing.append(name)
    return all_pass, swing


def resolve_config(benchmark):
    """Look a benchmark name up in the registry. Lists the available benchmarks
    and exits if the argument is missing or unknown."""
    if benchmark in BENCHMARKS:
        return BENCHMARKS[benchmark]
    if benchmark:
        print(f"Unknown benchmark: {benchmark!r}\n")
    else:
        print("Usage: python scripts/run.py <benchmark>\n")
    print("Available benchmarks:")
    for name in BENCHMARKS:
        print(f"  python scripts/run.py {name}")
    sys.exit(0 if not benchmark else 2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", nargs="?",
                        help="Benchmark name (e.g. terminal_bench). Omit to "
                             "list available benchmarks.")
    parser.add_argument("--pivots", type=int, default=None,
                        help="Override config pivots k (default 2)")
    parser.add_argument("--n-evaluations", type=int, default=None,
                        help="Override config repeated verifications K")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed for the random ring pass")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="verifier-call concurrency (default: 500 on "
                             "DeepSeek, 50 otherwise)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="use only the first N trials of each task "
                             "(best-of-N; default: every trial on disk)")
    args = parser.parse_args()

    cfg = resolve_config(args.benchmark)
    result = run_benchmark(
        cfg,
        pivots=args.pivots,
        n_evaluations=args.n_evaluations,
        seed=args.seed,
        max_workers=args.max_workers,
        n_trials=args.n_trials,
    )
    report(cfg, **result)


def run_benchmark(cfg, pivots=None, n_evaluations=None, seed=None,
                  max_workers=None, n_trials=None, cache_file=None,
                  results_file=None):
    """Score one benchmark end to end and return the metrics.

    Every argument except `cfg` overrides the benchmark's own setting, so a
    caller can pin a configuration (see scripts/run_bo3.py / run_bo5.py).
    `n_trials` truncates each task to its first N trials (best-of-5 data ->
    best-of-3 run). Returns the keyword arguments `report` expects.
    """
    k = pivots if pivots is not None else cfg.pivots
    n_reps = (n_evaluations if n_evaluations is not None
              else cfg.n_evaluations)
    seed = seed if seed is not None else cfg.seed
    # concurrency: explicit argument > benchmark config > backend default
    if max_workers is None:
        max_workers = (cfg.max_workers if cfg.max_workers is not None
                       else default_max_workers())

    # ---- criteria + ground-truth note ----
    note, all_criteria = load_prompts(cfg.prompts)
    criteria = select_criteria(all_criteria, cfg.criteria)
    criteria_ids = [c["id"] for c in criteria]

    # ---- data ----
    print(f"Loading {cfg.name} ...")
    tasks, n_runs = LOADERS[cfg.loader](cfg.data, ROOT_DIR)
    if n_trials:
        # First N trials per task (stable: the loader sorts by file name).
        tasks = {name: trials[:n_trials] for name, trials in tasks.items()}
        n_runs = min(n_runs, n_trials)
    n_tasks = len(tasks)
    all_pass, swing = classify(tasks)
    print(f"  tasks={n_tasks}  all-pass={len(all_pass)}  swing={len(swing)}  "
          f"N(trials)={n_runs}")
    print(f"  criteria={criteria_ids}  K={n_reps}  pivots={k}  seed={seed}  "
          f"max_workers={max_workers}")

    cache_file = _abs(cache_file or cfg.cache)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    lazy = LazyClient()
    usage_before = USAGE.copy()

    def score_fn(scores):
        def directed(a, b, t):
            return directed_reward(scores, t, a, b, criteria_ids, n_reps)
        return directed

    # ---- step 1: sample one ring per swing task (deterministic given seed) ----
    rng = random.Random(seed)
    rings = {task: ppt.ring_cycle(len(tasks[task]), rng)
             for task in swing}

    # ---- phase A: score all ring pairs ----
    print("Phase A: ring pass")
    scores = score_directed_pairs(
        lazy, tasks, rings, criteria, note, n_reps, max_workers,
        cache_file)
    directed = score_fn(scores)

    # ---- step 2: pivots = ring-pass leaders; collect pivot-round pairs ----
    pivots_by_task = {}
    pr_pairs = {}
    for task in swing:
        n = len(tasks[task])
        w, c = [0.0] * n, [0] * n
        ppt.accumulate(rings[task], lambda a, b, t=task: directed(a, b, t),
                       w, c)
        pivots = ppt.select_pivots(w, c, k)
        pivots_by_task[task] = pivots
        pr_pairs[task] = ppt.pivot_round_pairs(n, pivots)

    # ---- phase B: score all pivot-round pairs ----
    print("Phase B: pivot rounds")
    scores = score_directed_pairs(
        lazy, tasks, pr_pairs, criteria, note, n_reps, max_workers,
        cache_file)
    directed = score_fn(scores)

    # ---- step 3+4: aggregate ring + pivot rounds, select winner ----
    selected = 0
    total_comparisons = 0
    for task in swing:
        n = len(tasks[task])
        w, c = [0.0] * n, [0] * n
        ppt.accumulate(rings[task], lambda a, b, t=task: directed(a, b, t),
                       w, c)
        ppt.accumulate(pr_pairs[task], lambda a, b, t=task: directed(a, b, t),
                       w, c)
        best = max(range(n), key=lambda i: (w[i] / c[i] if c[i] else 0.0, -i))
        total_comparisons += len(rings[task]) + len(pr_pairs[task])
        if tasks[task][best]["reward"] == 1:
            selected += 1

    # ---- metrics ----
    pass1 = len(all_pass) + sum(
        sum(t["reward"] for t in tasks[tn]) / len(tasks[tn]) for tn in swing)
    verifier = len(all_pass) + selected
    oracle = len(all_pass) + len(swing)
    avg_cmp = total_comparisons / max(1, len(swing))

    return dict(
        n_tasks=n_tasks, n_runs=n_runs, n_swing=len(swing),
        criteria_ids=criteria_ids, n_reps=n_reps, k=k, seed=seed,
        pass1=pass1, verifier=verifier, oracle=oracle, avg_cmp=avg_cmp,
        usage=USAGE - usage_before,  # this call's usage only
        comparisons=total_comparisons, results_file=results_file,
    )


def report(cfg, n_tasks, n_runs, n_swing, criteria_ids, n_reps, k, seed,
           pass1, verifier, oracle, avg_cmp, usage=USAGE, comparisons=None,
           results_file=None):
    lines = [
        "",
        "=" * 72,
        cfg.name,
        f"  g{GRANULARITY}  criteria={criteria_ids}  K={n_reps}  "
        f"pivots={k}  seed={seed}",
        f"  tasks={n_tasks}  swing={n_swing}  N(trials)={n_runs}  "
        f"comparisons/task={avg_cmp:.1f}",
        "=" * 72,
        f"{'Method':<26s}  {'Score':>14s}  {'Rate':>7s}",
        "-" * 72,
        f"{'Pass@1':<26s}  {pass1:>8.2f}/{n_tasks}  "
        f"{100 * pass1 / n_tasks:>6.1f}%",
        f"{'LLM-as-a-Verifier':<26s}  {verifier:>8d}/{n_tasks}  "
        f"{100 * verifier / n_tasks:>6.1f}%",
        f"{'Oracle (Bo' + str(n_runs) + ')':<26s}  {oracle:>8d}/{n_tasks}  "
        f"{100 * oracle / n_tasks:>6.1f}%",
        "-" * 72,
        # Tokens used by THIS run; cached comparisons use none.
        *format_usage(usage),
        "",
    ]
    print("\n".join(lines))
    results_file = _abs(results_file or cfg.results)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Results saved: {results_file}")


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(ROOT_DIR, path)


if __name__ == "__main__":
    try:
        main()
    except MissingAPIKeyError as e:
        print(f"Error: {e}")
        sys.exit(1)
