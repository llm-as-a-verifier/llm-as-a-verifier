#!/usr/bin/env python3
"""
Reproduce the best-of-5 self-verification result.

Runs the tournament on all 5 trials of each task with pivots=2 and K=2
repeated verifications (5 ring + 7 pivot-round pairs per swing task).
Uses its own score cache (`<config cache>_bo5.json`) — never shared with the
best-of-3 run — so the reported token usage is what best-of-5 uses on its
own.

Usage:
    python scripts/run_bo5.py                    # terminal_bench_2.1
    python scripts/run_bo5.py swe_bench
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_verifier.fine_grained_reward import (  # noqa: E402
    MissingAPIKeyError,
    load_dotenv,
)
from run import report, resolve_config, run_benchmark  # noqa: E402

load_dotenv(ROOT_DIR)

LABEL = "bo5"
N_TRIALS = 5
PIVOTS = 2
N_EVALUATIONS = 2


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", nargs="?", default="terminal_bench_2.1",
                        help="Benchmark name (default: terminal_bench_2.1)")
    parser.add_argument("--n-evaluations", type=int, default=N_EVALUATIONS,
                        help=f"repeated verifications K (default: "
                             f"{N_EVALUATIONS})")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = resolve_config(args.benchmark)
    out = run_benchmark(
        cfg,
        pivots=PIVOTS,
        n_evaluations=args.n_evaluations,
        seed=args.seed,
        max_workers=args.max_workers,
        n_trials=N_TRIALS,
        cache_file=cfg.cache.replace(".json", f"_{LABEL}.json"),
        results_file=cfg.results.replace(".txt", f"_{LABEL}.txt"),
    )
    report(cfg, **out)


if __name__ == "__main__":
    try:
        main()
    except MissingAPIKeyError as e:
        print(f"Error: {e}")
        sys.exit(1)
