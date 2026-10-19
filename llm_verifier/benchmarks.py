"""Benchmark registry.

`scripts/run.py <name>` looks the name up in `BENCHMARKS`; CLI flags
(`--pivots`, `--n-evaluations`, `--seed`) override the values here at launch
time. Add a benchmark by adding an entry (and, if needed, a loader in
`loaders.py` plus a criteria file in `criteria/`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Benchmark:
    name: str                       # human-readable title shown in the report
    loader: str                     # key into llm_verifier.loaders.LOADERS
    prompts: str                    # criteria name (criteria/<name>.md at repo root) or a path
    data: dict                      # loader-specific data locations
    cache: str                      # path to the verifier-score cache (JSON)
    results: str                    # path to write the result table
    criteria: list = field(default_factory=list)  # criterion ids, in order
    criteria_weights: dict[str, float] = field(default_factory=dict)
    n_evaluations: int = 4        # repeated verifications K per criterion
    pivots: int = 2                 # number of pivots k in the tournament
    seed: int = 0                   # seed for the random ring pass
    max_workers: int | None = None  # verifier-call concurrency (None: auto)


BENCHMARKS = {
    "terminal_bench": Benchmark(
        name="TERMINAL-BENCH 2.0  (Capy · GPT-5.5, x5)",
        loader="terminal",
        prompts="terminal_bench",
        criteria=["specification", "output_match", "error_signals"],
        cache="cache/cache_terminal_capy_gpt-5.5.json",
        results="results/terminal_bench_capy.txt",
        data={"agent_dir": "data/terminal_bench_2.0_trajs/capy_gpt-5.5"},
    ),
    "terminal_bench_2.1": Benchmark(
        name="TERMINAL-BENCH 2.1  (mini-swe-agent · deepseek-v4-flash, x5)",
        loader="terminal",
        prompts="terminal_bench",
        criteria=["specification", "output_match", "error_signals"],
        cache="cache/cache_terminal_bench_2.1_mswea_deepseek.json",
        results="results/terminal_bench_2.1_mswea_deepseek.txt",
        data={"agent_dir":
              "data/terminal_bench_2.1_trajs/mini-swe-agent_deepseek-v4-flash"},
        n_evaluations=2,
        max_workers=2000,
    ),
    "terminal_bench_4.0": Benchmark(
        name="TERMINAL-BENCH 4.0  (Claude Code · Opus 5.5, x5, "
             "verified by deepseek-flash)",
        loader="terminal",
        prompts="terminalbench4",
        criteria=["specification", "evidence", "completion"],
        cache="cache/cache_terminal_bench_4.0_cc_opus55_deepseek.json",
        results="results/terminal_bench_4.0_cc_opus55_deepseek.txt",
        data={"agent_dir":
              "data/terminal_bench_4.0_trajs/claude-code_opus-5.5"},
        n_evaluations=2,
        max_workers=2000,
    ),
    "swe_bench": Benchmark(
        name="SWE-BENCH VERIFIED  (mini-swe-agent, x3)",
        loader="swe",
        prompts="swe_bench",
        criteria=["root_cause", "code_review", "verification"],
        cache="cache/cache_swebench.json",
        results="results/swe_bench.txt",
        # runs: defaults to every run directory found under trajs_dir
        data={"trajs_dir": "data/swebench_verified_trajs"},
    ),
    "swe_bench_logic": Benchmark(
        name="SWE-BENCH VERIFIED  (logic-risk ablation, x3)",
        loader="swe",
        prompts="swe_bench_logic",
        criteria=[
            "root_cause",
            "logic_l1",
            "logic_l2",
            "logic_l3",
            "logic_l4",
            "logic_l5",
            "logic_l6",
            "verification",
        ],
        criteria_weights={
            "root_cause": 1.0,
            "logic_l1": 1.0 / 6.0,
            "logic_l2": 1.0 / 6.0,
            "logic_l3": 1.0 / 6.0,
            "logic_l4": 1.0 / 6.0,
            "logic_l5": 1.0 / 6.0,
            "logic_l6": 1.0 / 6.0,
            "verification": 1.0,
        },
        cache="cache/cache_swebench_logic.json",
        results="results/swe_bench_logic.txt",
        data={"trajs_dir": "data/swebench_verified_trajs"},
    ),
    "medagentbench": Benchmark(
        name="MEDAGENTBENCH  (Opus 4.8 max-effort, x5)",
        loader="med",
        prompts="medagentbench",
        criteria=["query", "consistency", "structure"],
        cache="cache/cache_medagentbench_opus48max.json",
        results="results/medagentbench.txt",
        data={
            "test_data": "data/medagentbench_trajs/problems.json",
            "output_dir": "data/medagentbench_trajs/trajs/opus-4.8-max",
            "run_names": ["run1", "run2", "run3", "run4", "run5"],
        },
    ),
}
