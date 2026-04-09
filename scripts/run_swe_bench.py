#!/usr/bin/env python3
"""
SWE-bench Verified via LLM-as-a-Verifier TRM Selection.

Loads trajectories from data/swebench_verified_trajs/, scores pairwise
comparisons across runs with Gemini 2.5 Flash,runs a round-robin tournament 
per instance to select the best trial, and reports success rate vs Pass@1 and Oracle.

Data format: data/swebench_verified_trajs/<run>/data_cache.json
  Each item: {instance_id, reward, problem_statement, messages (JSON string), num_steps}

Usage:
    python scripts/run_swe_bench.py
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TRAJ_DIR = os.path.join(ROOT_DIR, "data", "swebench_verified_trajs")

sys.path.insert(0, SCRIPT_DIR)
from verifier_core import (
    create_gemini_client,
    evaluate_and_print,
    load_dotenv,
    score_all_trials,
)

load_dotenv(ROOT_DIR)


# ---------------------------------------------------------------------------
# SWE-bench-specific criteria + ground-truth note
# ---------------------------------------------------------------------------

GROUND_TRUTH_NOTE = (
    "**Do NOT trust the agent's self-assessment or claims that \"the patch looks correct\". "
    "Agents routinely declare success on patches that fix the wrong file, "
    "address only a symptom, or are subtly broken."
)

CRITERIA = [
    {
        "id": "root_cause",
        "name": "Root Cause Analysis",
        "description": (
            "Read the issue, identify the buggy behavior it describes, "
            "and trace it to the code that produces it. Decide whether "
            "the patch modifies the actual code path responsible for "
            "the bug, or only its symptoms. A patch that edits the "
            "buggy function or branch should score HIGH; a patch that "
            "catches the bad output downstream, special-cases the "
            "literal example in the issue, edits a caller to work "
            "around a buggy callee, or changes a default to dodge the "
            "broken path should score LOW. Judge by WHERE the change "
            "lands in the call stack — both small and larger fixes are "
            "valid as long as the edited lines are the ones whose "
            "behavior the issue actually depends on."
        ),
    },
    {
        "id": "code_review",
        "name": "Code Quality",
        "description": (
            "Review the agent's final patch (`diff --git ...`) as an "
            "experienced code reviewer would. Check syntactic validity, "
            "semantic correctness (right API, right types, right control "
            "flow, no off-by-one, no swapped arguments, no shadowed or "
            "unbound names), preservation of existing contracts "
            "(function signatures, return types, exception types and "
            "messages, output formats, default behavior), and "
            "consistency with surrounding code style. Pay attention to "
            "silent regressions in code paths the issue did not "
            "explicitly mention — these are the most common cause of a "
            "patch that looks fine but breaks something else. Judge "
            "the diff on its technical merits, not by length or "
            "apparent effort."
        ),
    },
    {
        "id": "verification",
        "name": "Empirical Verification",
        "description": (
            "Look at the commands the agent actually ran and what they "
            "printed, not what the agent claimed in its narration. "
            "Reward agents that (a) constructed a reproducer for the "
            "failure described in the issue, (b) observed the failure "
            "before applying the fix, (c) observed the expected correct "
            "behavior after the fix, and (d) ran the existing tests in "
            "the affected module without breaking them. Trust observed "
            "command output over the agent's narration of it. Penalize "
            "agents that declared success without running anything, "
            "misread their own command output (e.g. compared a literal "
            "string to itself, ignored a traceback, claimed a test "
            "passed when it errored), or edited the code again after "
            "the last successful verification step so the final patch "
            "is untested."
        ),
    },
]


# ---------------------------------------------------------------------------
# SWE-bench data loading
# ---------------------------------------------------------------------------

PR_DESC_RE = re.compile(r"<pr_description>(.*?)</pr_description>", re.DOTALL)
INSTRUCTIONS_RE = re.compile(r"<instructions>(.*?)</instructions>", re.DOTALL)


def extract_problem_from_messages(messages_str):
    """Extract the problem statement from the first user message by pulling
    the contents of the <pr_description> and <instructions> blocks."""
    try:
        messages = json.loads(messages_str)
    except (json.JSONDecodeError, TypeError):
        return ""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        c = msg.get("content", "")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            texts = []
            for x in c:
                if isinstance(x, dict) and x.get("type") == "text":
                    texts.append(x.get("text", ""))
                elif isinstance(x, str):
                    texts.append(x)
            text = "\n".join(texts)
        else:
            text = ""

        pr_match = PR_DESC_RE.search(text)
        instr_match = INSTRUCTIONS_RE.search(text)
        parts = []
        if pr_match:
            parts.append(
                f"<pr_description>\n{pr_match.group(1).strip()}\n</pr_description>"
            )
        if instr_match:
            parts.append(
                f"<instructions>\n{instr_match.group(1).strip()}\n</instructions>"
            )
        if parts:
            return "\n\n".join(parts)
        return text
    return ""


def _strip_problem_blocks(text):
    """Remove <pr_description>...</pr_description> and
    <instructions>...</instructions> blocks from a piece of text. These are
    the task-setup blocks already captured in the problem statement, so we
    don't want them duplicated inside the trace."""
    if not isinstance(text, str) or not text:
        return text
    text = PR_DESC_RE.sub("", text)
    text = INSTRUCTIONS_RE.sub("", text)
    return text.strip()


def format_swebench_trace(messages_str):
    """Format SWE-bench message log into readable text. Skips the initial
    PR description (extracted separately as the problem) and shows the
    agent's actions plus tool outputs."""
    try:
        messages = json.loads(messages_str)
    except (json.JSONDecodeError, TypeError):
        return "(no trajectory data)"

    parts = []
    step = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("system", "user") and step == 0:
            continue

        if role == "assistant":
            step += 1
            parts.append(f"--- Agent Step {step} ---")
            if isinstance(content, str) and content:
                content = _strip_problem_blocks(content)
                if not content:
                    parts.append("")
                    continue
                if len(content) > 2000:
                    parts.append(content[:2000] + "\n... (truncated)")
                else:
                    parts.append(content)
            parts.append("")

        elif role in ("tool", "user") and step > 0:
            if isinstance(content, str) and content:
                content = _strip_problem_blocks(content)
                if not content:
                    continue
                if len(content) > 2000:
                    parts.append(f"[Output]\n{content[:2000]}\n... (truncated)")
                else:
                    parts.append(f"[Output]\n{content}")

    return "\n".join(parts)


def load_swebench_tasks(trajs=None):
    """Load SWE-bench tasks across all runs."""
    if trajs:
        runs = sorted(trajs)
    else:
        runs = sorted(d for d in os.listdir(TRAJ_DIR)
                      if os.path.isdir(os.path.join(TRAJ_DIR, d)))

    # Load all runs (support both single data_cache.json and split data_cache{0,1,2,...}.json)
    run_data = {}
    for run in runs:
        single = os.path.join(TRAJ_DIR, run, "data_cache.json")
        if os.path.exists(single):
            run_data[run] = {
                item["instance_id"]: item
                for item in json.load(open(single))
            }
        else:
            items = []
            idx = 0
            while True:
                part = os.path.join(TRAJ_DIR, run, f"data_cache{idx}.json")
                if not os.path.exists(part):
                    break
                items.extend(json.load(open(part)))
                idx += 1
            if items:
                run_data[run] = {
                    item["instance_id"]: item for item in items
                }

    # Get all instances present in at least 2 runs
    id_counts = Counter()
    for data in run_data.values():
        id_counts.update(data.keys())
    all_ids = sorted(iid for iid, cnt in id_counts.items() if cnt >= 2)

    tasks = {}
    for iid in all_ids:
        trials = []
        for run in runs:
            if iid not in run_data[run]:
                continue
            item = run_data[run][iid]
            messages_str = item.get("messages", "[]")
            problem = extract_problem_from_messages(messages_str)
            if not problem:
                problem = f"(Instance: {iid})"
            trace = format_swebench_trace(messages_str)
            patch = (item.get("output_patch") or "").strip()
            patch_block = (
                f"\n\n--- Final Code Patch ---\n{patch}"
                if patch else "\n\n--- Final Code Patch ---\n(no patch produced)"
            )
            trace = trace + patch_block
            reward = 1 if item.get("reward", 0) == 1.0 else 0

            trials.append({
                "trial_name": f"{run}_{iid}",
                "reward": reward,
                "problem": problem,
                "trace": trace,
            })

        tasks[iid] = trials

    return tasks, runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWE-bench Verified downstream success rate via "
                    "LLM-as-a-Verifier TRM selection")
    parser.add_argument("--trajs", nargs="+", default=None,
                        help="Run directory names in data/swebench_verified_trajs/ "
                             "(default: all)")
    parser.add_argument("--granularity", type=int, default=20,
                        help="Score-token granularity g (only g=20 supported)")
    parser.add_argument("--n-verifications", type=int, default=4,
                        help="Independent verifications per (pair, criterion) "
                             "(default: 4)")
    parser.add_argument("--criteria", type=int, default=3,
                        help="Number of criteria from C1..C3 to use (1-3, "
                             "default: 3)")
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    if args.granularity != 20:
        parser.error("only --granularity 20 is currently supported")
    if not 1 <= args.criteria <= len(CRITERIA):
        parser.error(f"--criteria must be between 1 and {len(CRITERIA)}")
    criteria = CRITERIA[: args.criteria]

    print("Loading SWE-bench Verified trajectories...")
    tasks, runs = load_swebench_tasks(args.trajs)

    n_tasks = len(tasks)
    all_pass, all_fail, swing = [], [], []
    for iid, trials in sorted(tasks.items()):
        rewards = [t["reward"] for t in trials]
        if all(r == 1 for r in rewards):
            all_pass.append(iid)
        elif all(r == 0 for r in rewards):
            all_fail.append(iid)
        else:
            swing.append(iid)

    print(f"  Runs: {runs}")
    print(f"  Instances: {n_tasks}  All-pass: {len(all_pass)}  "
          f"All-fail: {len(all_fail)}  Swing: {len(swing)}")

    client = create_gemini_client()

    cache_file = args.cache or os.path.join(
        ROOT_DIR, "cache", "cache_swebench.json")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    scores = score_all_trials(
        client, tasks, swing, criteria, GROUND_TRUTH_NOTE,
        n_reps=args.n_verifications, max_workers=args.max_workers,
        cache_file=cache_file)

    results_file = os.path.join(ROOT_DIR, "results", "swebench_verified.txt")
    evaluate_and_print(
        "SWE-BENCH VERIFIED SUCCESS RATE",
        tasks, swing, all_pass, scores, criteria,
        n_reps=args.n_verifications, n_tasks=n_tasks, n_runs=len(runs),
        results_file=results_file)

    print(f"\nCache: {cache_file}")


if __name__ == "__main__":
    main()
