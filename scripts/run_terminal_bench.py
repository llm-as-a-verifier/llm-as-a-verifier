#!/usr/bin/env python3
"""
Terminal-Bench via LLM-as-a-Verifier TRM Selection.

For a given trajectory directory under data/terminal_trajs/, scores all
trajectory pairs per task with Gemini 2.5 Flash, runs a round-robin tournament 
per task to pick the best traj, and reports success rate vs Pass@1 and Oracle.

Usage:
    python scripts/run_terminal_bench.py --trajs forge_gpt54
"""

import argparse
import glob as globmod
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
TRAJ_DIR = os.path.join(ROOT_DIR, "data", "terminal_trajs")

sys.path.insert(0, SCRIPT_DIR)
from verifier_core import (
    create_gemini_client,
    evaluate_and_print,
    load_dotenv,
    score_all_trials,
)

load_dotenv(ROOT_DIR)


# ---------------------------------------------------------------------------
# Terminal-Bench-specific criteria + ground-truth note
# ---------------------------------------------------------------------------

GROUND_TRUTH_NOTE = (
    "**IMPORTANT:** Focus on TERMINAL OUTPUT as ground truth. "
    "Do NOT trust the agent's self-assessment or claims of success. "
    "Agents often claim success when the terminal shows errors."
)

CRITERIA = [
    {
        "id": "specification",
        "name": "Specification Adherence",
        "description": (
            "Re-read the task description and check the SPECIFIC "
            "requirements: exact file paths, install locations, output "
            "formats, naming, and any explicit constraints (e.g. \"no X11 "
            "support\", \"install to /usr/local/bin/X\", \"output JSON to "
            "/app/out.json\"). Did the agent meet these specific "
            "requirements, or did they produce a solution that solves a "
            "similar but different problem (right idea, wrong place / "
            "wrong format / missing constraint)?"
        ),
    },
    {
        "id": "output_match",
        "name": "Output Match",
        "description": (
            "Find the FINAL verification command the agent ran (the one "
            "that should prove the solution works). Compare its actual "
            "stdout/stderr output, character-by-character if needed, to "
            "what the task description says the output should look like. "
            "For example: if the task says it should print "
            "\"Results: X Y Z\" with integers, did the agent's last test "
            "actually print that? If the task asks for a JSON file, do the "
            "values look plausible and well-formed in the cat output? "
            "Reward trajectories whose terminal SHOWS the expected output "
            "literally. Ignore everything except whether the observed "
            "output matches the expected output."
        ),
    },
    {
        "id": "error_signals",
        "name": "Error Signal Detection",
        "description": (
            "Scan the trajectory — especially the later steps — for "
            "explicit failure markers: error messages, exception "
            "tracebacks, segmentation faults, \"command not found\", "
            "\"No such file or directory\", non-zero exit codes that the "
            "agent did not subsequently fix, compilation failures, test "
            "failures, etc. A trajectory that ends with unresolved errors "
            "is almost certainly broken even if the agent claims success. "
            "Conversely, a clean trajectory whose final commands all "
            "succeed without errors is a strong positive signal. Score "
            "based ONLY on the presence/absence of unresolved error signals."
        ),
    },
]


# ---------------------------------------------------------------------------
# Terminal-Bench data loading
# ---------------------------------------------------------------------------

def format_trace(trajectory):
    """Format trajectory to text."""
    if not trajectory:
        return "(no trajectory data)"
    parts = []
    for step in trajectory.get("steps", []):
        source = step.get("source", "")
        message = step.get("message", "")
        step_id = step.get("step_id", "?")
        if source in ("system", "user"):
            continue
        if source == "agent":
            parts.append(f"--- Agent Step {step_id} ---")
            if message:
                parts.append(message)
            for tc in step.get("tool_calls", []):
                keystrokes = tc.get("arguments", {}).get("keystrokes", "")
                if keystrokes:
                    parts.append(f"[Command] {keystrokes.rstrip()}")
            obs = step.get("observation", {})
            for result in obs.get("results", []):
                content = result.get("content", "")
                if content:
                    parts.append(f"[Output]\n{content}")
            parts.append("")
    return "\n".join(parts)


def extract_problem(steps, task_name):
    """Extract problem from trajectory steps."""
    for step in steps:
        if step.get("source") == "user":
            msg = step.get("message", "")
            if msg and not (msg.startswith("$") and len(msg) < 5):
                return msg
    parts = []
    for step in steps:
        if step.get("source") != "agent":
            continue
        msg = step.get("message", "")
        if msg:
            parts.append(msg)
        if len(parts) >= 2:
            break
    if parts:
        return (f"[Task: {task_name}]\n"
                "The original task instruction was not captured. Below is the "
                "agent's initial analysis:\n\n" + "\n\n".join(parts))
    return f"(Task: {task_name})"


def load_all_trials(agent_dir):
    """Load all trials grouped by task."""
    tasks = {}
    task_dirs = sorted(globmod.glob(os.path.join(agent_dir, "*/")))

    for task_dir in task_dirs:
        task_name = os.path.basename(task_dir.rstrip("/"))
        traj_files = sorted(globmod.glob(
            os.path.join(task_dir, "*_trajectory.json")))

        trials = []
        for traj_file in traj_files:
            with open(traj_file) as f:
                data = json.load(f)
            trajectory = data.get("trajectory")
            if not trajectory:
                continue
            steps = trajectory.get("steps", [])
            trials.append({
                "trial_name": data.get("trial_name", ""),
                "reward": data.get("reward", 0),
                "problem": extract_problem(steps, task_name),
                "trace": format_trace(trajectory),
            })

        if trials:
            tasks[task_name] = trials

    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Terminal-Bench downstream success rate via "
                    "LLM-as-a-Verifier TRM selection")
    parser.add_argument("--trajs", default="forge_gpt54",
                        help="Agent trajectory directory in data/terminal_trajs/ "
                             "(default: forge_gpt54)")
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

    agent_dir = os.path.join(TRAJ_DIR, args.trajs)
    if not os.path.isdir(agent_dir):
        matches = [d for d in os.listdir(TRAJ_DIR) if args.trajs in d]
        if matches:
            agent_dir = os.path.join(TRAJ_DIR, matches[0])
            print(f"Matched: {matches[0]}")
        else:
            print(f"Error: {agent_dir} not found")
            sys.exit(1)

    print(f"Loading trials from {os.path.basename(agent_dir)}...")
    tasks = load_all_trials(agent_dir)

    n_tasks = len(tasks)
    all_pass, all_fail, swing = [], [], []
    for task_name, trials in sorted(tasks.items()):
        rewards = [t["reward"] for t in trials]
        if all(r == 1 for r in rewards):
            all_pass.append(task_name)
        elif all(r == 0 for r in rewards):
            all_fail.append(task_name)
        else:
            swing.append(task_name)

    n_trials = len(next(iter(tasks.values())))
    print(f"  Tasks: {n_tasks}  All-pass: {len(all_pass)}  "
          f"All-fail: {len(all_fail)}  Swing: {len(swing)}")

    client = create_gemini_client()

    cache_file = args.cache or os.path.join(
        ROOT_DIR, "cache", f"cache_terminal_{os.path.basename(agent_dir)}.json")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    scores = score_all_trials(
        client, tasks, swing, criteria, GROUND_TRUTH_NOTE,
        n_reps=args.n_verifications, max_workers=args.max_workers,
        cache_file=cache_file)

    results_file = os.path.join(
        ROOT_DIR, "results",
        f"terminal_{os.path.basename(agent_dir)}.txt")
    evaluate_and_print(
        f"TERMINAL-BENCH SUCCESS RATE — {os.path.basename(agent_dir)}",
        tasks, swing, all_pass, scores, criteria,
        n_reps=args.n_verifications, n_tasks=n_tasks, n_runs=n_trials,
        results_file=results_file)

    print(f"\nCache: {cache_file}")


if __name__ == "__main__":
    main()
