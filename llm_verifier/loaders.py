"""
Dataset loaders for the three benchmarks.

Every loader takes the config's `data` block plus the repo root and returns
`(tasks, n_runs)` where `tasks` maps a task id to a list of trial dicts:

    {"trial_name": str, "reward": 0|1, "problem": str, "trace": str}

The verifier only ever sees `problem` and `trace`; `reward` is the held-out
ground truth used to score selection. Register new benchmarks by adding a
loader here and pointing a config's `loader:` field at its key in LOADERS.
"""

import glob as globmod
import json
import os
import re
from collections import Counter


def _abs(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


# ---------------------------------------------------------------------------
# Terminal-Bench 2
# ---------------------------------------------------------------------------

def _tb_format_trace(trajectory):
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


def _tb_extract_problem(steps, task_name):
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


def load_terminal(data, root):
    agent_dir = _abs(root, data["agent_dir"])
    tasks = {}
    for task_dir in sorted(globmod.glob(os.path.join(agent_dir, "*/"))):
        task_name = os.path.basename(os.path.normpath(task_dir))
        traj_files = sorted(globmod.glob(
            os.path.join(task_dir, "*_trajectory.json")))
        trials = []
        for traj_file in traj_files:
            with open(traj_file) as f:
                d = json.load(f)
            trajectory = d.get("trajectory")
            if not trajectory:
                continue
            steps = trajectory.get("steps", [])
            trials.append({
                "trial_name": d.get("trial_name", ""),
                "reward": d.get("reward", 0),
                "problem": _tb_extract_problem(steps, task_name),
                "trace": _tb_format_trace(trajectory),
            })
        if trials:
            tasks[task_name] = trials
    n_runs = max((len(v) for v in tasks.values()), default=0)
    return tasks, n_runs


# ---------------------------------------------------------------------------
# SWE-bench Verified
# ---------------------------------------------------------------------------

_PR_DESC_RE = re.compile(r"<pr_description>(.*?)</pr_description>", re.DOTALL)
_INSTRUCTIONS_RE = re.compile(r"<instructions>(.*?)</instructions>", re.DOTALL)


def _sb_extract_problem(messages_str):
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
        pr = _PR_DESC_RE.search(text)
        instr = _INSTRUCTIONS_RE.search(text)
        parts = []
        if pr:
            parts.append(
                f"<pr_description>\n{pr.group(1).strip()}\n</pr_description>")
        if instr:
            parts.append(
                f"<instructions>\n{instr.group(1).strip()}\n</instructions>")
        return "\n\n".join(parts) if parts else text
    return ""


def _sb_strip_blocks(text):
    if not isinstance(text, str) or not text:
        return text
    text = _PR_DESC_RE.sub("", text)
    text = _INSTRUCTIONS_RE.sub("", text)
    return text.strip()


def _sb_format_trace(messages_str):
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
                content = _sb_strip_blocks(content)
                if not content:
                    parts.append("")
                    continue
                parts.append(content[:2000] + "\n... (truncated)"
                             if len(content) > 2000 else content)
            parts.append("")
        elif role in ("tool", "user") and step > 0:
            if isinstance(content, str) and content:
                content = _sb_strip_blocks(content)
                if not content:
                    continue
                parts.append(f"[Output]\n{content[:2000]}\n... (truncated)"
                             if len(content) > 2000 else f"[Output]\n{content}")
    return "\n".join(parts)


def _sb_load_run(run_dir):
    single = os.path.join(run_dir, "data_cache.json")
    if os.path.exists(single):
        return {item["instance_id"]: item for item in json.load(open(single))}
    items, idx = [], 0
    while True:
        part = os.path.join(run_dir, f"data_cache{idx}.json")
        if not os.path.exists(part):
            break
        items.extend(json.load(open(part)))
        idx += 1
    return {item["instance_id"]: item for item in items} if items else {}


def load_swe(data, root):
    trajs_dir = _abs(root, data["trajs_dir"])
    runs = data.get("runs") or sorted(
        d for d in os.listdir(trajs_dir)
        if os.path.isdir(os.path.join(trajs_dir, d)))
    run_data = {}
    for run in runs:
        rd = _sb_load_run(os.path.join(trajs_dir, run))
        if rd:
            run_data[run] = rd

    id_counts = Counter()
    for rd in run_data.values():
        id_counts.update(rd.keys())
    all_ids = sorted(iid for iid, cnt in id_counts.items() if cnt >= 2)

    tasks = {}
    for iid in all_ids:
        trials = []
        for run in runs:
            if iid not in run_data.get(run, {}):
                continue
            item = run_data[run][iid]
            messages_str = item.get("messages", "[]")
            problem = _sb_extract_problem(messages_str) or f"(Instance: {iid})"
            trace = _sb_format_trace(messages_str)
            patch = (item.get("output_patch") or "").strip()
            trace += (f"\n\n--- Final Code Patch ---\n{patch}" if patch
                      else "\n\n--- Final Code Patch ---\n(no patch produced)")
            trials.append({
                "trial_name": f"{run}_{iid}",
                "reward": 1 if item.get("reward", 0) == 1.0 else 0,
                "problem": problem,
                "trace": trace,
            })
        tasks[iid] = trials
    return tasks, len(runs)


# ---------------------------------------------------------------------------
# MedAgentBench
# ---------------------------------------------------------------------------

def _med_format_trace(history):
    if not history:
        return "(no trajectory data)"
    parts = []
    step = 0
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "agent":
            step += 1
            parts.append(f"--- Agent Step {step} ---")
            parts.append(content)
            parts.append("")
        elif role == "user" and step > 0:
            parts.append("[FHIR Response / Feedback]")
            parts.append(content[:3000] + "\n... [truncated]"
                         if len(content) > 3000 else content)
            parts.append("")
    return "\n".join(parts)


def _med_extract_problem(task_data):
    problem = f"Question: {task_data.get('instruction', '')}"
    context = task_data.get("context", "")
    if context:
        problem += f"\nContext: {context}"
    return problem


def load_med(data, root):
    output_dir = _abs(root, data["output_dir"])
    with open(_abs(root, data["test_data"])) as f:
        test_data = json.load(f)
    run_names = data["run_names"]

    reward_lookup = {}
    for run_name in run_names:
        results_path = os.path.join(output_dir, run_name, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                overall = json.load(f)
            for r in overall.get("custom", overall).get("raw_results", []):
                status = r.get("status", "")
                is_correct = "Correct" in status and "Incorrect" not in status
                reward_lookup[(run_name, r.get("index"))] = 1 if is_correct else 0

    tasks = {}
    for run_name in run_names:
        trajs_path = os.path.join(
            output_dir, run_name, "trajectories.jsonl")
        if not os.path.exists(trajs_path):
            print(f"  Warning: {trajs_path} not found, skipping")
            continue
        with open(trajs_path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                idx = d.get("index")
                if idx is None:
                    continue
                history = d.get("output", {}).get("history", [])
                tasks.setdefault(f"task_{idx}", []).append({
                    "trial_name": f"{run_name}_task{idx}",
                    "reward": reward_lookup.get((run_name, idx), 0),
                    "problem": _med_extract_problem(test_data[idx]),
                    "trace": _med_format_trace(history),
                })
    return tasks, len(run_names)


LOADERS = {
    "terminal": load_terminal,
    "swe": load_swe,
    "med": load_med,
}
