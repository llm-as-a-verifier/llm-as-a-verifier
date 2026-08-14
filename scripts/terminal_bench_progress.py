#!/usr/bin/env python3
"""Re-score and plot the SUCCESS-vs-FAILED progress-tracking example.

Runs `llm_verifier.track()` (K=16) on the two Terminus 2 runs of the
Terminal-Bench task `pytorch-model-cli` in `data/progress_tracking/`, then
renders the figure. Scoring makes real verifier calls — requires a verifier
backend in `.env`.

Usage:
    python scripts/terminal_bench_progress.py                # re-score + plot
    python scripts/terminal_bench_progress.py --plot-only    # replot saved curves
    python scripts/terminal_bench_progress.py --online       # replay step-by-step
                                                             # via ProgressTracker

Writes `cache/progress_pytorch-model-cli_k16.json` and `progress_traces.png`.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ_DIR = os.path.join(ROOT_DIR, "data", "progress_tracking",
                        "pytorch-model-cli")
TRIALS = {"succ": "success", "fail": "fail"}
CURVES_JSON = os.path.join(ROOT_DIR, "cache",
                           "progress_pytorch-model-cli_k16.json")
OUT_PNG = "progress_traces.png"
GREEN, RED = "#3a7d34", "#c0392b"


# ---------------------------------------------------------------------------
# Scoring: raw trajectories -> track() curves
# ---------------------------------------------------------------------------

def agent_steps(trajectory):
    """One string per agent step: message + commands + observed output."""
    steps = []
    for s in trajectory.get("steps", []):
        if s.get("source") != "agent":
            continue
        parts = [s.get("message", "")]
        for tc in s.get("tool_calls", []):
            keystrokes = tc.get("arguments", {}).get("keystrokes", "")
            if keystrokes:
                parts.append(f"[Command] {keystrokes.rstrip()}")
        for result in s.get("observation", {}).get("results", []):
            content = result.get("content", "")
            if content:
                parts.append(f"[Output]\n{content}")
        steps.append("\n".join(p for p in parts if p))
    return steps


def rescore():
    import llm_verifier
    from llm_verifier.loaders import _tb_extract_problem

    curves = {}
    for key, trial in TRIALS.items():
        with open(os.path.join(TRAJ_DIR, f"{trial}_trajectory.json")) as f:
            d = json.load(f)
        trajectory = d["trajectory"]
        problem = _tb_extract_problem(trajectory["steps"], d["task_name"])
        steps = agent_steps(trajectory)
        print(f"{key}: {trial} ({len(steps)} agent steps, reward="
              f"{d.get('reward')}) — tracking with K=16 ...")
        r = llm_verifier.track(problem=problem, steps=steps,
                               n_evaluations=16)
        curves[key] = {"task": d["task_name"], "trial": trial,
                       "T": len(steps), "checkpoint_steps": r.steps,
                       "per_rep_scores": r.per_rep_scores}
    os.makedirs(os.path.dirname(CURVES_JSON), exist_ok=True)
    with open(CURVES_JSON, "w") as f:
        json.dump(curves, f, indent=2)
    print(f"saved {CURVES_JSON}")


def replay_online():
    """Feed each run's steps one at a time through ProgressTracker, printing
    the live progress score after every step — what an agent harness would
    see while the run is still going."""
    import llm_verifier
    from llm_verifier.loaders import _tb_extract_problem

    for key, trial in TRIALS.items():
        with open(os.path.join(TRAJ_DIR, f"{trial}_trajectory.json")) as f:
            d = json.load(f)
        trajectory = d["trajectory"]
        problem = _tb_extract_problem(trajectory["steps"], d["task_name"])
        steps = agent_steps(trajectory)
        print(f"\n{key}: {trial} ({len(steps)} agent steps, "
              f"reward={d.get('reward')}) — online tracking with K=4 ...")
        tracker = llm_verifier.ProgressTracker(problem, n_evaluations=4)
        for i, step in enumerate(steps, start=1):
            score = tracker.update(step)
            bar = "#" * int(round(score * 40))
            print(f"  step {i:>2}/{len(steps)}  {score:6.3f}  {bar}")


# ---------------------------------------------------------------------------
# Plotting: curves JSON -> figure
# ---------------------------------------------------------------------------

def curve(entry):
    """(normalized steps, mean over reps, std over reps) for one run."""
    reps = np.array([[0.0 if v is None else v for v in rep]
                     for rep in entry["per_rep_scores"]], float)
    steps = np.array(entry["checkpoint_steps"], float)
    x = (steps - steps.min()) / max(steps.max() - steps.min(), 1.0)
    return x, reps.mean(0), reps.std(0)


def plot():
    with open(CURVES_JSON) as f:
        payload = json.load(f)

    runs = []
    if "succ" in payload:
        runs.append(("SUCCESS", GREEN, "-o", 0.18, curve(payload["succ"])))
    if "fail" in payload:
        runs.append(("FAILED", RED, "-s", 0.15, curve(payload["fail"])))
    if not runs:
        sys.exit(f"no 'succ' or 'fail' entry in {CURVES_JSON}")

    # Normalize scores by the global max of the mean curves.
    g = max(m.max() for _, _, _, _, (_, m, _) in runs) or 1.0

    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for label, color, style, band_alpha, (x, m, s) in runs:
        m, s = m / g, s / g
        ax.plot(x, m, style, color=color, lw=2.4, ms=7, label=label, zorder=3)
        ax.fill_between(x, np.clip(m - s, 0, None), m + s, color=color,
                        alpha=band_alpha, zorder=1)

    ax.set_xlabel("Normalized Step", fontsize=16, fontweight="bold")
    ax.set_ylabel("Normalized Verifier Score", fontsize=16, fontweight="bold")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.05, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    print(f"saved {OUT_PNG}")


def main():
    if "--online" in sys.argv[1:]:
        replay_online()
        return
    if "--plot-only" not in sys.argv[1:]:
        rescore()
    plot()


if __name__ == "__main__":
    main()
