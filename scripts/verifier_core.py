"""
Shared verifier infrastructure for Terminal-Bench and SWE-bench Verified experiments.

Provides:
  - Gemini 2.5 Flash client + logprob-based call
  - g=20 scoring granularity
  - Per-criterion pairwise prompt + score extraction
  - Round-robin tournament best-traj selection
"""

import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations


# ---------------------------------------------------------------------------
# g=20 scoring scale
# ---------------------------------------------------------------------------

GRANULARITY = 20

SCALE = {
    "scale_description": (
        "Rate how likely the agent correctly solved the task on a "
        "20-point scale using letters A through T:\n"
        "  A = clearly and completely succeeded with verified output (best)\n"
        "  B-D = succeeded with only minor issues\n"
        "  E-G = above average, mostly correct with some issues\n"
        "  H-J = uncertain, leans toward success\n"
        "  K-M = uncertain, leans toward failure\n"
        "  N-P = below average, significant issues remain\n"
        "  Q-S = failed with some partial progress\n"
        "  T = clearly and completely failed (worst)"
    ),
    "score_format": "LETTER_A_TO_T",
    "valid_tokens": {
        **{chr(65 + i): float(GRANULARITY - i) for i in range(GRANULARITY)},
        **{chr(97 + i): float(GRANULARITY - i) for i in range(GRANULARITY)},
    },
}


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def load_dotenv(root_dir):
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def create_gemini_client():
    from google import genai
    vertex_key = os.environ.get("VERTEX_API_KEY")
    if vertex_key:
        return genai.Client(vertexai=True, api_key=vertex_key)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    print("Error: set GEMINI_API_KEY or VERTEX_API_KEY in .env or environment")
    sys.exit(1)


def call_gemini(client, prompt, top_logprobs=20):
    """Call Gemini 2.5 Flash with logprobs.
    Returns (text, tokens, position_logprobs)."""
    from google.genai.types import (
        Content, GenerateContentConfig, Part, ThinkingConfig)

    config = GenerateContentConfig(
        max_output_tokens=4096,
        temperature=1.0,
        response_logprobs=True,
        logprobs=top_logprobs,
        thinking_config=ThinkingConfig(thinking_budget=0),
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[Content(role="user", parts=[Part(text=prompt)])],
        config=config,
    )

    text = response.text or ""
    tokens = None
    position_logprobs = None

    candidate = response.candidates[0]
    if candidate.logprobs_result and candidate.logprobs_result.top_candidates:
        position_logprobs = []
        for pos in candidate.logprobs_result.top_candidates:
            alts = [(lp.token, lp.log_probability)
                    for lp in pos.candidates]
            position_logprobs.append(alts)
        if candidate.logprobs_result.chosen_candidates:
            tokens = [c.token for c in candidate.logprobs_result.chosen_candidates]

    return text, tokens, position_logprobs


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def _find_tag_logprobs(tokens, position_logprobs, tag):
    if not tokens or not position_logprobs:
        return None
    text_so_far = ""
    for i, tok in enumerate(tokens):
        text_so_far += tok
        if text_so_far.rstrip().endswith(tag):
            if i + 1 < len(position_logprobs):
                return position_logprobs[i + 1]
    return None


def extract_score(text, tokens, position_logprobs, tag):
    """Extract normalized [0,1] score from logprobs at the given tag."""
    valid_tokens = SCALE["valid_tokens"]

    tag_lp = _find_tag_logprobs(tokens, position_logprobs, tag)
    probs = {}
    if tag_lp:
        for tok_str, logprob in tag_lp:
            tok = tok_str.strip()
            if tok in valid_tokens:
                val = valid_tokens[tok]
                p = math.exp(logprob)
                probs[val] = max(probs.get(val, 0.0), p)

    if probs:
        unique_vals = sorted(set(valid_tokens.values()))
        min_val, max_val = min(unique_vals), max(unique_vals)
        total_p = sum(probs.values())
        expected = sum(v * p for v, p in probs.items()) / total_p
        return (expected - min_val) / (max_val - min_val) \
            if max_val > min_val else 0.5

    # Fallback: parse from text
    tag_name = tag.strip("<>")
    pattern = rf"<{re.escape(tag_name)}>\s*(.+?)\s*</{re.escape(tag_name)}>"
    match = re.search(pattern, text or "", re.IGNORECASE)
    if match:
        tok = match.group(1).strip()
        raw_val = valid_tokens.get(tok)
        if raw_val is None:
            for vt, val in valid_tokens.items():
                if tok.lower() == vt.lower():
                    raw_val = val
                    break
        if raw_val is not None:
            unique_vals = sorted(set(valid_tokens.values()))
            min_val, max_val = min(unique_vals), max(unique_vals)
            return (raw_val - min_val) / (max_val - min_val) \
                if max_val > min_val else 0.5

    return 0.5


# ---------------------------------------------------------------------------
# Pairwise prompt + scoring
# ---------------------------------------------------------------------------

def create_prompt_for_criterion(problem, trace_a, trace_b, criterion,
                                ground_truth_note):
    """One pairwise prompt focused on a single evaluation criterion."""
    return (
        "You are an expert evaluator of AI coding agents. "
        "You will see a task description and two agent trajectories. "
        f"Your job is to evaluate them on ONE specific criterion: "
        f"**{criterion['name']}**.\n\n"
        f"{ground_truth_note}\n\n"
        f"**Task:**\n{problem}\n\n"
        f"**Trajectory A:**\n{trace_a}\n\n"
        f"**Trajectory B:**\n{trace_b}\n\n"
        f"**Evaluation Guideline — {criterion['name']}:**\n"
        f"{criterion['description']}\n\n"
        f"Score each trajectory ONLY on this specific criterion. Ignore other "
        f"aspects of the trajectory that are not relevant to "
        f"\"{criterion['name']}\".\n\n"
        f"**Rating Scale:**\n{SCALE['scale_description']}\n\n"
        "Then output your final scores:\n"
        f"<score_A>{SCALE['score_format']}</score_A>\n"
        f"<score_B>{SCALE['score_format']}</score_B>\n\n"
        "Begin your analysis now."
    )


def score_pair_criterion(client, problem, trace_a, trace_b, criterion,
                         ground_truth_note):
    """Score (A, B) for a single criterion."""
    prompt = create_prompt_for_criterion(
        problem, trace_a, trace_b, criterion, ground_truth_note)
    text, tokens, position_logprobs = call_gemini(client, prompt)
    sa = extract_score(text, tokens, position_logprobs, "<score_A>")
    sb = extract_score(text, tokens, position_logprobs, "<score_B>")
    return sa, sb


# ---------------------------------------------------------------------------
# Cached batch scoring + tournament selection
# ---------------------------------------------------------------------------

def score_all_trials(client, tasks, swing_tasks, criteria, ground_truth_note,
                     n_reps, max_workers, cache_file):
    """Score every (pair_of_trials, criterion, rep) on swing tasks.

    Cache key: f"{crit_id}|{task_name}|{i},{j}|{rep}"
    """
    cached = {}
    if cache_file and os.path.exists(cache_file):
        with open(cache_file) as f:
            cached = json.load(f)

    jobs = []
    for task_name in swing_tasks:
        trials = tasks[task_name]
        n = len(trials)
        for i, j in combinations(range(n), 2):
            for crit in criteria:
                for rep in range(n_reps):
                    key = f"{crit['id']}|{task_name}|{i},{j}|{rep}"
                    if key not in cached:
                        jobs.append((key, trials[i]["problem"],
                                     trials[i]["trace"], trials[j]["trace"],
                                     crit))

    if not jobs:
        print(f"  All {len(cached)} scores cached")
        return cached

    print(f"  {len(jobs)} scoring jobs ({len(cached)} cached)")

    from tqdm import tqdm
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_pair_criterion, client, prob, ta, tb, crit,
                            ground_truth_note): key
            for key, prob, ta, tb, crit in jobs
        }
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Scoring")
        save_every = max(1, len(futures) // 20)
        done = 0

        for future in pbar:
            key = futures[future]
            try:
                sa, sb = future.result()
                cached[key] = {"score_i": sa, "score_j": sb}
            except Exception as e:
                errors += 1
                cached[key] = {"score_i": 0.5, "score_j": 0.5}
                if errors <= 3:
                    print(f"\n  Error: {e}")
            done += 1
            pbar.set_postfix(errors=errors)
            if cache_file and done % save_every == 0:
                with open(cache_file, "w") as f:
                    json.dump(cached, f)

    if cache_file:
        with open(cache_file, "w") as f:
            json.dump(cached, f)

    print(f"  Done ({errors} errors)")
    return cached


def select_best(tasks, swing_tasks, scores, criteria_ids, n_reps=1,
                rep_idx=None):
    """Select best trial per swing task via round-robin tournament,
    averaging scores across the given criteria. If rep_idx is given,
    only that rep is used; otherwise all n_reps reps are averaged."""
    selections = {}
    for task_name in swing_tasks:
        trials = tasks[task_name]
        n = len(trials)
        wins = [0.0] * n

        for i, j in combinations(range(n), 2):
            si_sum = sj_sum = 0
            count = 0
            for cid in criteria_ids:
                reps = [rep_idx] if rep_idx is not None else range(n_reps)
                for rep in reps:
                    key = f"{cid}|{task_name}|{i},{j}|{rep}"
                    entry = scores.get(key, {})
                    si_sum += entry.get("score_i", 0.5)
                    sj_sum += entry.get("score_j", 0.5)
                    count += 1

            si = si_sum / count if count else 0.5
            sj = sj_sum / count if count else 0.5

            if si > sj:
                wins[i] += 1
            elif sj > si:
                wins[j] += 1
            else:
                wins[i] += 0.5
                wins[j] += 0.5

        best_idx = max(range(n), key=lambda t: wins[t])
        selections[task_name] = {
            "idx": best_idx,
            "trial": trials[best_idx]["trial_name"],
            "reward": trials[best_idx]["reward"],
        }

    return selections


def evaluate_and_print(label, tasks, swing, all_pass, scores, criteria,
                       n_reps, n_tasks, n_runs, results_file=None):
    """Print Pass@1 / ensemble / oracle table. Also saves to results_file."""
    crit_ids = [c["id"] for c in criteria]

    def eval_config(criteria_ids):
        rep_totals = []
        for rep in range(n_reps):
            sel = select_best(tasks, swing, scores, criteria_ids,
                              n_reps=1, rep_idx=rep)
            sc = sum(1 for s in sel.values() if s["reward"] == 1)
            rep_totals.append(len(all_pass) + sc)
        mean = sum(rep_totals) / len(rep_totals)
        se = (sum((t - mean) ** 2 for t in rep_totals) / len(rep_totals)) \
            ** 0.5 / len(rep_totals) ** 0.5
        return mean, se

    random_ev = len(all_pass) + sum(
        sum(t["reward"] for t in tasks[tn]) / len(tasks[tn])
        for tn in swing)
    oracle = len(all_pass) + len(swing)
    mean_ens, se_ens = eval_config(crit_ids)

    lines = [
        "",
        "=" * 60,
        label,
        f"  Granularity: g{GRANULARITY}, "
        f"Criteria: {crit_ids}, Reps: {n_reps}",
        "=" * 60,
        "",
        f"{'Method':<40s}  {'Score':>14s}  {'Rate':>7s}",
        "-" * 67,
        f"{'Pass@1':<40s}  {random_ev:>5.1f}/{n_tasks}        "
        f"{100 * random_ev / n_tasks:>5.1f}%",
        f"{'LLM-as-a-Verifier':<40s}  "
        f"{mean_ens:>4.1f}\u00b1{se_ens:.1f}/{n_tasks}    "
        f"{100 * mean_ens / n_tasks:>5.1f}%",
        f"{'Oracle (Bo' + str(n_runs) + ')':<40s}  "
        f"{oracle:>5d}/{n_tasks}        "
        f"{100 * oracle / n_tasks:>5.1f}%",
    ]
    for line in lines:
        print(line)

    if results_file:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nResults saved: {results_file}")
