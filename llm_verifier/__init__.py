"""LLM-as-a-Verifier: fine-grained reward + pivot tournament selection.

Entry points: `select` (best of N trajectories), `compare` (raw pairwise
rewards), and `track` / `ProgressTracker` (per-step progress). For benchmark
runs use `scripts/run.py`.
"""

from __future__ import annotations

import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from llm_verifier import pivot_tournament as ppt
from llm_verifier.fine_grained_reward import (
    DEFAULT_MODEL,
    GRANULARITY,
    USAGE,
    LazyClient,
    MissingAPIKeyError,
    TokenUsage,
    create_client,
    default_max_workers,
    directed_reward,
    format_usage,
    load_dotenv,
    score_directed_pairs,
    score_pair_criterion,
    token_usage,
)
from llm_verifier.progress import ProgressResult, ProgressTracker, track
from llm_verifier.prompts import load_prompts, normalize_criteria

__version__ = "0.2.0"

__all__ = [
    "select",
    "compare",
    "track",
    "ProgressTracker",
    "VerifierResult",
    "ProgressResult",
    "MissingAPIKeyError",
    "GRANULARITY",
    "DEFAULT_MODEL",
    "load_dotenv",
    "USAGE",
    "TokenUsage",
    "token_usage",
    "format_usage",
]

# A criteria argument: bundled benchmark name or criteria-file path (str),
# {name: description} mapping, or a sequence of strings / criterion dicts.
CriteriaArg = Union[str, Mapping[str, str], Sequence[Union[str, Mapping[str, str]]]]

# An images argument: one image or a sequence of images, where each image is
# a local file path, an http(s) URL, or raw image bytes.
ImagesArg = Union[str, bytes, "os.PathLike[str]",
                  Sequence[Union[str, bytes, "os.PathLike[str]"]]]


@dataclass
class VerifierResult:
    """Outcome of `select`.

    Attributes:
        index: index of the winning trajectory in the input list.
        best: the winning trajectory itself.
        scores: per-trajectory mean preference (w_i / c_i) from the tournament.
        n_comparisons: number of directed verifier comparisons that were run.
        criteria: the criterion ids used for scoring.
    """

    index: int
    best: str
    scores: list[float] = field(repr=False)
    n_comparisons: int = 0
    criteria: list[str] = field(default_factory=list)

    @property
    def ranking(self) -> list[int]:
        """Trajectory indices sorted best-first by tournament score."""
        return sorted(range(len(self.scores)),
                      key=lambda i: (-self.scores[i], i))

    def __repr__(self) -> str:
        return (f"VerifierResult(index={self.index}, "
                f"n_comparisons={self.n_comparisons}, "
                f"criteria={self.criteria})")


def _resolve_criteria(criteria: CriteriaArg,
                      ground_truth_note: Optional[str]):
    """Turn any accepted `criteria` form into (note, criterion dicts)."""
    if isinstance(criteria, str):
        note, crits = load_prompts(criteria)
    else:
        note, crits = "", normalize_criteria(criteria)
    if ground_truth_note is not None:
        note = ground_truth_note
    return note, crits


def _default_progress(progress: Optional[bool]) -> bool:
    """`progress=None` means auto: show progress only in an interactive
    terminal, stay silent in servers / pipelines."""
    if progress is None:
        return sys.stderr.isatty()
    return progress


def select(
    problem: str,
    candidates: Sequence[str],
    *,
    criteria: CriteriaArg,
    images: Optional[ImagesArg] = None,
    ground_truth_note: Optional[str] = None,
    n_evaluations: int = 4,
    pivots: int = 2,
    seed: int = 0,
    max_workers: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    cache: Optional[str] = None,
    progress: Optional[bool] = None,
    on_error: str = "tie",
    client: Any = None,
) -> VerifierResult:
    """Select the best of N agent trajectories for a single task.

    Scores directed pairs with the fine-grained reward and aggregates them
    with a Probabilistic Pivot Tournament — O(Nk) verifier comparisons
    instead of a full O(N²) round-robin. Identical inputs with the same
    `seed` run the identical tournament.

    Args:
        problem: the task description shown to the verifier.
        candidates: list of N agent trajectories (strings) to rank.
        criteria: a bundled benchmark name (e.g. ``"swe_bench"``), a path to a
            ``*.md`` criteria file, a ``{name: description}`` dict, or a list
            of strings / ``{"id", "name", "description"}`` dicts.
        images: task-context image(s) the verifier sees with every comparison
            — one image or a list; each a file path, http(s) URL, or raw
            bytes. Requires a multimodal verifier model.
        ground_truth_note: optional note the verifier always sees; defaults
            to the note parsed from the prompt file (or empty).
        n_evaluations: repeated verifications K per criterion.
        pivots: number of pivots k. Cost grows as O(Nk); k is clamped to N.
        seed: seed for the random ring pass.
        max_workers: concurrency for verifier calls (default: 500 on
            DeepSeek, 50 otherwise).
        model: verifier model name (default ``gemini-2.5-flash``).
        cache: optional JSON score-cache path; re-runs only score
            comparisons not seen before.
        progress: show a progress bar. Default (``None``): only when stderr
            is a TTY.
        on_error: ``"tie"`` scores a failed verifier call 0.5/0.5 for this
            run (never persisted to the cache); ``"raise"`` re-raises it.
        client: a pre-built ``openai`` or ``google-genai`` client; by default
            one is built from the environment (``OPENAI_BASE_URL``, then
            ``DEEPSEEK_API_KEY``, then ``VERTEX_API_KEY``). The backend must
            expose token-level logprobs.

    Returns:
        A `VerifierResult` whose ``.index`` / ``.best`` is the chosen
        trajectory and ``.ranking`` orders all trajectories best-first.

    Raises:
        MissingAPIKeyError: no credentials and no `client` given (only when
            uncached comparisons actually need scoring).
    """
    note, crits = _resolve_criteria(criteria, ground_truth_note)
    criteria_ids = [c["id"] for c in crits]
    show_progress = _default_progress(progress)
    if max_workers is None:
        max_workers = default_max_workers()

    n = len(candidates)
    if n == 0:
        raise ValueError("need at least one candidate")
    if n == 1:
        return VerifierResult(0, candidates[0], [1.0], 0, criteria_ids)

    if cache:
        cache_dir = os.path.dirname(os.path.abspath(cache))
        os.makedirs(cache_dir, exist_ok=True)

    # One synthetic task holding the N candidate trajectories.
    task = "task"
    tasks = {task: [{"problem": problem, "trace": t, "reward": 0,
                     "images": images}
                    for t in candidates]}

    lazy = LazyClient()
    if client is not None:
        lazy._client = client

    def directed_for(scores):
        return lambda a, b: directed_reward(
            scores, task, a, b, criteria_ids, n_evaluations)

    def score_pairs(pairs):
        return score_directed_pairs(
            lazy, tasks, {task: pairs}, crits, note, n_evaluations,
            max_workers, cache, model=model, progress=show_progress,
            on_error=on_error)

    # Phase A: ring pass (slot bias cancels around the cycle).
    rng = random.Random(seed)
    ring = ppt.ring_cycle(n, rng)
    score = directed_for(score_pairs(ring))

    # Pivots = empirical leaders from the ring pass.
    w, c = [0.0] * n, [0] * n
    ppt.accumulate(ring, score, w, c)
    pivot_set = ppt.select_pivots(w, c, pivots)
    pr_pairs = ppt.pivot_round_pairs(n, pivot_set)

    # Phase B: score the pivot rounds, then aggregate everything.
    score = directed_for(score_pairs(pr_pairs))
    w, c = [0.0] * n, [0] * n
    ppt.accumulate(ring, score, w, c)
    ppt.accumulate(pr_pairs, score, w, c)
    best = max(range(n), key=lambda i: (w[i] / c[i] if c[i] else 0.0, -i))
    mean_pref = [w[i] / c[i] if c[i] else 0.0 for i in range(n)]
    return VerifierResult(best, candidates[best], mean_pref,
                          len(ring) + len(pr_pairs), criteria_ids)


def compare(
    problem: str,
    trace_a: str,
    trace_b: str,
    *,
    criteria: CriteriaArg,
    images: Optional[ImagesArg] = None,
    ground_truth_note: Optional[str] = None,
    n_evaluations: int = 1,
    max_workers: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    client: Any = None,
) -> Tuple[float, float]:
    """Fine-grained rewards (R_A, R_B) in [0, 1] for one directed comparison.

    The verifier sees `trace_a` in slot A and `trace_b` in slot B; rewards
    are averaged over all criteria and `n_evaluations` repeats. This is the
    raw pairwise reward `select` is built on — a single directed call does
    not cancel slot bias the way `select`'s ring pass does. `criteria` and
    `images` accept the same forms as `select`; a failed verifier call
    raises.
    """
    if n_evaluations < 1:
        raise ValueError("n_evaluations must be >= 1")
    note, crits = _resolve_criteria(criteria, ground_truth_note)
    if max_workers is None:
        max_workers = default_max_workers()
    if client is None:
        client = create_client()

    jobs = [crit for crit in crits for _ in range(n_evaluations)]
    if len(jobs) == 1:
        results = [score_pair_criterion(client, problem, trace_a, trace_b,
                                        jobs[0], note, model, images)]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers,
                                                len(jobs))) as executor:
            results = list(executor.map(
                lambda crit: score_pair_criterion(
                    client, problem, trace_a, trace_b, crit, note, model,
                    images),
                jobs))

    r_a = sum(r[0] for r in results) / len(results)
    r_b = sum(r[1] for r in results) / len(results)
    return r_a, r_b
