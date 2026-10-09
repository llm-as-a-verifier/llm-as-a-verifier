"""Tournament aggregation: both phases must reach the final scores.

Run: python tests/test_select.py   (or: pytest tests/)
No network and no API key — a stub verifier replaces the one scoring call,
and LazyClient never builds a real client.
"""

import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_verifier  # noqa: E402
from llm_verifier import fine_grained_reward as fgr  # noqa: E402

# A deterministic stand-in for the verifier: each candidate has a fixed
# "strength" and the pairwise reward is its share of the pair.
_STRENGTH = {}


def _stub_score(client, problem, trace_a, trace_b, criterion,
                ground_truth_note, model=None, images=None):
    a, b = _STRENGTH[trace_a], _STRENGTH[trace_b]
    return a / (a + b), b / (a + b)


def _select(candidates, seed, cache_path=None):
    real_score = fgr.score_pair_criterion
    fgr.score_pair_criterion = _stub_score
    try:
        return llm_verifier.select(
            "task", candidates, criteria={"C": "does it solve the task?"},
            seed=seed, n_evaluations=1, pivots=2, progress=False,
            cache=cache_path, client=object())
    finally:
        fgr.score_pair_criterion = real_score


def test_a_cache_path_does_not_change_the_result():
    """Regression: `select()` used to rebind its score dict to phase B's
    return value. score_directed_pairs returns the pairs it was asked for
    merged with whatever it read off disk, so with no cache file the ring
    pass vanished from the final aggregation and every ring lookup fell back
    to directed_reward's 0.5/0.5 default. The cache is an optimisation; it
    must never change what select() returns.
    """
    rng = random.Random(0)
    for trial in range(20):
        n = rng.randint(4, 7)
        candidates = [f"cand-{trial}-{i}" for i in range(n)]
        _STRENGTH.clear()
        for c in candidates:
            _STRENGTH[c] = rng.uniform(0.1, 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            cached = _select(candidates, 0, os.path.join(tmp, "scores.json"))
        uncached = _select(candidates, 0)

        assert uncached.index == cached.index, (
            f"trial {trial}: winner {uncached.index} without a cache, "
            f"{cached.index} with one")
        for i, (u, c) in enumerate(zip(uncached.scores, cached.scores)):
            assert abs(u - c) < 1e-9, (
                f"trial {trial}: candidate {i} scored {u} without a cache, "
                f"{c} with one")


def test_the_ring_pass_reaches_the_final_scores():
    """The same property stated directly: every pair the tournament reports
    having compared must be present in the dict the final aggregation reads,
    or its reward silently defaults to a tie."""
    _STRENGTH.clear()
    for i in range(4):
        _STRENGTH[f"c{i}"] = 0.2 + 0.2 * i

    missing = []
    real_directed_reward = fgr.directed_reward

    # *rest keeps the spy transparent to directed_reward's signature: it
    # inspects the arguments it needs and forwards everything untouched.
    def spy(scores, task_name, a, b, criteria_ids, n_reps, *rest):
        if not all(fgr.cache_key(cid, task_name, a, b, rep) in scores
                   for cid in criteria_ids for rep in range(n_reps)):
            missing.append((a, b))
        return real_directed_reward(scores, task_name, a, b, criteria_ids,
                                    n_reps, *rest)

    llm_verifier.directed_reward = spy
    try:
        _select(list(_STRENGTH), 0)
    finally:
        llm_verifier.directed_reward = real_directed_reward

    assert not missing, f"pairs absent from the score dict: {missing}"


def test_the_strongest_candidate_wins():
    """A sanity check on the tournament itself: with a stub verifier whose
    strengths are strictly ordered, the strongest candidate is selected."""
    _STRENGTH.clear()
    for i in range(6):
        _STRENGTH[f"c{i}"] = 0.1 + 0.15 * i
    result = _select(list(_STRENGTH), 0)
    assert result.index == 5, f"expected the strongest candidate, got {result.index}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
