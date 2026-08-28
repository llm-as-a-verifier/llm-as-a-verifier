"""Regression tests for `llm_verifier.select`'s two-phase tournament.

`select` scores the ring pass (phase A) and the pivot rounds (phase B) in
separate `score_directed_pairs` calls so the pivots can be chosen in between.
Each call returns only the comparisons it scored plus whatever was already on
disk, so phase B's return value carries the ring pass ONLY when a `cache=`
file is configured. Rebinding instead of merging therefore dropped every ring
comparison to `directed_reward`'s 0.5/0.5 default on the default
(`cache=None`) path — changing the winner in roughly 9% of tournaments.

These tests pin the fix: the aggregation must see both phases, must not
depend on `cache=`, and must not pay for the ring pass twice.
"""

import math
from itertools import combinations

import pytest

import llm_verifier
from llm_verifier import fine_grained_reward as fgr
from llm_verifier import pivot_tournament as ppt

# Deterministic candidate "qualities" the stub verifier reports. Chosen so
# the ranking is unambiguous and no two candidates tie.
QUALITIES = [0.91, 0.62, 0.55, 0.48, 0.12]
CANDIDATES = [f"candidate-{i}" for i in range(len(QUALITIES))]
QUALITY_OF = dict(zip(CANDIDATES, QUALITIES))

CRITERIA = {"Correctness": "Does it solve the task?"}
SEED = 0
PIVOTS = 2


@pytest.fixture
def stub_verifier(monkeypatch):
    """Replace the API call with a deterministic scorer and record calls.

    `score_directed_pairs` looks `score_pair_criterion` up as a module global,
    so patching it there intercepts every verifier call. Returns the list of
    (slot_A_trace, slot_B_trace) pairs actually scored.
    """
    calls = []

    def fake_score_pair_criterion(client, problem, trace_a, trace_b, criterion,
                                  ground_truth_note, model=None, images=None):
        calls.append((trace_a, trace_b))
        return QUALITY_OF[trace_a], QUALITY_OF[trace_b]

    monkeypatch.setattr(fgr, "score_pair_criterion", fake_score_pair_criterion)
    return calls


def run_select(cache=None, n_evaluations=1, candidates=CANDIDATES):
    return llm_verifier.select(
        "problem", candidates,
        criteria=CRITERIA,
        client=object(),          # never used: the stub ignores it
        n_evaluations=n_evaluations,
        pivots=PIVOTS,
        seed=SEED,
        progress=False,
        cache=cache,
    )


def reference_tournament(qualities, seed=SEED, pivots=PIVOTS):
    """Run PPT directly against the true qualities, with no scoring layer in
    between. This is what `select` must reproduce once every comparison
    reaches the aggregation."""
    import random

    n = len(qualities)

    def score(a, b):
        return qualities[a], qualities[b]

    ring = ppt.ring_cycle(n, random.Random(seed))
    w, c = [0.0] * n, [0] * n
    ppt.accumulate(ring, score, w, c)
    pivot_set = ppt.select_pivots(w, c, pivots)
    pr_pairs = ppt.pivot_round_pairs(n, pivot_set)

    w, c = [0.0] * n, [0] * n
    ppt.accumulate(ring, score, w, c)
    ppt.accumulate(pr_pairs, score, w, c)
    scores = [w[i] / c[i] if c[i] else 0.0 for i in range(n)]
    best = max(range(n), key=lambda i: (scores[i], -i))
    return best, scores, len(ring) + len(pr_pairs)


def test_ring_pass_reaches_aggregation_without_a_cache(stub_verifier):
    """The regression: with `cache=None` the ring comparisons must still
    carry their real rewards into the final scores, not 0.5/0.5 defaults."""
    expected_best, expected_scores, expected_cmps = reference_tournament(
        QUALITIES)

    result = run_select(cache=None)

    assert result.index == expected_best
    assert result.n_comparisons == expected_cmps
    assert result.scores == pytest.approx(expected_scores)


def test_cache_does_not_change_the_outcome(stub_verifier, tmp_path):
    """Passing `cache=` is a cost optimization, never a correctness fix:
    both paths must agree exactly."""
    without = run_select(cache=None)
    with_cache = run_select(cache=str(tmp_path / "scores.json"))

    assert without.index == with_cache.index
    assert without.ranking == with_cache.ranking
    assert without.scores == pytest.approx(with_cache.scores)


def test_no_candidate_is_left_at_the_default_tie(stub_verifier):
    """A dropped phase gives every candidate a block of 0.5/0.5 comparisons,
    which pulls scores toward exactly 0.5. With distinct qualities no
    candidate should land there."""
    result = run_select(cache=None)

    assert not any(math.isclose(s, 0.5, abs_tol=1e-9) for s in result.scores)


def test_ring_pass_is_not_scored_twice(stub_verifier):
    """Merging the two phases must not re-score the ring: the fix keeps
    phase A's results, it does not re-run them in phase B."""
    n, k = len(CANDIDATES), PIVOTS
    expected_comparisons = n + k * (n - k) + len(list(combinations(range(k), 2)))

    result = run_select(cache=None)

    assert result.n_comparisons == expected_comparisons
    # one verifier call per (comparison, criterion, repeat)
    assert len(stub_verifier) == expected_comparisons * len(CRITERIA)


def test_repeats_swap_slots_and_restore_candidate_order(stub_verifier):
    """With K >= 2 odd repeats swap the prompt slots; scores are recorded
    back in candidate order, so the outcome is unchanged and each directed
    pair is seen in both orderings."""
    expected_best, expected_scores, _ = reference_tournament(QUALITIES)

    result = run_select(cache=None, n_evaluations=2)

    assert result.index == expected_best
    assert result.scores == pytest.approx(expected_scores)
    # every scored pair appeared in both slot orderings
    seen = set(stub_verifier)
    assert all((b, a) in seen for a, b in seen)


def test_single_candidate_short_circuits(stub_verifier):
    """One candidate needs no verification at all."""
    result = run_select(cache=None, candidates=CANDIDATES[:1])

    assert result.index == 0
    assert result.n_comparisons == 0
    assert stub_verifier == []
