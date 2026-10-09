"""
Regression tests for `llm_verifier.select`.

Each test here fails on the pre-fix code and passes after. They use a stubbed
`score_pair_criterion`, so no API key, no network, and no verifier model are
needed -- `pytest` alone runs them.

Run with:  pytest tests/ -q
"""

import warnings

import pytest

import llm_verifier
import llm_verifier.fine_grained_reward as fgr


CRITERIA = {"Quality": "Is the candidate good?"}


@pytest.fixture
def stub_scorer(monkeypatch):
    """Replace the verifier call with a deterministic function of the
    candidate text, and record every comparison that was requested."""
    calls = []

    def make(quality):
        def scorer(client, problem, trace_a, trace_b, criterion,
                   ground_truth_note, model=None, images=None):
            calls.append((trace_a, trace_b))
            return quality[trace_a], quality[trace_b]

        monkeypatch.setattr(fgr, "score_pair_criterion", scorer)
        return calls

    make.calls = calls
    return make


# --------------------------------------------------------------------------- #
# 1. Phase A's ring scores must survive into the final aggregation.
# --------------------------------------------------------------------------- #

def test_ring_scores_are_not_lost_without_a_cache_file(stub_scorer, tmp_path):
    """`select` must produce identical results with and without `cache=`.

    Before the fix, phase B's returned score dict carried only the pivot-round
    comparisons, so the ring pairs fell back to 0.5/0.5 ties whenever no cache
    file was in play -- discarding N verifier calls that had already been made
    and changing the ranking.
    """
    candidates = [f"cand{i}" for i in range(5)]
    quality = dict(zip(candidates, [0.1, 0.2, 0.3, 0.4, 0.95]))
    stub_scorer(quality)

    kw = dict(criteria=CRITERIA, n_evaluations=1, client=object(),
              progress=False, seed=0)

    without_cache = llm_verifier.select("problem", candidates, **kw)
    with_cache = llm_verifier.select(
        "problem", candidates, cache=str(tmp_path / "scores.json"), **kw)

    assert without_cache.ranking == with_cache.ranking
    assert without_cache.scores == pytest.approx(with_cache.scores)


def test_every_ring_comparison_contributes(stub_scorer, monkeypatch):
    """No comparison that was scored may silently degrade to the 0.5 default.

    `directed_reward` returns 0.5/0.5 for a cache key it cannot find, so a
    dropped comparison is invisible in the output. Give every candidate a
    quality that is never 0.5, then assert that the final aggregation never
    saw the fallback.
    """
    candidates = [f"cand{i}" for i in range(5)]
    quality = dict(zip(candidates, [0.10, 0.20, 0.30, 0.40, 0.90]))
    stub_scorer(quality)

    lookups = []
    original = llm_verifier.directed_reward

    def spy(scores, task_name, a, b, criteria_ids, n_reps):
        out = original(scores, task_name, a, b, criteria_ids, n_reps)
        lookups.append(((a, b), out))
        return out

    monkeypatch.setattr(llm_verifier, "directed_reward", spy)
    llm_verifier.select("problem", candidates, criteria=CRITERIA,
                        n_evaluations=1, client=object(), progress=False)

    defaulted = [pair for pair, out in lookups if out == (0.5, 0.5)]
    assert not defaulted, (
        f"{len(defaulted)} comparison(s) fell back to the 0.5/0.5 default "
        f"despite having been scored: {sorted(set(defaulted))}")


# --------------------------------------------------------------------------- #
# 2. A shared cache file must not leak scores between different problems.
# --------------------------------------------------------------------------- #

def test_cache_is_not_shared_across_different_problems(stub_scorer, tmp_path):
    """Two `select` calls sharing one `cache=` file must not collide.

    The cache key used the literal task name "task" for every call, so the
    second problem read back the first problem's scores and returned a winner
    without making a single verifier call.
    """
    cache = str(tmp_path / "shared.json")

    first = [f"a{i}" for i in range(5)]
    calls = stub_scorer(dict(zip(first, [0.9, 0.1, 0.1, 0.1, 0.1])))
    llm_verifier.select("problem ONE", first, criteria=CRITERIA,
                        n_evaluations=1, client=object(), progress=False,
                        cache=cache)
    calls_after_first = len(calls)
    assert calls_after_first > 0

    # A completely different problem, whose best candidate is the LAST one.
    second = [f"b{i}" for i in range(5)]
    calls.clear()
    stub_scorer(dict(zip(second, [0.1, 0.1, 0.1, 0.1, 0.9])))
    result = llm_verifier.select("problem TWO", second, criteria=CRITERIA,
                                 n_evaluations=1, client=object(),
                                 progress=False, cache=cache)

    assert len(calls) > 0, "second problem was served entirely from the " \
                           "first problem's cached scores"
    assert result.best == "b4"


def test_cache_key_changes_with_the_verifier_model(stub_scorer, tmp_path):
    """Scores from one verifier model must not be reused for another."""
    cache = str(tmp_path / "shared.json")
    candidates = [f"c{i}" for i in range(4)]
    calls = stub_scorer({c: 0.5 for c in candidates})

    kw = dict(criteria=CRITERIA, n_evaluations=1, client=object(),
              progress=False, cache=cache)
    llm_verifier.select("problem", candidates, model="model-one", **kw)
    calls.clear()
    llm_verifier.select("problem", candidates, model="model-two", **kw)

    assert len(calls) > 0, "a different verifier model reused cached scores"


# --------------------------------------------------------------------------- #
# 3. Failed verifier calls must be visible.
# --------------------------------------------------------------------------- #

def test_failed_calls_are_counted_and_warned(monkeypatch):
    """A dead backend must not look like a confident selection.

    With `progress=False` (the default off a TTY) the error log was a no-op,
    so every call could fail and `select` would still return a winner with no
    exception, no warning and no error count.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("backend is down")

    monkeypatch.setattr(fgr, "score_pair_criterion", boom)
    candidates = [f"c{i}" for i in range(5)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = llm_verifier.select("problem", candidates, criteria=CRITERIA,
                                     n_evaluations=1, client=object(),
                                     progress=False)

    assert result.n_errors > 0, "failed verifier calls were not reported"
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
        "a fully failed run produced no warning"


def test_on_error_raise_still_propagates(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("backend is down")

    monkeypatch.setattr(fgr, "score_pair_criterion", boom)
    with pytest.raises(RuntimeError):
        llm_verifier.select("problem", ["a", "b", "c"], criteria=CRITERIA,
                            n_evaluations=1, client=object(), progress=False,
                            on_error="raise")


def test_clean_run_reports_no_errors(stub_scorer):
    candidates = [f"c{i}" for i in range(4)]
    stub_scorer(dict(zip(candidates, [0.1, 0.2, 0.3, 0.9])))
    result = llm_verifier.select("problem", candidates, criteria=CRITERIA,
                                 n_evaluations=1, client=object(),
                                 progress=False)
    assert result.n_errors == 0
    assert result.best == "c3"


# --------------------------------------------------------------------------- #
# 4. Cache durability.
# --------------------------------------------------------------------------- #

def test_cache_file_is_written_atomically(stub_scorer, tmp_path):
    """The cache must never be left as a truncated, unparseable file."""
    import json

    cache = tmp_path / "scores.json"
    candidates = [f"c{i}" for i in range(4)]
    stub_scorer({c: 0.5 for c in candidates})
    llm_verifier.select("problem", candidates, criteria=CRITERIA,
                        n_evaluations=1, client=object(), progress=False,
                        cache=str(cache))

    assert cache.exists()
    json.loads(cache.read_text())          # parses
    assert not (tmp_path / "scores.json.tmp").exists()  # temp cleaned up
