"""Regression tests for score-cache identity: an entry may only be reused for
the content it was produced from.

The scorer is stubbed, so these exercise `score_directed_pairs`' cache
bookkeeping (which entry is reused, which is re-scored) without any API."""

import json

import pytest

from llm_verifier import fine_grained_reward as fgr

CRITERION = {"id": "correctness", "name": "Correctness",
             "description": "Is it right?"}
NOTE = "the note"


def _tasks(problem, traces):
    return {"task": [{"problem": problem, "trace": t, "reward": 0,
                      "images": None} for t in traces]}


@pytest.fixture
def scorer(monkeypatch):
    """Stub verifier: records every comparison it is asked to score and
    returns a score derived from the traces, so a replayed entry is
    distinguishable from a freshly scored one."""
    calls = []

    def fake_score_pair_criterion(client, problem, trace_a, trace_b, crit,
                                  note, model, images=None):
        calls.append({"problem": problem, "trace_a": trace_a,
                      "trace_b": trace_b, "crit": crit["id"], "model": model})
        return float(len(trace_a)), float(len(trace_b))

    monkeypatch.setattr(fgr, "score_pair_criterion", fake_score_pair_criterion)
    return calls


def _score(tmp_path, tasks, scorer_model="model-x", note=NOTE,
           criterion=CRITERION, cache_name="cache.json"):
    lazy = fgr.LazyClient()
    lazy._client = object()  # the stubbed scorer never uses it
    return fgr.score_directed_pairs(
        lazy, tasks, {"task": [(0, 1)]}, [criterion], note,
        n_reps=1, max_workers=1, cache_file=str(tmp_path / cache_name),
        model=scorer_model, progress=False)


def test_a_different_problem_is_not_served_from_the_cache(tmp_path, scorer):
    """The reported bug: two `select()` calls share the synthetic task name
    `"task"`, so one cache file replayed the first call's scores."""
    first = _score(tmp_path, _tasks("reverse a string", ["aa", "bbb"]))
    calls_after_first = len(scorer)
    assert calls_after_first == 1

    second = _score(tmp_path, _tasks("solve a quadratic", ["cccc", "ddddd"]))

    assert len(scorer) == calls_after_first + 1, (
        "different content must be scored, not replayed")
    assert scorer[-1]["problem"] == "solve a quadratic"
    key = fgr.cache_key("correctness", "task", 0, 1, 0)
    assert first[key] != second[key]
    assert second[key] == {"score_A": 4.0, "score_B": 5.0}


def test_identical_content_still_hits_the_cache(tmp_path, scorer):
    tasks = _tasks("reverse a string", ["aa", "bbb"])
    first = _score(tmp_path, tasks)
    assert len(scorer) == 1

    second = _score(tmp_path, _tasks("reverse a string", ["aa", "bbb"]))

    assert len(scorer) == 1, "identical content must not be re-scored"
    assert second == first


@pytest.mark.parametrize("changed", ["trace", "criterion_text", "note",
                                     "model"])
def test_every_fingerprinted_input_invalidates_a_stale_entry(tmp_path, scorer,
                                                             changed):
    tasks = _tasks("reverse a string", ["aa", "bbb"])
    _score(tmp_path, tasks)
    assert len(scorer) == 1

    kwargs = {}
    if changed == "trace":
        tasks = _tasks("reverse a string", ["aa", "bbbb"])
    elif changed == "criterion_text":
        kwargs["criterion"] = dict(CRITERION, description="Is it correct?")
    elif changed == "note":
        kwargs["note"] = "a different note"
    elif changed == "model":
        kwargs["scorer_model"] = "model-y"

    _score(tmp_path, tasks, **kwargs)

    assert len(scorer) == 2, f"changing the {changed} must re-score"


def test_stale_entries_are_kept_on_disk_beside_the_new_ones(tmp_path, scorer):
    cache = tmp_path / "cache.json"
    _score(tmp_path, _tasks("reverse a string", ["aa", "bbb"]))
    after_first = json.loads(cache.read_text())
    assert len(after_first) == 1

    _score(tmp_path, _tasks("solve a quadratic", ["cccc", "ddddd"]))
    after_second = json.loads(cache.read_text())

    assert len(after_second) == 2, "the earlier entry must not be overwritten"
    assert set(after_first) < set(after_second)
    for key in after_second:
        assert key.startswith("correctness|task|0,1|0|")


def test_legacy_entries_without_a_fingerprint_are_ignored(tmp_path, scorer):
    """A cache written before content binding cannot be shown to match, so it
    is re-scored rather than trusted."""
    cache = tmp_path / "cache.json"
    legacy_key = fgr.cache_key("correctness", "task", 0, 1, 0)
    cache.write_text(json.dumps({legacy_key: {"score_A": 9.0,
                                              "score_B": 9.0}}))

    results = _score(tmp_path, _tasks("reverse a string", ["aa", "bbb"]))

    assert len(scorer) == 1, "a legacy entry must not be replayed"
    assert results[legacy_key] == {"score_A": 2.0, "score_B": 3.0}


@pytest.mark.parametrize("source", ["path", "url", "bytes"])
def test_image_content_changes_invalidate_cache(tmp_path, monkeypatch,
                                               scorer, source):
    image = tmp_path / "context.png"
    image.write_bytes(b"first image")
    content = [b"first image"]
    if source == "url":
        from io import BytesIO

        def urlopen(url):
            return BytesIO(content[0])

        monkeypatch.setattr("urllib.request.urlopen", urlopen)

    tasks = _tasks("describe image", ["aa", "bbb"])

    def set_images():
        value = (image if source == "path" else
                 "https://example.invalid/image.png" if source == "url" else
                 content[0])
        for trial in tasks["task"]:
            trial["images"] = value
        return value

    original = set_images()
    _score(tmp_path, tasks)
    _score(tmp_path, tasks)
    assert len(scorer) == 1, "unchanged image content must hit the cache"
    assert tasks["task"][0]["images"] == original

    content[0] = b"changed image content"
    image.write_bytes(content[0])
    set_images()
    _score(tmp_path, tasks)
    assert len(scorer) == 2, "changed image content must be scored again"


def test_scoring_uses_the_same_image_snapshot_as_the_fingerprint(
        tmp_path, monkeypatch):
    image = tmp_path / "context.png"
    image.write_bytes(b"first image")
    tasks = _tasks("describe image", ["aa", "bbb"])
    for trial in tasks["task"]:
        trial["images"] = image
    seen = []

    def score(client, problem, trace_a, trace_b, crit, note, model,
              images=None):
        image.write_bytes(b"replaced after fingerprinting")
        seen.append(fgr.load_image(fgr.as_image_list(images)[0])[0])
        return .8, .2

    monkeypatch.setattr(fgr, "score_pair_criterion", score)
    _score(tmp_path, tasks)
    assert seen == [b"first image"]
    assert tasks["task"][0]["images"] == image


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("on_error", ["tie", "raise"])
def test_image_preparation_respects_error_policy(
        tmp_path, monkeypatch, use_cache, on_error):
    tasks = _tasks("synthetic image failure", ["aa", "bbb"])
    tasks["task"][0]["images"] = tmp_path / "missing.png"
    cache = tmp_path / "cache.json"
    existing = {"unrelated": {"score_A": 0.8, "score_B": 0.2}}
    if use_cache:
        cache.write_text(json.dumps(existing))
    lazy = fgr.LazyClient()
    monkeypatch.setattr(lazy, "get", lambda: pytest.fail(
        "image preparation must not construct a provider client"))

    def score():
        return fgr.score_directed_pairs(
            lazy, tasks, {"task": [(0, 1)]}, [CRITERION], NOTE,
            n_reps=2, max_workers=1,
            cache_file=str(cache) if use_cache else None,
            progress=False, on_error=on_error)

    if on_error == "raise":
        with pytest.raises(FileNotFoundError):
            score()
    else:
        results = score()
        assert results == {
            fgr.cache_key("correctness", "task", 0, 1, rep):
                {"score_A": 0.5, "score_B": 0.5}
            for rep in range(2)
        }
    if use_cache:
        assert json.loads(cache.read_text()) == existing
    else:
        assert not cache.exists()


def test_image_preparation_failure_does_not_skip_valid_comparisons(
        tmp_path, scorer):
    tasks = _tasks("synthetic mixed comparisons", ["aa", "bbb", "cccc"])
    tasks["task"][0]["images"] = tmp_path / "missing.png"
    lazy = fgr.LazyClient()
    lazy._client = object()
    cache = tmp_path / "cache.json"
    results = fgr.score_directed_pairs(
        lazy, tasks, {"task": [(0, 1), (1, 2)]}, [CRITERION], NOTE,
        n_reps=1, max_workers=1, cache_file=str(cache),
        progress=False, on_error="tie")

    failed = fgr.cache_key("correctness", "task", 0, 1, 0)
    valid = fgr.cache_key("correctness", "task", 1, 2, 0)
    assert results[failed] == {"score_A": 0.5, "score_B": 0.5}
    assert results[valid] == {"score_A": 3.0, "score_B": 4.0}
    assert len(scorer) == 1
    disk = json.loads(cache.read_text())
    assert len(disk) == 1
    assert next(iter(disk)).startswith(valid + "|")


def test_unrequested_image_failure_does_not_abort_cache_translation(
        tmp_path, scorer):
    tasks = _tasks("synthetic unrelated image", ["aa", "bbb", "cccc"])
    tasks["task"][2]["images"] = tmp_path / "missing.png"
    first = _score(tmp_path, tasks)
    second = _score(tmp_path, tasks)
    assert first == second
    assert len(scorer) == 1
