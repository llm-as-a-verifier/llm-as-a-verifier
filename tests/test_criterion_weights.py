import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import pytest

import llm_verifier
from llm_verifier.benchmarks import BENCHMARKS
from llm_verifier.fine_grained_reward import (
    cache_key,
    directed_reward,
    normalize_criterion_weights,
)
from llm_verifier.prompts import load_prompts, select_criteria

_RUN_PATH = Path(__file__).parents[1] / "scripts" / "run.py"
_RUN_SPEC = importlib.util.spec_from_file_location("benchmark_runner", _RUN_PATH)
assert _RUN_SPEC is not None and _RUN_SPEC.loader is not None
benchmark_runner = importlib.util.module_from_spec(_RUN_SPEC)
_RUN_SPEC.loader.exec_module(benchmark_runner)


def test_weighted_directed_reward_preserves_macro_weight():
    scores = {
        cache_key("root", "task", 0, 1, 0): {
            "score_A": 1.0,
            "score_B": 0.0,
        },
        cache_key("root", "task", 0, 1, 1): {
            "score_A": 1.0,
            "score_B": 0.0,
        },
        cache_key("logic", "task", 0, 1, 0): {
            "score_A": 0.0,
            "score_B": 1.0,
        },
        cache_key("logic", "task", 0, 1, 1): {
            "score_A": 0.0,
            "score_B": 1.0,
        },
    }

    assert directed_reward(
        scores,
        "task",
        0,
        1,
        ["root", "logic"],
        2,
        {"root": 3.0, "logic": 1.0},
    ) == (0.75, 0.25)


@pytest.mark.parametrize("weight", [1e308, 5e-324])
def test_directed_reward_is_stable_for_extreme_relative_weights(weight):
    weights = {"root": weight, "logic": weight}
    scores = {
        cache_key(cid, "task", 0, 1, rep): {
            "score_A": 0.75,
            "score_B": 0.25,
        }
        for cid in weights
        for rep in range(2)
    }

    assert directed_reward(
        scores, "task", 0, 1, list(weights), 2, weights
    ) == (0.75, 0.25)
    assert weights == {"root": weight, "logic": weight}


def test_unweighted_directed_reward_keeps_equal_average():
    scores = {
        cache_key("root", "task", 0, 1, 0): {
            "score_A": 1.0,
            "score_B": 0.0,
        },
        cache_key("logic", "task", 0, 1, 0): {
            "score_A": 0.0,
            "score_B": 1.0,
        },
    }

    assert directed_reward(scores, "task", 0, 1, ["root", "logic"], 1) == (0.5, 0.5)


@pytest.mark.parametrize(
    "weights",
    [
        {"root": 1.0},
        {"root": 1.0, "logic": 1.0, "unknown": 1.0},
        {"root": 1.0, "logic": 0.0},
        {"root": 1.0, "logic": -1.0},
        {"root": 1.0, "logic": math.nan},
        {"root": 1.0, "logic": math.inf},
        {"root": 1.0, "logic": True},
        {"root": 1.0, "logic": "1"},
    ],
)
def test_invalid_criterion_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        normalize_criterion_weights(["root", "logic"], weights)


def test_normalized_weights_follow_criterion_order():
    assert normalize_criterion_weights(["root", "logic"], {"logic": 1, "root": 3}) == {
        "root": 3.0,
        "logic": 1.0,
    }


def test_single_candidate_still_validates_weights():
    with pytest.raises(ValueError):
        llm_verifier.select(
            "problem",
            ["only"],
            criteria={"Root": "criterion"},
            criterion_weights={"unknown": 1.0},
        )


def test_compare_applies_criterion_weights(monkeypatch):
    def fake_score(_client, _problem, _a, _b, criterion, *_args):
        if criterion["id"] == "root":
            return 1.0, 0.0
        return 0.0, 1.0

    monkeypatch.setattr(llm_verifier, "score_pair_criterion", fake_score)
    criteria = [
        {"id": "root", "name": "Root", "description": "root"},
        {"id": "logic", "name": "Logic", "description": "logic"},
    ]

    assert llm_verifier.compare(
        "problem",
        "A",
        "B",
        criteria=criteria,
        client=object(),
        criterion_weights={"root": 3.0, "logic": 1.0},
    ) == (0.75, 0.25)

    assert llm_verifier.compare(
        "problem",
        "A",
        "B",
        criteria=criteria,
        client=object(),
    ) == (0.5, 0.5)


@pytest.mark.parametrize("weight", [1e308, 5e-324])
def test_compare_is_stable_for_extreme_relative_weights(monkeypatch, weight):
    monkeypatch.setattr(
        llm_verifier, "score_pair_criterion", lambda *_args: (0.75, 0.25)
    )
    criteria = [
        {"id": "root", "name": "Root", "description": "root"},
        {"id": "logic", "name": "Logic", "description": "logic"},
    ]
    weights = {"root": weight, "logic": weight}

    assert llm_verifier.compare(
        "problem", "A", "B", criteria=criteria, client=object(),
        criterion_weights=weights, n_evaluations=2,
    ) == (0.75, 0.25)
    assert weights == {"root": weight, "logic": weight}


def test_swe_bench_logic_preserves_three_macro_weights():
    cfg = BENCHMARKS["swe_bench_logic"]
    _, available = load_prompts(cfg.prompts)
    selected = select_criteria(available, cfg.criteria)
    ids = [criterion["id"] for criterion in selected]

    assert ids == [
        "root_cause",
        "logic_l1",
        "logic_l2",
        "logic_l3",
        "logic_l4",
        "logic_l5",
        "logic_l6",
        "verification",
    ]
    assert cfg.criteria_weights["root_cause"] == 1.0
    assert sum(
        cfg.criteria_weights[f"logic_l{i}"] for i in range(1, 7)
    ) == pytest.approx(1.0)
    assert cfg.criteria_weights["verification"] == 1.0


def test_benchmark_runner_uses_configured_weights(monkeypatch):
    cfg = replace(
        BENCHMARKS["swe_bench_logic"],
        loader="criterion_weight_test",
        data={},
    )
    tasks = {
        "task": [
            {"problem": "problem", "trace": "pass", "reward": 1},
            {"problem": "problem", "trace": "fail", "reward": 0},
        ]
    }
    monkeypatch.setitem(
        benchmark_runner.LOADERS,
        "criterion_weight_test",
        lambda _data, _root: (tasks, 2),
    )
    monkeypatch.setattr(
        benchmark_runner,
        "score_directed_pairs",
        lambda *_args, **_kwargs: {},
    )
    observed_weights = []

    def fake_directed_reward(
        _scores, _task, _a, _b, _criteria_ids, _n_reps, criterion_weights
    ):
        observed_weights.append(criterion_weights)
        return 0.75, 0.25

    monkeypatch.setattr(benchmark_runner, "directed_reward", fake_directed_reward)

    benchmark_runner.run_benchmark(cfg)

    assert observed_weights
    assert all(weights == cfg.criteria_weights for weights in observed_weights)
