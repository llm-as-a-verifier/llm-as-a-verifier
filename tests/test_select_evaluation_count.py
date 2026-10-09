"""Regression tests for select evaluation-count validation."""
import unittest
from unittest.mock import patch

import llm_verifier
from llm_verifier import fine_grained_reward


class SelectEvaluationCountTests(unittest.TestCase):
    def test_nonpositive_evaluations_are_rejected_before_setup(self):
        for count in (0, -1):
            for candidates in (["only"], ["a", "b", "c"]):
                with self.subTest(count=count, candidates=candidates):
                    with patch("llm_verifier._resolve_criteria", wraps=llm_verifier._resolve_criteria) as resolve:
                        with self.assertRaisesRegex(
                            ValueError, "n_evaluations must be >= 1"
                        ):
                            llm_verifier.select(
                                "task", candidates, criteria={"C": "criterion"},
                                n_evaluations=count, progress=False,
                            )
                        resolve.assert_not_called()

    def test_positive_singleton_preserves_fast_path(self):
        result = llm_verifier.select(
            "task", ["only"], criteria={"C": "criterion"},
            n_evaluations=1, progress=False,
        )
        self.assertEqual(result.index, 0)
        self.assertEqual(result.scores, [1.0])
        self.assertEqual(result.n_comparisons, 0)

    def test_positive_evaluations_reach_scorer(self):
        for count in (1, 2):
            with self.subTest(count=count):
                with patch.object(
                    fine_grained_reward, "score_pair_criterion",
                    return_value=(0.8, 0.2),
                ) as scorer:
                    result = llm_verifier.select(
                        "task", ["a", "b"], criteria={"C": "criterion"},
                        n_evaluations=count, progress=False, client=object(),
                    )
                self.assertGreater(scorer.call_count, 0)
                self.assertEqual(len(result.scores), 2)


if __name__ == "__main__":
    unittest.main()
