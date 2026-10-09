"""Progress expectations must ignore non-score word alternatives."""
import math
import unittest
from unittest.mock import patch

from llm_verifier import progress


class ProgressTokenTests(unittest.TestCase):
    def test_words_are_not_score_letters(self):
        for word in ("The", "Task", "Agent", "Maybe", "answer", ">The"):
            with self.subTest(word=word):
                self.assertIsNone(progress._expected_value_from_alts([(word, 0.0)]))

    def test_valid_probability_mass_is_renormalized(self):
        alts = [("A", math.log(.3)), ("T", math.log(.2)), ("Task", math.log(.5))]
        self.assertAlmostEqual(progress._expected_value_from_alts(alts), .4)

    def test_fused_letters_and_whitespace_are_preserved(self):
        for token in ("B", " b ", ">B", " > B ", "B</c1>", ">B</c1>", "B</"):
            with self.subTest(token=token):
                self.assertAlmostEqual(
                    progress._expected_value_from_alts([(token, 0.0)]),
                    progress.LETTER_TO_VALUE["B"],
                )

    def test_duplicate_letter_spellings_keep_maximum_probability(self):
        alts = [("A", math.log(.2)), (" A", math.log(.3)), ("T", math.log(.5))]
        self.assertAlmostEqual(progress._expected_value_from_alts(alts), .625)

    def test_invalid_alternatives_use_text_fallback(self):
        self.assertEqual(progress.extract_progress_scores(
            "<c1>A</c1>", ["<c1>", "A", "</c1>"],
            [[], [("The", 0.0)], []], 1,
        ), [0.0])

    def test_public_tracking_ignores_word_alternatives(self):
        response = (
            "<c1>A</c1>", ["<c1>", "A", "</c1>"],
            [[], [("A", math.log(.6)), ("The", math.log(.4))], []],
        )
        with patch.object(progress, "call_verifier", return_value=response) as call:
            result = progress.track("task", ["No work done"], client=object())
            tracker = progress.ProgressTracker("task", client=object())
            score = tracker.update("No work done")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result.final, 0.0)
        self.assertEqual(score, 0.0)
        self.assertEqual(tracker.result().final, 0.0)


if __name__ == "__main__":
    unittest.main()
