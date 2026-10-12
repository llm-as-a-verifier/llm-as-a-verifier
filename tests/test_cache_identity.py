import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_verifier import select


class CacheIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.cache = str(Path(self.temp_dir.name) / "scores.json")
        self.criteria = [{
            "id": "correctness",
            "name": "Correctness",
            "description": "Prefer a correct solution.",
        }]

    def select_with_stubbed_verifier(self, **overrides):
        arguments = {
            "problem": "Fix the failing test.",
            "candidates": ["candidate zero", "candidate one"],
            "criteria": self.criteria,
            "n_evaluations": 1,
            "pivots": 1,
            "seed": 0,
            "max_workers": 1,
            "model": "verifier-v1",
            "cache": self.cache,
            "progress": False,
            "client": object(),
        }
        arguments.update(overrides)
        response = (
            "<score_A> A </score_A>\n<score_B> T </score_B>",
            None,
            None,
        )
        with patch("llm_verifier.fine_grained_reward.call_verifier",
                   return_value=response) as verifier:
            select(**arguments)
        return verifier.call_count

    def test_cache_is_reused_only_for_identical_verifier_inputs(self):
        self.assertGreater(self.select_with_stubbed_verifier(), 0)
        self.assertEqual(self.select_with_stubbed_verifier(), 0)

        changed_inputs = [
            {"problem": "Fix a different failing test."},
            {"candidates": ["changed candidate zero", "candidate one"]},
            {"criteria": [{
                "id": "correctness",
                "name": "Correctness",
                "description": "Require tests as evidence of correctness.",
            }]},
            {"model": "verifier-v2"},
            {"images": b"different image content"},
        ]
        for change in changed_inputs:
            with self.subTest(change=change):
                # Restore the baseline entry so this subtest isolates exactly
                # one changed verifier input.
                self.select_with_stubbed_verifier()
                self.assertGreater(self.select_with_stubbed_verifier(**change),
                                   0)

    def test_legacy_entries_without_an_input_fingerprint_are_rescored(self):
        self.assertGreater(self.select_with_stubbed_verifier(), 0)
        with open(self.cache, encoding="utf-8") as cache_file:
            entries = json.load(cache_file)
        for entry in entries.values():
            entry.pop("input_sha256", None)
        with open(self.cache, "w", encoding="utf-8") as cache_file:
            json.dump(entries, cache_file)

        self.assertGreater(self.select_with_stubbed_verifier(), 0)

    def test_local_image_content_change_invalidates_the_cache(self):
        image = Path(self.temp_dir.name) / "context.png"
        image.write_bytes(b"first image content")

        self.assertGreater(
            self.select_with_stubbed_verifier(images=str(image)), 0)
        self.assertEqual(
            self.select_with_stubbed_verifier(images=str(image)), 0)

        image.write_bytes(b"updated image content")
        self.assertGreater(
            self.select_with_stubbed_verifier(images=str(image)), 0)


if __name__ == "__main__":
    unittest.main()
