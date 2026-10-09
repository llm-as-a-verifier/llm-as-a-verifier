"""Regression coverage for whitespace-only prefill samples."""

from llm_verifier.fine_grained_reward import _find_tag_logprobs, extract_score


TAG = "<score_A>"
CLOSING = "</score_A>"
ALTS = [(" ", -0.116), (" A", -2.616), (" B", -4.366)]


def _build(sampled_content):
    letter = sampled_content.strip()
    text = "analysis\n" + TAG + letter + CLOSING
    tokens = ["\n" + TAG, letter, CLOSING]
    logprobs = [[("\n" + TAG, 0.0)], ALTS, [(CLOSING, 0.0)]]
    return text, tokens, logprobs


def test_whitespace_sample_keeps_distribution():
    text, tokens, logprobs = _build(" ")
    assert _find_tag_logprobs(tokens, logprobs, TAG) == ALTS
    assert extract_score(text, tokens, logprobs, TAG) != 0.5


def test_whitespace_sample_matches_letter_sample():
    assert extract_score(*_build(" "), TAG) == extract_score(*_build("A"), TAG)


def test_empty_distribution_remains_tie():
    text, tokens, _ = _build(" ")
    logprobs = [[("\n" + TAG, 0.0)], [], [(CLOSING, 0.0)]]
    assert extract_score(text, tokens, logprobs, TAG) == 0.5
