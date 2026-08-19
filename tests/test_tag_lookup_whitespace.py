"""Regression: the score distribution must survive a whitespace-only sample.

When the grammar-constrained prefill lands on a bare space -- a legal prefix of
" A" in the choice list, and common with tokenizers that emit a standalone space
token -- `letter` strips to "". The empty token leaves `text_so_far` unchanged in
`_find_tag_logprobs`, so the tag matches a SECOND time and `found` advances one
slot past the real distribution onto the closing tag's zero-logprob placeholder.

Measured on a self-hosted DeepSeek-V4-Flash: 78 of 357 cached scores (22%)
silently became exact 0.5 ties, with no exception raised and nothing in the run
output to indicate a problem.
"""
from llm_verifier.fine_grained_reward import extract_score, _find_tag_logprobs

TAG = "<score_A>"
CLOSING = "</" + TAG[1:]

# A real top-20 from a constrained prefill call: 19 of 20 alternatives are valid
# score letters, so the distribution is healthy no matter what was sampled.
ALTS = [(" ", -0.116), (" A", -2.616), (" B", -4.366),
        (" C", -5.054), (" P", -5.054), (" R", -5.491)]


def _build(sampled_content):
    """Exactly what _score_tags_by_prefill constructs for one tag."""
    letter = sampled_content.strip()
    text = "analysis" + f"\n{TAG}" + letter + CLOSING
    tokens = [f"\n{TAG}", letter, CLOSING]
    plp = [[(f"\n{TAG}", 0.0)], ALTS, [(CLOSING, 0.0)]]
    return text, tokens, plp


def test_bare_space_sample_still_uses_the_distribution():
    """A whitespace-only sample must not discard the distribution beside it."""
    text, tokens, plp = _build(" ")
    assert _find_tag_logprobs(tokens, plp, TAG) == ALTS
    assert extract_score(text, tokens, plp, TAG) != 0.5


def test_bare_space_matches_letter_sample_exactly():
    """Same distribution => same score, whichever token happened to be sampled.

    This is the strong form: the fix must not merely avoid 0.5, it must produce
    the identical value, proving correct reads are unaffected.
    """
    s_space = extract_score(*_build(" "), TAG)
    s_letter = extract_score(*_build("A"), TAG)
    assert s_space == s_letter


def test_genuinely_empty_distribution_still_ties():
    """Negative control: with nothing to read, 0.5 remains correct."""
    text, tokens, _ = _build(" ")
    plp = [[(f"\n{TAG}", 0.0)], [], [(CLOSING, 0.0)]]
    assert extract_score(text, tokens, plp, TAG) == 0.5


def test_second_tag_unaffected_by_first_tags_empty_token():
    """Cumulative token lists must not let tag A's empty token break tag B."""
    tag2, closing2 = "<score_B>", "</score_B>"
    alts2 = [(" ", -2.091), (" A", -0.591), (" C", -2.591)]
    tokens = [f"\n{TAG}", "", CLOSING, f"\n{tag2}", "", closing2]
    plp = [[(f"\n{TAG}", 0.0)], ALTS, [(CLOSING, 0.0)],
           [(f"\n{tag2}", 0.0)], alts2, [(closing2, 0.0)]]
    text = "analysis" + f"\n{TAG}" + CLOSING + f"\n{tag2}" + closing2
    assert _find_tag_logprobs(tokens, plp, tag2) == alts2
    assert extract_score(text, tokens, plp, tag2) != 0.5
