"""Tests for the score-token decoder: `extract_score` / `_find_tag_logprobs`.

These two functions turn a verifier response into the fine-grained reward, so
every backend quirk lands here: tokenizers that fuse the tag's closing ">"
with the score letter, reasoning parsers that emit empty tokens, models that
quote the output format mid-analysis before giving their real verdict, and
top-k alternatives that are truncated (and so don't sum to 1).

Each failure mode is silent — a mis-read distribution returns a plausible
number, not an error — which is why they are pinned here rather than left to
integration runs.

Scale reminder (`SCALE["valid_tokens"]`): A=20 (best) .. T=1 (worst), and
`extract_score` normalizes the expectation to [0, 1] as (E - 1) / 19. So a
certain A is 1.0, a certain T is 0.0.
"""

import math

import pytest

from llm_verifier.fine_grained_reward import _find_tag_logprobs, extract_score

TAG_A = "<score_A>"
TAG_B = "<score_B>"


def norm(expected_value):
    """The [0, 1] reward for a given expected letter value."""
    return (expected_value - 1.0) / 19.0


def alts(*pairs):
    """Build a position's top-k list from (token, probability) pairs."""
    return [(tok, math.log(p)) for tok, p in pairs]


# ---------------------------------------------------------------------------
# The expectation itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("letter,value", [
    ("A", 20.0),   # best
    ("B", 19.0),
    ("J", 11.0),
    ("T", 1.0),    # worst
])
def test_certain_letter_decodes_to_its_scale_value(letter, value):
    tokens = [TAG_A, letter]
    lps = [alts((TAG_A, 1.0)), alts((letter, 1.0))]

    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(norm(value))


def test_expectation_is_taken_over_the_whole_distribution():
    """The point of the framework: a split distribution gives a continuous
    score, not the argmax letter."""
    tokens = [TAG_A, "A"]
    lps = [alts((TAG_A, 1.0)), alts(("A", 0.8), ("T", 0.2))]

    # E = 20*0.8 + 1*0.2 = 16.2
    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(norm(16.2))


def test_truncated_top_k_is_renormalized():
    """Backends return only the top-20 alternatives, so the probabilities
    reaching us do not sum to 1. The expectation must divide by the mass it
    actually saw, not treat the rest as zero."""
    tokens = [TAG_A, "A"]
    lps = [alts((TAG_A, 1.0)), alts(("A", 0.25), ("T", 0.25))]

    # E = (20*0.25 + 1*0.25) / 0.5 = 10.5 -> exactly mid-scale
    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(0.5)


def test_lowercase_letters_share_their_uppercase_value():
    tokens = [TAG_A, "a"]
    lps = [alts((TAG_A, 1.0)), alts(("a", 1.0))]

    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(1.0)


def test_duplicate_spellings_take_the_max_not_the_sum():
    """A model may put mass on both "A" and " A" (with the leading space).
    They are the same score, so the larger is kept — summing them would
    double-count that letter against the rest of the distribution."""
    tokens = [TAG_A, "A"]
    lps = [alts((TAG_A, 1.0)),
           alts(("A", 0.3), (" A", 0.6), ("T", 0.4))]

    # A -> max(0.3, 0.6) = 0.6; total mass 1.0; E = 20*0.6 + 1*0.4 = 12.4
    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(norm(12.4))


def test_non_scale_tokens_are_ignored():
    """Punctuation, digits and words at the answer position contribute
    nothing; only the mass on real scale letters counts."""
    tokens = [TAG_A, "A"]
    lps = [alts((TAG_A, 1.0)),
           alts(("A", 0.5), ("the", 0.3), ("12", 0.1), ("!", 0.1))]

    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(1.0)


def test_letter_fused_with_the_closing_angle_bracket():
    """DeepSeek's tokenizer emits ">A" as one token at the answer position."""
    tokens = [TAG_A, ">A"]
    lps = [alts((TAG_A, 1.0)), alts((">A", 0.7), (">T", 0.3))]

    # E = 20*0.7 + 1*0.3 = 14.3
    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(norm(14.3))


# ---------------------------------------------------------------------------
# Locating the tag
# ---------------------------------------------------------------------------

def test_prefill_token_shape_is_decoded():
    """The exact shape `_score_tags_by_prefill` produces on the vLLM path:
    the tag arrives newline-prefixed, followed by the letter and closer."""
    tokens = ["\n" + TAG_A, "A", "</score_A>"]
    lps = [alts(("\n" + TAG_A, 1.0)), alts(("A", 1.0)),
           alts(("</score_A>", 1.0))]

    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(1.0)


def test_trailing_whitespace_after_the_tag_still_matches():
    tokens = [TAG_A + "\n", "A"]
    lps = [alts((TAG_A + "\n", 1.0)), alts(("A", 1.0))]

    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(1.0)


def test_last_occurrence_wins_over_a_quoted_format():
    """Models often restate the output format mid-analysis before committing.
    The verdict is the final score block, so an earlier mention must not
    shadow it."""
    tokens = [TAG_A, "T", " ...analysis... ", TAG_A, "A"]
    lps = [alts((TAG_A, 1.0)),
           alts(("T", 1.0)),                    # decoy: the quoted example
           alts((" ...analysis... ", 1.0)),
           alts((TAG_A, 1.0)),
           alts(("A", 1.0))]                    # the real verdict

    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(1.0)


def test_tag_fused_with_its_closing_bracket_falls_back_to_the_bare_tag():
    """Some tokenizers never emit "<score_A>" as a boundary — the ">" is
    fused into the next token. Matching the tag without its trailing ">"
    recovers the answer position."""
    tokens = ["<score_A", ">A"]
    lps = [alts(("<score_A", 1.0)), alts((">A", 0.7), (">T", 0.3))]

    assert _find_tag_logprobs(tokens, lps, TAG_A) is lps[1]
    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(norm(14.3))


@pytest.mark.parametrize("blank", ["", " ", "\n", "  \t "])
def test_blank_token_after_the_tag_does_not_shadow_the_distribution(blank):
    """Regression for #5 / #10 / #11: a reasoning parser can swallow the
    letter into an empty token, and a constrained sample can land on a bare
    space (a legal prefix of " A"). Either leaves the accumulated text
    unchanged, so the tag would match a SECOND time and the real
    distribution at the previous position would be shadowed by whatever
    follows the blank."""
    tokens = [TAG_A, blank, "A"]
    real = alts(("A", 0.7), ("T", 0.3))         # position right after the tag
    decoy = alts(("T", 1.0))                    # would win without the guard
    lps = [alts((TAG_A, 1.0)), real, decoy]

    assert _find_tag_logprobs(tokens, lps, TAG_A) is real
    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(norm(14.3))


def test_tag_as_the_final_token_has_no_answer_position():
    """Nothing follows the tag, so there is no distribution to read — this
    must fall through rather than index past the end."""
    tokens = ["some analysis", TAG_A]
    lps = [alts(("some analysis", 1.0)), alts((TAG_A, 1.0))]

    assert _find_tag_logprobs(tokens, lps, TAG_A) is None
    assert extract_score("", tokens, lps, TAG_A) == 0.5


@pytest.mark.parametrize("tokens,lps", [
    (None, None),
    ([], []),
    (["no tag here"], [[("no tag here", 0.0)]]),
])
def test_missing_tag_returns_none(tokens, lps):
    assert _find_tag_logprobs(tokens, lps, TAG_A) is None


def test_both_tags_are_read_from_one_response():
    """The realistic shape: a single reply carries both scores, and each tag
    must pick up its own distribution."""
    tokens = [TAG_A, "A", "\n", TAG_B, "T"]
    lps = [alts((TAG_A, 1.0)), alts(("A", 1.0)), alts(("\n", 1.0)),
           alts((TAG_B, 1.0)), alts(("T", 1.0))]

    assert extract_score("", tokens, lps, TAG_A) == pytest.approx(1.0)
    assert extract_score("", tokens, lps, TAG_B) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Text fallback — used when the backend returns no usable logprobs
# ---------------------------------------------------------------------------

def test_text_fallback_parses_the_tagged_letter():
    text = "Analysis here.\n<score_A> C </score_A>\n<score_B> S </score_B>"

    assert extract_score(text, None, None, TAG_A) == pytest.approx(norm(18.0))
    assert extract_score(text, None, None, TAG_B) == pytest.approx(norm(2.0))


def test_text_fallback_takes_the_last_block():
    text = "Format is <score_A> T </score_A>. Verdict:\n<score_A> A </score_A>"

    assert extract_score(text, None, None, TAG_A) == pytest.approx(1.0)


def test_text_fallback_is_case_insensitive():
    text = "<SCORE_A> b </SCORE_A>"

    assert extract_score(text, None, None, TAG_A) == pytest.approx(norm(19.0))


def test_logprobs_take_precedence_over_the_text():
    """When both are available the distribution is the better estimate — the
    sampled letter in the text is just its argmax."""
    tokens = [TAG_A, "A"]
    lps = [alts((TAG_A, 1.0)), alts(("A", 0.8), ("T", 0.2))]

    score = extract_score("<score_A> A </score_A>", tokens, lps, TAG_A)

    assert score == pytest.approx(norm(16.2))   # not 1.0


@pytest.mark.parametrize("text", [
    "",
    None,
    "The agent did well but I decline to score it.",
    "<score_A> Z </score_A>",          # not a letter on the A-T scale
    "<score_A> excellent </score_A>",  # not a single letter
])
def test_undecodable_response_scores_a_neutral_tie(text):
    """No distribution and no parsable letter: 0.5 for both sides, which the
    tournament reads as no preference."""
    assert extract_score(text, None, None, TAG_A) == 0.5
