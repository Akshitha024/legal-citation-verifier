from __future__ import annotations

from lcg.claims.decompose import (
    claims_from_answer,
    parse_citations,
    sentences,
)


def test_sentence_split_basic() -> None:
    s = sentences("Apple is red. Banana is yellow. Cherry is dark.")
    assert s == ["Apple is red.", "Banana is yellow.", "Cherry is dark."]


def test_sentence_split_handles_quote_after_punct() -> None:
    s = sentences('First sentence. "Second sentence" here.')
    assert len(s) == 2


def test_parse_citations_simple() -> None:
    text, cits = parse_citations("Section 16600 voids non-competes [s1].")
    assert text == "Section 16600 voids non-competes ."
    assert cits == ("s1",)


def test_parse_citations_multiple() -> None:
    text, cits = parse_citations("Many things [s1, s2, s3].")
    assert "many things" in text.lower()
    assert cits == ("s1", "s2", "s3")


def test_parse_citations_leaves_case_titles_alone() -> None:
    text, cits = parse_citations("In [Smith v. Jones], the court held that...")
    # the bracketed text isn't a list of ids so we keep it
    assert "[Smith v. Jones]" in text
    assert cits == ()


def test_claims_from_answer_extracts_citations() -> None:
    answer = "First fact [s1]. Second fact [s2, s3]. Unsourced fact."
    claims = claims_from_answer(answer)
    assert len(claims) == 3
    assert claims[0][1] == ("s1",)
    assert set(claims[1][1]) == {"s2", "s3"}
    assert claims[2][1] == ()
