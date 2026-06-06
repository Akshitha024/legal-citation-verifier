from __future__ import annotations

from lcg.nli.scorer import HeuristicNLI
from lcg.types import CitedAnswer, Claim, Source
from lcg.verifier.check import Verifier


def _ans(
    claims_with_citations: list[tuple[str, tuple[str, ...]]], sources: list[Source]
) -> CitedAnswer:
    return CitedAnswer(
        qid="q",
        question="?",
        claims=tuple(Claim(text=t, citations=c) for t, c in claims_with_citations),
        sources=tuple(sources),
    )


def test_clean_attribution() -> None:
    src = Source(sid="s1", text="Apples are red and round.")
    a = _ans([("Apples are red and round.", ("s1",))], [src])
    v = Verifier(nli=HeuristicNLI(), threshold=0.3)
    verdict = v.verify(a)
    assert verdict.per_claim[0].cited_supports is True
    assert verdict.hallucination_rate == 0.0


def test_bad_citation_flagged() -> None:
    src_correct = Source(sid="s1", text="Apples are red.")
    src_off = Source(sid="s2", text="Bananas are yellow.")
    # claim cites s2 even though s1 is the right source
    a = _ans([("Apples are red.", ("s2",))], [src_correct, src_off])
    v = Verifier(nli=HeuristicNLI(), threshold=0.4)
    verdict = v.verify(a)
    # cited (s2) doesn't support but s1 in the global pool does
    assert verdict.per_claim[0].cited_supports is False
    assert verdict.per_claim[0].any_source_supports is True
    assert verdict.per_claim[0].flagged is False  # global supports rescues it


def test_pure_hallucination_flagged() -> None:
    src = Source(sid="s1", text="Apples are red.")
    a = _ans([("The moon is made of cheese.", ("s1",))], [src])
    v = Verifier(nli=HeuristicNLI(), threshold=0.4)
    verdict = v.verify(a)
    assert verdict.per_claim[0].flagged is True
    assert verdict.hallucination_rate == 1.0


def test_empty_answer_zero_rate() -> None:
    a = _ans([], [Source(sid="s1", text="x")])
    v = Verifier(nli=HeuristicNLI())
    verdict = v.verify(a)
    assert verdict.hallucination_rate == 0.0
