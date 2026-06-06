from __future__ import annotations

from lcg.nli.scorer import HeuristicNLI


def test_high_overlap() -> None:
    nli = HeuristicNLI()
    score = nli.entailment(
        "Apples are red and crunchy",
        "Apples are red",
    )
    assert score > 0.5


def test_low_overlap() -> None:
    nli = HeuristicNLI()
    score = nli.entailment(
        "Apples are red",
        "The moon orbits the earth",
    )
    assert score < 0.2


def test_empty_hypothesis() -> None:
    nli = HeuristicNLI()
    assert nli.entailment("anything", "") == 0.0
