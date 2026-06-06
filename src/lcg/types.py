"""Core dataclasses.

A citation-grounded answer = a sequence of claims, each with zero or more
citation indices pointing into the retrieved source corpus.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    sid: str
    text: str
    title: str | None = None


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[str, ...]  # source ids the claim cites


@dataclass(frozen=True)
class CitedAnswer:
    qid: str
    question: str
    claims: tuple[Claim, ...]
    sources: tuple[Source, ...]


# verifier verdict per claim
@dataclass
class ClaimVerdict:
    claim_idx: int
    claim_text: str
    entailment: float  # 0..1 from NLI head
    cited_supports: bool  # any cited source entails the claim?
    any_source_supports: bool  # any source in the candidate set entails it?
    flagged: bool  # = (not cited_supports) and (not any_source_supports)
    rationale: str


@dataclass
class AnswerVerdict:
    qid: str
    per_claim: list[ClaimVerdict]
    citation_precision: float  # of cited sources, fraction that entail
    citation_recall: (
        float  # of claims that need support, fraction with at least one cited supporter
    )
    hallucination_rate: float  # fraction of claims flagged
