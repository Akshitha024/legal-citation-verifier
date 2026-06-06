"""The verifier loop.

For each claim in a cited answer:
  1. Run NLI(any cited source, claim) -> max entailment over cited sources.
  2. Run NLI(any source in the candidate set, claim) -> max over all sources.
  3. flagged := (cited_supports < THRESHOLD) and (any_source_supports < THRESHOLD)

A flagged claim is either un-attributable (no source supports it) OR
attributed-but-not-actually-supported (the cite is wrong). Both are
hallucination signals; the per-flag split is in the verdict so downstream
code can route them differently.
"""

from __future__ import annotations

from loguru import logger

from ..claims.decompose import claims_from_answer
from ..nli.scorer import HeuristicNLI, NLIHead
from ..types import AnswerVerdict, CitedAnswer, ClaimVerdict, Source

DEFAULT_THRESHOLD = 0.5


class Verifier:
    def __init__(
        self, nli: NLIHead | HeuristicNLI | None = None, threshold: float = DEFAULT_THRESHOLD
    ) -> None:
        self.nli = nli or HeuristicNLI()
        self.threshold = threshold

    def verify(self, answer: CitedAnswer) -> AnswerVerdict:
        per_claim: list[ClaimVerdict] = []
        # the answer may have its claims pre-decomposed (CitedAnswer.claims) or
        # not; if not, we re-derive from the joined text. For now CitedAnswer
        # always has .claims set by the caller.
        for i, claim in enumerate(answer.claims):
            cited_max = _max_entailment(
                self.nli,
                _pick_sources(answer.sources, claim.citations),
                claim.text,
            )
            global_max = _max_entailment(self.nli, answer.sources, claim.text)
            cited_supports = cited_max >= self.threshold
            global_supports = global_max >= self.threshold
            flagged = (not cited_supports) and (not global_supports)
            rationale = (
                f"cited_entail_max={cited_max:.3f}; "
                f"global_entail_max={global_max:.3f}; "
                f"threshold={self.threshold:.2f}"
            )
            per_claim.append(
                ClaimVerdict(
                    claim_idx=i,
                    claim_text=claim.text,
                    entailment=cited_max,
                    cited_supports=cited_supports,
                    any_source_supports=global_supports,
                    flagged=flagged,
                    rationale=rationale,
                )
            )
        return _aggregate(answer.qid, per_claim)


def _pick_sources(sources: tuple[Source, ...], cite_ids: tuple[str, ...]) -> tuple[Source, ...]:
    by_id = {s.sid: s for s in sources}
    return tuple(by_id[c] for c in cite_ids if c in by_id)


def _max_entailment(nli: NLIHead | HeuristicNLI, sources: tuple[Source, ...], claim: str) -> float:
    if not sources:
        return 0.0
    best = 0.0
    for s in sources:
        v = nli.entailment(s.text, claim)
        if v > best:
            best = v
    return best


def _aggregate(qid: str, per_claim: list[ClaimVerdict]) -> AnswerVerdict:
    if not per_claim:
        return AnswerVerdict(
            qid=qid,
            per_claim=[],
            citation_precision=0.0,
            citation_recall=0.0,
            hallucination_rate=0.0,
        )
    needs_support = per_claim  # treat every claim as needing support; tighten later
    recall = sum(1 for v in needs_support if v.cited_supports) / len(needs_support)
    cited_with_any_cite = [
        v for v in per_claim if (v.entailment > 0 or v.cited_supports or v.any_source_supports)
    ]
    precision = (
        sum(1 for v in cited_with_any_cite if v.cited_supports) / len(cited_with_any_cite)
        if cited_with_any_cite
        else 0.0
    )
    halluc = sum(1 for v in per_claim if v.flagged) / len(per_claim)
    logger.debug(
        "{}: P={:.3f} R={:.3f} halluc={:.3f}",
        qid,
        precision,
        recall,
        halluc,
    )
    return AnswerVerdict(
        qid=qid,
        per_claim=per_claim,
        citation_precision=precision,
        citation_recall=recall,
        hallucination_rate=halluc,
    )


def verify_text(
    qid: str,
    question: str,
    answer_text: str,
    sources: tuple[Source, ...],
    nli: NLIHead | HeuristicNLI | None = None,
) -> AnswerVerdict:
    """Convenience: parse the answer text, build the CitedAnswer, verify."""
    from ..types import CitedAnswer, Claim

    pairs = claims_from_answer(answer_text)
    claims = tuple(Claim(text=t, citations=c) for t, c in pairs)
    ca = CitedAnswer(qid=qid, question=question, claims=claims, sources=sources)
    v = Verifier(nli=nli)
    return v.verify(ca)
