"""NLI entailment scorer.

Default model: ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` (a 184M
DeBERTa-v3 fine-tuned on MNLI + FEVER + ANLI). It is the smallest open
NLI model I know that holds up on out-of-domain text, which matters for
legal premises. The fallback is a token-overlap heuristic if torch is
unavailable.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ..claims.decompose import _CITATION  # noqa: F401  (used to clean claims)


class NLIHead:
    name = "deberta-mnli"

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    ) -> None:
        self.model_name = model_name
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            logger.info("loading NLI model {}", self.model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
            )
        return self._pipeline

    def entailment(self, premise: str, hypothesis: str) -> float:
        """Return P(entailment) for premise -> hypothesis.

        DeBERTa-MNLI uses labels {entailment, neutral, contradiction}. We
        return the entailment probability. The model expects the SEP
        sequence; the HF pipeline handles it for us.
        """
        pipe = self._load()
        out = pipe(
            f"{premise}",
            text_pair=hypothesis,
        )
        # out is a list of dicts [{label: 'entailment', score: ..}, ...]
        for r in out:
            if str(r["label"]).lower().startswith("entail"):
                return float(r["score"])
        return 0.0


class HeuristicNLI:
    """Fallback NLI: token Jaccard. Coarse but works without a model."""

    name = "jaccard"

    def entailment(self, premise: str, hypothesis: str) -> float:
        import re

        def toks(t: str) -> set[str]:
            return {w for w in re.findall(r"\w+", t.lower()) if len(w) > 2}

        p = toks(premise)
        h = toks(hypothesis)
        if not h:
            return 0.0
        return len(p & h) / len(h)
