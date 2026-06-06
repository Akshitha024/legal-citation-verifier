"""Atomic claim decomposition.

The naive approach: split on sentences (`nltk.sent_tokenize`). This is the
right default for legal answers, which are typically short and well-formed.
For long-form generations the FActScore-style "atomic-fact" decomposition
gives finer granularity (one sentence -> N facts) but requires an LLM call;
that variant is offered behind `decompose_atomic`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def sentences(text: str) -> list[str]:
    """Sentence split without nltk (avoid the data-download dependency)."""
    text = text.strip()
    if not text:
        return []
    # break on . ! ? followed by whitespace+capital, also on newlines
    parts: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if buf:
                parts.append(" ".join(buf))
                buf = []
            continue
        # cheap sentence break on punctuation + space + capital
        line_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", line)
        for p in line_parts:
            p = p.strip()
            if p:
                parts.append(p)
    if buf:
        parts.append(" ".join(buf))
    return [p for p in parts if p]


_CITATION = re.compile(r"\[([^\]]+)\]")


def parse_citations(claim: str) -> tuple[str, tuple[str, ...]]:
    """Strip [sid1, sid2] markers from a claim, return (clean_claim, sids).

    The convention is that citations live inline at the end of the claim,
    in square brackets, comma-separated source ids. Anything inside the
    brackets that doesn't look like an id (purely [Foo v. Bar]) is left
    alone in the text.
    """
    sids: list[str] = []
    cleaned = claim
    for m in _CITATION.finditer(claim):
        inside = m.group(1)
        # split on commas; keep only tokens that look like ids
        for tok in (t.strip() for t in inside.split(",")):
            if tok and re.fullmatch(r"[A-Za-z0-9_\-]+", tok):
                sids.append(tok)
        # only strip the bracket span if every token looked like an id
        if all(re.fullmatch(r"[A-Za-z0-9_\-]+", t.strip()) for t in inside.split(",") if t.strip()):
            cleaned = cleaned.replace(m.group(0), "").strip()
    return cleaned.strip(), tuple(sids)


def claims_from_answer(answer: str) -> list[tuple[str, tuple[str, ...]]]:
    """Sentence-decompose an answer; pull citations from each sentence."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for s in sentences(answer):
        text, cits = parse_citations(s)
        if text:
            out.append((text, cits))
    return out


def merge_citation_lists(claim_to_cits: Iterable[tuple[str, tuple[str, ...]]]) -> set[str]:
    out: set[str] = set()
    for _, cits in claim_to_cits:
        out.update(cits)
    return out
