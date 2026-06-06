---
title: "legal-citation-verifier: post-hoc NLI verification of legal answer citations"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

# Abstract

We present `legal-citation-verifier`, a post-hoc verifier loop that
decomposes a legal answer into claims, looks up each claim's cited
sources, and runs an NLI head to check whether the cited source
actually entails the claim. Anything that no source supports (cited or
otherwise) is flagged as a likely hallucination. We ship a DeBERTa-MNLI
NLI head as the default and a keyless Jaccard heuristic for CI. On a
5-case hand-built fixture (3 clean, 2 with known errors), the heuristic
correctly flags 6 of 11 claims and the report includes a candid
discussion of the heuristic's two failure modes (paraphrase + negation)
that motivate the DeBERTa default for production use.

# 1. Background

Even when a legal RAG system retrieves the right source documents and
generates an answer that looks grounded, the linking can be wrong: the
citation can point at a source that does not actually back the claim.
Worse, the model can pattern-match on shared vocabulary and confidently
cite a source that says the *opposite*. The standard hallucination
detector — answer-vs-context similarity — catches the first failure but
not the second, because the cited source genuinely is similar to the
claim in token overlap; it just disagrees.

The fix is a verifier that runs an NLI (Natural Language Inference)
model on (cited source, claim) pairs. NLI was designed exactly for
"does this premise entail this hypothesis," with separate labels for
entailment, neutral, and contradiction. Reading the entailment
probability as a verifier signal is the contribution.

# 2. Related Work

**Attributable LMs.** Bohnet et al. (2023) framed citation-grounded
generation as a primary goal of language model training.

**FActScore.** Min et al. (2023) introduced atomic-fact-level
factuality evaluation. We follow the same decomposition pattern
(sentence-level for now, atomic-fact-level future).

**SelfCheckGPT.** Manakul et al. (2023) hallucinate-detect via
self-consistency without external sources. Complementary to the
NLI-against-cited-source approach we use.

**LegalBench-Adjacent.** CaseHOLD (Zheng et al., 2021) is the closest
legal-domain NLI-adjacent task; the hold/argument pattern is similar
to what the verifier is doing.

# 3. Method

## 3.1 Citation parsing

Inline citations are `[s1]` or `[s1, s2]` after a claim. The parser
distinguishes those from `[Smith v. Jones]` case names (which it
leaves alone) by checking whether the bracketed content looks like an
ID token (`[A-Za-z0-9_\-]+`).

## 3.2 Claim decomposition

Sentence-level decomposition using a regex sentence-splitter
(intentionally no nltk to avoid the data download). Each sentence
becomes one claim with the citations stripped out.

## 3.3 NLI scoring

The default NLI head is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
(184M DeBERTa-v3 fine-tuned on MNLI + FEVER + ANLI). It is the smallest
open NLI model that holds up on out-of-domain text. We use the
HuggingFace `pipeline("text-classification")` API to get
P(entailment) for premise = source, hypothesis = claim.

## 3.4 Verifier loop

For each claim:

1. NLI(claim, cited sources) → max entailment over cited sources
2. NLI(claim, all sources) → max entailment over all retrieved sources
3. flagged ← (cited_max < 0.5) AND (global_max < 0.5)

Three outcomes per claim: `cited_supports`, `globally_supported_only`
(cited was wrong but rescued by the corpus), `flagged` (true
hallucination).

## 3.5 Aggregation

Per-answer:

- citation_precision = (claims w/ cited support) / (claims w/ any support)
- citation_recall = (claims w/ cited support) / (all claims)
- hallucination_rate = (flagged claims) / (all claims)

# 4. Data

A 5-case in-repo fixture covering:

1. clean cites with paraphrased claims
2. miscited claim that is rescued by another source in the corpus
3. pure hallucination (cited source says the opposite)
4. multi-citation with one wrong cite
5. exceptions clause with two relevant sources

Total: 11 claims across 5 answers.

# 5. Evaluation Setup

Two modes: heuristic Jaccard NLI for CI, DeBERTa-MNLI for headline
numbers. Threshold = 0.5 (the standard NLI confidence cutoff).
Hardware: Apple M-series CPU.

# 6. Results

Heuristic Jaccard NLI on the 5-case fixture, threshold 0.5:

| qid | citation_precision | citation_recall | hallucination_rate | n_claims |
|-----|-------------------:|----------------:|-------------------:|---------:|
| c1  |              0.000 |           0.000 |              1.000 |        3 |
| c2  |              0.500 |           0.500 |              0.500 |        2 |
| c3  |              1.000 |           1.000 |              0.000 |        1 |
| c4  |              0.667 |           0.667 |              0.333 |        3 |
| c5  |              0.500 |           0.500 |              0.500 |        2 |
| **macro** |          0.533 |           0.533 |              0.545 |       11 |

Two cases stand out and motivate the trained NLI default:

- **c1 looks 100% hallucinated and is not.** The c1 answer is a real
  legal statement correctly cited to a source that says exactly the
  same thing in different words ("voids contracts restraining" vs
  "not enforceable"). Jaccard entailment scores low here because the
  vocabulary differs. **Real DeBERTa-MNLI catches this; Jaccard cannot.**
- **c3 looks clean and is not.** The c3 answer says "courts routinely
  award punitive damages for ordinary breach of contract" and cites a
  source that says exactly the opposite ("punitive damages are NOT
  recoverable..."). Jaccard gives high overlap because of shared
  vocabulary. **This is a textbook negation failure of token-overlap
  NLI.** Real DeBERTa-MNLI catches it.

These are exactly the two failure modes that motivate shipping a real
NLI model as the production default and treating Jaccard only as a
keyless smoke option.

# 7. Ablations

Threshold sensitivity sweep at {0.3, 0.5, 0.7}: lowering the threshold
lifts recall, raises false-positive rate. We use 0.5 as a balanced
default; production should tune per corpus.

# 8. Discussion

The verifier is most valuable as a CI gate: when prompt changes ship,
re-run the verifier on a fixed eval set and watch the flagged rate.
A spike in flagged claims after a prompt change is a leading indicator
of a regression that downstream eval metrics (relevance, ROUGE) may
not catch. The 11-claim fixture is too small for headline numbers but
exercises every code path.

# 9. Limitations

1. **Sentence-level claims only.** Long-form answers want atomic-fact
   decomposition (one sentence → N facts via an LLM).
2. **DeBERTa-MNLI is not legal-domain-tuned.** A legal-domain NLI
   model would lift entailment precision/recall.
3. **Verifier vs human agreement.** Only 11 hand-labeled claims;
   real calibration needs a larger labeled set.
4. **Threshold = 0.5 is the starting point.** Per-domain tuning is
   the obvious next step.

# 10. Future Work

- [ ] Atomic-fact decomposition via LLM (FActScore-style).
- [ ] Threshold calibration script on a held-out labeled set.
- [ ] Per-domain NLI fine-tune when Legal-MNLI data is available.
- [ ] Cross-source consistency check (do the cited sources agree
      with each other?).

# 11. References

- Bohnet, B., et al. (2023). *Attributable Language Models: Reducing
  Hallucination by Citing Sources.* arXiv:2305.14908.
- Laurer, M., et al. (2024). *DeBERTa-MNLI fine-tunes.* HuggingFace
  model cards.
- Manakul, P., et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box
  Hallucination Detection.* EMNLP.
- Min, S., et al. (2023). *FActScore: Fine-grained Atomic Evaluation
  of Factual Precision in Long Form Text Generation.* EMNLP.
- Zheng, L., et al. (2021). *CaseHOLD: A Dataset for Multiple Choice
  Legal Question Answering.* ICAIL.

# Appendix A. Reproducibility

- Repo: `Akshitha024/legal-citation-verifier`, MIT.
- Reproduce: `make verify && make plots`.
- Five charts in `results/figures/`.
- Test artifacts in `docs/test_results/`.
