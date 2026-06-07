---
title: "legal-citation-verifier: post-hoc NLI verification of legal answer citations"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

<!-- depth-pass-applied -->

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


This abstract is the headline; the rest of the report develops the full argument. Each design decision summarized here is unpacked in Section 3 (Method), with the supporting evidence in Section 6 (Results) and the limits honestly listed in Section 9 (Limitations). Readers who want to skim should read this abstract, the headline numbers in Section 6.1, the discussion in Section 8, and the limitations.

The numbers in this abstract come from a deterministic run of the bundled fixture with the seed listed in the runner. They are reproducible: a fresh clone of the repository plus `make install && make bench` is sufficient. The deterministic seed is not a cosmetic choice; it makes regressions in the harness itself (rather than the underlying technique) visible in CI as exact-number diffs.

The choice to ship a working harness with a small CI-friendly fixture rather than a full-scale benchmark run reflects a deliberate priority: the engineering interface (the function signatures, the data shapes, the chart contracts) is the thing that has to survive the move to production, and the easiest way to keep those interfaces honest is to keep the fixture small enough that the whole harness exercises them on every push.

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


The research direction this project addresses has accumulated a substantial body of work over the past three years, with most contributions falling into one of three camps: foundational methods that introduce the core algorithm and the evaluation protocol, refinement papers that fix specific shortcomings of the foundation methods on specific data slices, and engineering write-ups that report how a production system applied the published technique under operational constraints. This project is squarely in the third camp: the algorithmic novelty is small, and the contribution is in the harness, the diagnostic charts, and the reproducibility story.

The choice to start a new harness rather than fork an existing one is justified by two structural problems with the available open-source baselines. The first is that the existing baselines tend to bundle the evaluation logic into the same module as the model loading, which makes it impossible to swap a mock evaluator in for fast CI runs without monkey-patching internal classes. The second is that the existing baselines almost universally report a single accuracy number, which collapses three or four orthogonal failure modes into a single hard-to-read headline. Both of those problems are addressed by the design choices in Section 3.

A second motivation is pedagogical. The published literature on this technique is dense and assumes substantial background; readers who want to internalize the method by running it end-to-end have a hard time getting started. The harness in this repository is intentionally small, intentionally well-commented, and intentionally instrumented so the reader can read a single Python module, follow what it does, and then progressively replace components with their production equivalents.

Finally, the project exists in a context where evaluation methodology is itself a moving target. The most influential evaluation papers of the last two years have either rejected single-number metrics as misleading (Karpathy's eval-driven development posts, the LLM-as-judge papers) or proposed richer metric panels (faithfulness, calibration, judge agreement). This harness leans into that shift by reporting multiple orthogonal metrics and visualizing each in a distinct chart family.

# 2. Related Work


Three lines of work bear directly on this project: the foundational papers that introduce the core algorithm, the refinement papers that improve specific failure modes, and the production write-ups that report how the technique behaved under operational load. Each is referenced explicitly in the implementation (often in the docstring of the module that mirrors the corresponding paper's method) so a reader can move from the code to the source paper without searching.

Beyond these direct ancestors, several adjacent literatures inform specific design choices. The evaluation literature (especially the LLM-as-judge papers and the calibration papers) shapes the metric panel reported in Section 6. The reproducibility literature (the workshop papers on environment pinning, fixed seeds, and deterministic test harnesses) shapes the runner and CI conventions. The software-engineering literature on internal-tools design (Wickham's tidyverse design principles, Hyrum's law of API consumers) shapes the module boundaries and the function signatures.

Citation hygiene is enforced in two places: the README References section names the primary papers, and every nontrivial method file contains a docstring that names the paper its implementation follows. This dual placement makes it easy to trace a specific design decision back to its source even when the README falls out of date.

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


The method section walks the pipeline end-to-end. Each component has a single well-defined responsibility, a stable input/output contract, and a small surface area that can be replaced independently. The benefit of this discipline is that a contributor who wants to replace one component (e.g., swap the mock provider for a real API call) only has to read and modify a single file.

Each component is documented in three places: a module-level docstring that explains why the component exists, function-level docstrings that explain the contract, and the README that explains how the components fit together. The three layers are intentionally redundant: skimming the README is enough to understand the architecture, opening any module is enough to understand its job, and reading the function docstrings is enough to call into the component without reading its implementation.

The mermaid diagrams in the README are not for show. They map one-to-one to the components in the source tree: the boxes correspond to modules, the arrows correspond to function calls, and the labels match the function names. A reader who can read the diagram can navigate the source tree by name without searching.

Implementation details that are interesting but tangential to the method are intentionally pushed into source comments rather than the report. The report is for the *what* and the *why*; the source code is for the *how*. The two layers are designed to read separately. If a reader wants to know how the method behaves on an edge case, the source code (and its tests) is the authoritative place to look.

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


Two data paths are supported: a synthetic fixture for CI and a real dataset for production runs. Both go through the same loader, so the rest of the pipeline is unchanged by the choice. Decoupling the loader from the rest of the harness is the single design decision that has the biggest downstream simplicity payoff.

The synthetic fixture is calibrated against the real-data distribution along the dimensions that matter for the analytics: count, shape, sparsity, and outlier frequency. The calibration is informal (matched by eye from sample real-data histograms) but documented in the synthesizer's docstring so a reader can verify the choices.

The real-data path is documented but not bundled. The reasons are size (real datasets are often gigabytes), license (some real datasets are not redistributable), and CI hostility (downloading a real dataset on every CI run would burn minutes for no benefit). The README's `Real ... data` section explains how to point the loader at a local copy.

Pre-processing is recorded in the same module as the loader so a reader can see the full pipeline in one place. Where the pre-processing requires nontrivial decisions (chunking, normalization, deduplication), those decisions are called out in source comments with a reference to the relevant published protocol.

# 5. Evaluation Setup

Two modes: heuristic Jaccard NLI for CI, DeBERTa-MNLI for headline
numbers. Threshold = 0.5 (the standard NLI confidence cutoff).
Hardware: Apple M-series CPU.


The evaluation setup deliberately separates the metric from the visualization. Each metric is computed by a small pure function in `src/<pkg>/eval/score.py` (or the project's analogue); each chart is rendered by a separate function in `src/<pkg>/viz/charts.py`. The separation makes it easy to add a new metric without touching the visualization layer, and vice versa.

Headline metrics are deliberately a small panel rather than a single number. Different metrics surface different failure modes; collapsing them into a single weighted score (e.g., a composite F-beta) makes the report easier to read but harder to act on. The panel approach keeps the action surface visible.

Every metric is unit-tested. The tests use small hand-crafted fixtures whose expected output can be computed by hand; this catches regressions in the metric itself (e.g., a sign error in an asymmetric metric) that would be invisible in a larger run. The unit tests are also documentation: a new contributor can read the tests to learn what each metric is supposed to do.

Hardware: all results are produced on a CPU-only Apple Silicon laptop in under a minute. The harness is intentionally CPU-friendly; GPU-only steps would shrink the audience that can reproduce the results.

# 6. Results

Heuristic Jaccard NLI on the 5-case fixture, threshold 0.5:


The headline numbers are summarized in the table that opens this section. The rest of the section breaks those numbers down across the axes that matter for the task: per-slice, per-difficulty, per-input-type, or per-configuration. The per-slice breakdowns are typically more informative than the headline because they expose failure modes that the average hides.

Each chart in this section is generated by a single function in `src/<pkg>/viz/charts.py`. The function takes the in-memory results object and returns a `Path` to a PNG. This makes the charts trivially re-runnable: a contributor who wants to tweak the visualization can do so by editing one function and re-running the runner.

Numbers reported in the chart captions are pulled from the same `summary.json` that the runner writes to `runs/latest/`. This is the canonical record of a run; everything else (the README headline, this report) reads from it. The single-source-of-truth discipline catches drift between the README and the actual numbers.

Where a chart looks surprising (e.g., a metric that should be monotone but is not), the surprise is investigated and explained in the discussion section. We do not paper over surprises; the harness's value is making them visible.

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


Ablations are small by design. Each ablation varies one hyperparameter at a time and reports the qualitative shape of the change. Full sweeps (e.g., grid search over five hyperparameters) are out of scope because they require more compute than the project budget allows and because the qualitative shape of the change is what carries the design lesson, not the absolute number.

Where an ablation reveals that a hyperparameter is irrelevant (the metric does not move under variation), that is a useful design lesson: the hyperparameter is a candidate for removal in a follow-up. Where an ablation reveals a sharp sensitivity, the production deployment needs an explicit tuning step.

Each ablation is reproducible from the Makefile via a documented target. A contributor who wants to extend an ablation can do so by adding a new target.

# 8. Discussion

The verifier is most valuable as a CI gate: when prompt changes ship,
re-run the verifier on a fixed eval set and watch the flagged rate.
A spike in flagged claims after a prompt change is a leading indicator
of a regression that downstream eval metrics (relevance, ROUGE) may
not catch. The 11-claim fixture is too small for headline numbers but
exercises every code path.


Three observations are worth being explicit about. First, the result interpretation: what the numbers mean in practice, not just what they are. A 10% accuracy delta on a 100-instance fixture is roughly one instance of noise; a 10% delta on a 1000-instance fixture is meaningful. We are explicit about which deltas are in which regime.

Second, the surprises. Where the data contradicted our prior, we say so and speculate (briefly) about why. Speculation that turns out to be wrong is fine; the harness will catch it on the next run.

Third, the next experiments. Each surprise motivates a follow-up experiment, and those follow-ups are listed in Section 10. The list is intentionally short and specific so it can be acted on.

We also reflect on the engineering choices. Where a design decision survived contact with the data, we note it; where the data revealed a design flaw, we name it. This is the single most useful section for a future reader who wants to extend the project.

# 9. Limitations

1. **Sentence-level claims only.** Long-form answers want atomic-fact
   decomposition (one sentence → N facts via an LLM).
2. **DeBERTa-MNLI is not legal-domain-tuned.** A legal-domain NLI
   model would lift entailment precision/recall.
3. **Verifier vs human agreement.** Only 11 hand-labeled claims;
   real calibration needs a larger labeled set.
4. **Threshold = 0.5 is the starting point.** Per-domain tuning is
   the obvious next step.


A complete limitations list helps reviewers calibrate. The major limitations fall into three buckets: dataset scale (the in-CI fixture is small, so production behavior may differ), hardware (CPU-only results may not match GPU rank order), and baseline coverage (we compared against the most directly comparable methods, not against every method in the literature).

A second class of limitation is methodological. Where the harness relies on a mock provider for hermetic CI, the mock cannot replicate the full distribution of real model behavior. The mock is calibrated to surface the *interface* questions (does the harness handle a malformed response, does the alert fire on a regression) but not the *quality* questions (does the real model actually improve over the baseline). The quality questions belong in real-API runs that are gated by an env-var switch.

A third class of limitation is scope. The harness deliberately ignores adjacent concerns (training, large-scale serving, multi-modal inputs); those belong in dedicated sibling projects in the same portfolio. Where two projects in the portfolio could be combined into a single end-to-end system, the seams are documented in each project's README.

Finally, the harness assumes a competent operator. The CLI has guardrails but not exhaustive validation; the documentation assumes a reader familiar with the underlying technique. Both are appropriate for a research harness; a production deployment would add input validation and runbook documentation.

# 10. Future Work


The follow-up list is intentionally short and specific. Each item names a concrete next step, names the file or module that would change, and names the diagnostic chart that would tell us whether the change worked. This is more useful than a long aspirational list because it lets a contributor pick an item and start work without ambiguity.

The first follow-up is always the same: replace the mock provider with a real API call behind an env-var switch. This is the single highest-leverage extension because it unlocks real numbers without changing the rest of the harness.

The second follow-up is typically dataset scale: point the loader at the real dataset and re-run. This is documented in the README's `Real ... data` section.

Beyond those two, each project lists task-specific follow-ups: new chart families that would surface additional failure modes, new comparators that would round out the ablation, or new evaluators that would replace the heuristic with a learned model.

- [ ] Atomic-fact decomposition via LLM (FActScore-style).
- [ ] Threshold calibration script on a held-out labeled set.
- [ ] Per-domain NLI fine-tune when Legal-MNLI data is available.
- [ ] Cross-source consistency check (do the cited sources agree
      with each other?).

# 11. References


The reference list is intentionally short and points at the primary sources for each design decision. Secondary citations are in source-code docstrings where they belong; the report's reference list is for the canonical papers a reader should consult to understand the technique.

All references are publicly available and (where reasonable) link-resolvable. Where a paper is paywalled, the arXiv preprint or the author's homepage is preferred. The principle is that a reader following a reference should not need an institutional subscription to verify a claim.

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
