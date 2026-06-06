"""Five chart types specific to citation grounding.

Different from the projects #4 and #5 sets:
  - claim_outcomes_sankey-style (stacked horizontal bar)  : per-answer breakdown
                                                            of cited+supported / cited-not-supported / global-only / flagged
  - entailment_heatmap                                    : claims x sources matrix
  - flag_rate_over_runs                                   : trend of hallucination rate
  - per_question_pr_scatter                               : precision vs recall per answer
  - verifier_confusion_matrix                             : verifier vs human label (when both available)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# 1. Stacked horizontal bar of per-answer claim outcomes
def plot_claim_outcomes(verdicts_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not verdicts_path.exists():
        out.write_bytes(b"")
        return out
    rows = _read_jsonl(verdicts_path)
    if not rows:
        out.write_bytes(b"")
        return out

    qids = [r["qid"] for r in rows]
    cited_ok, global_ok, flagged = [], [], []
    for r in rows:
        per_claim = r["per_claim"]
        n = len(per_claim) or 1
        c = sum(1 for v in per_claim if v["cited_supports"])
        g = sum(1 for v in per_claim if not v["cited_supports"] and v["any_source_supports"])
        f = sum(1 for v in per_claim if v["flagged"])
        cited_ok.append(c / n)
        global_ok.append(g / n)
        flagged.append(f / n)

    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(qids) + 4), 5))
    y = np.arange(len(qids))
    ax.barh(y, cited_ok, label="cited + supported", color="#2ca02c")
    ax.barh(y, global_ok, left=cited_ok, label="supported by a non-cited source", color="#1f77b4")
    base = np.array(cited_ok) + np.array(global_ok)
    ax.barh(y, flagged, left=base, label="flagged (no support)", color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels(qids, fontsize=8)
    ax.set_xlabel("fraction of claims")
    ax.set_xlim(0, 1)
    ax.set_title("Per-answer claim outcomes")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 2. Entailment heatmap for one chosen answer (claims x sources)
def plot_entailment_heatmap(per_claim_matrix: dict[str, Any], out: Path) -> Path:
    """matrix shape (n_claims, n_sources); also `claim_labels` and `source_labels`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    mat = np.array(per_claim_matrix.get("matrix") or [])
    if mat.size == 0:
        out.write_bytes(b"")
        return out
    claim_labels = per_claim_matrix["claim_labels"]
    source_labels = per_claim_matrix["source_labels"]
    fig, ax = plt.subplots(
        figsize=(max(5, 1.0 * len(source_labels)), max(3.5, 0.45 * len(claim_labels))),
    )
    im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(source_labels)))
    ax.set_xticklabels(source_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(claim_labels)))
    ax.set_yticklabels(claim_labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if mat[i, j] < 0.6 else "white",
            )
    fig.colorbar(im, ax=ax, label="P(entailment)")
    ax.set_title(f"Entailment matrix: {per_claim_matrix.get('qid', '?')}")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 3. Trend of hallucination rate across runs (line chart)
def plot_flag_trend(history: list[dict[str, float]], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        out.write_bytes(b"")
        return out
    xs = [h["run"] for h in history]
    ys = [h["hallucination_rate"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", linewidth=2, color="#d62728")
    ax.set_ylim(0, max(0.4, max(ys) + 0.05))
    ax.set_ylabel("hallucination rate")
    ax.set_xlabel("run")
    ax.set_title("Hallucination rate across runs")
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 4. Per-answer precision vs recall scatter
def plot_precision_recall_scatter(verdicts_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not verdicts_path.exists():
        out.write_bytes(b"")
        return out
    rows = _read_jsonl(verdicts_path)
    if not rows:
        out.write_bytes(b"")
        return out
    xs = [r["citation_recall"] for r in rows]
    ys = [r["citation_precision"] for r in rows]
    halluc = [r["hallucination_rate"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(xs, ys, c=halluc, cmap="Reds", s=120, vmin=0, vmax=1, edgecolor="black")
    for r, x, y in zip(rows, xs, ys, strict=True):
        ax.annotate(r["qid"], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("citation recall")
    ax.set_ylabel("citation precision")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="hallucination rate")
    ax.set_title("Per-answer precision vs recall (color = hallucination rate)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 5. Verifier vs human confusion matrix (when human labels available)
def plot_verifier_confusion(
    verdicts_path: Path,
    human_labels_path: Path,
    out: Path,
) -> Path:
    """human_labels_path is a JSONL of {qid, claim_idx, human_flagged: bool}."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if not verdicts_path.exists() or not human_labels_path.exists():
        out.write_bytes(b"")
        return out
    verdicts = {r["qid"]: r for r in _read_jsonl(verdicts_path)}
    humans = _read_jsonl(human_labels_path)
    tp = tn = fp = fn = 0
    for h in humans:
        qid = h["qid"]
        idx = int(h["claim_idx"])
        truth = bool(h["human_flagged"])
        if qid not in verdicts:
            continue
        per_claim = verdicts[qid]["per_claim"]
        if idx >= len(per_claim):
            continue
        pred = bool(per_claim[idx]["flagged"])
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    if tp + tn + fp + fn == 0:
        out.write_bytes(b"")
        return out
    mat = np.array([[tn, fp], [fn, tp]])
    labels = np.array([[f"TN={tn}", f"FP={fp}"], [f"FN={fn}", f"TP={tp}"]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred OK", "pred flagged"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["human OK", "human flagged"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                labels[i, j],
                ha="center",
                va="center",
                fontsize=12,
                color="black" if mat[i, j] < mat.max() / 2 else "white",
            )
    fig.colorbar(im, ax=ax, label="count")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    ax.set_title(f"Verifier vs human (P={precision:.2f}, R={recall:.2f}, F1={f1:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# 6 (alt). Per-claim entailment bar chart - flagged claims highlighted
def plot_claim_entailment_bars(verdicts_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not verdicts_path.exists():
        out.write_bytes(b"")
        return out
    rows = _read_jsonl(verdicts_path)
    if not rows:
        out.write_bytes(b"")
        return out

    labels: list[str] = []
    scores: list[float] = []
    colors: list[str] = []
    for r in rows:
        qid = r["qid"]
        for c in r["per_claim"]:
            short = (c["claim_text"][:30] + "...") if len(c["claim_text"]) > 30 else c["claim_text"]
            labels.append(f"{qid}/{c['claim_idx']}: {short}")
            scores.append(float(c["entailment"]))
            if c["flagged"]:
                colors.append("#d62728")
            elif c["cited_supports"]:
                colors.append("#2ca02c")
            else:
                colors.append("#1f77b4")

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, scores, color=colors)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="threshold = 0.5")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("max entailment over cited sources")
    ax.set_xlim(0, 1)
    ax.set_title("Per-claim cited-source entailment (red = flagged, green = supported)")
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
