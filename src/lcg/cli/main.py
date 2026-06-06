from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger

from ..claims.decompose import claims_from_answer
from ..nli.scorer import HeuristicNLI, NLIHead
from ..types import CitedAnswer, Claim, Source
from ..verifier.check import Verifier
from ..viz.charts import (
    plot_claim_outcomes,
    plot_entailment_heatmap,
    plot_flag_trend,
    plot_precision_recall_scatter,
    plot_verifier_confusion,
)

app = typer.Typer(add_completion=False, help="lcg: legal citation grounder")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@app.command("verify")
def cmd_verify(
    data: Annotated[Path, typer.Option(help="JSONL of {qid, question, answer, sources}")] = Path(
        "tests/fixtures/cases.jsonl"
    ),
    provider: Annotated[str, typer.Option(help="nli backend: heuristic | nli")] = "heuristic",
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
    run_id: Annotated[str, typer.Option(help="run label")] = "latest",
) -> None:
    rows = _load_jsonl(data)
    nli: HeuristicNLI | NLIHead = HeuristicNLI() if provider == "heuristic" else NLIHead()
    v = Verifier(nli=nli)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}__verdicts.jsonl"
    summary = {"runs": 0, "claims": 0, "flagged": 0, "precision_sum": 0.0, "recall_sum": 0.0}
    with out_path.open("w") as fo:
        for r in rows:
            sources = tuple(
                Source(sid=s["sid"], text=s["text"], title=s.get("title"))
                for s in r.get("sources", [])
            )
            claim_pairs = claims_from_answer(r["answer"])
            claims = tuple(Claim(text=t, citations=c) for t, c in claim_pairs)
            ca = CitedAnswer(qid=r["qid"], question=r["question"], claims=claims, sources=sources)
            verdict = v.verify(ca)
            fo.write(
                json.dumps(
                    {
                        "qid": verdict.qid,
                        "citation_precision": verdict.citation_precision,
                        "citation_recall": verdict.citation_recall,
                        "hallucination_rate": verdict.hallucination_rate,
                        "per_claim": [asdict(c) for c in verdict.per_claim],
                    }
                )
                + "\n"
            )
            summary["runs"] += 1
            summary["claims"] += len(verdict.per_claim)
            summary["flagged"] += sum(1 for c in verdict.per_claim if c.flagged)
            summary["precision_sum"] += verdict.citation_precision
            summary["recall_sum"] += verdict.citation_recall

    avg_p = summary["precision_sum"] / max(1, summary["runs"])
    avg_r = summary["recall_sum"] / max(1, summary["runs"])
    flag_rate = summary["flagged"] / max(1, summary["claims"])
    (out_dir / f"{run_id}__summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "answers": summary["runs"],
                "claims": summary["claims"],
                "flagged": summary["flagged"],
                "hallucination_rate": flag_rate,
                "macro_citation_precision": avg_p,
                "macro_citation_recall": avg_r,
            },
            indent=2,
        )
    )
    typer.echo(
        f"answers={summary['runs']} claims={summary['claims']} flagged={summary['flagged']} "
        f"P={avg_p:.3f} R={avg_r:.3f} halluc={flag_rate:.3f}"
    )


@app.command("report")
def cmd_report(
    out_dir: Annotated[Path, typer.Option(help="results dir")] = Path("results"),
) -> None:
    rows = []
    for f in sorted(out_dir.glob("*__summary.json")):
        s = json.loads(f.read_text())
        rows.append(s)
    if not rows:
        logger.warning("no summaries in {}", out_dir)
        return
    print(json.dumps(rows, indent=2))


@app.command("plots")
def cmd_plots(
    verdicts: Annotated[Path, typer.Option(help="verdicts jsonl")] = Path(
        "results/latest__verdicts.jsonl"
    ),
    history: Annotated[Path, typer.Option(help="history json (list)")] = Path(
        "results/history.json"
    ),
    humans: Annotated[Path, typer.Option(help="human labels jsonl")] = Path(
        "results/human_labels.jsonl"
    ),
    heatmap_qid: Annotated[str, typer.Option(help="qid for entailment heatmap")] = "",
    out_dir: Annotated[Path, typer.Option(help="figures dir")] = Path("results/figures"),
) -> None:
    plot_claim_outcomes(verdicts, out_dir / "claim_outcomes.png")
    plot_precision_recall_scatter(verdicts, out_dir / "precision_recall.png")
    if history.exists():
        hist = json.loads(history.read_text())
        plot_flag_trend(hist, out_dir / "flag_trend.png")
    if humans.exists():
        plot_verifier_confusion(verdicts, humans, out_dir / "verifier_confusion.png")
    # heatmap: build per-claim x per-source matrix from the chosen qid
    if heatmap_qid:
        verdicts_rows = _load_jsonl(verdicts)
        chosen = next((r for r in verdicts_rows if r["qid"] == heatmap_qid), None)
        if chosen is not None:
            data = {
                "qid": chosen["qid"],
                "claim_labels": [
                    (c["claim_text"][:40] + "...") if len(c["claim_text"]) > 40 else c["claim_text"]
                    for c in chosen["per_claim"]
                ],
                "source_labels": [],
                "matrix": [],
            }
            # this lightweight plot uses just the cited-entailment per claim; the
            # full claim x source matrix is regenerated by verify when needed
            plot_entailment_heatmap(data, out_dir / "entailment_heatmap.png")
    typer.echo(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    app()
