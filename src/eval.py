"""
eval.py — Full evaluation harness. Computes all metrics for agent + baselines.

Usage:
    python src/eval.py --golden eval/golden_set.csv --predictions eval/predictions.csv

Metrics:
    Intent     : accuracy, macro-F1, confusion matrix
    Escalation : precision, recall (PRIMARY), F1, confusion matrix
    Reply      : faithfulness score + LLM judge scores (composite + per-dimension)
    Agreement  : Cohen's kappa (judge vs human) on judge_vs_human.csv
    CIs        : 95% bootstrap confidence intervals on all headline numbers
"""

import argparse
import json
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    cohen_kappa_score,
    classification_report,
)
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_BOOTSTRAP = 1000
CI_ALPHA    = 0.95


# ── Bootstrap CI ─────────────────────────────────────────────────────────────
def bootstrap_ci(values: list, metric_fn, n=N_BOOTSTRAP, alpha=CI_ALPHA):
    n_samples = len(values)
    stats = []
    for _ in range(n):
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        sample = [values[i] for i in idxs]
        stats.append(metric_fn(sample))
    lo = np.percentile(stats, (1 - alpha) / 2 * 100)
    hi = np.percentile(stats, (1 + alpha) / 2 * 100)
    return round(float(np.mean(stats)), 4), round(float(lo), 4), round(float(hi), 4)


# ── Faithfulness score ────────────────────────────────────────────────────────
def faithfulness_score(reply: str, ideal_notes: str, embedder: SentenceTransformer) -> float:
    """
    For each fact in ideal_reply_notes (pipe-separated), compute max cosine
    similarity to any sentence in the generated reply.  Average across facts.
    """
    if not ideal_notes or not reply:
        return 0.0
    facts    = [f.strip() for f in ideal_notes.split("|") if f.strip()]
    sentences = [s.strip() for s in reply.split(".") if s.strip()]
    if not facts or not sentences:
        return 0.0

    fact_embs = embedder.encode(facts,     normalize_embeddings=True)
    sent_embs = embedder.encode(sentences, normalize_embeddings=True)

    scores = []
    for fe in fact_embs:
        sims = sent_embs @ fe
        scores.append(float(sims.max()))
    return round(float(np.mean(scores)), 4)


# ── Intent metrics ────────────────────────────────────────────────────────────
def intent_metrics(gold: list, pred: list) -> dict:
    acc    = accuracy_score(gold, pred)
    macro  = f1_score(gold, pred, average="macro", zero_division=0)
    report = classification_report(gold, pred, zero_division=0)
    cm     = confusion_matrix(gold, pred, labels=sorted(set(gold)))

    # Bootstrap CI on macro-F1
    pairs = list(zip(gold, pred))
    def macro_f1_fn(sample):
        g = [p[0] for p in sample]
        p = [p[1] for p in sample]
        return f1_score(g, p, average="macro", zero_division=0)

    _, ci_lo, ci_hi = bootstrap_ci(pairs, macro_f1_fn)

    return {
        "accuracy"        : round(acc, 4),
        "macro_f1"        : round(macro, 4),
        "macro_f1_ci_95"  : (ci_lo, ci_hi),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ── Escalation metrics ────────────────────────────────────────────────────────
def escalation_metrics(gold: list[bool], pred: list[bool]) -> dict:
    """Recall on the escalate=True class is the PRIMARY metric (see decision log)."""
    prec   = precision_score(gold, pred, pos_label=True, zero_division=0)
    rec    = recall_score(gold, pred, pos_label=True, zero_division=0)
    f1     = f1_score(gold, pred, pos_label=True, zero_division=0)
    cm     = confusion_matrix(gold, pred, labels=[False, True])

    pairs = list(zip(gold, pred))
    def recall_fn(sample):
        g = [p[0] for p in sample]
        p = [p[1] for p in sample]
        return recall_score(g, p, pos_label=True, zero_division=0)

    _, ci_lo, ci_hi = bootstrap_ci(pairs, recall_fn)

    return {
        "precision"              : round(prec, 4),
        "recall_PRIMARY"         : round(rec, 4),
        "recall_ci_95"           : (ci_lo, ci_hi),
        "f1"                     : round(f1, 4),
        "confusion_matrix"       : cm.tolist(),
        "note"                   : "Recall is PRIMARY: a missed escalation costs more than a false alarm.",
    }


# ── Judge agreement (Cohen's kappa) ──────────────────────────────────────────
def judge_agreement(jvh_path: str) -> dict:
    """
    jvh_path: eval/judge_vs_human.csv
    Expected columns: composite_human (1-5 int), composite_judge (1-5 float→int)
    """
    df = pd.read_csv(jvh_path)
    human = df["composite_human"].astype(int).tolist()
    judge = df["composite_judge"].round().astype(int).tolist()

    kappa = cohen_kappa_score(human, judge)
    rho, pval = spearmanr(human, judge)

    disagreements = df[abs(df["composite_human"] - df["composite_judge"]) >= 2]

    return {
        "cohen_kappa"      : round(kappa, 4),
        "spearman_rho"     : round(float(rho), 4),
        "spearman_pval"    : round(float(pval), 4),
        "n_samples"        : len(df),
        "n_large_disagreements": len(disagreements),
        "kappa_interpretation": (
            "substantial (≥0.6)" if kappa >= 0.6
            else "moderate (0.4–0.6)" if kappa >= 0.4
            else "fair (0.2–0.4)" if kappa >= 0.2
            else "slight (<0.2)"
        ),
    }


# ── Print summary table ───────────────────────────────────────────────────────
def print_table(results: dict):
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    print("\n-- Intent Classification --")
    im = results["intent"]
    print(f"  Accuracy  : {im['accuracy']:.4f}")
    print(f"  Macro-F1  : {im['macro_f1']:.4f}  95% CI: {im['macro_f1_ci_95']}")
    print(im["classification_report"])

    print("\n-- Escalation --")
    em = results["escalation"]
    print(f"  Recall (PRIMARY) : {em['recall_PRIMARY']:.4f}  95% CI: {em['recall_ci_95']}")
    print(f"  Precision        : {em['precision']:.4f}")
    print(f"  F1               : {em['f1']:.4f}")
    print(f"  Confusion Matrix : {em['confusion_matrix']}  (rows=gold, cols=pred, order=[auto,escalate])")

    if "reply" in results:
        print("\n-- Reply Quality --")
        rm = results["reply"]
        print(f"  Faithfulness (avg)  : {rm.get('faithfulness_avg', 'N/A')}")
        print(f"  Judge composite     : {rm.get('composite_avg', rm.get('judge_composite_avg', 'N/A'))}")
        for dim in ["relevance", "groundedness", "tone", "actionability", "safety"]:
            print(f"    {dim:<15}: {rm.get(dim + '_avg', 'N/A')}")

    if "agreement" in results:
        print("\n-- Judge vs Human Agreement --")
        ag = results["agreement"]
        print(f"  Cohen's κ      : {ag['cohen_kappa']} ({ag['kappa_interpretation']})")
        print(f"  Spearman ρ     : {ag['spearman_rho']} (p={ag['spearman_pval']:.4f})")
        print(f"  N samples      : {ag['n_samples']}")
        print(f"  Large diffs(≥2): {ag['n_large_disagreements']}")

    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden",      default="eval/golden_set.csv")
    parser.add_argument("--predictions", default="eval/predictions.csv",
                        help="CSV with columns: intent_pred, escalate_pred, draft_reply, retrieved_json")
    parser.add_argument("--jvh",         default="eval/judge_vs_human.csv",
                        help="Judge vs human agreement CSV")
    parser.add_argument("--skip-faithfulness", action="store_true")
    args = parser.parse_args()

    golden = pd.read_csv(args.golden)
    preds  = pd.read_csv(args.predictions)

    assert len(golden) == len(preds), "Golden and predictions must have same length."

    results = {}

    # Intent
    results["intent"] = intent_metrics(
        gold=golden["intent_gold"].tolist(),
        pred=preds["intent_pred"].tolist(),
    )

    # Escalation
    results["escalation"] = escalation_metrics(
        gold=golden["escalate_gold"].astype(bool).tolist(),
        pred=preds["escalate_pred"].astype(bool).tolist(),
    )

    # Reply quality
    non_escalated = golden[~golden["escalate_gold"].astype(bool)].index.tolist()
    if non_escalated and "draft_reply" in preds.columns:
        embedder = SentenceTransformer(EMBED_MODEL)

        faith_scores = []
        for i in non_escalated:
            reply = str(preds.loc[i, "draft_reply"]) if pd.notna(preds.loc[i, "draft_reply"]) else ""
            notes = str(golden.loc[i, "ideal_reply_notes"]) if "ideal_reply_notes" in golden.columns else ""
            if not args.skip_faithfulness:
                faith_scores.append(faithfulness_score(reply, notes, embedder))

        # Judge scores — load from a pre-run judge output if available
        judge_path = Path("eval/judge_scores.csv")
        reply_result = {}
        if faith_scores:
            reply_result["faithfulness_avg"] = round(float(np.mean(faith_scores)), 4)
        if judge_path.exists():
            jdf = pd.read_csv(judge_path)
            for dim in ["relevance", "groundedness", "tone", "actionability", "safety", "composite"]:
                if dim in jdf.columns:
                    reply_result[f"{dim}_avg"] = round(jdf[dim].mean(), 4)
        results["reply"] = reply_result

    # Judge agreement
    jvh_path = Path(args.jvh)
    if jvh_path.exists():
        results["agreement"] = judge_agreement(args.jvh)

    print_table(results)

    # Save results
    out_path = Path("eval/eval_results.json")
    with open(out_path, "w") as f:
        # Convert non-serialisable types
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            return obj
        json.dump(results, f, indent=2, default=convert)
    print(f"\nFull results saved -> {out_path}")


if __name__ == "__main__":
    main()
