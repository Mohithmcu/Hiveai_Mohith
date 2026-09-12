"""
run_judge.py — Run the LLM judge over agent predictions, save scored output.

Usage:
    python src/run_judge.py --predictions eval/predictions.csv --out eval/judge_scores.csv
"""

import argparse, json
import pandas as pd
import sys, io
from pathlib import Path
from tqdm import tqdm

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.judge import Judge
except ModuleNotFoundError:
    from judge import Judge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="eval/predictions.csv")
    parser.add_argument("--golden",      default="eval/golden_set.csv")
    parser.add_argument("--out",         default="eval/judge_scores.csv")
    args = parser.parse_args()

    preds  = pd.read_csv(args.predictions)
    golden = pd.read_csv(args.golden)
    judge  = Judge()

    rows = []
    for i, (_, pred_row) in enumerate(tqdm(preds.iterrows(), total=len(preds), desc="Judging")):
        reply = str(pred_row.get("draft_reply", ""))
        if not reply or pred_row.get("escalate_pred", False):
            rows.append({"row_index": i, "skipped": True})
            continue

        tweet = str(golden.iloc[i]["tweet_text"])
        retrieved_raw = pred_row.get("retrieved_json", "[]")
        try:
            retrieved = json.loads(retrieved_raw)
        except Exception:
            retrieved = []

        scores = judge.score(tweet, reply, retrieved)
        scores["row_index"] = i
        rows.append(scores)

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Judge scores saved -> {args.out}")
    if "skipped" in out_df.columns:
        scored = out_df[~out_df["skipped"].fillna(False)]
    else:
        scored = out_df
    if len(scored):
        print(f"Composite avg: {scored['composite'].mean():.2f}  (n={len(scored)})")


if __name__ == "__main__":
    main()
