"""
run_agent.py — Run the SpotifyAgent over the golden set and save predictions.

Usage:
    python src/run_agent.py --golden eval/golden_set.csv --out eval/predictions.csv
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
    from src.agent import SpotifyAgent
except ModuleNotFoundError:
    from agent import SpotifyAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden_set.csv")
    parser.add_argument("--out",    default="eval/predictions.csv")
    parser.add_argument("--force",  action="store_true", help="Overwrite existing predictions")
    args = parser.parse_args()

    golden = pd.read_csv(args.golden)
    agent  = SpotifyAgent()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if out_path.exists() and not args.force:
        try:
            existing = pd.read_csv(out_path)
            if len(existing) == len(golden):
                print(f"Predictions already complete ({len(existing)} rows) -> {out_path}")
                return
            rows = existing.to_dict(orient="records")
            print(f"Resuming from checkpoint at index {len(rows)}/{len(golden)} ...")
        except Exception:
            rows = []

    start_idx = len(rows)
    for i in tqdm(range(start_idx, len(golden)), desc="Running agent", initial=start_idx, total=len(golden)):
        row = golden.iloc[i]
        result = agent.run(str(row["tweet_text"]))
        rows.append({
            "tweet_text"   : row["tweet_text"],
            "intent_pred"  : result["intent"],
            "confidence"   : result["confidence"],
            "escalate_pred": result["escalation"]["decision"] == "escalate",
            "escalate_reason": result["escalation"]["reason"],
            "triggered_rule" : result["escalation"]["triggered_rule"],
            "draft_reply"  : result["draft_reply"] or "",
            "retrieved_json": json.dumps(result["retrieved"]),
        })
        # Checkpoint after each row
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"Saved {len(rows)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
