"""
sample_golden.py — Stratified sampling of golden evaluation set.

Sampling strategy (documented for report):
  - Proportional by intent cluster (capped at 40 per class to avoid dominance)
  - Stratified by message length: short (<50 chars), medium (50-150), long (>150)
  - Deliberate edge-case slice: sarcasm, multi-issue, angry/threat, very short

Output: eval/golden_set_template.csv  (you hand-label the *_gold columns)

Usage:
    python src/sample_golden.py
"""

import re, sys, io
from pathlib import Path
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CACHE_DIR     = Path("data/cache")
TAXONOMY_PATH = CACHE_DIR / "taxonomy.parquet"
OUTPUT_PATH   = Path("eval/golden_set_template.csv")

TARGET_TOTAL  = 160
CAP_PER_CLASS = 30
EDGE_CASE_N   = 28   # reserved for deliberate edge-case slice

SARCASM_PATTERNS = re.compile(
    r"\b(wow|great job|thanks for nothing|oh sure|yeah right|"
    r"love how|amazing how|brilliant|so helpful)\b", re.I
)
MULTI_ISSUE = re.compile(
    r"(and also|also.{0,20}(can't|won't|doesn't)|multiple|two (issues|problems))", re.I
)
ANGER_PATTERNS = re.compile(
    r"\b(furious|disgusting|outrageous|scam|pathetic|useless|horrible|worst)\b", re.I
)


def tag_length(text: str) -> str:
    n = len(text)
    if n < 50:   return "short"
    if n < 150:  return "medium"
    return "long"


def tag_edge(text: str) -> str:
    if SARCASM_PATTERNS.search(text): return "sarcasm"
    if MULTI_ISSUE.search(text):      return "multi_issue"
    if ANGER_PATTERNS.search(text):   return "angry"
    return "clean"


def main():
    Path("eval").mkdir(exist_ok=True)

    df = pd.read_parquet(TAXONOMY_PATH)
    df = df[["customer_tweet_id", "customer_msg", "brand_reply", "intent_label"]].copy()
    df["length_bucket"] = df["customer_msg"].apply(tag_length)
    df["edge_tag"]      = df["customer_msg"].apply(tag_edge)

    sampled = []

    # 1. Edge-case slice (deliberate)
    edge = df[df["edge_tag"] != "clean"]
    edge_sample = edge.sample(min(EDGE_CASE_N, len(edge)), random_state=42)
    sampled.append(edge_sample)
    remaining = df.drop(edge_sample.index)

    # 2. Stratified by intent (proportional, capped)
    per_class = (
        remaining.groupby("intent_label")
        .apply(lambda g: g.sample(min(CAP_PER_CLASS, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    # Balance to hit TARGET_TOTAL
    still_need = TARGET_TOTAL - len(edge_sample)
    per_class  = per_class.sample(min(still_need, len(per_class)), random_state=42)
    sampled.append(per_class)

    result = pd.concat(sampled).drop_duplicates(subset=["customer_tweet_id"])
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    # Add empty gold columns for hand-labelling
    result["intent_gold"]        = ""   # fill by hand
    result["ideal_reply_notes"]  = ""   # pipe-separated facts a good reply must contain
    result["escalate_gold"]      = ""   # True / False
    result["escalate_reason_gold"]= ""  # why escalated (if True)
    result["difficulty"]         = ""   # clean / ambiguous / adversarial

    # Rename for clarity
    result = result.rename(columns={"customer_msg": "tweet_text"})
    result = result[[
        "customer_tweet_id", "tweet_text", "intent_label",
        "length_bucket", "edge_tag",
        "intent_gold", "ideal_reply_notes",
        "escalate_gold", "escalate_reason_gold", "difficulty",
    ]]

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} examples -> {OUTPUT_PATH}")
    print("\nSampling breakdown:")
    print(f"  Edge cases : {len(edge_sample)}")
    print(f"  By intent  : {len(per_class)}")
    print(f"  Total      : {len(result)}")
    print("\nIntent distribution:")
    print(result["intent_label"].value_counts().to_string())
    print("\nLength distribution:")
    print(result["length_bucket"].value_counts().to_string())
    print("\nEdge-case distribution:")
    print(result["edge_tag"].value_counts().to_string())
    print(f"\nNEXT STEP: Open {OUTPUT_PATH} and fill in the *_gold columns by hand.")


if __name__ == "__main__":
    main()
