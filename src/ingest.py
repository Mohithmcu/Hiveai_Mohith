"""
ingest.py — Filter SpotifyCares from twcs.csv, reconstruct threads, clean, cache to parquet.

Usage:
    python src/ingest.py

Output:
    data/cache/spotify_threads.parquet
"""

import sys, io, re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Windows console encoding safeguard
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW_CSV   = Path("data/raw/twcs.csv")
CACHE_DIR = Path("data/cache")
OUTPUT    = CACHE_DIR / "spotify_threads.parquet"
BRAND     = "SpotifyCares"

# Boilerplate phrases to filter out (brand replies that add no grounding value)
BOILERPLATE = [
    "please dm", "send us a dm", "send a dm", "direct message",
    "we're sorry to hear", "we are sorry to hear",
]

def clean_text(text: str) -> str:
    """Strip @handles, URLs, and excess whitespace. Keep emoji — they carry sentiment."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"@\w+", "", text)               # remove @handles
    text = re.sub(r"http\S+|www\.\S+", "", text)   # remove URLs
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_boilerplate(text: str) -> bool:
    t = text.lower()
    return any(bp in t for bp in BOILERPLATE)

def load_raw() -> pd.DataFrame:
    print(f"Reading {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV, encoding="utf-8", encoding_errors="replace", low_memory=False)
    # Strip potential float suffix .0 from IDs (e.g. 119253.0 -> 119253)
    df["tweet_id"]                = df["tweet_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["response_tweet_id"]       = df["response_tweet_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df["inbound"]                 = df["inbound"].astype(str).str.lower().str.strip() == "true"
    print(f"  Total rows : {len(df):,}")
    return df

def reconstruct_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each SpotifyCares reply, retrieve the customer tweet it directly
    responds to. We use the *direct* customer tweet (not the thread root)
    intentionally: it is the most proximal context for the brand reply.
    This is a named limitation recorded in the decision log.
    """
    text_map    = df.set_index("tweet_id")["text"].to_dict()
    inbound_map = df.set_index("tweet_id")["inbound"].to_dict()

    brand_df = df[df["author_id"].astype(str).str.lower() == BRAND.lower()].copy()
    print(f"  SpotifyCares replies: {len(brand_df):,}")

    pairs = []
    for _, row in tqdm(brand_df.iterrows(), total=len(brand_df), desc="Building pairs"):
        cid = row["in_response_to_tweet_id"]
        if not cid or cid not in text_map:
            continue
        if not inbound_map.get(cid, False):
            continue                            # not a customer tweet

        c_raw = text_map[cid]
        b_raw = row["text"]
        c_clean = clean_text(c_raw)
        b_clean = clean_text(b_raw)

        if len(c_clean) < 15 or len(b_clean) < 15:
            continue
        if is_boilerplate(b_clean):
            continue

        pairs.append({
            "customer_tweet_id" : cid,
            "brand_tweet_id"    : row["tweet_id"],
            "customer_msg"      : c_clean,
            "brand_reply"       : b_clean,
            "customer_msg_raw"  : c_raw,
            "brand_reply_raw"   : b_raw,
            "created_at"        : row["created_at"],
        })

    result = pd.DataFrame(pairs).drop_duplicates(subset=["brand_reply"])
    print(f"  Thread pairs (after dedup): {len(result):,}")
    return result

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        print(f"Cache already exists: {OUTPUT}")
        df = pd.read_parquet(OUTPUT)
        print(f"  Loaded {len(df):,} cached pairs. Delete file to re-run.")
        return

    df = load_raw()
    pairs = reconstruct_pairs(df)
    pairs.to_parquet(OUTPUT, index=False)
    print(f"\nSaved -> {OUTPUT}  ({len(pairs):,} pairs)")
    if len(pairs) > 0:
        print(pairs[["customer_msg", "brand_reply"]].head(3).to_string())

if __name__ == "__main__":
    main()
