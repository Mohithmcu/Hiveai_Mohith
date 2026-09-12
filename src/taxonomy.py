"""
taxonomy.py — Embed customer messages, cluster into intents, save label mapping.

Usage:
    python src/taxonomy.py

Output:
    data/cache/embeddings.npy
    data/cache/taxonomy.parquet   (threads + cluster_id + intent_label)
    data/cache/intent_map.json
"""

import sys, io, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Windows console encoding safeguard
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CACHE_DIR      = Path("data/cache")
THREADS_PATH   = CACHE_DIR / "spotify_threads.parquet"
EMB_PATH       = CACHE_DIR / "embeddings.npy"
TAXONOMY_PATH  = CACHE_DIR / "taxonomy.parquet"
INTENT_MAP     = CACHE_DIR / "intent_map.json"

INTENT_LABELS = {
    0: "App / Playback Bug",
    1: "Account / Login Issue",
    2: "Premium / Billing",
    3: "Content / Playlist Issue",
    4: "Feature Request",
    5: "General Complaint / Sentiment",
    6: "Other / Unclassifiable",
}

N_CLUSTERS  = 7
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLE_SIZE = 10_000

def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    print(f"Embedding {len(texts):,} texts ...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if TAXONOMY_PATH.exists() and INTENT_MAP.exists():
        print("Taxonomy cache exists. Delete to re-run.")
        return

    df = pd.read_parquet(THREADS_PATH)
    print(f"Loaded {len(df):,} thread pairs.")

    sample = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)
    model = SentenceTransformer(MODEL_NAME)

    if EMB_PATH.exists():
        print(f"Loading cached embeddings from {EMB_PATH}")
        sample_emb = np.load(EMB_PATH)
    else:
        sample_emb = embed(sample["customer_msg"].tolist(), model)
        np.save(EMB_PATH, sample_emb)
        print(f"Saved embeddings -> {EMB_PATH}")

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    sample["cluster_id"] = km.fit_predict(sample_emb)

    print("Assigning all rows to nearest centroid ...")
    all_emb = embed(df["customer_msg"].tolist(), model)
    df["cluster_id"] = km.predict(all_emb)
    df["intent_label"] = df["cluster_id"].map(INTENT_LABELS).fillna("Other / Unclassifiable")

    df.to_parquet(TAXONOMY_PATH, index=False)
    print(f"Saved taxonomy -> {TAXONOMY_PATH}")

    dist = df["intent_label"].value_counts().to_dict()
    intent_map = {v: {"cluster_id": k, "count": dist.get(v, 0)} for k, v in INTENT_LABELS.items()}
    with open(INTENT_MAP, "w") as f:
        json.dump(intent_map, f, indent=2)
    print(f"Saved intent map -> {INTENT_MAP}")

    print("\nIntent distribution:")
    print(df["intent_label"].value_counts().to_string())

if __name__ == "__main__":
    main()
