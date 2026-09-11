"""
index.py — Build a FAISS retrieval index from (customer_msg, brand_reply) pairs.

Usage:
    python src/index.py

Output:
    data/cache/retrieval_index.index   (FAISS index)
    data/cache/retrieval_corpus.parquet (ordered corpus matching index rows)
"""

import sys, io
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Windows console encoding safeguard
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CACHE_DIR      = Path("data/cache")
TAXONOMY_PATH  = CACHE_DIR / "taxonomy.parquet"
INDEX_PATH     = CACHE_DIR / "retrieval_index.index"
CORPUS_PATH    = CACHE_DIR / "retrieval_corpus.parquet"

MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
MAX_INDEX_SIZE = 8_000

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists() and CORPUS_PATH.exists():
        print("Retrieval index already exists. Delete to rebuild.")
        return

    df = pd.read_parquet(TAXONOMY_PATH)
    corpus = df.sample(min(MAX_INDEX_SIZE, len(df)), random_state=42).reset_index(drop=True)
    print(f"Building index from {len(corpus):,} pairs ...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        corpus["customer_msg"].tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    corpus.to_parquet(CORPUS_PATH, index=False)

    print(f"FAISS index saved  -> {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")
    print(f"Corpus saved       -> {CORPUS_PATH}")

if __name__ == "__main__":
    main()
