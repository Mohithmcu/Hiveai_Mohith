"""
baselines.py — Two baselines evaluated against the golden set.

Baseline 1 (Trivial):
  - Intent: always predict majority class
  - Reply:  single canned template for every message
  - Escalate: never

Baseline 2 (Simple):
  - Intent: TF-IDF + Logistic Regression
  - Reply:  verbatim nearest-neighbour brand reply (BM25-style TF-IDF cosine)
  - Escalate: keyword-only rules (subset of agent hard rules)

Usage:
    python src/baselines.py --golden eval/golden_set.csv
"""

import sys, io, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# Windows console encoding safeguard
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CACHE_DIR    = Path("data/cache")
TAXONOMY_PATH = CACHE_DIR / "taxonomy.parquet"

CANNED_REPLY = (
    "Hey! Sorry to hear you're having trouble. "
    "Could you DM us your account details so we can take a look? /SC"
)

KEYWORD_ESCALATE = re.compile(
    r"\b(sue|lawsuit|refund|hacked|cancel.{0,10}subscription|"
    r"charge|stolen|fraud|hurt myself)\b", re.I
)

# ── Baseline 1: Trivial ──────────────────────────────────────────────────────
class TrivialBaseline:
    name = "Trivial (majority class + canned reply)"

    def __init__(self, majority_class: str):
        self.majority_class = majority_class

    def predict(self, tweets: list[str]) -> list[dict]:
        return [
            {
                "intent"     : self.majority_class,
                "confidence" : 1.0,
                "draft_reply": CANNED_REPLY,
                "escalation" : {"decision": "auto", "reason": "trivial baseline never escalates", "triggered_rule": None},
            }
            for _ in tweets
        ]

# ── Baseline 2: Simple (TF-IDF + LR + nearest-neighbour retrieval) ──────────
class SimpleBaseline:
    name = "Simple (TF-IDF + LR intent; NN reply; keyword escalation)"

    def __init__(self, corpus: pd.DataFrame, intent_col: str = "intent_label"):
        self.corpus     = corpus.reset_index(drop=True)
        self.intent_col = intent_col
        self._fit()

    def _fit(self):
        X = self.corpus["customer_msg"].tolist()
        y = self.corpus[self.intent_col].tolist()

        self.clf = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30_000, sublinear_tf=True)),
            ("lr",    LogisticRegression(max_iter=1000, C=5.0, random_state=42)),
        ])
        self.clf.fit(X, y)

        self.retrieval_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30_000, sublinear_tf=True)
        self.corpus_matrix = self.retrieval_vec.fit_transform(X)

    def retrieve_reply(self, tweet: str) -> str:
        q_vec  = self.retrieval_vec.transform([tweet])
        sims   = cosine_similarity(q_vec, self.corpus_matrix).flatten()
        best   = int(np.argmax(sims))
        return self.corpus.iloc[best]["brand_reply"]

    def escalate(self, tweet: str) -> dict:
        if KEYWORD_ESCALATE.search(tweet):
            return {"decision": "escalate", "reason": "Keyword escalation rule matched", "triggered_rule": "KEYWORD"}
        return {"decision": "auto", "reason": "No escalation keywords found", "triggered_rule": None}

    def predict(self, tweets: list[str]) -> list[dict]:
        intents    = self.clf.predict(tweets)
        probs      = self.clf.predict_proba(tweets).max(axis=1)
        results    = []
        for tweet, intent, conf in zip(tweets, intents, probs):
            esc = self.escalate(tweet)
            reply = self.retrieve_reply(tweet) if esc["decision"] == "auto" else None
            results.append({
                "intent"     : intent,
                "confidence" : float(conf),
                "draft_reply": reply,
                "escalation" : esc,
            })
        return results

def evaluate_baseline(baseline, golden: pd.DataFrame) -> dict:
    tweets      = golden["tweet_text"].tolist()
    predictions = baseline.predict(tweets)

    intent_preds   = [p["intent"]              for p in predictions]
    escalate_preds = [p["escalation"]["decision"] == "escalate" for p in predictions]
    reply_preds    = [p["draft_reply"] or ""   for p in predictions]

    return {
        "intent_preds"   : intent_preds,
        "escalate_preds" : escalate_preds,
        "reply_preds"    : reply_preds,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="eval/golden_set.csv")
    args = parser.parse_args()

    golden = pd.read_csv(args.golden)
    corpus = pd.read_parquet(TAXONOMY_PATH)

    majority_class = corpus["intent_label"].value_counts().idxmax()
    print(f"Majority class: {majority_class}")

    trivial = TrivialBaseline(majority_class=majority_class)
    simple  = SimpleBaseline(corpus=corpus)

    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

    g_intent = golden["intent_gold"].tolist()
    g_esc    = golden["escalate_gold"].astype(bool).tolist()

    for bl in [trivial, simple]:
        print(f"\n-- {bl.name} --")
        preds = evaluate_baseline(bl, golden)
        acc  = accuracy_score(g_intent, preds["intent_preds"])
        mf1  = f1_score(g_intent, preds["intent_preds"], average="macro", zero_division=0)
        rec  = recall_score(g_esc, preds["escalate_preds"], pos_label=True, zero_division=0)
        prec = precision_score(g_esc, preds["escalate_preds"], pos_label=True, zero_division=0)
        print(f"  Intent Accuracy     : {acc:.4f}")
        print(f"  Intent Macro-F1     : {mf1:.4f}")
        print(f"  Escalation Recall   : {rec:.4f} (PRIMARY)")
        print(f"  Escalation Precision: {prec:.4f}")

    print("\nBaselines evaluated successfully.")

if __name__ == "__main__":
    main()
