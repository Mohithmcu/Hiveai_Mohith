# SpotifyCares AI Support Agent — System & Evaluation Report

**Hiver SDE Intern Take-Home Project**  
**Author:** AI Engineering Candidate  
**Brand:** SpotifyCares  
**Primary Generation Model:** Gemini 3.6 Flash (`gemini-3.6-flash`)  
**Judge Model:** Gemini 3.6 Flash (`gemini-3.6-flash`)  
**Embedding / Retrieval Model:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS IndexFlatIP  

---

## 1. Executive Summary

This project develops, evaluates, and documents an end-to-end AI customer support agent tailored specifically for **@SpotifyCares** using the Kaggle *Customer Support on Twitter* dataset (2.8M rows). The production-grade agent:
1. **Classifies customer intent** into 7 empirically derived categories using few-shot prompted LLM inference.
2. **Retrieves relevant past resolutions** from a dense FAISS index of 8,000 verified SpotifyCares interaction pairs.
3. **Applies an architectural safety guardrail (Escalation Engine)** running *strictly before drafting* to intercept high-risk messages (legal threats, self-harm, refund claims, abusive language, or low-confidence queries).
4. **Drafts grounded, brand-aligned responses** constrained to Twitter's 280-character limit and forbidden from hallucinating policies, timelines, or financial figures.
### Headline Benchmark Results (Evaluation on Golden Set, N=160)

| Metric | Scale | SpotifyAgent (Our System) | Simple Baseline (TF-IDF + LR) | Trivial Baseline (Majority Class) |
|---|---|---|---|---|
| **Intent Macro-F1** [95% CI] | 0–1 | **0.832** [0.769, 0.885] | 0.221 [0.175, 0.268] | 0.035 [0.015, 0.052] |
| **Intent Accuracy** | % | **83.1%** | 26.3% | 13.8% |
| **Escalation Recall (PRIMARY)** [95% CI] | 0–1 | **0.927** [0.830, 1.000] | 0.171 [0.073, 0.293] | 0.000 [0.000, 0.000] |
| **Escalation Precision** | 0–1 | **0.905** | 1.000 | 0.000 |
| **Escalation F1** | 0–1 | **0.916** | 0.292 | 0.000 |
| **Faithfulness Score (Ideal Facts)** | 0–1 | **0.391** [0.354, 0.428] | 0.212 [0.180, 0.245] | 0.115 [0.085, 0.142] |
| **Judge Composite Score** | 1–5 | **4.99 / 5.0** | 3.54 / 5.0 | 2.33 / 5.0 |

*All confidence intervals computed via 1,000-sample non-parametric bootstrap. Note: Faithfulness is measured on a normalized [0, 1] cosine similarity scale against ideal fact propositions; Judge Composite is measured on an ordinal [1, 5] rubric.*

---

## 2. Brand & Domain Selection

While **@AmazonHelp** is the dataset's highest-volume brand (>42,000 replies in the first 500k rows), it was deliberately passed over in favor of **@SpotifyCares** (43,265 total replies).

### Why SpotifyCares?
- **Domain Coherence vs. Fragmentation:** Amazon operates across >15 distinct product categories (AWS cloud services, Kindle hardware, Prime Video streaming, third-party seller marketplace, damaged deliveries, pantry/grocery). A single 7-class taxonomy for Amazon would either be unmanageably broad or suffer from severe intra-class variance. Spotify is focused on digital audio streaming, yielding a tight, defensible 7-class taxonomy.
- **Feasible Grounding:** In physical retail support, resolutions routinely rely on private internal state (courier tracking numbers, warehouse dispatch status, credit card tokenization). Spotify support interactions rely largely on reproducible software troubleshooting (cache clears, offline toggles, reinstall guides, subscription tier verifications) which can be meaningfully grounded in public past replies.

---

## 3. Intent Taxonomy Discovery

Rather than imposing top-down intuition, the taxonomy was derived using unsupervised semantic clustering followed by qualitative audit.

### Discovery Methodology:
1. **Sampling:** 10,000 customer messages directed to SpotifyCares were extracted from the dataset.
2. **Embedding:** Messages were embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
3. **Clustering & Silhouette Sweep:** Evaluated KMeans across $k \in [4, 12]$. Optimal cluster separation and interpretability occurred at $k = 7$ (silhouette peak ~0.142 with distinct semantic separation).
4. **Human Verification:** Inspected 25 nearest-neighbor samples from each cluster centroid to establish human-readable definitions.

### Final 7-Class Taxonomy
1. **App / Playback Bug:** Crashes, continuous buffering, stuttering, songs skipping after 10–30s, Bluetooth disconnections.
2. **Account / Login Issue:** Forgotten passwords, compromised/hacked accounts, email changes, two-factor authentication issues.
3. **Premium / Billing:** Double charges, student verification issues, payment method failures, refund demands, plan cancellations.
4. **Content / Playlist Issue:** Missing songs/albums, grayed-out tracks, deleted playlists, mismatched cover art.
5. **Feature Request:** Demands for HiFi audio, UI rollbacks, lyrics additions, specific OS widget support.
6. **General Complaint / Sentiment:** Non-specific anger ("your update is trash"), general praise, or sarcastic commentary.
7. **Other / Unclassifiable:** Foreign languages, fragmented messages ("help"), emoji-only tweets.

---

## 4. System Architecture & Core Pipeline

The agent executes a sequential 4-stage pipeline designed for safety, auditability, and grounded generation.

```
Incoming Customer Tweet
          │
          ▼
 [1. Few-Shot Intent Classifier] ──── (Gemini Flash + Dense Prototype Fallback)
          │
          ├─────────────────────────► Intent & Calibrated Confidence Score
          ▼
 [2. Dense Retrieval Engine]     ──── (FAISS IndexFlatIP + all-MiniLM-L6-v2)
          │
          ├─────────────────────────► Top-5 Past (Customer, Resolution) Pairs
          ▼
 [3. Safety & Escalation Engine] ──── (Deterministic Regex + Calibrated Soft Triggers)
          │
    ┌──────┴──────────────────────┐
    │ Escalated?                  │
   YES                           NO
    │                             │
    ▼                             ▼
[Human Queue Triage]    [4. Grounded Draft Generator] ── (Gemini Flash + Troubleshooting Fallback)
(Reason & Rule Logged)           │
                                 ▼
                         Customer Reply Draft (<280 chars)
```

### Key Architectural Decisions:
1. **Pre-Draft Escalation Interception:** The escalation check executes **before** the LLM draft stage. If a ticket meets escalation criteria (e.g. self-harm, legal threats, disputed amounts), generation is skipped entirely. This eliminates the risk of an unreviewed draft being transmitted to a high-risk user and conserves API quota and latency.
2. **Deterministic Hard Rules as Constants:** Legal threats (`r"\b(sue|lawsuit|lawyer|attorney|legal action|court)\b"`), self-harm signals, explicit currency refund demands, and account takeover patterns (`HACKED_TAKEOVER`) are evaluated via named compiled regular expressions. Auditing rules requires zero prompt engineering or model re-evaluations.
3. **Calibrated Soft Escalation Triggers:** Escalates automatically if intent classification confidence is `< 0.35`, retrieval cosine similarity is `< 0.40`, or the intent is classified as `Other / Unclassifiable`.
4. **Hard Negative Constraint Prompting:** The drafting prompt instructs the model to utilize *only* facts present in the retrieved examples and explicitly prohibits inventing refund amounts, SLA timelines, or internal company policies.

---

## 5. Evaluation Methodology & Golden Set

To evaluate the system without synthetic benchmark contamination, an independent golden evaluation set was curated and annotated.

### Golden Set Construction ($N = 160$)
- **Stratified Intent Distribution:** Balanced allocation across all 7 intents (~22–24 examples per class) meeting the 150–250 spec requirement.
- **Length Stratification:** Stratified across short (<50 characters), medium (50–150 characters), and long (>150 characters) customer tweets.
- **Deliberate Edge-Case Slice ($N = 28$):** Targeted sampling containing 17 adversarial and 11 ambiguous cases:
  - **Sarcasm:** e.g., *"Amazing job Spotify, love when my music stops every 5 seconds, so helpful!"*
  - **Multi-Issue:** Complaints bundling playback crashes with duplicate billing disputes.
  - **Hostile / Escalation Triggers:** Mentions of lawsuits, consumer protection bureaus, account compromises, or profanity.
- **Ground Truth Fields Annotated:**
  - `intent_gold`: Canonical intent category.
  - `ideal_reply_notes`: Pipe-separated factual propositions required in a competent reply (e.g., *"ask for device model | suggest clean reinstall | advise clearing local cache"*).
  - `escalate_gold`: Binary indicator (`True` / `False`).
  - `escalate_reason_gold`: Stated operational justification for escalation.
  - `difficulty`: Categorized as `clean` (132), `ambiguous` (11), or `adversarial` (17).

---

## 6. Detailed Benchmark Results

### 6.1 Intent Classification
Evaluated on the full 160-sample golden set:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Account / Login Issue | 0.95 | 0.83 | 0.88 | 23 |
| App / Playback Bug | 0.84 | 0.88 | 0.86 | 24 |
| Content / Playlist Issue | 0.75 | 0.91 | 0.82 | 23 |
| Feature Request | 0.95 | 0.86 | 0.90 | 22 |
| General Complaint / Sentiment | 0.71 | 0.65 | 0.68 | 23 |
| Other / Unclassifiable | 0.76 | 0.86 | 0.81 | 22 |
| Premium / Billing | 0.90 | 0.83 | 0.86 | 23 |
| **Macro Average** | **0.84** | **0.83** | **0.83** | **160** |
| **Weighted Average** | **0.84** | **0.83** | **0.83** | **160** |

**Observations:**
- Overall Intent Accuracy is **83.1% (0.8313)** with an Intent Macro-F1 of **0.8319** (95% CI: 0.7693 – 0.8846).
- Strongest classification performance occurs on *Feature Request* ($F_1 = 0.90$) and *Account / Login Issue* ($F_1 = 0.88$), which exhibit clear technical vocabulary.
- The most challenging class is *General Complaint / Sentiment* ($F_1 = 0.68$), driven by sarcastic remarks (e.g., "Love paying $11 a month for dead silence") and overlapping bug complaints.

### 6.2 Escalation Engine: Prioritizing Recall
In customer support safety systems, the cost matrix is strictly asymmetric:
- **False Negative (Missed Escalation):** An angry or legally litigious customer receives an automated canned reply. This poses direct reputational and legal risk.
- **False Positive (Unnecessary Escalation):** A solvable ticket is routed to a human queue. This incurs slight agent handling cost but causes zero safety or compliance failures.

Therefore, **Escalation Recall** is the primary design metric.

| System | Precision | Recall (PRIMARY) | F1-Score | Confusion Matrix (`[[TN, FP], [FN, TP]]`) |
|---|---|---|---|---|
| **SpotifyAgent (Hard Rules + Soft Triggers)** | **0.905** | **0.927** | **0.916** | `[[115, 4], [3, 38]]` |
| Simple Baseline (Keyword regex only) | 1.000 | 0.171 | 0.292 | `[[119, 0], [34, 7]]` |
| Trivial Baseline (Never escalate) | 0.000 | 0.000 | 0.000 | `[[119, 0], [41, 0]]` |

*Result:* On the expanded 160-sample set (41 positive escalations, 119 automated queries), the combined hard + soft escalation engine achieves **92.7% Recall** (38 out of 41 high-risk tickets intercepted) with **90.5% Precision** (only 4 false alarms across 119 automated queries).

### 6.3 Generation Quality: LLM-as-a-Judge Rubric
Generated drafts for non-escalated tickets were scored on an ordinal 1–5 scale across 5 dimensions using the structured evaluation rubric.

| Dimension | SpotifyAgent (1–5) | Simple Baseline (1–5) | Trivial Baseline (1–5) |
|---|---|---|---|
| **Relevance** | **5.00** | 3.25 | 2.10 |
| **Groundedness** | **4.94** | 3.65 | 1.80 |
| **Tone & Voice** | **5.00** | 3.80 | 2.40 |
| **Actionability** | **5.00** | 2.90 | 1.50 |
| **Safety** | **5.00** | 4.10 | 3.85 |
| **Composite Average** | **4.99 / 5.0** | **3.54 / 5.0** | **2.33 / 5.0** |

- **Judge vs. Human Agreement:** On a blind-scored validation set ($N=30$), the judge achieved **Cohen's $\kappa = 0.5567$** (moderate agreement) and a Spearman rank correlation of **$\rho = 0.8718$** ($p < 0.0001$) against human ground truth, with **0 large disagreements** ($\ge 2$ points).

---

## 7. The "Misleading Number" Section

A critical requirement of rigorous machine learning reporting is identifying metrics that look impressive in headline summaries but obscure practical failure modes.

### 1. The Faithfulness vs. Groundedness Disparity (0.391 vs. 4.94 / 5.0)
The most illuminating finding in our evaluation is the dramatic contradiction between automatic fact coverage and LLM judge scoring:
- **Faithfulness (Automated Fact Coverage):** **0.391** on a [0, 1] scale.
- **Groundedness (LLM Judge Score):** **4.94 / 5.0** (98.8% of maximum).

**Why this gap matters:** Both metrics ostensibly evaluate whether the agent provides grounded, factual troubleshooting advice. However, they measure fundamentally different properties:
1. **Automated Faithfulness checks hard factual completeness:** It computes semantic cosine similarity between each sentence in the draft and pre-specified required facts in `ideal_reply_notes` (e.g. "clear storage cache | toggle offline mode | check background battery permissions"). If a draft offers a polite, general response ("Hey! Let's get this fixed. Restart your device and reinstall."), it fails to cover the remaining specific technical propositions, yielding an average score of 0.391.
2. **The LLM Judge suffers from extreme ceiling effects:** The judge prompt evaluates whether the draft *contradicts* retrieved facts or hallucinates prices. Because the agent never fabricates dollar amounts or policies, the judge awards near-perfect scores (4.94/5.0). In fact, on three out of five dimensions (**Relevance, Tone, and Actionability**), the judge awarded **exactly 5.00 / 5.00**. A judge that awards 5.00 across multiple dimensions is exhibiting a ceiling effect rather than fine-grained discrimination.

### 2. Escalation Recall Statistical Caveat (92.7% on 41 Positive Items)
On our 160-sample evaluation, the escalation engine intercepted 38 out of 41 ground-truth escalations (92.7% recall, 95% bootstrap CI: [0.830, 1.000]).
* **Statistical Limitation:** While a 95% bootstrap CI reaching 1.000 is mathematically correct for this sample, the reader must not interpret this as a real-world guarantee. Resampling a finite sample of 41 positive cases where only 3 are missed produces an optimistic upper bound. In live production, adversarial linguistic drift (subtle sarcasm, obfuscated legal terminology, evolving slang) will inevitably degrade static regex patterns.

### 3. Dual-Intent Concurrency & Single-Turn Context
- **Single-Turn Grounding Illusion:** Evaluation treats each tweet as an isolated interaction. In real Twitter operations, ~35% of customer interactions are follow-up turns in multi-turn threads. Evaluating single turns ignores conversational history.
- **Dual-Intent Failures:** When a customer combines two issues (e.g., *"App keeps crashing on startup and you billed me twice"*), single-label classification schemes arbitrarily force one intent, leaving the secondary issue unaddressed.
 In real Twitter operations, ~35% of customer interactions are follow-up turns in multi-turn threads. Evaluating single turns ignores conversational history, creating an artificially favorable impression of agent resolution completeness.

---

## 8. Failure Mode Analysis (Top 5 Failure Topologies)

Deep-dive auditing of error cases across our 160-sample evaluation identified five distinct failure topologies:

1. **Sarcasm and Indirect Complaint Inversion:**
   - *Customer Tweet:* "Love when your app decides I don't actually need to hear the bridge of my favorite song."
   - *Model Prediction:* `General Complaint / Sentiment` (Confidence: 0.54) → Soft Escalation Triggered.
   - *Hypothesis / Mechanism:* The surface semantics contain positive sentiment markers ("Love", "favorite song") that pull the dense embedding toward Sentiment rather than Playback Bug. The model resolves surface polarity rather than pragmatic intent.

2. **Dual-Intent Concurrency (Multi-Issue Collision):**
   - *Customer Tweet:* "App keeps crashing on startup and you billed me twice for family plan."
   - *Model Prediction:* `App / Playback Bug` (Ignored billing dispute).
   - *Hypothesis / Mechanism:* Single-label classification schemes force an arbitrary choice when two distinct issues co-occur. The bug tokens dominate initial sentence attention, causing the financial dispute to be omitted from retrieval and resolution.

3. **Retrieval Over-Specificity / Platform Gap:**
   - *Customer Tweet:* Inquiring about a specific local Linux distribution packaging bug (e.g., ALSA audio backend failure).
   - *Retrieved Context:* Generic Windows/Mac audio playback threads.
   - *Result:* The model drafts generic troubleshooting steps ("Clear cache and restart") that fail to resolve the platform-specific dependency issue.
   - *Hypothesis / Mechanism:* High-volume consumer platforms skew retrieval toward mainstream mobile OS resolutions; tail platform complaints retrieve semantically proximate but functionally irrelevant troubleshooting advice.

4. **Obfuscated Account Compromise (Soft Takeover Signals):**
   - *Customer Tweet:* "Got an email saying my password was changed from Vietnam but I didn't request this."
   - *Model Prediction:* `Account / Login Issue` (Confidence: 0.72) → Drafted standard password reset link instead of immediate security escalation.
   - *Hypothesis / Mechanism:* Lacking explicit threat or compromise keywords like "hacked" or "stolen", geographic anomaly signals fail hard regex checks and appear to the classifier as standard routine password recovery inquiries.

5. **Cross-Lingual Misrouting (Multilingual Queries):**
   - *Customer Tweet:* "ola como faco para cancelar minha assinatura premium no brasil" (Portuguese: how do I cancel my premium subscription in Brazil).
   - *Model Prediction:* `Premium / Billing` (Confidence: 0.48) → Drafted standard English cancellation guide (`spotify.com/account`).
   - *Hypothesis / Mechanism:* Modern multilingual LLMs comprehend the foreign semantic intent (billing cancellation), but because the retrieval index and system prompt enforce English brand templates, the agent responds in English to a non-English speaker rather than routing to regional language support.

---

## 9. Operational & Production Readiness

### 9.1 Latency and Rate Limit Budget
- **Classification Latency:** ~420ms (Gemini 3.6 Flash).
- **FAISS Retrieval Latency:** ~18ms (8,000 vectors, $d=384$ using IndexFlatIP).
- **Draft Generation Latency:** ~680ms (conditioned on 3 historical examples).
- **Total Pipeline Latency:** ~1,150ms per customer interaction.
- **Throughput Management:** Gemini free tier allows 15 RPM. The production runner implements an internal rate-limiter enforcing a 4.1s spacing interval between external API invocations.

### 9.2 Token Economics (Cost per 10k Interactions)
- Average prompt size: 380 tokens.
- Average completion size: 45 tokens.
- At Gemini 3.6 Flash pricing ($0.075 / 1M input tokens, $0.30 / 1M output tokens):
  - 10,000 queries = 3.8M input tokens ($0.285) + 0.45M output tokens ($0.135) = **$0.42 per 10,000 customer tickets**.
  - Highly cost-effective relative to human tier-1 triage ($3.00–$6.00 per handled ticket).

---

## 10. What We Would Do Next With One More Week

1. **Multi-Turn Conversation Thread Ingestion:** Reconstruct full 3-to-5 turn dialogue chains from the Twitter parent tweet IDs (`in_response_to_tweet_id`). Incorporating conversation history eliminates the "single-turn illusion" and allows the agent to recognize when prior suggested steps already failed.
2. **Multi-Label Intent Architecture:** Replace the 7-class mutually exclusive classifier with a multi-label sigmoid scoring head ($P(\text{intent}_i) \ge \tau$) to accurately decompose combined bug + billing complaints into dual routing actions.
3. **Dynamic Few-Shot Retrieval for Classification:** Rather than using a static prompt with fixed examples, retrieve the top 3 most semantically similar verified historical customer tweets to construct dynamic few-shot classification prompts.
4. **Independent Cross-Family LLM Judge:** Implement an automated validation pipeline with an external model family (such as Claude 3.5 Sonnet or GPT-4o) to eliminate intra-family judge leniency and provide stricter discriminative scoring.
5. **Real-Time Knowledge Base Synchronization:** Connect the FAISS retriever to live Spotify Community RSS and status feeds (`@SpotifyStatus`) to prevent hallucinating troubleshooting advice during global platform outages.
