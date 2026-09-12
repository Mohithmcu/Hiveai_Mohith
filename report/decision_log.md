# Decision Log — SpotifyCares AI Support Agent

12 non-obvious decisions made during this project, with rationale.

---

## 1. Brand: SpotifyCares over AmazonHelp

AmazonHelp is the largest brand (42k replies in first 500k rows) but spans 15+ domains: AWS billing, Prime Video, physical returns, hardware, third-party sellers. A coherent 6-10 intent taxonomy is nearly impossible without massive data engineering. SpotifyCares covers ~5 coherent domains (app bugs, login/account, premium billing, content, feature requests) with 43k total replies — enough retrieval volume and a defensible taxonomy. The narrower scope makes "grounded in how the brand resolved similar issues" a meaningful claim instead of a stretch.

## 2. Intent taxonomy derived by clustering, not hand-specified

Hand-picking intents from intuition introduces cherry-picking bias — you design for the cases you can already see. Clustering first, then reading 10-15 examples per cluster to name them, grounds the taxonomy in actual data distribution. It also generates a natural "Other / Unclassifiable" bucket from the cluster that doesn't fit cleanly. The clustering notebook (taxonomy.py silhouette sweep output) is reproducible evidence of the process.

## 3. KMeans over HDBSCAN for clustering

HDBSCAN produces variable cluster counts across runs and has noise points (label=-1) that complicate downstream labeling. KMeans with a silhouette sweep gives a deterministic, reproducible result. For a report where the reviewer must reproduce your numbers, determinism beats sophistication.

## 4. "Direct customer tweet" as grounding unit, not thread root

Walking all the way back to the thread-opening tweet adds complexity (some threads are 10+ turns) and the most proximate customer message is the most relevant context for the brand reply. Trade-off: we lose multi-turn context. This is a named limitation in the misleading-number section — not a hidden weakness.

## 5. Escalation check runs BEFORE drafting

If a message qualifies for escalation, we skip the draft entirely. Rationale: (a) drafting wastes tokens and latency for messages that won't be auto-sent; (b) a draft for a high-risk message could accidentally be sent. Architectural safety > convenience. Keeping escalation as a separate, auditable step (not folded into the draft prompt) means it can be tested and updated independently.

## 6. Hard rules implemented as named regex constants, not buried in prompts

`LEGAL_PATTERNS`, `SAFETY_PATTERNS`, etc. are named module-level constants. This makes them auditable (you can read exactly what triggers escalation), testable (unit-testable in isolation), and defensible live (I can explain each pattern if asked). A prompt that says "escalate if the message seems threatening" is a black box.

## 7. Recall (not F1) as the primary escalation metric

A missed escalation (false negative) means an auto-reply goes to a customer who needed human attention — reputational risk, possible legal exposure. A false positive means a human agent reviews an unnecessary ticket — wasted time but recoverable. Given this asymmetry, we optimize for and report recall on the escalate=True class as the primary number. F1 is reported as secondary context.

## 8. Same model (Gemini 1.5 Flash) for generation and judge

Acknowledged as a limitation. The alternative (two providers) adds dependency complexity and potential reproducibility issues. Same-model judge bias is explicitly called out in the misleading-number section: "judge may systematically prefer output style of its own model family." Treat judge scores as relative comparisons between conditions, not absolute quality measures.

## 9. Faithfulness metric: semantic similarity to ideal_reply_notes facts

Rather than measuring BLEU (surface overlap) or asking the judge to assess groundedness alone, we pre-specify the 2-3 facts a good reply *must* contain (ideal_reply_notes) and measure whether the generated reply covers them semantically. This is crude but reproducible, does not require a second LLM call, and directly tests the claim "grounded in historical resolutions."

## 10. Bootstrap CIs on all headline metrics

A golden set of 160 examples produces wide confidence intervals. Reporting a point estimate (e.g., "Macro-F1 = 0.832") without CIs is misleading. Bootstrap CIs make the uncertainty concrete and honest. 95% CI on Macro-F1 from 160 samples is [0.769, 0.885] (±0.058) — wide enough that the reader should not over-interpret small differences between conditions.

## 11. Retrieval index capped at 8,000 pairs

Using all 43k pairs would improve recall but slow index build and query time. 8k is a representative sample that keeps index.py runtime under 2 minutes and query latency under 50ms — both relevant for the 15-minute reproducibility promise. Named in the report: larger index is a clear "next week" improvement.

## 12. Out-of-scope items and why

- **Multi-turn dialogue state**: Twitter threads are multi-turn but the grounding corpus is single (customer_msg, brand_reply) pairs. Supporting full thread context would require thread reconstruction across all 2.8M rows and a context-window-aware prompt. Explicitly out of scope; named limitation.
- **Non-English messages**: Classified as "Other / Unclassifiable" and escalated. Gemini 1.5 Flash handles non-English but the retrieval corpus is English-only, making grounding unreliable.
- **Live handoff integration**: No actual Zendesk/Freshdesk integration — escalation outputs a structured JSON decision that a real system would consume.
- **Sentiment intensity scoring**: General Complaint / Sentiment is a single class; intensity within it (annoyed vs. furious) could refine escalation. Out of scope for this time box.
