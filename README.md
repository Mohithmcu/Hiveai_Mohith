# SpotifyCares AI Support Agent

Model: Gemini 3.6 Flash**

## What this builds

An AI support agent for SpotifyCares that:
1. **Classifies** incoming customer tweets into 7 intents
2. **Drafts** replies grounded in 8,000 real historical SpotifyCares resolutions
3. **Decides** auto-handle vs. escalate with a stated reason

## Reproduce headline results in <15 minutes

### Prerequisites
- Python 3.10+
- `data/raw/twcs.csv` (download from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter))
- Gemini API key (free at [aistudio.google.com](https://aistudio.google.com/app/apikey))

### Setup
```powershell
# 1. Clone and enter repo
cd Hiver

# 2. Install dependencies 
pip install -r requirements.txt

# 3. Add API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 4. Run full pipeline 
# Windows:
.\run.ps1

# Linux / macOS:
make run
```

### What run.ps1 does (in order)
| Step | Script | Time |
|------|--------|------|
| Filter SpotifyCares + build parquet cache | `src/ingest.py` | ~3 min |
| Embed + cluster → intent taxonomy | `src/taxonomy.py` | ~3 min |
| Build FAISS retrieval index | `src/index.py` | ~1 min |
| Run agent over golden set | `src/run_agent.py` | ~3 min |
| Run LLM judge | `src/run_judge.py` | ~2 min |
| Compute all metrics + CIs | `src/eval.py` | ~30 sec |

**Headline results table printed to stdout and saved to `eval/eval_results.json`.**

> **Note:** `ingest.py` and `index.py` cache results to `data/cache/`. Re-runs skip these steps automatically (use `--SkipIngest --SkipIndex` flags).

---

## Run individual steps

```powershell
# Just ingest
python src/ingest.py

# Just taxonomy
python src/taxonomy.py

# Just build index
python src/index.py

# Sample golden set template (then hand-label it)
python src/sample_golden.py

# Run agent on a single tweet
python src/agent.py

# Run baselines only
python src/baselines.py --golden eval/golden_set.csv

# Run judge
python src/run_judge.py --predictions eval/predictions.csv --out eval/judge_scores.csv

# Full eval metrics
python src/eval.py --golden eval/golden_set.csv --predictions eval/predictions.csv
```

---

## Repo structure

```
data/
  raw/twcs.csv              # gitignored — download from Kaggle
  cache/                    # auto-generated parquet + index files
src/
  ingest.py                 # filter + thread reconstruction + parquet
  taxonomy.py               # embed + cluster + intent labels
  index.py                  # FAISS retrieval index
  agent.py                  # main pipeline: classify→retrieve→escalate→draft
  baselines.py              # trivial + simple baselines
  judge.py                  # Gemini LLM-as-judge rubric
  eval.py                   # all metrics + bootstrap CIs
  run_agent.py              # batch runner over golden set
  run_judge.py              # batch judge runner
  sample_golden.py          # stratified golden set sampler
eval/
  golden_set.csv            # 160 hand-labelled examples (N=160, spec 150–250)
  predictions.csv           # agent outputs (auto-generated)
  judge_scores.csv          # judge dimension scores (auto-generated)
  judge_vs_human.csv        # agreement study subset
  eval_results.json         # headline metrics (auto-generated)
report/
  report.md                 # full report
  decision_log.md           # 12 non-obvious decisions + rationale
run.ps1                     # single-command reproducer
requirements.txt            # pinned versions
.env.example                # API key template
```

---

## Intent taxonomy (7 classes)

Derived by clustering 10,000 customer messages with KMeans (k chosen via silhouette sweep):

| # | Intent | Description |
|---|--------|-------------|
| 0 | App / Playback Bug | Crashes, skipping, buffering, audio issues |
| 1 | Account / Login Issue | Password reset, account access, hacking |
| 2 | Premium / Billing | Charges, cancellation, refunds |
| 3 | Content / Playlist Issue | Missing songs, removed content, playlist bugs |
| 4 | Feature Request | Suggestions, missing features |
| 5 | General Complaint / Sentiment | Vague frustration, praise |
| 6 | Other / Unclassifiable | Non-English, empty, unclear |

---

## Escalation rules

**Hard rules** (immediate escalation, pre-draft):
- Safety / self-harm language
- Legal threat language
- Public threat / reputational risk
- Specific refund amount mentioned
- High-intensity profanity + non-sentiment intent
- Account takeover / compromise signals

**Soft triggers** (escalate when confidence is low):
- Intent = "Other / Unclassifiable"
- Classification confidence < 0.35 (calibrated softmax)
- Max retrieval similarity < 0.40

---

## Golden set

- **160 hand-labelled examples** in `eval/golden_set.csv` (satisfies 150–250 requirement)
- Stratified by intent (proportional, balanced across all 7 classes)
- Includes 28 deliberate edge cases: sarcasm, multi-issue, angry/threat language (17 adversarial + 11 ambiguous)
- Self-consistency: 10% re-labelled blind after 24h; agreement reported in `report/report.md`

---

## Key design decisions

See `report/decision_log.md` for all 12 decisions with rationale.

Top 3:
1. **SpotifyCares over AmazonHelp** — tighter scope (5 domains vs 15+) → coherent taxonomy
2. **Escalation before drafting** — never create a draft for a high-risk message
3. **Same Gemini Flash model for judge** — acknowledged as limitation in report; judge independence via different prompt role and temperature

---

## Headline benchmark results (Evaluation on Golden Set, N=160)

| Metric | Scale | SpotifyAgent (Our System) | Simple Baseline (TF-IDF + LR) | Trivial Baseline (Majority Class) |
|---|---|---|---|---|
| **Intent Macro-F1** [95% CI] | 0–1 | **0.832** [0.769, 0.885] | 0.221 [0.175, 0.268] | 0.035 [0.015, 0.052] |
| **Intent Accuracy** | % | **83.1%** | 26.3% | 13.8% |
| **Escalation Recall (PRIMARY)** [95% CI] | 0–1 | **0.927** [0.830, 1.000] | 0.171 [0.073, 0.293] | 0.000 [0.000, 0.000] |
| **Escalation Precision** | 0–1 | **0.905** | 1.000 | 0.000 |
| **Escalation F1** | 0–1 | **0.916** | 0.292 | 0.000 |
| **Faithfulness Score (Ideal Facts)** | 0–1 | **0.391** [0.354, 0.428] | 0.212 [0.180, 0.245] | 0.115 [0.085, 0.142] |
| **Judge Composite Score** | 1–5 | **4.99 / 5.0** | 3.54 / 5.0 | 2.33 / 5.0 |

*All confidence intervals computed via 1,000-sample non-parametric bootstrap.*
