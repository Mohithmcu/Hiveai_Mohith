# SpotifyCares AI Support Agent

**Model: Gemini 1.5 Flash** | **Dataset: 32,363 SpotifyCares tweet pairs** | **Golden Set: 160 hand-labelled examples**

## What this builds

An AI support agent for SpotifyCares that:
1. **Classifies** incoming customer tweets into 7 intents (Gemini 1.5 Flash, few-shot)
2. **Retrieves** top-5 most similar historical resolutions (FAISS semantic search)
3. **Escalates** or **auto-handles** with a stated reason (hard rules + soft triggers, runs BEFORE drafting)
4. **Drafts** replies grounded in 8,000 real SpotifyCares resolutions

---

## Reproduce headline results in <15 minutes

### Prerequisites
- Python 3.10+
- Gemini API key — free at [aistudio.google.com](https://aistudio.google.com/app/apikey)
- **No dataset download needed** — `data/spotify_twcs.csv` is included in this repo

### Setup & Run
```powershell
# 1. Clone and enter repo
git clone https://github.com/Mohithmcu/Hiveai_Mohith.git
cd Hiveai_Mohith

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key
copy .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here

# 4. Run the full pipeline (Windows)
.\run.ps1

# Linux / macOS
make run
```

### What `run.ps1` does (in order)

| Step | Script | What it does | Time |
|------|--------|--------------|------|
| [1/7] | `pip install` | Install dependencies | ~1 min |
| [2/7] | `src/ingest.py` | Load `spotify_twcs.csv` → clean → parquet cache | ~30 sec |
| [3/7] | `src/taxonomy.py` | Embed + KMeans cluster → intent labels | ~3 min |
| [4/7] | `src/index.py` | Build FAISS retrieval index (8,000 pairs) | ~1 min |
| [5/7] | `src/run_agent.py` | Run agent over 160-example golden set | ~3 min |
| [6/7] | `src/baselines.py` | Evaluate trivial + simple baselines | ~30 sec |
| [7/7] | `src/run_judge.py` + `src/eval.py` | LLM judge + all metrics + bootstrap CIs | ~3 min |

> **Caching:** Steps 2–4 skip automatically if cache files already exist. Use `-SkipIngest`, `-SkipIndex`, `-SkipJudge` flags to skip individual steps.

**Results printed to stdout and saved to `eval/eval_results.json`.**

---

## Headline Results (Golden Set, N=160)

| Metric | Scale | **SpotifyAgent** | Simple Baseline (TF-IDF + LR) | Trivial Baseline (Majority) |
|--------|-------|-----------------|-------------------------------|------------------------------|
| **Intent Macro-F1** [95% CI] | 0–1 | **0.832** [0.769, 0.885] | 0.221 [0.175, 0.268] | 0.035 [0.015, 0.052] |
| **Intent Accuracy** | % | **83.1%** | 26.3% | 13.8% |
| **Escalation Recall ★** [95% CI] | 0–1 | **0.927** [0.833, 1.000] | 0.171 [0.073, 0.293] | 0.000 |
| **Escalation Precision** | 0–1 | **0.905** | 1.000 | 0.000 |
| **Escalation F1** | 0–1 | **0.916** | 0.292 | 0.000 |
| **Faithfulness Score** | 0–1 | **0.391** [0.354, 0.428] | 0.212 [0.180, 0.245] | 0.115 [0.085, 0.142] |
| **Judge Composite Score** | 1–5 | **4.99 / 5.0** | 3.54 / 5.0 | 2.33 / 5.0 |

★ **Escalation Recall is the PRIMARY metric** — a missed escalation costs more than a false alarm.

*All confidence intervals via 1,000-sample non-parametric bootstrap.*

---

## Repo Structure

```
data/
  spotify_twcs.csv          ← SpotifyCares-only dataset (32,363 pairs, committed to repo)
  cache/                    ← auto-generated parquet + index files (gitignored)
src/
  ingest.py                 ← load spotify_twcs.csv → clean → parquet cache
  taxonomy.py               ← embed + KMeans cluster → 7 intent labels
  index.py                  ← build FAISS flat-IP retrieval index
  agent.py                  ← main pipeline: classify → retrieve → escalate → draft
  baselines.py              ← trivial (majority class) + simple (TF-IDF + LR) baselines
  judge.py                  ← Gemini 1.5 Flash LLM-as-judge (5 rubric dimensions)
  eval.py                   ← intent, escalation, reply metrics + bootstrap CIs
  run_agent.py              ← batch runner: agent over golden set → predictions.csv
  run_judge.py              ← batch runner: judge over predictions → judge_scores.csv
  sample_golden.py          ← stratified golden set sampler (edge-case aware)
eval/
  golden_set.csv            ← 160 hand-labelled examples
  predictions.csv           ← agent outputs (auto-generated)
  judge_scores.csv          ← judge dimension scores (auto-generated)
  judge_vs_human.csv        ← 30-example agreement study subset
  eval_results.json         ← headline metrics (auto-generated)
report/
  report.md                 ← full technical report
  decision_log.md           ← 12 non-obvious design decisions with rationale
run.ps1                     ← single-command reproducer (Windows PowerShell)
Makefile                    ← single-command reproducer (Linux/macOS)
requirements.txt            ← pinned dependency versions
.env.example                ← API key template
```

---

## Intent Taxonomy (7 Classes)

Derived by embedding 10,000 customer messages and clustering with KMeans (k=7):

| # | Intent | Examples |
|---|--------|---------|
| 0 | App / Playback Bug | crashes, skipping, buffering, audio cuts |
| 1 | Account / Login Issue | password reset, account access, hacking |
| 2 | Premium / Billing | double charges, cancellation, refunds |
| 3 | Content / Playlist Issue | missing songs, removed content, playlist bugs |
| 4 | Feature Request | suggestions, missing features |
| 5 | General Complaint / Sentiment | vague frustration, praise |
| 6 | Other / Unclassifiable | non-English, too short, unclear |

---

## Escalation Rules

### Hard Rules (immediate escalation, checked BEFORE drafting)
| Rule | Trigger |
|------|---------|
| `SAFETY` | Self-harm / suicide language |
| `LEGAL` | Lawsuit, lawyer, trading standards |
| `THREAT` | Public exposure, social media blast |
| `REFUND_AMT` | Specific dollar amount mentioned |
| `HACKED_TAKEOVER` | Account compromise / data deletion |
| `PROFANITY` | High-intensity profanity |
| `FRAGMENT` | Unclassifiable short fragment (≤3 words) |

### Soft Triggers (escalate when grounding is unreliable)
| Trigger | Threshold |
|---------|-----------|
| Intent = `Other / Unclassifiable` | always |
| Classification confidence | < 0.35 |
| Max retrieval similarity | < 0.40 |

---

## Run Individual Steps

```powershell
# Ingest only
python src/ingest.py

# Taxonomy only
python src/taxonomy.py

# Build FAISS index only
python src/index.py

# Test agent on sample tweets
python src/agent.py

# Run agent over golden set
python src/run_agent.py --golden eval/golden_set.csv --out eval/predictions.csv

# Run baselines
python src/baselines.py --golden eval/golden_set.csv

# Run LLM judge
python src/run_judge.py --predictions eval/predictions.csv --out eval/judge_scores.csv

# Compute all metrics
python src/eval.py --golden eval/golden_set.csv --predictions eval/predictions.csv
```

---

## Key Design Decisions

See [`report/decision_log.md`](report/decision_log.md) for all 12 decisions with rationale. Top 3:

1. **SpotifyCares over AmazonHelp** — tighter scope (5 domains vs 15+) → coherent taxonomy
2. **Escalation before drafting** — never create a draft for a high-risk message
3. **Recall as primary escalation metric** — missed escalations cost more than false alarms

---

## Dependencies

```
pandas>=2.0.0          # data handling
pyarrow>=14.0.0        # parquet I/O
numpy>=1.24.0          # numerics
sentence-transformers>=2.2.0  # embeddings (all-MiniLM-L6-v2)
faiss-cpu>=1.7.0       # retrieval index
scikit-learn>=1.3.0    # clustering, baselines, metrics
google-generativeai>=0.7.0    # Gemini 1.5 Flash
scipy>=1.10.0          # Spearman correlation
python-dotenv>=1.0.0   # .env loading
tqdm>=4.65.0           # progress bars
```
