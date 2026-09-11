# Hiver Take-Home — Implementation Plan

## Brand: SpotifyCares | Model: Gemini 3.6 Flash

### Data confirmed
- Full dataset: 2,811,774 rows
- SpotifyCares replies: 43,265

### Repo structure to build
```
/data/raw/          twcs.csv (gitignored)
/data/cache/        spotify_threads.parquet (committed)
/notebooks/         01_explore.ipynb, 02_taxonomy.ipynb, 03_golden_sampling.ipynb
/src/
  ingest.py         filter + thread reconstruction + parquet cache
  taxonomy.py       embed + cluster + intent mapping
  index.py          build FAISS retrieval index
  agent.py          classify -> retrieve -> draft -> escalate
  baselines.py      trivial + simple baselines
  judge.py          Gemini Pro judge rubric
  eval.py           all metrics, confusion matrices, CI bootstrapping
/eval/
  golden_set.csv    150-250 hand-labeled examples
  judge_vs_human.csv  blind-scored subset for agreement study
/report/
  report.md
  decision_log.md
Makefile
README.md
requirements.txt
.env.example
```

### Intent taxonomy (target 7 classes)
1. App / Playback Bug
2. Account / Login Issue
3. Premium / Billing
4. Content / Playlist Issue
5. Feature Request
6. General Complaint / Sentiment
7. Other / Unclassifiable

### Execution order
1. ingest.py → cache parquet
2. taxonomy.py → clusters → hand-name
3. golden set sampling + labeling
4. index.py → FAISS index
5. agent.py → full pipeline
6. baselines.py
7. eval.py + judge.py
8. report + decision log + README
