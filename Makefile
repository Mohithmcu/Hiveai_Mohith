# SpotifyCares Support Agent — Makefile
.PHONY: setup ingest taxonomy index sample eval baselines clean run

setup:
	pip install -r requirements.txt

ingest:
	python src/ingest.py

taxonomy:
	python src/taxonomy.py

index:
	python src/index.py

sample:
	python src/sample_golden.py

agent:
	python src/run_agent.py --golden eval/golden_set.csv --out eval/predictions.csv

baselines:
	python src/baselines.py --golden eval/golden_set.csv

judge:
	python src/run_judge.py --predictions eval/predictions.csv --out eval/judge_scores.csv

eval:
	python src/eval.py --golden eval/golden_set.csv --predictions eval/predictions.csv

run: ingest taxonomy index agent baselines judge eval

clean:
	rm -rf data/cache/*
