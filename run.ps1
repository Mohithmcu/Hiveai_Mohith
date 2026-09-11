# Spotify Support Agent — run.ps1
# Reproduces headline results in <15 minutes on a clean checkout.
# Usage: .\run.ps1

param(
    [switch]$SkipIngest,
    [switch]$SkipIndex,
    [switch]$SkipJudge
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."

function Run-Command {
    param([string]$cmd)
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Step failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`n=== Spotify AI Support Agent ===" -ForegroundColor Cyan
Write-Host "Brand: SpotifyCares | Model: gemini-3.6-flash`n"

# 1. Check .env
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example" -ForegroundColor Yellow
    Write-Host "IMPORTANT: Add your GEMINI_API_KEY to .env before continuing." -ForegroundColor Red
    exit 1
}

# Check if key is placeholder
$envContent = Get-Content ".env" -Raw
if ($envContent -match "your_gemini_api_key_here") {
    Write-Host "WARNING: GEMINI_API_KEY in .env is still set to placeholder." -ForegroundColor Yellow
    Write-Host "Please add your real Gemini API key to .env if you plan to run agent or judge." -ForegroundColor Yellow
}

# 2. Install deps
Write-Host "[1/7] Checking dependencies..." -ForegroundColor Green
Run-Command "pip install -r requirements.txt -q"

# 3. Ingest
if (-not $SkipIngest -and (-not (Test-Path "data/cache/spotify_threads.parquet"))) {
    Write-Host "[2/7] Ingesting SpotifyCares threads..." -ForegroundColor Green
    Run-Command "python src/ingest.py"
} else {
    Write-Host "[2/7] Skipping ingest (cache exists or -SkipIngest specified)" -ForegroundColor Yellow
}

# 4. Taxonomy
if (-not (Test-Path "data/cache/taxonomy.parquet")) {
    Write-Host "[3/7] Building intent taxonomy (embed + cluster)..." -ForegroundColor Green
    Run-Command "python src/taxonomy.py"
} else {
    Write-Host "[3/7] Skipping taxonomy (cache exists)" -ForegroundColor Yellow
}

# 5. Index
if (-not $SkipIndex -and (-not (Test-Path "data/cache/retrieval_index.index"))) {
    Write-Host "[4/7] Building FAISS retrieval index..." -ForegroundColor Green
    Run-Command "python src/index.py"
} else {
    Write-Host "[4/7] Skipping index build (cache exists or -SkipIndex specified)" -ForegroundColor Yellow
}

# 6. Check golden set
if (-not (Test-Path "eval/golden_set.csv") -or (Get-Content "eval/golden_set.csv").Count -le 1) {
    Write-Host "`n[5/7] Golden set not found. Generating sampling template..." -ForegroundColor Yellow
    Run-Command "python src/sample_golden.py"
    Write-Host "IMPORTANT: Hand-label eval/golden_set_template.csv, save as eval/golden_set.csv, then re-run." -ForegroundColor Red
    exit 0
}

# 7. Run agent over golden set
Write-Host "[5/7] Running agent over golden set..." -ForegroundColor Green
Run-Command "python src/run_agent.py --golden eval/golden_set.csv --out eval/predictions.csv"

# 8. Run baselines
Write-Host "[6/7] Running baselines..." -ForegroundColor Green
Run-Command "python src/baselines.py --golden eval/golden_set.csv"

# 9. Judge + Eval
if (-not $SkipJudge) {
    Write-Host "[7/7] Running LLM judge..." -ForegroundColor Green
    Run-Command "python src/run_judge.py --predictions eval/predictions.csv --out eval/judge_scores.csv"
}

Write-Host "[7/7] Computing metrics..." -ForegroundColor Green
Run-Command "python src/eval.py --golden eval/golden_set.csv --predictions eval/predictions.csv"

Write-Host "`n=== Done. Results in eval/eval_results.json ===" -ForegroundColor Cyan
