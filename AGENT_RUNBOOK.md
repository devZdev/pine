# Agent Runbook — Automated Pipeline Execution

This document is written for an AI agent coming in cold. Read it completely before executing any step. Do not skip steps. Do not proceed past a gate check that fails — stop and report the failure with the exact error output.

---

## What you are doing

You are running a quantitative trading research pipeline that ends with Pine Script v6 files the user loads directly into TradingView. Your job is to execute every phase in order, validate each output before moving on, and leave a clean set of artifacts. The user will not be watching — leave detailed logs and a final artifact manifest so they can pick up exactly where you left off.

---

## Pre-flight checklist

Run these checks before touching anything else. If any fail, stop and report — do not attempt to proceed.

### 1. Python version
```bash
python --version
```
Required: 3.11 or higher. If lower, stop.

### 2. Dependencies installed
```bash
pip install -r requirements.txt
```
If any package fails to install, report the exact error. Do not continue with missing packages.

### 3. Credentials present
```bash
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
required = ['COINBASE_API_KEY','COINBASE_API_SECRET','ALPACA_API_KEY','ALPACA_SECRET_KEY']
missing = [k for k in required if not os.getenv(k) or 'YOUR_' in os.getenv(k,'')]
print('MISSING:', missing) if missing else print('OK: all Phase 1 credentials present')
"
```
If any variable is missing or still contains a placeholder value, stop and tell the user which keys need to be filled in `.env`.

### 4. Data directory
```bash
mkdir -p data/raw backtest/results
```
Always safe to run — creates directories if they don't exist.

---

## Phase 1 — Data ingestion

**Run when:** First time, or to update data with new candles since the last run.
**The pipeline is incremental** — re-running only fetches candles newer than the last saved timestamp. It is always safe to re-run.

### Command
```bash
python main.py \
  --symbols BTC TSLA \
  --timeframes 1m 5m \
  --start 2020-01-01 \
  --log-level INFO
```

Expected runtime: 20–40 minutes on first run. Subsequent incremental runs: 1–3 minutes.

### Gate check — do not proceed until this passes
```bash
python -c "
import pandas as pd, os
files = {
    'BTC_USD_1m':  'data/raw/BTC_USD_1m.parquet',
    'BTC_USD_5m':  'data/raw/BTC_USD_5m.parquet',
    'TSLA_1m':     'data/raw/TSLA_1m.parquet',
    'TSLA_5m':     'data/raw/TSLA_5m.parquet',
}
for name, path in files.items():
    if not os.path.exists(path):
        print(f'FAIL: {path} missing')
        continue
    df = pd.read_parquet(path)
    required_cols = ['open','high','low','close','volume','atr_14','sma_200','bb_pct']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f'FAIL: {name} missing columns: {missing}')
    elif len(df) < 1000:
        print(f'FAIL: {name} has only {len(df)} rows — likely incomplete')
    else:
        print(f'OK: {name} — {len(df):,} rows, {df.index.min().date()} to {df.index.max().date()}')
"
```
All four files must report OK before proceeding. If any fail, re-run Phase 1 or report the error.

---

## Phase 2 — Black Box regime classifier

**Status: NOT YET BUILT.** This phase will be implemented in a future session.

**What to do now:** Skip this phase. The backtest and Pine Script will run without it. When Phase 2 is built, a `main_regime.py` file will appear in the repo root and this section will be updated with run instructions.

**Do not attempt to start a FastAPI server or run any AI model — that code does not exist yet.**

---

## Phase 3 — Backtest

**Run when:** After Phase 1 data is confirmed present. Run the fast version first to validate, then the full WFO run to lock in parameters.

### Step 3a — Fast validation run (no WFO)
```bash
python main_backtest.py \
  --no-wfo \
  --bb-pct 0.20 \
  --hurst 0.45 \
  --log-level INFO
```
Expected runtime: 5–10 minutes.

### Gate check 3a — do not proceed to WFO until this passes
```bash
python -c "
import os, pandas as pd
path = 'backtest/results/performance_matrix.csv'
if not os.path.exists(path):
    print('FAIL: performance_matrix.csv not found')
else:
    df = pd.read_csv(path)
    print('OK: performance matrix present')
    print(df.to_string())
"
```
Inspect the printed metrics. If Total Return is negative for both IS and OOS periods, stop and report — do not run the full WFO. The strategy may need parameter adjustment.

### Step 3b — Full walk-forward optimization run
```bash
python main_backtest.py \
  --log-level INFO
```
Expected runtime: 30–60 minutes.

### Gate check 3b
```bash
python -c "
import os, pandas as pd
for fname in ['performance_matrix.csv', 'trade_log.csv']:
    path = f'backtest/results/{fname}'
    if not os.path.exists(path):
        print(f'FAIL: {fname} missing')
    else:
        df = pd.read_csv(path)
        print(f'OK: {fname} — {len(df)} rows')
"
```
Both files must be present before proceeding.

---

## Phase 4 — Pine Script generation

**Status: NOT YET BUILT.** This phase will be implemented in a future session.

**What to do now:** Skip execution. When Phase 4 is built, two files will appear:
- `tradingview/lib_atr_mean_reversion.pine` — the reusable library script
- `tradingview/strategy_csp.pine` — the strategy script that fires webhook alerts

This section will be updated with exact TradingView load instructions when those files exist.

---

## After all phases complete — produce the artifact manifest

Run this after completing all available phases. It generates a summary file the user can read to understand exactly what was produced.

```bash
python -c "
import os, pandas as pd
from datetime import datetime

lines = []
lines.append('# Pipeline Artifact Manifest')
lines.append(f'Generated: {datetime.utcnow().strftime(\"%Y-%m-%d %H:%M UTC\")}')
lines.append('')

# Phase 1 artifacts
lines.append('## Phase 1 — Data')
parquet_files = [
    'data/raw/BTC_USD_1m.parquet',
    'data/raw/BTC_USD_5m.parquet',
    'data/raw/TSLA_1m.parquet',
    'data/raw/TSLA_5m.parquet',
]
for path in parquet_files:
    if os.path.exists(path):
        df = pd.read_parquet(path)
        size_mb = os.path.getsize(path) / 1_000_000
        lines.append(f'- {path}: {len(df):,} rows | {df.index.min().date()} to {df.index.max().date()} | {size_mb:.1f} MB')
    else:
        lines.append(f'- {path}: MISSING')

lines.append('')

# Phase 3 artifacts
lines.append('## Phase 3 — Backtest')
for path in ['backtest/results/performance_matrix.csv', 'backtest/results/trade_log.csv']:
    if os.path.exists(path):
        df = pd.read_csv(path)
        lines.append(f'- {path}: {len(df)} rows')
    else:
        lines.append(f'- {path}: MISSING')

lines.append('')

# Phase 2 / 4 status
lines.append('## Phase 2 — Black Box Regime Classifier')
lines.append('- Status: Not yet built')
lines.append('')
lines.append('## Phase 4 — Pine Script Files')
pine_files = [
    'tradingview/lib_atr_mean_reversion.pine',
    'tradingview/strategy_csp.pine',
]
for path in pine_files:
    if os.path.exists(path):
        lines.append(f'- {path}: READY — load into TradingView Pine Script Editor')
    else:
        lines.append(f'- {path}: Not yet built')

lines.append('')
lines.append('## Next steps for the user')
lines.append('1. Review backtest/results/performance_matrix.csv — check Calmar and Sortino ratios')
lines.append('2. Review backtest/results/trade_log.csv — inspect individual trades for sanity')
lines.append('3. When Phase 4 is complete: open TradingView → Pine Editor → load each .pine file')
lines.append('4. Set up webhook alert in TradingView pointing to your server URL + WEBHOOK_SECRET')

manifest = '\n'.join(lines)
with open('ARTIFACT_MANIFEST.md', 'w') as f:
    f.write(manifest)
print(manifest)
print()
print('Saved to ARTIFACT_MANIFEST.md')
"
```

---

## How the Pine Script files get into TradingView

*(This section applies once Phase 4 is built and the `.pine` files exist.)*

**File 1 — Library script** (`tradingview/lib_atr_mean_reversion.pine`)
1. Open TradingView → Pine Script Editor (bottom panel)
2. Delete default content → paste the full contents of the library file
3. Click **Publish Script** → choose **Publish to Account (Private)**
4. Note the published library name — the strategy script imports it by this name

**File 2 — Strategy script** (`tradingview/strategy_csp.pine`)
1. Open a new Pine Script Editor tab
2. Paste the full contents of the strategy file
3. Click **Add to Chart**
4. The strategy will appear as an overlay on your BTC or TSLA chart

**Setting up the webhook alert**
1. Click the **Alert** (clock) icon → Create Alert
2. Set Condition: select your strategy → "Order fills only"
3. Set Webhook URL: `https://your-server.com/webhook?secret=YOUR_WEBHOOK_SECRET`
4. The alert payload fires automatically when the strategy signals a put-selling opportunity

---

## Failure handling

| Situation | What to do |
|---|---|
| Phase 1 rate limit error (429) | Wait 60 seconds, re-run. The pipeline resumes from the last saved candle. |
| Phase 1 missing candles warning | Expected for weekends (TSLA) and exchange downtime. Log the warning and continue. |
| Phase 3 negative OOS return | Stop. Report the full performance matrix. Do not proceed to TradingView with a losing strategy. |
| Any unhandled Python exception | Report the full traceback. Do not attempt to work around it — the user needs to see the real error. |
| `.env` credentials rejected by API | Stop. Report which API returned an auth error. Do not retry more than twice. |

---

## Artifact checklist — what a successful run leaves behind

```
data/raw/
  BTC_USD_1m.parquet          ← Phase 1
  BTC_USD_5m.parquet          ← Phase 1
  TSLA_1m.parquet             ← Phase 1
  TSLA_5m.parquet             ← Phase 1

backtest/results/
  performance_matrix.csv      ← Phase 3
  trade_log.csv               ← Phase 3

tradingview/                  ← Phase 4 (not yet built)
  lib_atr_mean_reversion.pine
  strategy_csp.pine

ARTIFACT_MANIFEST.md          ← generated by this runbook
```

The user's final deliverable is the two `.pine` files. Everything else is research infrastructure that supports and validates those files.
