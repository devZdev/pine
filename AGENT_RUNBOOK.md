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

**Run when:** After Phase 1 data is confirmed present. Keep running in the background while using TradingView.

### Pre-flight check
```bash
python -c "
from dotenv import load_dotenv; import os; load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN', '')
print('FAIL: HUGGINGFACE_TOKEN missing or placeholder' if not token or 'YOUR_' in token else 'OK: HUGGINGFACE_TOKEN present')
"
```

### Command — Docker (recommended)
```bash
# First run: builds image, downloads torch CPU (~200MB) + model weights (~300MB)
# Expect 5–10 min on first build. Model is cached — subsequent starts are fast.
docker compose up --build

# Subsequent starts
docker compose up -d
```

### Command — local (no Docker)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chronos-forecasting
python main_regime.py
```

### Gate check — do not proceed until this passes
```bash
curl -s http://localhost:8000/health | python -m json.tool
```
Must return `"loaded": true`. If `"loaded": false`, check Docker logs: `docker compose logs -f`.

### Smoke test
```bash
curl -s "http://localhost:8000/regime?symbol=BTC" | python -m json.tool
curl -s "http://localhost:8000/regime?symbol=TSLA" | python -m json.tool
```
Both must return a valid regime response with `regime` set to `TRENDING` or `MEAN_REVERTING`.

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

## Phase 4 — Pine Script delivery

**Status: BUILT.** Pine Script files are checked into `tradingview/` — they are the user's final deliverable. There is no Python runtime for this phase. The agent's job here is to **verify the files exist and are syntactically clean**, then point the user to the TradingView setup guide.

### Verification check
```bash
ls -la tradingview/lib_atr_mean_reversion.pine tradingview/strategy_csp.pine tradingview/README_TRADINGVIEW.md && \
grep -c "^//@version=6" tradingview/lib_atr_mean_reversion.pine tradingview/strategy_csp.pine
```
Both files must report `1` for the `//@version=6` count. Missing files = report failure to user.

### Hand-off to user
Tell the user:
> Phase 4 deliverables are at `tradingview/`. Read `tradingview/README_TRADINGVIEW.md` for the full TradingView load procedure — publish the library script first, replace `<USERNAME>` in the strategy with your TradingView handle, then attach the strategy to a BTC or TSLA chart at 5m timeframe. Slack webhook setup is in the same file.

**Do not attempt to load Pine Scripts yourself — only the user can do this through the TradingView UI.**

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

# Phase 2 status — running service, no on-disk artifacts
lines.append('## Phase 2 — Black Box Regime Classifier')
import urllib.request, json as _json
try:
    with urllib.request.urlopen('http://localhost:8000/health', timeout=2) as r:
        h = _json.loads(r.read())
    lines.append(f'- Service: RUNNING — model {h.get(\"model\")}, loaded={h.get(\"loaded\")}')
except Exception as e:
    lines.append(f'- Service: NOT RUNNING (start with: docker compose up -d)')

lines.append('')
lines.append('## Phase 4 — Pine Script Files')
pine_files = [
    'tradingview/lib_atr_mean_reversion.pine',
    'tradingview/strategy_csp.pine',
    'tradingview/README_TRADINGVIEW.md',
]
for path in pine_files:
    if os.path.exists(path):
        lines.append(f'- {path}: READY — load into TradingView Pine Script Editor')
    else:
        lines.append(f'- {path}: MISSING')

lines.append('')
lines.append('## Next steps for the user')
lines.append('1. Review backtest/results/performance_matrix.csv — check Calmar and Sortino ratios')
lines.append('2. Review backtest/results/trade_log.csv — inspect individual trades for sanity')
lines.append('3. Confirm regime API responds: curl http://localhost:8000/regime?symbol=BTC')
lines.append('4. Open TradingView → Pine Editor → publish lib_atr_mean_reversion.pine first')
lines.append('5. Replace <USERNAME> in strategy_csp.pine, attach to BTC or TSLA 5m chart')
lines.append('6. Create Slack incoming webhook → wire it into TradingView alert (see tradingview/README_TRADINGVIEW.md)')

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

The full step-by-step setup guide is at [tradingview/README_TRADINGVIEW.md](tradingview/README_TRADINGVIEW.md). Quick overview:

**File 1 — Library script** (`tradingview/lib_atr_mean_reversion.pine`)
1. Open TradingView → Pine Script Editor (bottom panel)
2. Delete default content → paste the full contents of the library file
3. Click **Publish Script** → choose **Publish to Account (Private)**
4. Note your TradingView username — needed for the strategy import

**File 2 — Strategy script** (`tradingview/strategy_csp.pine`)
1. Open a new Pine Script Editor tab
2. Paste the full contents of the strategy file
3. Replace `<USERNAME>` in the import line with your TradingView handle
4. Click **Add to Chart** on a BTC/USD or TSLA chart, 5m timeframe (matches the backtest)
5. The strategy will overlay BB bands, SMA200, ATR trail, and a Hurst HUD

**Setting up the Slack webhook alert**
1. Create a Slack incoming webhook at [api.slack.com/messaging/webhooks](https://api.slack.com/messaging/webhooks) — copy the URL
2. In TradingView: click the **Alert** (clock) icon → Create Alert
3. Condition: select the strategy → **Any alert() call**
4. Webhook URL: paste the Slack webhook URL
5. Message: leave blank — Pine's `alert()` populates the body with Block Kit JSON
6. The Slack message renders with a header, all signal fields, and a context line

---

## Failure handling

| Situation | What to do |
|---|---|
| Phase 1 rate limit error (429) | Wait 60 seconds, re-run. The pipeline resumes from the last saved candle. |
| Phase 1 missing candles warning | Expected for weekends (TSLA) and exchange downtime. Log the warning and continue. |
| Phase 2 `loaded: false` on `/health` | Check `docker compose logs -f`. Most likely: missing `HUGGINGFACE_TOKEN` or first-time model download still running (give it 5 min). |
| Phase 2 Docker build fails | Confirm Docker daemon is running. If torch CPU wheel download times out, retry — the download is large. |
| Phase 3 negative OOS return | Stop. Report the full performance matrix. Do not proceed to TradingView with a losing strategy. |
| Phase 4 `<USERNAME>` not replaced | The strategy will fail to compile in Pine Editor. Tell the user to substitute their TradingView handle in the import line. |
| Any unhandled Python exception | Report the full traceback. Do not attempt to work around it — the user needs to see the real error. |
| `.env` credentials rejected by API | Stop. Report which API returned an auth error. Do not retry more than twice. |

---

## Artifact checklist — what a successful run leaves behind

```
data/raw/                                  ← Phase 1
  BTC_USD_1m.parquet
  BTC_USD_5m.parquet
  TSLA_1m.parquet
  TSLA_5m.parquet

backtest/results/                          ← Phase 3
  performance_matrix.csv
  trade_log.csv

http://localhost:8000/health               ← Phase 2 (running service, not a file)

tradingview/                               ← Phase 4 (committed to repo)
  lib_atr_mean_reversion.pine
  strategy_csp.pine
  README_TRADINGVIEW.md

ARTIFACT_MANIFEST.md                       ← generated by this runbook
```

The user's final deliverable is the two `.pine` files. The Phase 2 regime API runs in Docker on the user's machine; the Phase 1 parquets and Phase 3 backtest CSVs validate that the strategy has alpha before committing real capital.
