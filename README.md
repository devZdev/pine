# Hybrid Quant Trading System

Identifies high-probability price floors and volatility expansions in BTC and TSLA to optimize timing and strike selection for **selling cash-secured puts**.

The system pits a deterministic Glass Box (ATR/math-based) model against a probabilistic Black Box (SOTA Hugging Face time series) model, with rigorous walk-forward backtesting and a TradingView execution layer.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full explanation of design decisions, agent responsibilities, and the overall system map.

Looking for automated end-to-end execution? See [AGENT_RUNBOOK.md](AGENT_RUNBOOK.md) — a self-contained playbook written for an AI agent to run the full pipeline and leave TradingView-ready artifacts without human supervision.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended. Use a virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Open `.env` and substitute your keys. The file is in `.gitignore` and will never be committed.

| Variable | Where to get it | Required for | Cost |
|---|---|---|---|
| `COINBASE_API_KEY` | [coinbase.com/settings/api](https://www.coinbase.com/settings/api) | Phase 1 — BTC data | Free |
| `COINBASE_API_SECRET` | Same page | Phase 1 — BTC data | Free |
| `ALPACA_API_KEY` | [alpaca.markets](https://alpaca.markets) | Phase 1 — TSLA data | Free |
| `ALPACA_SECRET_KEY` | Same page | Phase 1 — TSLA data | Free |
| `HUGGINGFACE_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Phase 2 — Black Box engine | Free |

Phase 4 (TradingView) doesn't need an env var — the Slack webhook URL is configured directly in the TradingView alert UI, not stored in `.env`.

**Coinbase note:** API keys use the Advanced Trade format: `organizations/<org-id>/apiKeys/<key-id>`. The secret is an EC private key beginning with `-----BEGIN EC PRIVATE KEY-----`.

**Alpaca note:** Use the paper trading keys (not live). The key format is `PKXXXXXXXXXXXXXXXXXXXXXXXX`.

### 3. Run the pipeline

```bash
# Full historical backfill — BTC (Coinbase) + TSLA (Alpaca), 1m and 5m, from 2020
# Expect 20–40 min for BTC 1m (~3.3M rows across ~11K paginated requests)
python main.py --symbols BTC TSLA --timeframes 1m 5m --start 2020-01-01

# Quick smoke test — TSLA 5m, 3 months, skip slow Hurst computation
python main.py --symbols TSLA --timeframes 5m --start 2024-01-01 --end 2024-03-01 --no-hurst

# BTC only, 1m, no Hurst
python main.py --symbols BTC --timeframes 1m --start 2020-01-01 --no-hurst
```

Output lands in `data/raw/` as Snappy-compressed Parquet:
```
data/raw/
  BTC_USD_1m.parquet
  BTC_USD_5m.parquet
  TSLA_1m.parquet
  TSLA_5m.parquet
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--symbols` | `BTC TSLA` | Space-separated symbols to process |
| `--timeframes` | `1m 5m` | Space-separated timeframes |
| `--start` | `2020-01-01` | Historical start date |
| `--end` | today | Historical end date |
| `--data-dir` | `data/raw` | Output directory for Parquet files |
| `--no-hurst` | off | Skip DFA Hurst computation (much faster) |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

Interrupt at any time with `Ctrl+C` — partial progress is saved automatically.

---

## End-to-End Workflow

This section explains what you are building, why each step exists, and exactly how running a few Python scripts results in a live TradingView strategy that tells you when to sell a cash-secured put.

---

### The big picture

```
Coinbase API ─┐
               ├─► [Step 1] Data Pipeline ──► Parquet files
Alpaca API ───┘                                     │
                                                    ▼
                                      [Step 2] Backtest ──► Does the strategy have alpha?
                                                    │
                                                    ▼
                                      [Step 3] Black Box API ──► Is the market trending or mean-reverting right now?
                                                    │
                                                    ▼
                                      [Step 4] TradingView ──► Pine Script fires a webhook alert
                                                    │
                                                    ▼
                                           You sell the put
```

---

### Step 1 — Pull and clean historical data (`main.py`)

**What it does:** Connects to Coinbase (for BTC) and Alpaca (for TSLA), pulls every 1-minute and 5-minute candle going back to 2020, fills gaps, and computes five technical features on top of the raw price data: ATR, 200 SMA, Bollinger Bands, and the Hurst Exponent.

**Why it matters:** Everything downstream — the backtest, the AI model, and the live TradingView strategy — runs on these same features. Computing them once from clean data ensures every layer of the system is looking at identical numbers.

**What you get:** Four Parquet files in `data/raw/` — one per asset per timeframe. Think of these as your research database.

```bash
# One-time historical backfill (~20–40 min for BTC 1m)
python main.py --symbols BTC TSLA --timeframes 1m 5m --start 2020-01-01

# Re-run anytime to fetch only new candles since your last pull
python main.py --symbols BTC TSLA --timeframes 1m 5m
```

---

### Step 2 — Prove the strategy has alpha (`main_backtest.py`)

**What it does:** Simulates selling cash-secured puts on BTC and TSLA from 2020 to present using the Glass Box signal logic: sell a 0.20-delta put when price is near the lower Bollinger Band, above the 200 SMA, and the Hurst Exponent confirms the market is mean-reverting. Models real options P&L using Black-Scholes, applies 0.1% slippage, and sizes positions with Half-Kelly. Runs walk-forward optimization so the result is honest out-of-sample performance, not curve-fitted hindsight.

**Why it matters:** Before you commit real capital, you need evidence the signal works. This step tells you the Calmar Ratio, Sortino Ratio, Max Drawdown, and win rate — against a simple Buy & Hold benchmark. If the backtest doesn't beat the benchmark, you adjust the parameters before touching TradingView.

**What you get:** `backtest/results/performance_matrix.csv` (summary metrics) and `backtest/results/trade_log.csv` (every trade, entry/exit price, premium, P&L).

```bash
# Full walk-forward backtest
python main_backtest.py

# Fast run to spot-check results without WFO
python main_backtest.py --no-wfo --bb-pct 0.20 --hurst 0.45
```

---

### Step 3 — Run the Black Box regime classifier (Phase 2)

**What it does:** A FastAPI service running in Docker that loads Amazon Chronos-T5 (CPU-only) and exposes `GET /regime?symbol=BTC` — returning `TRENDING` or `MEAN_REVERTING` plus a confidence score and 10-bar forecast range.

**Why it matters:** The Glass Box signal tells you *where* price is (near a support band). The Black Box tells you *what the market is doing* at a macro level. Stacking both signals reduces false positives significantly. The Pine Script also runs its own R/S Hurst check on-chart, and you cross-reference its output against this API for confidence before manually executing.

**What you get:** A local HTTP service at `http://localhost:8000` that you can query any time.

```bash
# First run — builds image, downloads torch CPU + model weights (~5–10 min)
docker compose up --build

# Subsequent starts (model is cached)
docker compose up -d

# Query
curl "http://localhost:8000/regime?symbol=BTC"
```

---

### Step 4 — Trade from TradingView (Phase 4)

**What it does:** Two Pine Script v6 files load into TradingView:
1. A **library script** ([tradingview/lib_atr_mean_reversion.pine](tradingview/lib_atr_mean_reversion.pine)) — reusable ATR trailing stop, Bollinger Bands, R/S Hurst, regime label
2. A **strategy script** ([tradingview/strategy_csp.pine](tradingview/strategy_csp.pine)) — imports the library, runs the Glass Box logic, fires Slack webhook alerts

**Why it matters:** TradingView watches the live tape; when the strategy fires, you get a Slack message in seconds with the strike hint, regime, ATR, and Bollinger %B — all formatted with Slack Block Kit so it's readable at a glance. You execute the put manually in your broker.

**What you get:** Slack alerts that look like this:

```
📉 Sell Cash-Secured Put — BTCUSD

Action: SELL_PUT          Symbol: BTCUSD
Strike Hint: $94250.50    DTE: 30 days
Delta Target: 0.20        Spot: $96120.25
ATR: $1832.40             Regime: MEAN_REVERTING
Hurst: 0.41               BB %B: 0.18

Bar time: 2026-05-07 14:32 UTC | Strategy: Glass Box CSP
```

**TradingView setup** (full guide at [tradingview/README_TRADINGVIEW.md](tradingview/README_TRADINGVIEW.md)):
1. Pine Editor → paste `lib_atr_mean_reversion.pine` → **Publish Script** → **Publish to Account (Private)**
2. New Pine Editor tab → paste `strategy_csp.pine` → replace `<USERNAME>` with your TradingView handle → **Add to Chart**
3. Create a Slack incoming webhook at [api.slack.com/messaging/webhooks](https://api.slack.com/messaging/webhooks)
4. Click the alert clock → Create Alert → Condition: strategy → **Any alert() call** → paste Slack webhook URL → leave message blank
5. Pine's `alert()` populates the Block Kit JSON automatically

Requires TradingView **Pro+ tier** for webhook URLs and unlimited alerts.

---

### Recommended run order

| # | Command | Time | Gate |
|---|---|---|---|
| 1 | `pip install -r requirements.txt` | 2 min | Once |
| 2 | Fill in `.env` with your API keys | 5 min | Once |
| 3 | `python main.py --symbols BTC TSLA --timeframes 1m 5m --start 2020-01-01` | 20–40 min | Run once, then incrementally |
| 4 | `python main_backtest.py --no-wfo` | 5–10 min | Confirm metrics look reasonable |
| 5 | `python main_backtest.py` | 30–60 min | Full WFO — lock in final parameters |
| 6 | `docker compose up -d` (regime API) | 5–10 min first build | Keep running in background |
| 7 | Add Pine Scripts to TradingView, wire Slack webhook | 10 min | Set alerts, go live |

---

## Project Structure

```
pipeline/
  coinbase_ingestor.py   # Async CCXT Coinbase BTC fetcher
  alpaca_ingestor.py     # Async Alpaca TSLA fetcher (market-hours aware)
  feature_engineer.py    # ATR-14, SMA-200, Bollinger %B, Hurst DFA
  storage.py             # Parquet read/write helpers
  utils.py               # Rate limiter, backoff, gap filler, logging
backtest/
  options_math.py        # Black-Scholes pricing, realized vol, strike solver
  signal_generator.py    # Glass Box entry/exit signals
  kelly_sizer.py         # Half-Kelly position sizing
  simulator.py           # Trade-by-trade P&L, equity curve
  wfo_engine.py          # 4-fold walk-forward optimization
  performance.py         # Calmar, Sortino, Max Drawdown, benchmark
  backtester.py          # Orchestrator
  results/               # performance_matrix.csv + trade_log.csv (runtime)
regime/
  model.py               # Chronos-T5 forecaster wrapper
  classifier.py          # Regime classification (Chronos + Hurst)
  data_loader.py         # Parquet caching
  router.py              # FastAPI routes
tradingview/
  lib_atr_mean_reversion.pine    # Pine v6 library (ATR, BB, R/S Hurst)
  strategy_csp.pine              # Pine v6 strategy with Slack webhooks
  README_TRADINGVIEW.md          # TradingView setup guide
main.py                  # Data pipeline CLI
main_backtest.py         # Backtest CLI
main_regime.py           # Regime API entry point
Dockerfile               # Regime API container (CPU-only torch)
docker-compose.yml       # Regime API service
.env                     # Your credentials (never committed)
.env.example             # Credential reference (committed, no real keys)
requirements.txt
ARCHITECTURE.md          # Full system design doc
AGENT_RUNBOOK.md         # Autonomous agent execution playbook
```

---

## Pine Script Indicators (original)

`ia_mean_reversion.pine` — InvestAnswers-inspired Mean Reversion strategy overlay, visualizing multiple standard deviation bands around a configurable moving average. Used for options strategy context in TradingView.
