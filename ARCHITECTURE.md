# System Architecture

## Goal

Identify high-probability price floors and volatility expansions in **BTC** and **TSLA** to optimize the timing and strike selection for **selling cash-secured puts**.

The system is a four-agent pipeline coordinated by a Lead Quant Architect (Claude). Each agent owns one layer of the stack. No agent starts until the previous one is approved.

---

## Agent Map

```
Agent 1: Data Engineer          →  parquet files with features
Agent 2: AI Researcher          →  Black Box regime classifier (FastAPI)
Agent 3: Quant Backtester       →  walk-forward performance matrix
Agent 4: Execution Engineer     →  Pine Script v6 library + strategy + webhooks
```

---

## Phase 1 — Data Pipeline (complete)

### Data Sources

| Asset | Source | Library | Notes |
|---|---|---|---|
| BTC/USD | Coinbase Advanced Trade | `ccxt` async | ~3.3M rows at 1m from 2020 |
| TSLA | Alpaca Markets | `aiohttp` direct | Market hours only (9:30–16:00 ET) |

Both ingestors are **incremental**: on re-run they load the existing Parquet, detect the last timestamp, and fetch only the missing tail.

### Rate Limiting

A token-bucket `AsyncRateLimiter` caps Coinbase at 9 req/s and Alpaca at 3 req/s. Every network call is wrapped in `@async_retry` with full-jitter exponential backoff (7 attempts, 1s → 120s cap). A `Ctrl+C` interrupt saves partial progress before exiting.

### Gap Handling

- Gaps ≤ 5 candles: forward-fill OHLCV silently.
- Gaps > 5 candles: forward-fill and log a `WARNING` with the timestamp and gap size.
- TSLA uses a market-hours-aware gap filler that only inserts valid trading minutes — no phantom pre-market or weekend bars.

### Features Computed (zero look-ahead bias)

All features are appended to the Parquet files after ingestion.

| Column | Method | Detail |
|---|---|---|
| `atr_14` | Wilder's EWM ATR | `close.shift(1)` for prev close; `ewm(com=13, adjust=False)` |
| `sma_200` | Rolling mean | `rolling(200, min_periods=200).mean()` |
| `bb_upper` / `bb_lower` | Bollinger Bands | 20-period, 2σ, `center=False` explicit |
| `bb_pct` | Bollinger %B | `(close - lower) / (upper - lower)` |
| `hurst_dfa` | DFA via `nolds.dfa()` | 512-bar rolling window, `raw=True` NumPy array per call |

**Look-ahead bias controls:**
- ATR: `df["close"].shift(1)` — previous close never bleeds forward
- Bollinger Bands: `center=False` is set explicitly, not relied on as default
- Hurst: `rolling().apply(raw=True)` — each window only contains past bars

### Output Schema

Every Parquet file shares this column layout:

```
timestamp (index, UTC)
open, high, low, close, volume   ← raw OHLCV
is_filled                        ← bool, True if gap-filled candle
atr_14
sma_200
bb_upper, bb_lower, bb_pct
hurst_dfa                        ← NaN for first 511 bars, and if --no-hurst
```

---

## Phase 2 — Black Box Engine (complete)

Agent 2 runs **Amazon Chronos-T5-tiny** (CPU-only) as a zero-shot time series forecaster, combined with the Phase 1 Hurst exponent to classify the current market regime.

### Inference
- Model: `amazon/chronos-t5-tiny` via `chronos-forecasting` (not AutoGluon — ~1 GB lighter)
- Context: last 512 bars of `close` prices from the 5m Parquet
- Forecast: 10-step ahead with quantiles [0.1, 0.5, 0.9], `num_samples=20`
- CPU inference: ~2–4 seconds per request

### Regime Classification
Two signals combined:

| Signal | Logic |
|---|---|
| Chronos spread | `mean(q90 - q10) / atr_14 > 1.5` → TRENDING |
| Hurst exponent | `< 0.45` → MEAN_REVERTING, `> 0.55` → TRENDING |

Confidence: `0.55 + 0.30*(both_agree) + 0.15*min(1, |hurst-0.5|/0.1)`

### API Endpoints
- `GET /regime?symbol=BTC&lookback=512` — regime + confidence + forecast range
- `GET /health` — Docker healthcheck
- `GET /symbols` — lists symbols with loaded data
- `POST /refresh` — reloads Parquets from disk after a pipeline re-run

### Deployment — Docker
```bash
# First run (downloads ~200MB torch CPU + model weights ~300MB — cached after)
docker compose up --build

# Subsequent starts (uses cached model)
docker compose up

# Query
curl "http://localhost:8000/regime?symbol=BTC"
```

### Files
```
regime/
  model.py         # ChronosForecaster wrapper, ForecastResult dataclass
  classifier.py    # classify_regime(), Regime enum, RegimeResult dataclass
  data_loader.py   # DataStore (load/cache/refresh), symbol → parquet mapping
  router.py        # FastAPI routes, Pydantic response models
main_regime.py     # FastAPI app, lifespan startup sequence
Dockerfile         # python:3.11-slim, torch CPU-only, chronos-forecasting
docker-compose.yml # mounts data/ + persistent HF model cache volume
```

---

## Phase 3 — Glass Box Backtest (complete)

Agent 3 simulates **cash-secured put selling** on BTC and TSLA as a combined portfolio.

### Options Simulation Model
- **Strike**: 0.20Δ put solved via `scipy.optimize.brentq` on the Black-Scholes delta formula
- **DTE**: 30 calendar days
- **IV proxy**: 30-day rolling realized vol (annualized log returns) — causal, no look-ahead
- **Premium**: Black-Scholes put price at entry
- **P&L at close**: `premium` if `exit_price >= strike`; `premium - (strike - exit_price)` if assigned
- **Max loss**: `strike - premium` (cash-secured, no leverage)

### Position Sizing — Half-Kelly
```
p   = 0.80  (PoP = 1 - delta)
b   = premium / (strike - premium)
f   = p - (1-p)/b  →  f_half = f/2
cap = 25% of portfolio per trade
```

### Execution Constraints
- **Slippage**: 0.1% on underlying at entry and exit (propagates into strike + premium)
- **Latency**: `signal.shift(1)` — trade executes on the bar after the signal fires

### Walk-Forward Optimization
- **In-sample**: 2020–2023 | **Out-of-sample**: 2024–2026
- 4 expanding folds optimizing `bb_pct_entry` ∈ [0.10, 0.15, 0.20, 0.25] and `hurst_threshold` ∈ [0.40, 0.45, 0.50] on Calmar Ratio
- Final OOS run uses parameters from Fold 4 (most recent training window)

### Performance Matrix
Calmar Ratio, Sortino Ratio, Sharpe Ratio, Max Drawdown, CAGR, Win Rate, Avg Trade P&L — reported for IS, OOS, each WFO fold, and Buy & Hold benchmark.

### Files
```
backtest/
  options_math.py      # Black-Scholes, realized vol, strike solver (brentq)
  signal_generator.py  # Vectorized entry/exit signals, ATR stop level
  kelly_sizer.py       # Half-Kelly fraction + position sizing
  simulator.py         # Trade-by-trade P&L, equity curve construction
  wfo_engine.py        # 4-fold WFO, parameter grid search
  performance.py       # All 8 metrics + benchmark comparison
  backtester.py        # Orchestrator
  results/             # performance_matrix.csv + trade_log.csv (runtime)
main_backtest.py       # CLI entry point
```

### To run
```bash
# Full run with WFO (default)
python main_backtest.py

# Fast run without WFO
python main_backtest.py --no-wfo --bb-pct 0.20 --hurst 0.45

# Custom paths
python main_backtest.py --data-dir /path/to/parquets --output-dir /path/to/out
```

Requires Phase 1 parquets at `data/raw/BTC_USD_5m.parquet` and `data/raw/TSLA_5m.parquet`.

---

## Phase 4 — TradingView Execution Layer (planned)

Agent 4 delivers two Pine Script v6 files:

1. **`library()` script** — reusable ATR Trailing Stop and Mean Reversion band functions
2. **`strategy()` script** — imports the library, implements entry/exit logic, fires webhook alerts

Webhook alert payload (JSON):
```json
{
  "symbol": "{{ticker}}",
  "action": "SELL_PUT",
  "strike_hint": "{{close}} * 0.95",
  "expiry": "weekly",
  "atr": "{{plot_0}}",
  "regime": "MEAN_REVERTING",
  "timestamp": "{{time}}"
}
```

---

## Key Design Constraints

- **No yfinance** — Alpaca and CCXT only
- **No look-ahead bias** — enforced structurally, not by convention
- **No unvectorized loops** — all feature engineering and backtesting uses pandas/numpy/vectorbt vectorized APIs
- **No procedural spaghetti** — each file has a single responsibility; ingestors do not compute features, feature engineer does not write files

---

## Resuming After a Disconnect

If this session is interrupted, the next session should:

1. Read this file to understand system state and conventions
2. Check `data/raw/` for existing Parquet files — the pipeline is incremental and will resume from the last timestamp
3. Confirm which phase was last approved before continuing
4. Current status: **Phase 2 complete, awaiting explicit approval before Phase 4 (Pine Script)**

The master prompt lives at [master-prompt.md](master-prompt.md).
