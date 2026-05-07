# Glass Box CSP — TradingView Setup

Two Pine Script v6 files form the live execution layer for the Glass Box cash-secured put strategy:

- `lib_atr_mean_reversion.pine` — reusable library (Bollinger, ATR trailing stop, R/S Hurst, regime label)
- `strategy_csp.pine` — strategy that imports the library, computes signals, and fires Slack webhooks

The Python research stack (Phases 1–3) is upstream of this. TradingView is the live trigger surface only.

---

## 1. Publish the library

Libraries must be published before any strategy can `import` them.

1. Open TradingView → **Pine Editor**.
2. Paste the contents of `lib_atr_mean_reversion.pine`.
3. Click **Save** and give it the name `AtrMeanReversion`.
4. Click **Publish Script** → choose **Publish to Account (Private)**.
5. Confirm. The library now lives at `<your-username>/AtrMeanReversion/1` (the trailing `1` is the version number — it bumps with every republish).

Take note of your TradingView username — you need it in step 2.

## 2. Import the library into the strategy

1. In Pine Editor open a new tab and paste `strategy_csp.pine`.
2. Find this line near the top:
   ```pinescript
   import <USERNAME>/AtrMeanReversion/1 as amr
   ```
3. Replace `<USERNAME>` with your TradingView handle, e.g.:
   ```pinescript
   import johndoe/AtrMeanReversion/1 as amr
   ```
4. Click **Save**, name it `Glass Box CSP — BTC/TSLA`, then **Add to Chart**.

If you republish the library after edits, increment the version (`/2`, `/3`, …) in the import.

## 3. Add the strategy to a chart

- **Symbols**: `BINANCE:BTCUSDT` (or `COINBASE:BTCUSD`) and `NASDAQ:TSLA`.
- **Timeframe**: 5-minute, matching the offline backtest.
- **Chart type**: regular candles (not Heikin Ashi — strategies on HA charts give misleading fills).

You should see Bollinger bands, the 200 SMA, an ATR trailing stop (rendered only when in a position), a green background tint while in position, and a Hurst/regime HUD in the bottom-right.

## 4. Set up the Slack webhook

### a. Create the Slack incoming webhook

1. Go to <https://api.slack.com/messaging/webhooks>.
2. Create a Slack app (or reuse one), enable **Incoming Webhooks**, add a webhook to the target channel.
3. Copy the webhook URL — it looks like `https://hooks.slack.com/services/T000/B000/xxxx`.

### b. Wire the TradingView alert

1. With the strategy loaded on a chart, click the alert clock icon → **Create Alert**.
2. **Condition**: select your strategy → choose **Any alert() function call** (this captures both entry and exit).
3. **Options**: leave as default (`Once Per Bar Close` — the strategy already requests this freq).
4. **Notifications → Webhook URL**: paste the Slack webhook URL.
5. **Message**: leave blank. Pine's `alert()` already supplies the JSON Block Kit body; TradingView forwards it verbatim.
6. Save.

Repeat for each symbol you want to monitor (one alert per chart).

## 5. Required TradingView plan

- **Pro+** is the minimum tier that supports webhook URLs and gives enough concurrent alerts to run BTC + TSLA simultaneously.
- **Premium** is recommended if you also intend to monitor 1-minute charts or stack multiple strategies.

## 6. Verify the alert flow

Before trusting the live pipeline:

1. **Slack-side test**: from a terminal, post a manual JSON payload to confirm the channel renders Block Kit:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"hello from glass box"}' \
        https://hooks.slack.com/services/T000/B000/xxxx
   ```
2. **TradingView-side test**: temporarily lower `bb_pct_entry` to a value that's currently true (e.g. `0.95`) and `hurst_threshold` to `0.99`. Save the strategy, recreate the alert, and wait one bar close. Reset the inputs once you've confirmed Slack received the message.
3. **Order-fill check**: open the **Strategy Tester** tab → **List of Trades** to confirm entries/exits match the alert pings.

## 7. Customisation

| Input | Tune when… |
|---|---|
| `bb_length`, `bb_mult` | Underlying has shifted volatility regime — wider bands for crypto, tighter for equities. |
| `bb_pct_entry` | You want stricter (lower) or looser (higher) oversold gating. |
| `bb_pct_exit` | Earlier vs later mean-reversion exits. |
| `sma_length` | Long-trend filter horizon — shorten for faster regime adaptation. |
| `atr_length`, `atr_mult` | Volatility-stop tightness. Wider mult during high realised vol. |
| `hurst_length` | Window for R/S Hurst. Below 50 the estimate is noisy; above 200 it lags. |
| `hurst_threshold` | Stricter mean-reversion gating (lower) vs looser (higher). |
| `delta_target`, `dte` | Adjust the strike-hint and payload — does not change Pine's underlying P&L. |

The Python research stack should remain the source of truth for parameter selection; this strategy is the live trigger, not the optimiser.
