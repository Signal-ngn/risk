# risk

Layered exit-decision logic for the [Signal NGN](https://signal-ngn.com) trading engine and backtester. Zero non-stdlib dependencies.

## Install

```bash
go get github.com/Signal-ngn/risk
```

## Overview

The package evaluates an open position against a prioritised set of exit layers on every price tick or candle close and returns an exit decision when any layer fires.

```
Layer 1 — Signal SL       strategy-supplied stop-loss price
Layer 2 — Hard stop       leverage-scaled circuit-breaker (always active)
Layer 4 — Trailing stop   breakeven + trailing stop for ML strategies
Layer 5 — Time exit       max hold duration per strategy/granularity
Layer 6 — Signal TP       take-profit for rule-based strategies
```

Layer 3 (conviction drop) is handled upstream by the signal engine and is not evaluated here.

## Usage

```go
pos := &risk.Position{
    EntryPrice:  50000,
    Side:        "long",
    StopLoss:    48000,
    TakeProfit:  55000,
    HardStop:    risk.ComputeHardStop(50000, "long", 2, "futures"),
    Leverage:    2,
    Strategy:    "ml_transformer",
    Granularity: "FIVE_MINUTES",
    MarketType:  "futures",
    OpenedAt:    time.Now(),
}

// Call on every tick or candle close.
decision, shouldExit := risk.Evaluate(pos, high, low, close, now)
if shouldExit {
    fmt.Println(decision.ExitReason)
    // e.g. "Layer 1: signal SL — price $47800.0000 hit stop $48000.0000"
}
```

`Evaluate` mutates `pos.PeakPrice` and `pos.TrailingStop` in-place as the trailing stop advances. Persist these fields between calls so state survives restarts.

## API

### `Evaluate(pos, high, low, close, now) (ExitDecision, bool)`

Checks all exit layers in priority order. Pass the candle's intra-bar high/low for accurate SL/TP detection, or pass `currentPrice` for all three in tick mode.

Returns `(decision, true)` when an exit fires, `({}, false)` otherwise.

### `ComputeHardStop(entryPrice, side, leverage, marketType) float64`

Computes the immutable circuit-breaker price at entry time.

| Market | Formula |
|--------|---------|
| Spot / leverage ≤ 1 | entry × 7% adverse |
| Futures (leverage ≥ 2) | entry × (30% / leverage) adverse |

### `MaxHoldDuration(strategy, granularity) time.Duration`

Returns the maximum hold duration before a time-based exit fires.

| Strategy | Granularity | Max hold |
|----------|-------------|----------|
| `ml_xgboost*` | `FIVE_MINUTES` | 2h 30m |
| `ml_transformer*` | `FIVE_MINUTES` | 2h |
| `ml_transformer*` | `ONE_HOUR` | 12h |
| rule-based | `FIVE_MINUTES` | 4h |
| rule-based | `ONE_HOUR` | 24h |
| anything else | — | 48h |

### `IsMLStrategy(strategy) bool`

Returns `true` when the strategy name has the `ml_` prefix. Used internally to gate trailing stop and TP behaviour.

## Exit decision fields

```go
type ExitDecision struct {
    Layer      int    // 1, 2, 4, 5, or 6
    Label      string // "signal SL", "hard stop", "trailing stop", "time exit", "signal TP"
    Detail     string // human-readable detail
    ExitReason string // "Layer N: label — detail"
}
```
