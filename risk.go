// Package risk implements the layered exit-decision logic for the Signal NGN
// trading engine and backtester. Zero non-stdlib dependencies.
package risk

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// Position holds the risk state for a single open position.
// Static fields are set at entry and never mutate. Mutable fields
// (PeakPrice, TrailingStop) are updated in-place by Evaluate.
type Position struct {
	// Static — set at entry
	EntryPrice  float64
	Side        string    // "long" | "short"
	StopLoss    float64   // 0 = absent
	TakeProfit  float64   // 0 = absent
	HardStop    float64   // pre-computed via ComputeHardStop
	Leverage    int
	Strategy    string
	Granularity string    // e.g. "FIVE_MINUTES", "ONE_HOUR"
	MarketType  string    // "spot" | "futures"
	OpenedAt    time.Time

	// Mutable — updated by Evaluate as trailing stop advances
	PeakPrice    float64
	TrailingStop float64
}

// ExitDecision describes why a position should be closed.
type ExitDecision struct {
	Layer      int
	Label      string
	Detail     string
	ExitReason string // "Layer N: label — detail"
}

// exitReason formats a structured exit reason string.
func exitReason(layer int, label, detail string) string {
	return fmt.Sprintf("Layer %d: %s — %s", layer, label, detail)
}

// ComputeHardStop returns the hard stop price for an entry.
//
// max_adverse_pct:
//   - Spot (leverage ≤ 1): 7% flat (no leverage scaling)
//   - Futures (leverage ≥ 2): 30% / leverage  (e.g. 2×=15%, 3×=10%, 5×=6%)
//
// The formula is applied as an adverse-move threshold from entryPrice.
func ComputeHardStop(entryPrice float64, side string, leverage int, marketType string) float64 {
	if leverage <= 0 {
		leverage = 1
	}
	var adversePct float64
	if marketType == "spot" || leverage <= 1 {
		adversePct = 0.07
	} else {
		adversePct = 0.30 / float64(leverage)
	}
	if side == "long" {
		return entryPrice * (1 - adversePct)
	}
	return entryPrice * (1 + adversePct)
}

// IsMLStrategy returns true if the strategy has the "ml_" prefix.
func IsMLStrategy(strategy string) bool {
	return strings.HasPrefix(strategy, "ml_")
}

// MaxHoldDuration returns the max hold duration for a strategy/granularity pair.
func MaxHoldDuration(strategy, granularity string) time.Duration {
	switch {
	case strings.HasPrefix(strategy, "ml_xgboost") && granularity == "FIVE_MINUTES":
		return 2*time.Hour + 30*time.Minute
	case strings.HasPrefix(strategy, "ml_transformer") && granularity == "FIVE_MINUTES":
		return 2 * time.Hour
	case strings.HasPrefix(strategy, "ml_transformer") && granularity == "ONE_HOUR":
		return 12 * time.Hour
	case !strings.HasPrefix(strategy, "ml_") && granularity == "FIVE_MINUTES":
		return 4 * time.Hour
	case !strings.HasPrefix(strategy, "ml_") && granularity == "ONE_HOUR":
		return 24 * time.Hour
	default:
		return 48 * time.Hour
	}
}

// slDistance returns the stop-loss distance for trailing stop calculations.
// Falls back to EntryPrice × 0.04 when StopLoss is zero.
func slDistance(pos *Position) float64 {
	if pos.StopLoss == 0 {
		return pos.EntryPrice * 0.04
	}
	return math.Abs(pos.EntryPrice - pos.StopLoss)
}

// granularityDuration returns the candle duration for a granularity string.
func granularityDuration(granularity string) time.Duration {
	switch granularity {
	case "FIVE_MINUTES":
		return 5 * time.Minute
	case "ONE_HOUR":
		return time.Hour
	case "FOUR_HOURS":
		return 4 * time.Hour
	case "SIX_HOURS":
		return 6 * time.Hour
	case "ONE_DAY":
		return 24 * time.Hour
	default:
		return time.Hour
	}
}

// isCoarseGranularity returns true for granularities where intrabar wicks are
// large enough to wick-fire a signal stop-loss even when the candle closes above
// the stop level. For these granularities the signal SL is checked against the
// candle close rather than the candle low (hard stop still uses low).
func isCoarseGranularity(granularity string) bool {
	switch granularity {
	case "FOUR_HOURS", "SIX_HOURS", "ONE_DAY":
		return true
	}
	return false
}

// candleCountForDuration returns the candle count for a hold duration and granularity.
func candleCountForDuration(hold time.Duration, granularity string) int {
	d := granularityDuration(granularity)
	if d == 0 {
		return 0
	}
	return int(hold / d)
}

// formatDuration formats a duration as a human-readable string (e.g. "12h4m").
func formatDuration(d time.Duration) string {
	s := d.Round(time.Minute).String()
	// Strip trailing "0s" that time.Duration.String() appends after minutes.
	s = strings.TrimSuffix(s, "0s")
	return s
}

// Evaluate checks all exit layers in priority order for the given price range.
//
// high/low are the candle's intra-bar extremes (use currentPrice for both in
// tick mode). close is the candle close (used for trailing stop advancement
// and exit check). now is the candle timestamp (used for time-based exit).
//
// Returns (decision, true) when an exit should fire. Also mutates pos.PeakPrice
// and pos.TrailingStop in-place when the trailing stop advances without firing.
// The caller is responsible for persisting the updated state.
func Evaluate(pos *Position, high, low, close float64, now time.Time) (ExitDecision, bool) {
	// Layer 2: Hard stop — checked before signal SL (hardstop is always active)
	if pos.HardStop > 0 {
		if pos.Side == "long" && low <= pos.HardStop {
			adversePct := math.Abs(low-pos.EntryPrice) / pos.EntryPrice * 100
			lev := pos.Leverage
			if lev <= 0 {
				lev = 1
			}
			detail := fmt.Sprintf("%.1f%% adverse move at %d× leverage", adversePct, lev)
			reason := exitReason(2, "hard stop", detail)
			return ExitDecision{Layer: 2, Label: "hard stop", Detail: detail, ExitReason: reason}, true
		}
		if pos.Side == "short" && high >= pos.HardStop {
			adversePct := math.Abs(high-pos.EntryPrice) / pos.EntryPrice * 100
			lev := pos.Leverage
			if lev <= 0 {
				lev = 1
			}
			detail := fmt.Sprintf("%.1f%% adverse move at %d× leverage", adversePct, lev)
			reason := exitReason(2, "hard stop", detail)
			return ExitDecision{Layer: 2, Label: "hard stop", Detail: detail, ExitReason: reason}, true
		}
	}

	// Layer 1: Signal SL
	// For coarse granularities (4H, 6H, 1D) check against the candle close to
	// avoid wick-fired false stops — the intrabar range is wide enough that the
	// wick can touch the stop level while the candle closes well above it.
	// For fine granularities (5m, 1H) the intrabar range is smaller so we check
	// the full intrabar low/high as before.
	// Note: the hard stop (Layer 2 above) always uses low/high regardless of granularity.
	if pos.StopLoss > 0 {
		slCheckPrice := low
		if isCoarseGranularity(pos.Granularity) {
			slCheckPrice = close
		}
		slCheckHigh := high
		if isCoarseGranularity(pos.Granularity) {
			slCheckHigh = close
		}
		if pos.Side == "long" && slCheckPrice <= pos.StopLoss {
			detail := fmt.Sprintf("price $%.4f hit stop $%.4f", slCheckPrice, pos.StopLoss)
			reason := exitReason(1, "signal SL", detail)
			return ExitDecision{Layer: 1, Label: "signal SL", Detail: detail, ExitReason: reason}, true
		}
		if pos.Side == "short" && slCheckHigh >= pos.StopLoss {
			detail := fmt.Sprintf("price $%.4f hit stop $%.4f", slCheckHigh, pos.StopLoss)
			reason := exitReason(1, "signal SL", detail)
			return ExitDecision{Layer: 1, Label: "signal SL", Detail: detail, ExitReason: reason}, true
		}
	}

	// Layer 4: Trailing stop (ML strategies only)
	// Update PeakPrice/TrailingStop from close, then check for breach.
	if IsMLStrategy(pos.Strategy) {
		dist := slDistance(pos)

		if pos.Side == "long" {
			// Advance peak price
			if close > pos.PeakPrice {
				pos.PeakPrice = close
			}
			profit := close - pos.EntryPrice

			if profit >= 2*dist {
				// Active trailing: trail 1× slDistance behind peak
				newTrailing := pos.PeakPrice - dist
				if newTrailing > pos.TrailingStop {
					pos.TrailingStop = newTrailing
				}
			} else if profit >= dist {
				// Breakeven: move stop to entry (only if entry is above current stop)
				if pos.EntryPrice > pos.TrailingStop {
					pos.TrailingStop = pos.EntryPrice
				}
			}

			// Check if trailing stop is breached
			if pos.TrailingStop > 0 && close <= pos.TrailingStop {
				var detail string
				if pos.TrailingStop == pos.EntryPrice {
					detail = fmt.Sprintf("breakeven triggered, stop at entry $%.4f", pos.EntryPrice)
				} else {
					detail = fmt.Sprintf("trailing at $%.4f, best price $%.4f", pos.TrailingStop, pos.PeakPrice)
				}
				reason := exitReason(4, "trailing stop", detail)
				return ExitDecision{Layer: 4, Label: "trailing stop", Detail: detail, ExitReason: reason}, true
			}

		} else { // short
			// Advance peak price (lowest price seen for short)
			if pos.PeakPrice == 0 || close < pos.PeakPrice {
				pos.PeakPrice = close
			}
			profit := pos.EntryPrice - close

			if profit >= 2*dist {
				// Active trailing: trail 1× slDistance above peak
				newTrailing := pos.PeakPrice + dist
				if pos.TrailingStop == 0 || newTrailing < pos.TrailingStop {
					pos.TrailingStop = newTrailing
				}
			} else if profit >= dist {
				// Breakeven: move stop to entry (only if entry is below current stop)
				if pos.TrailingStop == 0 || pos.EntryPrice < pos.TrailingStop {
					pos.TrailingStop = pos.EntryPrice
				}
			}

			// Check if trailing stop is breached
			if pos.TrailingStop > 0 && close >= pos.TrailingStop {
				var detail string
				if pos.TrailingStop == pos.EntryPrice {
					detail = fmt.Sprintf("breakeven triggered, stop at entry $%.4f", pos.EntryPrice)
				} else {
					detail = fmt.Sprintf("trailing at $%.4f, best price $%.4f", pos.TrailingStop, pos.PeakPrice)
				}
				reason := exitReason(4, "trailing stop", detail)
				return ExitDecision{Layer: 4, Label: "trailing stop", Detail: detail, ExitReason: reason}, true
			}
		}
	}

	// Layer 5: Time-based exit
	maxHold := MaxHoldDuration(pos.Strategy, pos.Granularity)
	held := now.Sub(pos.OpenedAt)
	if held > maxHold {
		count := candleCountForDuration(maxHold, pos.Granularity)
		heldStr := formatDuration(held)
		detail := fmt.Sprintf("%d-candle hold limit reached (held %s)", count, heldStr)
		reason := exitReason(5, "time exit", detail)
		return ExitDecision{Layer: 5, Label: "time exit", Detail: detail, ExitReason: reason}, true
	}

	// Layer 6: Signal TP (rule-based strategies only)
	if !IsMLStrategy(pos.Strategy) && pos.TakeProfit > 0 {
		if pos.Side == "long" && close >= pos.TakeProfit {
			detail := fmt.Sprintf("price $%.4f hit take-profit $%.4f", close, pos.TakeProfit)
			reason := exitReason(6, "signal TP", detail)
			return ExitDecision{Layer: 6, Label: "signal TP", Detail: detail, ExitReason: reason}, true
		}
		if pos.Side == "short" && close <= pos.TakeProfit {
			detail := fmt.Sprintf("price $%.4f hit take-profit $%.4f", close, pos.TakeProfit)
			reason := exitReason(6, "signal TP", detail)
			return ExitDecision{Layer: 6, Label: "signal TP", Detail: detail, ExitReason: reason}, true
		}
	}

	return ExitDecision{}, false
}
