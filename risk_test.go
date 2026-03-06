package risk

import (
	"math"
	"strings"
	"testing"
	"time"
)

// ── helpers ───────────────────────────────────────────────────────────────────

func approx(a, b float64) bool { return math.Abs(a-b) < 0.0001 }

func basePos(side string) *Position {
	return &Position{
		EntryPrice:  1.000,
		Side:        side,
		HardStop:    ComputeHardStop(1.000, side, 2, "futures"),
		Leverage:    2,
		Strategy:    "ml_transformer",
		Granularity: "ONE_HOUR",
		MarketType:  "futures",
		OpenedAt:    time.Now().Add(-1 * time.Minute),
	}
}

// ── Task 1.9: ComputeHardStop ─────────────────────────────────────────────────

func TestComputeHardStop_Spot_Long(t *testing.T) {
	// spot / leverage=1 → 7% floor
	got := ComputeHardStop(1.000, "long", 1, "spot")
	want := 1.000 * 0.93
	if !approx(got, want) {
		t.Errorf("spot long hard stop: got %.4f, want %.4f", got, want)
	}
}

func TestComputeHardStop_Spot_Short(t *testing.T) {
	got := ComputeHardStop(1.000, "short", 1, "spot")
	want := 1.000 * 1.07
	if !approx(got, want) {
		t.Errorf("spot short hard stop: got %.4f, want %.4f", got, want)
	}
}

func TestComputeHardStop_Futures2x_Long(t *testing.T) {
	// 30% / 2 = 15%
	got := ComputeHardStop(1.000, "long", 2, "futures")
	want := 1.000 * 0.85
	if !approx(got, want) {
		t.Errorf("2x long hard stop: got %.4f, want %.4f", got, want)
	}
}

func TestComputeHardStop_Futures2x_Short(t *testing.T) {
	got := ComputeHardStop(1.000, "short", 2, "futures")
	want := 1.000 * 1.15
	if !approx(got, want) {
		t.Errorf("2x short hard stop: got %.4f, want %.4f", got, want)
	}
}

func TestComputeHardStop_Futures3x_Long(t *testing.T) {
	// 30% / 3 = 10%
	got := ComputeHardStop(1.000, "long", 3, "futures")
	want := 1.000 * 0.90
	if !approx(got, want) {
		t.Errorf("3x long hard stop: got %.4f, want %.4f", got, want)
	}
}

func TestComputeHardStop_Futures3x_Short(t *testing.T) {
	got := ComputeHardStop(1.000, "short", 3, "futures")
	want := 1.000 * 1.10
	if !approx(got, want) {
		t.Errorf("3x short hard stop: got %.4f, want %.4f", got, want)
	}
}

func TestComputeHardStop_Futures5x_Long(t *testing.T) {
	// 30% / 5 = 6% (NOT floored to 7% for futures)
	got := ComputeHardStop(1.000, "long", 5, "futures")
	want := 1.000 * 0.94 // 6% adverse → 0.94
	if !approx(got, want) {
		t.Errorf("5x long hard stop: got %.4f, want %.4f (spec table: 6%% adverse)", got, want)
	}
}

func TestComputeHardStop_Futures5x_Short(t *testing.T) {
	// 30% / 5 = 6%
	got := ComputeHardStop(1.000, "short", 5, "futures")
	want := 1.000 * 1.06
	if !approx(got, want) {
		t.Errorf("5x short hard stop: got %.4f, want %.4f", got, want)
	}
}

// ── Task 1.10: MaxHoldDuration ────────────────────────────────────────────────

func TestMaxHoldDuration_MLXGBoost5m(t *testing.T) {
	d := MaxHoldDuration("ml_xgboost", "FIVE_MINUTES")
	want := 2*time.Hour + 30*time.Minute
	if d != want {
		t.Errorf("ml_xgboost/5m: got %v, want %v", d, want)
	}
}

func TestMaxHoldDuration_MLTransformer5m(t *testing.T) {
	d := MaxHoldDuration("ml_transformer", "FIVE_MINUTES")
	want := 2 * time.Hour
	if d != want {
		t.Errorf("ml_transformer/5m: got %v, want %v", d, want)
	}
}

func TestMaxHoldDuration_MLTransformer1h(t *testing.T) {
	d := MaxHoldDuration("ml_transformer", "ONE_HOUR")
	want := 12 * time.Hour
	if d != want {
		t.Errorf("ml_transformer/1h: got %v, want %v", d, want)
	}
}

func TestMaxHoldDuration_RuleBased5m(t *testing.T) {
	d := MaxHoldDuration("rsi_divergence", "FIVE_MINUTES")
	want := 4 * time.Hour
	if d != want {
		t.Errorf("rule-based/5m: got %v, want %v", d, want)
	}
}

func TestMaxHoldDuration_RuleBased1h(t *testing.T) {
	d := MaxHoldDuration("rsi_divergence", "ONE_HOUR")
	want := 24 * time.Hour
	if d != want {
		t.Errorf("rule-based/1h: got %v, want %v", d, want)
	}
}

func TestMaxHoldDuration_Default(t *testing.T) {
	d := MaxHoldDuration("unknown_strategy", "UNKNOWN_GRAN")
	want := 48 * time.Hour
	if d != want {
		t.Errorf("default: got %v, want %v", d, want)
	}
}

func TestMaxHoldDuration_MLXGBoostVariant(t *testing.T) {
	// Strategy variant with suffix should still match prefix
	d := MaxHoldDuration("ml_xgboost_short", "FIVE_MINUTES")
	want := 2*time.Hour + 30*time.Minute
	if d != want {
		t.Errorf("ml_xgboost_short/5m: got %v, want %v", d, want)
	}
}

// ── Task 1.11: hard stop fires on Low (long) / High (short) ──────────────────

func TestEvaluate_HardStop_Long_FiresOnLow(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		HardStop:   0.850,
		Leverage:   2,
		Strategy:   "rsi",
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	// high=0.900, low=0.847 → low <= 0.850
	decision, exit := Evaluate(pos, 0.900, 0.847, 0.860, time.Now())
	if !exit {
		t.Fatal("hard stop should fire when Low <= HardStop")
	}
	if decision.Layer != 2 {
		t.Errorf("expected Layer 2, got %d", decision.Layer)
	}
	if !strings.Contains(decision.ExitReason, "Layer 2: hard stop") {
		t.Errorf("exit reason should contain 'Layer 2: hard stop', got: %q", decision.ExitReason)
	}
}

func TestEvaluate_HardStop_Long_NoFireOnCloseAlone(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		HardStop:   0.850,
		Leverage:   2,
		Strategy:   "rsi",
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	// high=0.900, low=0.860, close=0.848 (below hard stop, but LOW is not)
	// low=0.860 > hard stop=0.850 → no fire
	_, exit := Evaluate(pos, 0.900, 0.860, 0.848, time.Now())
	if exit {
		t.Fatal("hard stop should NOT fire when Low > HardStop (even if Close < HardStop)")
	}
}

func TestEvaluate_HardStop_Long_AboveNotTriggered(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		HardStop:   0.850,
		Leverage:   2,
		Strategy:   "rsi",
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	_, exit := Evaluate(pos, 0.920, 0.851, 0.900, time.Now())
	if exit {
		t.Fatal("hard stop should NOT fire when Low > HardStop")
	}
}

func TestEvaluate_HardStop_Short_FiresOnHigh(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "short",
		HardStop:   1.150,
		Leverage:   2,
		Strategy:   "rsi",
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	// high=1.153 >= 1.150
	decision, exit := Evaluate(pos, 1.153, 0.980, 1.140, time.Now())
	if !exit {
		t.Fatal("hard stop should fire when High >= HardStop (short)")
	}
	if decision.Layer != 2 {
		t.Errorf("expected Layer 2, got %d", decision.Layer)
	}
}

func TestEvaluate_HardStop_Short_NotTriggered(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "short",
		HardStop:   1.150,
		Leverage:   2,
		Strategy:   "rsi",
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	_, exit := Evaluate(pos, 1.148, 0.990, 1.100, time.Now())
	if exit {
		t.Fatal("hard stop should NOT fire when High < HardStop (short)")
	}
}

// ── Task 1.12: signal SL fires correctly ─────────────────────────────────────

func TestEvaluate_SignalSL_Long(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0.960,
		HardStop:   0.850,
		Strategy:   "rsi",
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	// low=0.958 <= SL=0.960
	decision, exit := Evaluate(pos, 0.990, 0.958, 0.975, time.Now())
	if !exit {
		t.Fatal("signal SL should fire")
	}
	if decision.Layer != 1 {
		t.Errorf("expected Layer 1 (signal SL), got %d", decision.Layer)
	}
	if !strings.Contains(decision.ExitReason, "Layer 1: signal SL") {
		t.Errorf("exit reason: %q", decision.ExitReason)
	}
}

func TestEvaluate_SignalSL_Short(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "short",
		StopLoss:   1.040,
		HardStop:   1.150,
		Strategy:   "rsi",
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	// high=1.042 >= SL=1.040
	decision, exit := Evaluate(pos, 1.042, 0.990, 1.030, time.Now())
	if !exit {
		t.Fatal("signal SL should fire for short")
	}
	if decision.Layer != 1 {
		t.Errorf("expected Layer 1, got %d", decision.Layer)
	}
}

func TestEvaluate_SignalSL_Zero_NoFire(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0, // absent
		HardStop:   0.850,
		Strategy:   "rsi",
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	_, exit := Evaluate(pos, 0.950, 0.870, 0.900, time.Now())
	if exit {
		t.Fatal("SL=0 should not fire signal SL layer (hard stop may fire if low <= HardStop)")
	}
}

// ── Task 1.13: trailing stop scenarios ───────────────────────────────────────

func TestEvaluate_TrailingStop_BreakevenActivation(t *testing.T) {
	// Long, entry=1.000, SL=0.960 → dist=0.040
	// 1×dist profit → breakeven → TrailingStop = EntryPrice = 1.000
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0.960,
		HardStop:   0.700,
		Strategy:   "ml_transformer",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	// close=1.040 → profit=0.040 = 1×dist → breakeven
	_, exit := Evaluate(pos, 1.050, 1.035, 1.040, time.Now())
	if exit {
		t.Fatal("breakeven activation should NOT cause an exit")
	}
	if !approx(pos.TrailingStop, 1.000) {
		t.Errorf("breakeven: trailing stop should be entry=1.000, got %.4f", pos.TrailingStop)
	}
}

func TestEvaluate_TrailingStop_BelowBreakeven_NotActivated(t *testing.T) {
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0.960,
		HardStop:   0.700,
		Strategy:   "ml_transformer",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	// close=1.039 → profit=0.039 < dist=0.040 → no activation
	_, exit := Evaluate(pos, 1.050, 1.035, 1.039, time.Now())
	if exit {
		t.Fatal("should not exit before 1×dist threshold")
	}
	if pos.TrailingStop != 0 {
		t.Errorf("trailing stop should not be set below threshold, got %.4f", pos.TrailingStop)
	}
}

func TestEvaluate_TrailingStop_ActiveTrailing_LongAdvances(t *testing.T) {
	// Long, entry=1.000, dist=0.040. Peak reaches 1.080 (2×dist profit).
	// Trailing stop = 1.080 - 0.040 = 1.040
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0.960,
		HardStop:   0.700,
		Strategy:   "ml_transformer",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	_, exit := Evaluate(pos, 1.090, 1.075, 1.080, time.Now())
	if exit {
		t.Fatal("should not exit when close=1.080 and trailing stop=1.040")
	}
	if !approx(pos.TrailingStop, 1.040) {
		t.Errorf("active trailing: expected 1.040, got %.4f", pos.TrailingStop)
	}
	if !approx(pos.PeakPrice, 1.080) {
		t.Errorf("peak should be 1.080, got %.4f", pos.PeakPrice)
	}
}

func TestEvaluate_TrailingStop_DoesNotRetreat(t *testing.T) {
	// Establish trailing stop at 1.060 (peak 1.100)
	pos := &Position{
		EntryPrice:   1.000,
		Side:         "long",
		StopLoss:     0.960,
		HardStop:     0.700,
		Strategy:     "ml_transformer",
		Leverage:     2,
		OpenedAt:     time.Now().Add(-1 * time.Minute),
		PeakPrice:    1.100,
		TrailingStop: 1.060,
	}
	// Price falls to 1.090 (still above trailing stop)
	_, exit := Evaluate(pos, 1.095, 1.088, 1.090, time.Now())
	if exit {
		t.Fatal("should not exit when close=1.090 > trailing stop=1.060")
	}
	// PeakPrice stays at 1.100 (not updated since 1.090 < 1.100)
	if !approx(pos.PeakPrice, 1.100) {
		t.Errorf("peak should remain 1.100, got %.4f", pos.PeakPrice)
	}
	// TrailingStop stays at 1.060 (not retreated)
	if !approx(pos.TrailingStop, 1.060) {
		t.Errorf("trailing stop should remain 1.060, got %.4f", pos.TrailingStop)
	}
}

func TestEvaluate_TrailingStop_ExitFiresOnClose(t *testing.T) {
	// Trailing stop at 1.040, close=1.038 → exit
	pos := &Position{
		EntryPrice:   1.000,
		Side:         "long",
		StopLoss:     0.960,
		HardStop:     0.700,
		Strategy:     "ml_transformer",
		Leverage:     2,
		OpenedAt:     time.Now().Add(-1 * time.Minute),
		PeakPrice:    1.080,
		TrailingStop: 1.040,
	}
	decision, exit := Evaluate(pos, 1.045, 1.035, 1.038, time.Now())
	if !exit {
		t.Fatal("trailing stop should fire when close <= TrailingStop")
	}
	if decision.Layer != 4 {
		t.Errorf("expected Layer 4 (trailing stop), got %d", decision.Layer)
	}
	if !strings.Contains(decision.ExitReason, "Layer 4: trailing stop") {
		t.Errorf("exit reason: %q", decision.ExitReason)
	}
}

func TestEvaluate_TrailingStop_NotHit(t *testing.T) {
	pos := &Position{
		EntryPrice:   1.000,
		Side:         "long",
		StopLoss:     0.960,
		HardStop:     0.700,
		Strategy:     "ml_transformer",
		Leverage:     2,
		OpenedAt:     time.Now().Add(-1 * time.Minute),
		PeakPrice:    1.080,
		TrailingStop: 1.040,
	}
	_, exit := Evaluate(pos, 1.050, 1.042, 1.041, time.Now())
	if exit {
		t.Fatal("trailing stop should NOT fire when close > TrailingStop")
	}
}

func TestEvaluate_TrailingStop_Short_Breakeven(t *testing.T) {
	// Short, entry=1.000, SL=1.040 → dist=0.040
	// close=0.960 → profit=0.040 = 1×dist → breakeven at entry
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "short",
		StopLoss:   1.040,
		HardStop:   1.300,
		Strategy:   "ml_transformer",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	_, exit := Evaluate(pos, 1.005, 0.955, 0.960, time.Now())
	if exit {
		t.Fatal("breakeven should not cause exit")
	}
	if !approx(pos.TrailingStop, 1.000) {
		t.Errorf("short breakeven: trailing stop should be 1.000, got %.4f", pos.TrailingStop)
	}
}

func TestEvaluate_TrailingStop_Short_ActiveTrailing(t *testing.T) {
	// Short, entry=1.000, dist≈0.040. Price falls to 0.915 (profit > 2×dist).
	// Trailing stop = 0.915 + 0.040 = 0.955
	// Use 0.915 to avoid floating-point edge case at exactly 2×dist boundary.
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "short",
		StopLoss:   1.040,
		HardStop:   1.300,
		Strategy:   "ml_transformer",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	_, exit := Evaluate(pos, 0.920, 0.912, 0.915, time.Now())
	if exit {
		t.Fatal("should not exit when close is above trailing stop")
	}
	dist := math.Abs(1.000 - 1.040)
	want := 0.915 + dist
	if !approx(pos.TrailingStop, want) {
		t.Errorf("short active trailing: expected %.4f, got %.4f", want, pos.TrailingStop)
	}
}

func TestEvaluate_TrailingStop_Short_ExitFires(t *testing.T) {
	// Short trailing stop at 0.960, close=0.962 → exit
	pos := &Position{
		EntryPrice:   1.000,
		Side:         "short",
		StopLoss:     1.040,
		HardStop:     1.300,
		Strategy:     "ml_transformer",
		Leverage:     2,
		OpenedAt:     time.Now().Add(-1 * time.Minute),
		PeakPrice:    0.920,
		TrailingStop: 0.960,
	}
	decision, exit := Evaluate(pos, 0.965, 0.958, 0.962, time.Now())
	if !exit {
		t.Fatal("short trailing stop should fire when close >= TrailingStop")
	}
	if decision.Layer != 4 {
		t.Errorf("expected Layer 4, got %d", decision.Layer)
	}
}

// ── Task 1.14: time exit ──────────────────────────────────────────────────────

func TestEvaluate_TimeExit_FiresAfterLimit(t *testing.T) {
	// ml_transformer / ONE_HOUR → 12h limit
	pos := &Position{
		EntryPrice:  1.000,
		Side:        "long",
		HardStop:    0.700,
		Strategy:    "ml_transformer",
		Granularity: "ONE_HOUR",
		Leverage:    2,
		OpenedAt:    time.Now().Add(-(12*time.Hour + 5*time.Minute)),
	}
	decision, exit := Evaluate(pos, 1.010, 0.995, 1.000, time.Now())
	if !exit {
		t.Fatal("time exit should fire after hold limit")
	}
	if decision.Layer != 5 {
		t.Errorf("expected Layer 5 (time exit), got %d", decision.Layer)
	}
	if !strings.Contains(decision.ExitReason, "Layer 5: time exit") {
		t.Errorf("exit reason: %q", decision.ExitReason)
	}
	if !strings.Contains(decision.Detail, "12-candle") {
		t.Errorf("detail should mention candle count: %q", decision.Detail)
	}
}

func TestEvaluate_TimeExit_NoFireBeforeLimit(t *testing.T) {
	// ml_transformer / ONE_HOUR → 12h. Held for 11h55m.
	pos := &Position{
		EntryPrice:  1.000,
		Side:        "long",
		HardStop:    0.700,
		Strategy:    "ml_transformer",
		Granularity: "ONE_HOUR",
		Leverage:    2,
		OpenedAt:    time.Now().Add(-(11*time.Hour + 55*time.Minute)),
	}
	_, exit := Evaluate(pos, 1.010, 0.995, 1.000, time.Now())
	if exit {
		t.Fatal("time exit should NOT fire before hold limit")
	}
}

func TestEvaluate_TimeExit_RuleBased1h(t *testing.T) {
	// rule-based / ONE_HOUR → 24h. Held for 23h55m → no exit.
	pos := &Position{
		EntryPrice:  1.000,
		Side:        "long",
		HardStop:    0.700,
		Strategy:    "rsi_divergence",
		Granularity: "ONE_HOUR",
		OpenedAt:    time.Now().Add(-(23*time.Hour + 55*time.Minute)),
	}
	_, exit := Evaluate(pos, 1.010, 0.995, 1.000, time.Now())
	if exit {
		t.Fatal("rule-based 1h should not exit at 23h55m")
	}
}

// ── Task 1.15: priority ordering ─────────────────────────────────────────────

func TestEvaluate_Priority_HardStopBeatsSignalSL(t *testing.T) {
	// Both hard stop and signal SL fire on same candle.
	// Hard stop = 0.850, SL = 0.920. Low = 0.840 (breaches both).
	// Layer 2 (hard stop) beats Layer 1 (signal SL).
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0.920,
		HardStop:   0.850,
		Strategy:   "rsi",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	decision, exit := Evaluate(pos, 0.960, 0.840, 0.870, time.Now())
	if !exit {
		t.Fatal("should exit")
	}
	if decision.Layer != 2 {
		t.Errorf("hard stop (Layer 2) should beat signal SL (Layer 1), got Layer %d", decision.Layer)
	}
}

func TestEvaluate_Priority_TrailingBeatsTimeExit(t *testing.T) {
	// Position held for 13h (above 12h ml_transformer/1h limit).
	// Trailing stop also fires.
	// Layer 4 (trailing stop) beats Layer 5 (time exit).
	pos := &Position{
		EntryPrice:   1.000,
		Side:         "long",
		HardStop:     0.700,
		Strategy:     "ml_transformer",
		Granularity:  "ONE_HOUR",
		Leverage:     2,
		OpenedAt:     time.Now().Add(-13 * time.Hour),
		PeakPrice:    1.080,
		TrailingStop: 1.040,
	}
	// close=1.038 → trailing stop fires (close <= 1.040)
	decision, exit := Evaluate(pos, 1.045, 1.035, 1.038, time.Now())
	if !exit {
		t.Fatal("should exit")
	}
	if decision.Layer != 4 {
		t.Errorf("trailing stop (Layer 4) should beat time exit (Layer 5), got Layer %d", decision.Layer)
	}
}

func TestEvaluate_Priority_SignalTP_SkippedForML(t *testing.T) {
	// ML strategy: TP should be skipped even if price reaches it.
	pos := &Position{
		EntryPrice:  1.000,
		Side:        "long",
		TakeProfit:  1.100,
		HardStop:    0.700,
		Strategy:    "ml_transformer",
		Granularity: "ONE_HOUR",
		Leverage:    2,
		OpenedAt:    time.Now().Add(-1 * time.Minute),
	}
	// close=1.120 >= TP=1.100 — but ML strategy skips TP
	_, exit := Evaluate(pos, 1.130, 1.110, 1.120, time.Now())
	if exit {
		t.Fatal("ML strategy should NOT exit on signal TP")
	}
}

func TestEvaluate_Priority_SignalTP_FiredForRuleBased(t *testing.T) {
	// Rule-based strategy: TP fires.
	pos := &Position{
		EntryPrice:  1.000,
		Side:        "long",
		TakeProfit:  1.100,
		HardStop:    0.700,
		Strategy:    "rsi_divergence",
		Granularity: "ONE_HOUR",
		OpenedAt:    time.Now().Add(-1 * time.Minute),
	}
	decision, exit := Evaluate(pos, 1.130, 1.110, 1.110, time.Now())
	if !exit {
		t.Fatal("rule-based strategy should exit on signal TP")
	}
	if decision.Layer != 6 {
		t.Errorf("expected Layer 6 (signal TP), got %d", decision.Layer)
	}
}

// ── Task 1.16: SL=0 fallback ──────────────────────────────────────────────────

func TestEvaluate_SLZero_HardStopStillFires(t *testing.T) {
	// SL=0 but hard stop is active
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0, // absent
		HardStop:   0.850,
		Strategy:   "rsi",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	decision, exit := Evaluate(pos, 0.900, 0.840, 0.860, time.Now())
	if !exit {
		t.Fatal("hard stop should fire even when signal SL=0")
	}
	if decision.Layer != 2 {
		t.Errorf("expected Layer 2 (hard stop), got %d", decision.Layer)
	}
}

func TestEvaluate_SLZero_TrailingStop_Uses4PctFallback(t *testing.T) {
	// SL=0 → slDistance = entryPrice × 0.04 = 0.040
	// close=1.040 → profit=0.040 = 1×dist → breakeven activated
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0, // absent → fallback dist = 0.040
		HardStop:   0.700,
		Strategy:   "ml_transformer",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	_, exit := Evaluate(pos, 1.050, 1.035, 1.040, time.Now())
	if exit {
		t.Fatal("breakeven activation should not exit immediately")
	}
	// Breakeven should be at entry price (1.000)
	if !approx(pos.TrailingStop, 1.000) {
		t.Errorf("SL=0 fallback: trailing stop (breakeven) should be 1.000, got %.4f", pos.TrailingStop)
	}
}

func TestEvaluate_SLZero_TrailingStop_FallbackActiveTrail(t *testing.T) {
	// SL=0 → dist = 0.040. Active trailing when profit >= 2×dist = 0.080.
	// close=1.090 → profit=0.090 >= 0.080. Trail = peak - dist = 1.090 - 0.040 = 1.050
	pos := &Position{
		EntryPrice: 1.000,
		Side:       "long",
		StopLoss:   0,
		HardStop:   0.700,
		Strategy:   "ml_transformer",
		Leverage:   2,
		OpenedAt:   time.Now().Add(-1 * time.Minute),
	}
	_, exit := Evaluate(pos, 1.095, 1.085, 1.090, time.Now())
	if exit {
		t.Fatal("should not exit when close is above trailing stop")
	}
	if !approx(pos.TrailingStop, 1.050) {
		t.Errorf("SL=0 active trail: expected 1.050, got %.4f", pos.TrailingStop)
	}
}

// ── helper: verify exit reason format ────────────────────────────────────────

func TestExitReasonFormat(t *testing.T) {
	reason := exitReason(2, "hard stop", "15.3% adverse move at 2× leverage")
	want := "Layer 2: hard stop — 15.3% adverse move at 2× leverage"
	if reason != want {
		t.Errorf("exit reason format: got %q, want %q", reason, want)
	}
}
