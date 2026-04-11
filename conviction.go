package risk

import (
	"fmt"
	"math"
	"time"
)

// OHLCV represents a single candlestick with open, high, low, close, volume.
type OHLCV struct {
	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume float64
}

// PositionContext holds the context of an open position for conviction scoring.
type PositionContext struct {
	Side       string        // "long" or "short"
	EntryPrice float64       // price at position entry
	EntryATR   float64       // ATR at time of entry (from MarkEntry)
	SignalAge  time.Duration // time since last 4H TFT signal
}

// ErosionSignal describes a single active erosion signal contributing to score decay.
type ErosionSignal struct {
	Name   string  // machine-readable signal name
	Weight float64 // penalty applied (negative)
	Value  string  // human-readable current value
}

// --- Internal ring buffer (minimal, self-contained) ---

type ringBuf struct {
	data  []float64
	cap   int
	head  int
	count int
	sum   float64
}

func newRingBuf(capacity int) *ringBuf {
	return &ringBuf{data: make([]float64, capacity), cap: capacity}
}

func (rb *ringBuf) push(v float64) {
	if rb.count == rb.cap {
		rb.sum -= rb.data[rb.head]
	} else {
		rb.count++
	}
	rb.data[rb.head] = v
	rb.sum += v
	rb.head = (rb.head + 1) % rb.cap
}

func (rb *ringBuf) full() bool { return rb.count == rb.cap }
func (rb *ringBuf) len() int   { return rb.count }

func (rb *ringBuf) values() []float64 {
	result := make([]float64, rb.count)
	start := (rb.head - rb.count + rb.cap) % rb.cap
	for i := 0; i < rb.count; i++ {
		result[i] = rb.data[(start+i)%rb.cap]
	}
	return result
}

func (rb *ringBuf) max() float64 {
	if rb.count == 0 {
		return 0
	}
	start := (rb.head - rb.count + rb.cap) % rb.cap
	m := rb.data[start]
	for i := 1; i < rb.count; i++ {
		v := rb.data[(start+i)%rb.cap]
		if v > m {
			m = v
		}
	}
	return m
}

func (rb *ringBuf) min() float64 {
	if rb.count == 0 {
		return 0
	}
	start := (rb.head - rb.count + rb.cap) % rb.cap
	m := rb.data[start]
	for i := 1; i < rb.count; i++ {
		v := rb.data[(start+i)%rb.cap]
		if v < m {
			m = v
		}
	}
	return m
}

// --- Internal indicator states ---

type rsiState struct {
	period  int
	avgGain float64
	avgLoss float64
	prev    float64
	count   int
}

func newRSIState(period int) *rsiState {
	return &rsiState{period: period}
}

func (r *rsiState) update(close float64) (float64, bool) {
	r.count++
	if r.count == 1 {
		r.prev = close
		return 0, false
	}
	change := close - r.prev
	r.prev = close
	gain := math.Max(change, 0)
	loss := math.Max(-change, 0)

	if r.count <= r.period {
		r.avgGain += gain
		r.avgLoss += loss
		return 0, false
	}
	if r.count == r.period+1 {
		r.avgGain = (r.avgGain + gain) / float64(r.period)
		r.avgLoss = (r.avgLoss + loss) / float64(r.period)
	} else {
		r.avgGain = (r.avgGain*float64(r.period-1) + gain) / float64(r.period)
		r.avgLoss = (r.avgLoss*float64(r.period-1) + loss) / float64(r.period)
	}
	if r.avgLoss == 0 {
		return 100, true
	}
	rs := r.avgGain / r.avgLoss
	return 100 - (100 / (1 + rs)), true
}

type emaState struct {
	period     int
	multiplier float64
	value      float64
	count      int
	sum        float64
}

func newEMAState(period int) *emaState {
	return &emaState{period: period, multiplier: 2.0 / float64(period+1)}
}

func (e *emaState) update(v float64) (float64, bool) {
	e.count++
	if e.count <= e.period {
		e.sum += v
		if e.count == e.period {
			e.value = e.sum / float64(e.period)
			return e.value, true
		}
		return 0, false
	}
	e.value = v*e.multiplier + e.value*(1-e.multiplier)
	return e.value, true
}

type macdState struct {
	ema12  *emaState
	ema26  *emaState
	signal *emaState
}

type macdResult struct {
	line      float64
	signal    float64
	histogram float64
}

func newMACDState() *macdState {
	return &macdState{
		ema12:  newEMAState(12),
		ema26:  newEMAState(26),
		signal: newEMAState(9),
	}
}

func (m *macdState) update(close float64) (macdResult, bool) {
	ema12, v12 := m.ema12.update(close)
	ema26, v26 := m.ema26.update(close)
	if !v12 || !v26 {
		return macdResult{}, false
	}
	line := ema12 - ema26
	sig, vSig := m.signal.update(line)
	if !vSig {
		return macdResult{}, false
	}
	return macdResult{line: line, signal: sig, histogram: line - sig}, true
}

type stochState struct {
	kPeriod int
	dPeriod int
	highs   *ringBuf
	lows    *ringBuf
	kValues *ringBuf
}

type stochResult struct {
	k float64
	d float64
}

func newStochState(kPeriod, dPeriod int) *stochState {
	return &stochState{
		kPeriod: kPeriod,
		dPeriod: dPeriod,
		highs:   newRingBuf(kPeriod),
		lows:    newRingBuf(kPeriod),
		kValues: newRingBuf(dPeriod),
	}
}

func (s *stochState) update(high, low, close float64) (stochResult, bool) {
	s.highs.push(high)
	s.lows.push(low)
	if !s.highs.full() {
		return stochResult{}, false
	}
	hh := s.highs.max()
	ll := s.lows.min()
	var k float64
	if hh == ll {
		k = 50.0
	} else {
		k = (close - ll) / (hh - ll) * 100
	}
	s.kValues.push(k)
	if !s.kValues.full() {
		return stochResult{}, false
	}
	d := s.kValues.sum / float64(s.dPeriod)
	return stochResult{k: k, d: d}, true
}

type atrState struct {
	period    int
	prevClose float64
	sum       float64
	value     float64
	count     int
}

func newATRState(period int) *atrState {
	return &atrState{period: period}
}

func (a *atrState) update(high, low, close float64) (float64, bool) {
	a.count++
	var tr float64
	if a.count == 1 {
		tr = high - low
	} else {
		tr = math.Max(high-low, math.Max(math.Abs(high-a.prevClose), math.Abs(low-a.prevClose)))
	}
	a.prevClose = close

	if a.count <= a.period {
		a.sum += tr
		if a.count == a.period {
			a.value = a.sum / float64(a.period)
			return a.value, true
		}
		return 0, false
	}
	a.value = (a.value*float64(a.period-1) + tr) / float64(a.period)
	return a.value, true
}

type supertrendState struct {
	atr            *atrState
	multiplier     float64
	prevFinalUpper float64
	prevFinalLower float64
	prevClose      float64
	isBullish      bool
	valid          bool
}

type supertrendResult struct {
	value     float64
	isBullish bool
}

func newSupertrendState(atrPeriod int, multiplier float64) *supertrendState {
	return &supertrendState{
		atr:        newATRState(atrPeriod),
		multiplier: multiplier,
		isBullish:  true,
	}
}

func (s *supertrendState) update(high, low, close float64) (supertrendResult, bool) {
	atrVal, atrValid := s.atr.update(high, low, close)
	if !atrValid {
		s.prevClose = close
		return supertrendResult{}, false
	}

	mid := (high + low) / 2.0
	basicUpper := mid + s.multiplier*atrVal
	basicLower := mid - s.multiplier*atrVal

	var finalUpper, finalLower float64
	if !s.valid {
		finalUpper = basicUpper
		finalLower = basicLower
	} else {
		if s.prevClose <= s.prevFinalUpper {
			finalUpper = math.Min(basicUpper, s.prevFinalUpper)
		} else {
			finalUpper = basicUpper
		}
		if s.prevClose >= s.prevFinalLower {
			finalLower = math.Max(basicLower, s.prevFinalLower)
		} else {
			finalLower = basicLower
		}
	}

	if s.valid {
		if s.isBullish && close < finalLower {
			s.isBullish = false
		} else if !s.isBullish && close > finalUpper {
			s.isBullish = true
		}
	}

	s.prevFinalUpper = finalUpper
	s.prevFinalLower = finalLower
	s.prevClose = close
	s.valid = true

	var value float64
	if s.isBullish {
		value = finalLower
	} else {
		value = finalUpper
	}
	return supertrendResult{value: value, isBullish: s.isBullish}, true
}

type bollingerState struct {
	period int
	stdDev float64
	buffer *ringBuf
}

type bollingerResult struct {
	upper    float64
	middle   float64
	lower    float64
	percentB float64
}

func newBollingerState(period int, stddev float64) *bollingerState {
	return &bollingerState{period: period, stdDev: stddev, buffer: newRingBuf(period)}
}

func (b *bollingerState) update(close float64) (bollingerResult, bool) {
	b.buffer.push(close)
	if !b.buffer.full() {
		return bollingerResult{}, false
	}
	middle := b.buffer.sum / float64(b.period)
	vals := b.buffer.values()
	var sumSqDiff float64
	for _, v := range vals {
		diff := v - middle
		sumSqDiff += diff * diff
	}
	sd := math.Sqrt(sumSqDiff / float64(b.period))
	upper := middle + b.stdDev*sd
	lower := middle - b.stdDev*sd

	var pctB float64
	bw := upper - lower
	if bw > 0 {
		pctB = (close - lower) / bw
	}
	return bollingerResult{upper: upper, middle: middle, lower: lower, percentB: pctB}, true
}

// --- ConvictionScorer ---

// convictionWarmup is the number of candles needed before all indicators are valid.
const convictionWarmup = 26

// Erosion signal weights (as penalties).
const (
	weightMACDHistDecline   = 0.15
	weightRSICross          = 0.10
	weightStochCross        = 0.10
	weightSupertrendFlip    = 0.25
	weightATRExpansion      = 0.15
	weightBollingerReject   = 0.10
	weightAdverseDrawdown   = 0.10
	weightSignalStaleness   = 0.05
)

// ConvictionScorer evaluates conviction health for an open position using 15M candles.
// It maintains its own rolling indicator state and outputs a 0.0-1.0 health score.
type ConvictionScorer struct {
	rsi        *rsiState
	macd       *macdState
	stoch      *stochState
	atr        *atrState
	supertrend *supertrendState
	bollinger  *bollingerState

	// MACD histogram history for decline detection
	histBuf *ringBuf

	// Tracking state
	count    int
	entryATR float64

	// Latest indicator values
	lastRSI            float64
	lastRSIValid       bool
	lastMACD           macdResult
	lastMACDValid      bool
	lastStoch          stochResult
	lastStochValid     bool
	lastATR            float64
	lastATRValid       bool
	lastSupertrend     supertrendResult
	lastSupertrendValid bool
	lastBollinger      bollingerResult
	lastBollingerValid bool
	lastClose          float64
}

// NewConvictionScorer creates a fresh ConvictionScorer with standard indicator parameters.
func NewConvictionScorer() *ConvictionScorer {
	return &ConvictionScorer{
		rsi:        newRSIState(14),
		macd:       newMACDState(),
		stoch:      newStochState(14, 3),
		atr:        newATRState(14),
		supertrend: newSupertrendState(10, 3.0),
		bollinger:  newBollingerState(20, 2.0),
		histBuf:    newRingBuf(4), // track last 4 histogram values for 3-bar decline
	}
}

// Update feeds a single 15M candle and updates all indicator state.
func (cs *ConvictionScorer) Update(candle OHLCV) {
	cs.count++
	cs.lastClose = candle.Close

	// RSI
	rsi, rsiValid := cs.rsi.update(candle.Close)
	if rsiValid {
		cs.lastRSI = rsi
		cs.lastRSIValid = true
	}

	// MACD
	macd, macdValid := cs.macd.update(candle.Close)
	if macdValid {
		cs.lastMACD = macd
		cs.lastMACDValid = true
		cs.histBuf.push(macd.histogram)
	}

	// Stochastic
	stoch, stochValid := cs.stoch.update(candle.High, candle.Low, candle.Close)
	if stochValid {
		cs.lastStoch = stoch
		cs.lastStochValid = true
	}

	// ATR
	atrVal, atrValid := cs.atr.update(candle.High, candle.Low, candle.Close)
	if atrValid {
		cs.lastATR = atrVal
		cs.lastATRValid = true
	}

	// Supertrend
	st, stValid := cs.supertrend.update(candle.High, candle.Low, candle.Close)
	if stValid {
		cs.lastSupertrend = st
		cs.lastSupertrendValid = true
	}

	// Bollinger
	boll, bollValid := cs.bollinger.update(candle.Close)
	if bollValid {
		cs.lastBollinger = boll
		cs.lastBollingerValid = true
	}
}

// IsWarm returns true once enough candles have been received for all indicators to be valid.
func (cs *ConvictionScorer) IsWarm() bool {
	return cs.count >= convictionWarmup
}

// MarkEntry captures the current ATR as entry ATR and returns it.
// Returns 0 if the scorer is not yet warm.
func (cs *ConvictionScorer) MarkEntry() float64 {
	if !cs.lastATRValid {
		cs.entryATR = 0
		return 0
	}
	cs.entryATR = cs.lastATR
	return cs.entryATR
}

// Score evaluates conviction health for the given position context.
// Returns (score, activeSignals). Score is 0.0-1.0 where 1.0 = full conviction.
func (cs *ConvictionScorer) Score(pos PositionContext) (float64, []ErosionSignal) {
	if !cs.IsWarm() {
		return 1.0, nil
	}

	var signals []ErosionSignal

	// 1. MACD histogram decline (3+ consecutive bars)
	if sig, ok := cs.checkMACDDecline(pos.Side); ok {
		signals = append(signals, sig)
	}

	// 2. RSI cross
	if sig, ok := cs.checkRSICross(pos.Side); ok {
		signals = append(signals, sig)
	}

	// 3. Stochastic cross
	if sig, ok := cs.checkStochCross(pos.Side); ok {
		signals = append(signals, sig)
	}

	// 4. Supertrend flip
	if sig, ok := cs.checkSupertrendFlip(pos.Side); ok {
		signals = append(signals, sig)
	}

	// 5. ATR expansion + adverse move
	if sig, ok := cs.checkATRExpansion(pos); ok {
		signals = append(signals, sig)
	}

	// 6. Bollinger rejection
	if sig, ok := cs.checkBollingerReject(pos.Side); ok {
		signals = append(signals, sig)
	}

	// 7. Adverse drawdown > 2%
	if sig, ok := cs.checkAdverseDrawdown(pos); ok {
		signals = append(signals, sig)
	}

	// 8. Signal staleness
	if sig, ok := cs.checkStaleness(pos.SignalAge); ok {
		signals = append(signals, sig)
	}

	// Compute score: 1.0 + sum(penalties), clamped to [0, 1]
	score := 1.0
	for _, s := range signals {
		score += s.Weight
	}
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score, signals
}

// --- Individual erosion signal detectors ---

func (cs *ConvictionScorer) checkMACDDecline(side string) (ErosionSignal, bool) {
	if !cs.lastMACDValid || cs.histBuf.len() < 4 {
		return ErosionSignal{}, false
	}

	vals := cs.histBuf.values()
	// Need at least 4 values to detect 3 consecutive declines
	n := len(vals)
	if n < 4 {
		return ErosionSignal{}, false
	}

	// Check last 3 transitions (4 values)
	declining := true
	rising := true
	for i := n - 3; i < n; i++ {
		if vals[i] >= vals[i-1] {
			declining = false
		}
		if vals[i] <= vals[i-1] {
			rising = false
		}
	}

	if side == "long" && declining {
		return ErosionSignal{
			Name:   "macd_histogram_decline",
			Weight: -weightMACDHistDecline,
			Value:  formatFloat(cs.lastMACD.histogram),
		}, true
	}
	if side == "short" && rising {
		return ErosionSignal{
			Name:   "macd_histogram_decline",
			Weight: -weightMACDHistDecline,
			Value:  formatFloat(cs.lastMACD.histogram),
		}, true
	}

	return ErosionSignal{}, false
}

func (cs *ConvictionScorer) checkRSICross(side string) (ErosionSignal, bool) {
	if !cs.lastRSIValid {
		return ErosionSignal{}, false
	}

	if side == "long" && cs.lastRSI < 50 {
		return ErosionSignal{
			Name:   "rsi_cross",
			Weight: -weightRSICross,
			Value:  formatFloat(cs.lastRSI),
		}, true
	}
	if side == "short" && cs.lastRSI > 50 {
		return ErosionSignal{
			Name:   "rsi_cross",
			Weight: -weightRSICross,
			Value:  formatFloat(cs.lastRSI),
		}, true
	}

	return ErosionSignal{}, false
}

func (cs *ConvictionScorer) checkStochCross(side string) (ErosionSignal, bool) {
	if !cs.lastStochValid {
		return ErosionSignal{}, false
	}

	if side == "long" && cs.lastStoch.k < cs.lastStoch.d {
		return ErosionSignal{
			Name:   "stochastic_cross",
			Weight: -weightStochCross,
			Value:  formatFloat(cs.lastStoch.k) + "/" + formatFloat(cs.lastStoch.d),
		}, true
	}
	if side == "short" && cs.lastStoch.k > cs.lastStoch.d {
		return ErosionSignal{
			Name:   "stochastic_cross",
			Weight: -weightStochCross,
			Value:  formatFloat(cs.lastStoch.k) + "/" + formatFloat(cs.lastStoch.d),
		}, true
	}

	return ErosionSignal{}, false
}

func (cs *ConvictionScorer) checkSupertrendFlip(side string) (ErosionSignal, bool) {
	if !cs.lastSupertrendValid {
		return ErosionSignal{}, false
	}

	// For long: bearish when supertrend value > close (price below the upper band)
	// For short: bullish when supertrend value < close (price above the lower band)
	if side == "long" && !cs.lastSupertrend.isBullish {
		return ErosionSignal{
			Name:   "supertrend_flip",
			Weight: -weightSupertrendFlip,
			Value:  formatFloat(cs.lastSupertrend.value),
		}, true
	}
	if side == "short" && cs.lastSupertrend.isBullish {
		return ErosionSignal{
			Name:   "supertrend_flip",
			Weight: -weightSupertrendFlip,
			Value:  formatFloat(cs.lastSupertrend.value),
		}, true
	}

	return ErosionSignal{}, false
}

func (cs *ConvictionScorer) checkATRExpansion(pos PositionContext) (ErosionSignal, bool) {
	if !cs.lastATRValid || pos.EntryATR <= 0 {
		return ErosionSignal{}, false
	}

	ratio := cs.lastATR / pos.EntryATR
	if ratio < 1.5 {
		return ErosionSignal{}, false
	}

	// Also check adverse move
	if pos.Side == "long" && cs.lastClose >= pos.EntryPrice {
		return ErosionSignal{}, false
	}
	if pos.Side == "short" && cs.lastClose <= pos.EntryPrice {
		return ErosionSignal{}, false
	}

	return ErosionSignal{
		Name:   "atr_expansion",
		Weight: -weightATRExpansion,
		Value:  formatFloat(ratio) + "x entry ATR",
	}, true
}

func (cs *ConvictionScorer) checkBollingerReject(side string) (ErosionSignal, bool) {
	if !cs.lastBollingerValid {
		return ErosionSignal{}, false
	}

	if side == "long" && cs.lastBollinger.percentB < 0.5 {
		return ErosionSignal{
			Name:   "bollinger_rejection",
			Weight: -weightBollingerReject,
			Value:  "%B=" + formatFloat(cs.lastBollinger.percentB),
		}, true
	}
	if side == "short" && cs.lastBollinger.percentB > 0.5 {
		return ErosionSignal{
			Name:   "bollinger_rejection",
			Weight: -weightBollingerReject,
			Value:  "%B=" + formatFloat(cs.lastBollinger.percentB),
		}, true
	}

	return ErosionSignal{}, false
}

func (cs *ConvictionScorer) checkAdverseDrawdown(pos PositionContext) (ErosionSignal, bool) {
	if pos.EntryPrice <= 0 {
		return ErosionSignal{}, false
	}

	var drawdown float64
	if pos.Side == "long" {
		drawdown = (pos.EntryPrice - cs.lastClose) / pos.EntryPrice
	} else {
		drawdown = (cs.lastClose - pos.EntryPrice) / pos.EntryPrice
	}

	if drawdown > 0.02 {
		return ErosionSignal{
			Name:   "adverse_drawdown",
			Weight: -weightAdverseDrawdown,
			Value:  formatFloat(drawdown*100) + "%",
		}, true
	}

	return ErosionSignal{}, false
}

func (cs *ConvictionScorer) checkStaleness(signalAge time.Duration) (ErosionSignal, bool) {
	if signalAge <= 0 {
		return ErosionSignal{}, false
	}

	fourHours := 4 * time.Hour
	ratio := float64(signalAge) / float64(fourHours)
	if ratio > 1 {
		ratio = 1
	}

	penalty := weightSignalStaleness * ratio
	if penalty < 1e-10 {
		return ErosionSignal{}, false
	}

	return ErosionSignal{
		Name:   "signal_staleness",
		Weight: -penalty,
		Value:  signalAge.Round(time.Minute).String(),
	}, true
}

// --- Accessor methods for testing ---

// RSI returns the current RSI value and validity.
func (cs *ConvictionScorer) RSI() (float64, bool) {
	return cs.lastRSI, cs.lastRSIValid
}

// MACD returns the current MACD line, signal, histogram, and validity.
func (cs *ConvictionScorer) MACD() (line, signal, histogram float64, valid bool) {
	return cs.lastMACD.line, cs.lastMACD.signal, cs.lastMACD.histogram, cs.lastMACDValid
}

// Stochastic returns the current %K, %D, and validity.
func (cs *ConvictionScorer) Stochastic() (k, d float64, valid bool) {
	return cs.lastStoch.k, cs.lastStoch.d, cs.lastStochValid
}

// ATR returns the current ATR value and validity.
func (cs *ConvictionScorer) ATR() (float64, bool) {
	return cs.lastATR, cs.lastATRValid
}

// Supertrend returns the current Supertrend value, direction, and validity.
func (cs *ConvictionScorer) Supertrend() (value float64, isBullish, valid bool) {
	return cs.lastSupertrend.value, cs.lastSupertrend.isBullish, cs.lastSupertrendValid
}

// Bollinger returns upper, middle, lower, %B, and validity.
func (cs *ConvictionScorer) Bollinger() (upper, middle, lower, percentB float64, valid bool) {
	r := cs.lastBollinger
	return r.upper, r.middle, r.lower, r.percentB, cs.lastBollingerValid
}

// CandleCount returns the number of candles fed so far.
func (cs *ConvictionScorer) CandleCount() int {
	return cs.count
}

// formatFloat formats a float to a reasonable precision string.
func formatFloat(v float64) string {
	if v == 0 {
		return "0"
	}
	abs := math.Abs(v)
	var s string
	switch {
	case abs >= 1000:
		s = fmt.Sprintf("%.1f", v)
	case abs >= 1:
		s = fmt.Sprintf("%.2f", v)
	default:
		s = fmt.Sprintf("%.4f", v)
	}
	// Trim trailing zeros after decimal point
	for len(s) > 1 && s[len(s)-1] == '0' && s[len(s)-2] != '.' {
		s = s[:len(s)-1]
	}
	return s
}
