# Trading Indicators

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Indicators: 65+](https://img.shields.io/badge/indicators-65+-green.svg)](#supported-indicators)
[![TA-Lib](https://img.shields.io/badge/powered%20by-TA--Lib-orange.svg)](https://ta-lib.org/)

**Technical indicators library with automatic frame synchronization** built on top of [trading-frame](https://github.com/Morgiver/trading-frame).

**65+ professional-grade technical indicators** including momentum, trend, volatility, volume, cycle, price, and statistical analysis tools, all with automatic synchronization and event-driven updates.

## Overview

`trading-indicators` provides a powerful and simple way to add technical indicators to your trading frames. Indicators automatically synchronize with frame periods through event-driven architecture, eliminating manual calculation management.

## Features

- 🔄 **Automatic synchronization**: Indicators update automatically when frames receive new candles
- 📊 **Period-by-period calculation**: Efficient computation, no full recalculation
- 📈 **TA-Lib powered**: Leverages industry-standard TA-Lib for calculations
- 🎯 **Simple API**: Create indicator, bind to frame, access values with attributes
- 💾 **Dynamic storage**: Uses `_data` dict pattern like `trading-frame` Period
- 🔔 **Event-driven**: Built on the same event system as TimeFrame
- 🔗 **Composite indicators**: Indicators can depend on other indicators

## Supported Indicators

### Momentum & Oscillators (30 indicators)
- **ADX** (Average Directional Movement Index) - Trend strength measurement
- **ADXR** (ADX Rating) - Smoothed trend strength
- **APO** (Absolute Price Oscillator) - Absolute difference between EMAs
- **AROON** (Aroon Up/Down) - Trend change detection
- **AROONOSC** (Aroon Oscillator) - Trend direction oscillator
- **BOP** (Balance of Power) - Buyer/seller strength
- **CCI** (Commodity Channel Index) - Cyclical trend detection
- **CMO** (Chande Momentum Oscillator) - Momentum with sum of gains/losses
- **DX** (Directional Movement Index) - Raw directional movement
- **MACD** (Moving Average Convergence Divergence) - Trend following momentum
- **MACDEXT** (MACD Extended) - MACD with customizable MA types
- **MACDFIX** (MACD Fix) - Fixed 12/26 MACD
- **MFI** (Money Flow Index) - Volume-weighted RSI
- **MINUS_DI** (Minus Directional Indicator) - Downward movement indicator
- **MINUS_DM** (Minus Directional Movement) - Downward directional movement
- **MOM** (Momentum) - Rate of change indicator
- **PLUS_DI** (Plus Directional Indicator) - Upward movement indicator
- **PLUS_DM** (Plus Directional Movement) - Upward directional movement
- **PPO** (Percentage Price Oscillator) - Percentage difference between EMAs
- **ROC** (Rate of Change) - ((price/prevPrice)-1)*100
- **ROCP** (Rate of Change Percentage) - (price-prevPrice)/prevPrice
- **ROCR** (Rate of Change Ratio) - price/prevPrice
- **ROCR100** (ROC Ratio 100) - (price/prevPrice)*100
- **RSI** (Relative Strength Index) - Overbought/oversold detection
- **STOCH** (Stochastic) - %K and %D oscillator
- **STOCHF** (Stochastic Fast) - Fast stochastic oscillator
- **STOCHRSI** (Stochastic RSI) - Stochastic applied to RSI
- **TRIX** (Triple EMA ROC) - 1-day ROC of triple smooth EMA
- **ULTOSC** (Ultimate Oscillator) - Multi-period oscillator
- **WILLR** (Williams %R) - Momentum indicator (-100 to 0)

### Trend
- **SMA** (Simple Moving Average) - Basic trend identification
- **EMA** (Exponential Moving Average) - Responsive trend following
- **DEMA** (Double Exponential Moving Average) - Reduced lag trend following
- **TEMA** (Triple Exponential Moving Average) - Ultra-responsive moving average
- **WMA** (Weighted Moving Average) - Linear weighted moving average
- **TRIMA** (Triangular Moving Average) - Double-smoothed moving average
- **KAMA** (Kaufman Adaptive Moving Average) - Adaptive noise-filtering MA
- **MAMA** (MESA Adaptive Moving Average) - Cycle-adaptive MA with FAMA
- **T3** (Triple Exponential Moving Average T3) - Tillson's smooth responsive MA
- **HT_TRENDLINE** (Hilbert Transform Trendline) - Instantaneous trendline
- **MA** (Moving Average) - Generic MA with multiple types (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/MAMA/T3)
- **MAVP** (Moving Average Variable Period) - Dynamic period moving average
- **MIDPOINT** (MidPoint over period) - Midpoint of price range
- **MIDPRICE** (Midpoint Price) - Midpoint of high/low range
- **SAR** (Parabolic SAR) - Stop and Reverse indicator
- **SAREXT** (Parabolic SAR Extended) - Enhanced SAR with separate long/short parameters
- **Bollinger Bands** (BBANDS) - Volatility and price level analysis
- **Pivot Points** - Swing High/Low detection with alternation rule
- **FVG** (Fair Value Gap) - ICT/SMC gap detection

### Volatility (1 indicator)
- **ATR** (Average True Range) - Volatility measurement

### Volume (3 indicators)
- **AD** (Chaikin A/D Line) - Accumulation/Distribution line
- **ADOSC** (Chaikin A/D Oscillator) - A/D momentum oscillator
- **OBV** (On Balance Volume) - Cumulative volume-based indicator

### Cycle (5 indicators - Hilbert Transform)
- **HT_DCPERIOD** (Dominant Cycle Period) - Identifies the dominant cycle period (number of bars in current cycle)
- **HT_DCPHASE** (Dominant Cycle Phase) - Current phase angle of the dominant cycle (degrees)
- **HT_PHASOR** (Phasor Components) - InPhase and Quadrature components for cycle analysis
- **HT_SINE** (Sine Wave) - Sine and LeadSine waves for cycle prediction and timing
- **HT_TRENDMODE** (Trend vs Cycle Mode) - Binary indicator: 1 = trending market, 0 = cycling market

### Price (4 indicators)
- **AVGPRICE** (Average Price) - (Open + High + Low + Close) / 4
- **MEDPRICE** (Median Price) - (High + Low) / 2
- **TYPPRICE** (Typical Price) - (High + Low + Close) / 3
- **WCLPRICE** (Weighted Close Price) - (High + Low + 2*Close) / 4

### Statistical (9 indicators)
- **BETA** (Beta Coefficient) - Measures relative volatility between two assets
- **CORREL** (Pearson's Correlation) - Linear correlation coefficient between two series (-1 to 1)
- **LINEARREG** (Linear Regression) - Linear regression line fitted to price data
- **LINEARREG_ANGLE** (Linear Regression Angle) - Angle of regression line in degrees
- **LINEARREG_INTERCEPT** (Linear Regression Intercept) - Y-intercept of regression line
- **LINEARREG_SLOPE** (Linear Regression Slope) - Slope (rate of change) of regression line
- **STDDEV** (Standard Deviation) - Volatility and dispersion measurement
- **TSF** (Time Series Forecast) - Forecasted value using linear regression
- **VAR** (Variance) - Statistical variance (square of standard deviation)

## Installation

```bash
# Install from source
git clone https://github.com/Morgiver/trading-indicators.git
cd trading-indicators
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy
- TA-Lib

## Quick Start

```python
from trading_frame import TimeFrame, Candle
from trading_indicators import (
    RSI, SMA, EMA, MACD, BollingerBands, ATR,
    ADX, STOCH, CCI, MFI, AROON
)

# Create a frame
frame = TimeFrame('5T', max_periods=100)

# Create indicators (automatically bind to frame)
rsi = RSI(frame=frame, length=14, column_name='RSI_14')
sma20 = SMA(frame=frame, period=20, column_name='SMA_20')
ema20 = EMA(frame=frame, period=20, column_name='EMA_20')
macd = MACD(frame=frame, fast=12, slow=26, signal=9)
bb = BollingerBands(frame=frame, period=20)
atr = ATR(frame=frame, length=14, column_name='ATR_14')
adx = ADX(frame=frame, length=14, column_name='ADX_14')
stoch = STOCH(frame=frame, fastk_period=5, slowk_period=3, slowd_period=3)
cci = CCI(frame=frame, length=14, column_name='CCI_14')
mfi = MFI(frame=frame, length=14, column_name='MFI_14')
aroon = AROON(frame=frame, length=14)

# Feed candles - all indicators update automatically
for candle in candles:
    frame.feed(candle)

# Access indicator values with named attributes
print(f"RSI: {rsi.periods[-1].RSI_14}")
print(f"SMA: {sma20.periods[-1].SMA_20}")
print(f"EMA: {ema20.periods[-1].EMA_20}")
print(f"MACD Line: {macd.periods[-1].MACD_LINE}")
print(f"BB Upper: {bb.periods[-1].BB_UPPER}")
print(f"ATR: {atr.periods[-1].ATR_14}")
print(f"ADX: {adx.periods[-1].ADX_14}")
print(f"Stochastic %K: {stoch.periods[-1].STOCH_K}")
print(f"CCI: {cci.periods[-1].CCI_14}")
print(f"MFI: {mfi.periods[-1].MFI_14}")
print(f"Aroon Up: {aroon.periods[-1].AROON_UP}")
```

## Usage Examples

### RSI (Relative Strength Index)

```python
from trading_indicators import RSI

# Create RSI indicator
rsi = RSI(frame=frame, length=14, column_name='RSI_14')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_rsi = rsi.get_latest()
print(f"Current RSI: {latest_rsi}")

# Check conditions
if rsi.is_overbought(threshold=70):
    print("RSI indicates overbought condition")

if rsi.is_oversold(threshold=30):
    print("RSI indicates oversold condition")

# Export to NumPy
rsi_array = rsi.to_numpy()

# Export normalized for ML
rsi_normalized = rsi.to_normalize()  # [0, 1] range
```

### SMA (Simple Moving Average)

```python
from trading_indicators import SMA

# Create multiple SMAs
sma20 = SMA(frame=frame, period=20, column_name='SMA_20')
sma50 = SMA(frame=frame, period=50, column_name='SMA_50')
sma200 = SMA(frame=frame, period=200, column_name='SMA_200')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Detect golden cross
if sma20.get_latest() > sma50.get_latest():
    print("Golden Cross detected (bullish)")

# Access via periods
for period in sma20.periods:
    print(f"Date: {period.open_date}, SMA20: {period.SMA_20}")
```

### EMA (Exponential Moving Average)

```python
from trading_indicators import EMA

# Create EMA indicator (more responsive than SMA)
ema20 = EMA(frame=frame, period=20, column_name='EMA_20', price_field='close')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_ema = ema20.get_latest()
print(f"Current EMA: {latest_ema}")

# Export to NumPy
ema_array = ema20.to_numpy()
```

### MACD (Moving Average Convergence Divergence)

```python
from trading_indicators import MACD

# Create MACD indicator
macd = MACD(
    frame=frame,
    fast=12,
    slow=26,
    signal=9,
    column_names=['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST']
)

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest = macd.get_latest()
print(f"MACD Line: {latest['MACD_LINE']}")
print(f"Signal: {latest['MACD_SIGNAL']}")
print(f"Histogram: {latest['MACD_HIST']}")

# Detect crossovers
if macd.is_bullish_crossover():
    print("Bullish MACD crossover")

if macd.is_bearish_crossover():
    print("Bearish MACD crossover")

# Export to NumPy (returns dict)
macd_arrays = macd.to_numpy()
print(macd_arrays['MACD_LINE'])
print(macd_arrays['MACD_SIGNAL'])
print(macd_arrays['MACD_HIST'])
```

### Bollinger Bands

```python
from trading_indicators import BollingerBands

# Create Bollinger Bands
bb = BollingerBands(
    frame=frame,
    period=20,
    nbdevup=2.0,
    nbdevdn=2.0,
    column_names=['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
)

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest = bb.get_latest()
print(f"Upper: {latest['BB_UPPER']}")
print(f"Middle: {latest['BB_MIDDLE']}")
print(f"Lower: {latest['BB_LOWER']}")

# Calculate bandwidth (volatility measure)
bandwidth = bb.get_bandwidth()
print(f"Bandwidth: {bandwidth:.4f}")

# Calculate %B (position within bands)
percent_b = bb.get_percent_b()
print(f"%B: {percent_b:.2f}")

# Detect squeeze (low volatility)
if bb.is_squeeze(threshold=0.02):
    print("Bollinger Band squeeze detected")
```

### ATR (Average True Range)

```python
from trading_indicators import ATR

# Create ATR indicator
atr = ATR(frame=frame, length=14, column_name='ATR_14')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_atr = atr.get_latest()
print(f"Current ATR: {latest_atr}")

# Use for position sizing or stop-loss calculation
stop_distance = latest_atr * 2  # 2 ATR stop loss
print(f"Stop distance: {stop_distance}")

# Export to NumPy
atr_array = atr.to_numpy()
```

### Pivot Points (Swing High/Low Detection)

```python
from trading_indicators import PivotPoints

# Create Pivot Points indicator
pivots = PivotPoints(
    frame=frame,
    left_bars=5,
    right_bars=5,
    column_names=['SWING_HIGH', 'SWING_LOW']
)

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access pivot points (marked with lag of right_bars)
for period in pivots.periods:
    swing_high = getattr(period, 'SWING_HIGH', None)
    swing_low = getattr(period, 'SWING_LOW', None)

    if swing_high is not None:
        print(f"Swing High at {period.open_date}: {swing_high}")
    if swing_low is not None:
        print(f"Swing Low at {period.open_date}: {swing_low}")

# Export to NumPy (returns dict)
pivots_data = pivots.to_numpy()
```

### FVG (Fair Value Gap)

```python
from trading_indicators import FVG

# Create FVG indicator
fvg = FVG(
    frame=frame,
    column_names=['FVG_TOP', 'FVG_BOTTOM', 'FVG_TYPE']
)

# Feed candles
for candle in candles:
    frame.feed(candle)

# Check for gaps
for period in fvg.periods:
    fvg_type = getattr(period, 'FVG_TYPE', None)

    if fvg_type is not None:
        fvg_top = period.FVG_TOP
        fvg_bottom = period.FVG_BOTTOM

        if fvg.is_bullish_fvg(period):
            print(f"Bullish FVG at {period.open_date}: [{fvg_bottom}, {fvg_top}]")
        elif fvg.is_bearish_fvg(period):
            print(f"Bearish FVG at {period.open_date}: [{fvg_bottom}, {fvg_top}]")

# Export to NumPy (returns dict)
fvg_data = fvg.to_numpy()
```

### ADX (Average Directional Movement Index)

```python
from trading_indicators import ADX

# Create ADX indicator
adx = ADX(frame=frame, length=14, column_name='ADX_14')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_adx = adx.get_latest()
print(f"Current ADX: {latest_adx}")

# Check trend strength
if adx.is_trending(threshold=20):
    print("Market is trending")

if adx.is_strong_trend(threshold=40):
    print("Strong trend detected")

# Export to NumPy
adx_array = adx.to_numpy()
adx_normalized = adx.to_normalize()  # [0, 1] range
```

### Stochastic Oscillator

```python
from trading_indicators import STOCH

# Create Stochastic indicator
stoch = STOCH(
    frame=frame,
    fastk_period=5,
    slowk_period=3,
    slowk_ma_type=0,
    slowd_period=3,
    slowd_ma_type=0,
    column_names=['STOCH_K', 'STOCH_D']
)

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest = stoch.get_latest()
print(f"%K: {latest['STOCH_K']}")
print(f"%D: {latest['STOCH_D']}")

# Check conditions
if stoch.is_overbought(threshold=80):
    print("Stochastic indicates overbought")

if stoch.is_oversold(threshold=20):
    print("Stochastic indicates oversold")

# Detect crossovers
if stoch.is_bullish_crossover():
    print("Bullish %K/%D crossover")

if stoch.is_bearish_crossover():
    print("Bearish %K/%D crossover")
```

### CCI (Commodity Channel Index)

```python
from trading_indicators import CCI

# Create CCI indicator
cci = CCI(frame=frame, length=14, column_name='CCI_14')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_cci = cci.get_latest()
print(f"Current CCI: {latest_cci}")

# Check conditions
if cci.is_overbought(threshold=100):
    print("CCI indicates overbought (strong uptrend)")

if cci.is_oversold(threshold=-100):
    print("CCI indicates oversold (strong downtrend)")

# Export to NumPy
cci_array = cci.to_numpy()
```

### MFI (Money Flow Index)

```python
from trading_indicators import MFI

# Create MFI indicator (requires volume)
mfi = MFI(frame=frame, length=14, column_name='MFI_14')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_mfi = mfi.get_latest()
print(f"Current MFI: {latest_mfi}")

# Check conditions
if mfi.is_overbought(threshold=80):
    print("MFI indicates overbought")

if mfi.is_oversold(threshold=20):
    print("MFI indicates oversold")

# Export to NumPy
mfi_array = mfi.to_numpy()
mfi_normalized = mfi.to_normalize()  # [0, 1] range
```

### AROON Indicator

```python
from trading_indicators import AROON, AROONOSC

# Create AROON indicator
aroon = AROON(
    frame=frame,
    length=14,
    column_names=['AROON_DOWN', 'AROON_UP']
)

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest = aroon.get_latest()
print(f"Aroon Up: {latest['AROON_UP']}")
print(f"Aroon Down: {latest['AROON_DOWN']}")

# Check trend
if aroon.is_uptrend(up_threshold=70, down_threshold=30):
    print("Strong uptrend detected")

if aroon.is_downtrend(down_threshold=70, up_threshold=30):
    print("Strong downtrend detected")

# Or use Aroon Oscillator for simpler analysis
aroonosc = AROONOSC(frame=frame, length=14)
osc_value = aroonosc.get_latest()
print(f"Aroon Oscillator: {osc_value}")

if aroonosc.is_bullish():
    print("Bullish (Aroon Up > Aroon Down)")
```

### AD (Chaikin A/D Line)

```python
from trading_indicators import AD

# Create AD indicator
ad = AD(frame=frame, column_name='AD')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_ad = ad.get_latest()
print(f"Current A/D Line: {latest_ad}")

# Check accumulation/distribution
if ad.is_accumulating():
    print("Accumulation detected (buying pressure)")

if ad.is_distributing():
    print("Distribution detected (selling pressure)")

# Detect divergences
if ad.is_bullish_divergence(lookback=10):
    print("Bullish divergence: Price falling but A/D rising")

if ad.is_bearish_divergence(lookback=10):
    print("Bearish divergence: Price rising but A/D falling")

# Export to NumPy
ad_array = ad.to_numpy()
```

### ADOSC (Chaikin A/D Oscillator)

```python
from trading_indicators import ADOSC

# Create ADOSC indicator
adosc = ADOSC(frame=frame, fast=3, slow=10, column_name='ADOSC')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_adosc = adosc.get_latest()
print(f"Current A/D Oscillator: {latest_adosc}")

# Check conditions
if adosc.is_bullish():
    print("ADOSC > 0: Buying pressure dominant")

if adosc.is_bearish():
    print("ADOSC < 0: Selling pressure dominant")

# Detect crossovers
if adosc.is_bullish_crossover():
    print("ADOSC crossed above zero (bullish signal)")

if adosc.is_bearish_crossover():
    print("ADOSC crossed below zero (bearish signal)")

# Detect divergences
if adosc.is_bullish_divergence(lookback=10):
    print("Bullish divergence detected")

# Export to NumPy
adosc_array = adosc.to_numpy()
```

### OBV (On Balance Volume)

```python
from trading_indicators import OBV

# Create OBV indicator
obv = OBV(frame=frame, column_name='OBV')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access values
latest_obv = obv.get_latest()
print(f"Current OBV: {latest_obv}")

# Check trend direction
if obv.is_rising():
    print("OBV rising (buying pressure)")

if obv.is_falling():
    print("OBV falling (selling pressure)")

# Check trend confirmation
if obv.confirms_trend(lookback=5):
    print("OBV confirms price trend (healthy trend)")

# Get trend strength
strength = obv.get_trend_strength(lookback=10)
if strength is not None:
    print(f"Trend strength: {strength:.2f}%")

# Detect divergences
if obv.is_bullish_divergence(lookback=10):
    print("Bullish divergence: Price falling but OBV rising")

if obv.is_bearish_divergence(lookback=10):
    print("Bearish divergence: Price rising but OBV falling")

# Export to NumPy
obv_array = obv.to_numpy()
```

### HT_TRENDMODE (Hilbert Transform - Trend vs Cycle Mode)

```python
from trading_indicators import HT_TRENDMODE

# Create HT_TRENDMODE indicator
ht_trendmode = HT_TRENDMODE(frame=frame, column_name='HT_TRENDMODE')

# Feed candles (need 63+ periods for calculation)
for candle in candles:
    frame.feed(candle)

# Access values
latest = ht_trendmode.get_latest()
print(f"Trend Mode: {latest}")  # 1 = trending, 0 = cycling

# Check market mode
if ht_trendmode.is_trending():
    print("Market is in TREND mode - use trend-following strategies")
elif ht_trendmode.is_cycling():
    print("Market is in CYCLE mode - use mean-reversion strategies")

# Detect mode changes
mode_change = ht_trendmode.mode_changed()
if mode_change == 'to_trend':
    print("Market transitioned to TREND mode")
elif mode_change == 'to_cycle':
    print("Market transitioned to CYCLE mode")

# Get mode stability
stability = ht_trendmode.get_mode_stability(lookback=10)
print(f"Mode stability: {stability:.1f}%")

# Get consecutive duration
if ht_trendmode.is_trending():
    duration = ht_trendmode.get_trend_duration()
    print(f"Trending for {duration} periods")

# Get human-readable mode
mode_str = ht_trendmode.get_mode_string()
print(f"Current mode: {mode_str}")  # 'TREND' or 'CYCLE'

# Export to NumPy
trendmode_array = ht_trendmode.to_numpy()
```

### HT_DCPERIOD (Hilbert Transform - Dominant Cycle Period)

```python
from trading_indicators import HT_DCPERIOD

# Create HT_DCPERIOD indicator
ht_dcperiod = HT_DCPERIOD(frame=frame, column_name='HT_DCPERIOD')

# Feed candles (need 32+ periods for calculation)
for candle in candles:
    frame.feed(candle)

# Access values
latest = ht_dcperiod.get_latest()
print(f"Dominant Cycle Period: {latest:.2f} bars")

# Get cycle type
cycle_type = ht_dcperiod.get_cycle_type()
print(f"Cycle type: {cycle_type}")  # 'fast', 'normal', or 'slow'

# Check if cycle is changing
if ht_dcperiod.is_cycle_shortening(lookback=5):
    print("Cycle period is shortening - market becoming more volatile")

if ht_dcperiod.is_cycle_lengthening(lookback=5):
    print("Cycle period is lengthening - market slowing down")

# Get average cycle period
avg_cycle = ht_dcperiod.get_average_cycle(lookback=20)
print(f"Average cycle over 20 periods: {avg_cycle:.2f} bars")

# Export to NumPy
dcperiod_array = ht_dcperiod.to_numpy()
```

### HT_DCPHASE (Hilbert Transform - Dominant Cycle Phase)

```python
from trading_indicators import HT_DCPHASE

# Create HT_DCPHASE indicator
ht_dcphase = HT_DCPHASE(frame=frame, column_name='HT_DCPHASE')

# Feed candles (need 63+ periods for calculation)
for candle in candles:
    frame.feed(candle)

# Access values
latest = ht_dcphase.get_latest()
print(f"Dominant Cycle Phase: {latest:.2f}°")

# Get phase quadrant
quadrant = ht_dcphase.get_phase_quadrant()
print(f"Phase quadrant: {quadrant}")  # 'Q1', 'Q2', 'Q3', or 'Q4'

# Check trend phase
if ht_dcphase.is_early_uptrend():
    print("Early uptrend phase (Q1: 0-90°)")
elif ht_dcphase.is_late_uptrend():
    print("Late uptrend phase (Q2: 90-180°) - potential top")
elif ht_dcphase.is_early_downtrend():
    print("Early downtrend phase (Q3: 180-270°)")
elif ht_dcphase.is_late_downtrend():
    print("Late downtrend phase (Q4: 270-360°) - potential bottom")

# Get phase velocity (rate of change)
velocity = ht_dcphase.get_phase_velocity(lookback=3)
print(f"Phase velocity: {velocity:.2f}°/period")

# Check if phase is changing rapidly
if ht_dcphase.is_phase_accelerating():
    print("Phase accelerating - strong momentum")

# Export to NumPy
dcphase_array = ht_dcphase.to_numpy()
```

### HT_SINE (Hilbert Transform - Sine Wave)

```python
from trading_indicators import HT_SINE

# Create HT_SINE indicator
ht_sine = HT_SINE(frame=frame, column_names=['HT_SINE', 'HT_LEADSINE'])

# Feed candles (need 63+ periods for calculation)
for candle in candles:
    frame.feed(candle)

# Access values
latest = ht_sine.get_latest()
print(f"Sine: {latest['HT_SINE']:.4f}")
print(f"LeadSine: {latest['HT_LEADSINE']:.4f}")

# Detect crossovers (cycle turning points)
if ht_sine.is_bullish_crossover():
    print("Bullish crossover: Sine crossed above LeadSine")

if ht_sine.is_bearish_crossover():
    print("Bearish crossover: Sine crossed below LeadSine")

# Export to NumPy (returns dict)
sine_data = ht_sine.to_numpy()
sine_array = sine_data['HT_SINE']
leadsine_array = sine_data['HT_LEADSINE']
```

### HT_PHASOR (Hilbert Transform - Phasor Components)

```python
from trading_indicators import HT_PHASOR

# Create HT_PHASOR indicator
ht_phasor = HT_PHASOR(frame=frame, column_names=['HT_INPHASE', 'HT_QUADRATURE'])

# Feed candles (need 63+ periods for calculation)
for candle in candles:
    frame.feed(candle)

# Access values
latest = ht_phasor.get_latest()
print(f"InPhase: {latest['HT_INPHASE']:.4f}")
print(f"Quadrature: {latest['HT_QUADRATURE']:.4f}")

# Get phasor magnitude (cycle strength)
magnitude = ht_phasor.get_magnitude()
print(f"Phasor magnitude: {magnitude:.4f}")

# Get phase angle
angle = ht_phasor.get_phase_angle()
print(f"Phase angle: {angle:.2f}°")

# Export to NumPy (returns dict)
phasor_data = ht_phasor.to_numpy()
inphase_array = phasor_data['HT_INPHASE']
quadrature_array = phasor_data['HT_QUADRATURE']
```

### Price Indicators

```python
from trading_indicators import AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE

# Create price indicators
avgprice = AVGPRICE(frame=frame, column_name='AVGPRICE')
medprice = MEDPRICE(frame=frame, column_name='MEDPRICE')
typprice = TYPPRICE(frame=frame, column_name='TYPPRICE')
wclprice = WCLPRICE(frame=frame, column_name='WCLPRICE')

# Feed candles
for candle in candles:
    frame.feed(candle)

# Access average price (O+H+L+C)/4
avg = avgprice.get_latest()
print(f"Average Price: {avg:.2f}")

# Check if average price is above/below close
if avgprice.is_above_close():
    print("Bearish bar - average above close")
elif avgprice.is_below_close():
    print("Bullish bar - average below close")

# Get spread from close
spread = avgprice.get_spread_from_close()
print(f"AVGPRICE - Close spread: {spread:.2f}")

# Access median price (H+L)/2
med = medprice.get_latest()
print(f"Median Price: {med:.2f}")

# Check where close is in the range
position = medprice.get_close_position_in_range()
print(f"Close position in range: {position:.2%}")  # 0 = at low, 1 = at high

# Get range size
range_size = medprice.get_range_size()
print(f"High-Low range: {range_size:.2f}")

# Access typical price (H+L+C)/3
typ = typprice.get_latest()
print(f"Typical Price: {typ:.2f}")

# Check trend direction
if typprice.is_rising(lookback=3):
    print("Typical price rising over last 3 periods")
elif typprice.is_falling(lookback=3):
    print("Typical price falling over last 3 periods")

# Access weighted close price (H+L+2*C)/4
wcl = wclprice.get_latest()
print(f"Weighted Close Price: {wcl:.2f}")

# Get momentum
momentum = wclprice.get_momentum(lookback=5)
print(f"5-period momentum: {momentum:.2f}")

# Export to NumPy
avgprice_array = avgprice.to_numpy()
medprice_array = medprice.to_numpy()
typprice_array = typprice.to_numpy()
wclprice_array = wclprice.to_numpy()
```

### Statistical Indicators

```python
from trading_indicators.stats import (
    BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_SLOPE,
    STDDEV, TSF, VAR
)
import numpy as np

# Statistical indicators can work in two modes:
# 1. Auto-synced with frame (like other indicators)
# 2. Static utility mode for quick calculations

# Mode 1: Auto-synced with frame
frame = TimeFrame('5T', max_periods=100)

# Standard deviation for volatility measurement
stddev = STDDEV(frame=frame, length=20, nbdev=1, column_name='STDDEV')

# Linear regression for trend analysis
linearreg = LINEARREG(frame=frame, length=14, column_name='LINEARREG')
lr_angle = LINEARREG_ANGLE(frame=frame, length=14, column_name='REG_ANGLE')
lr_slope = LINEARREG_SLOPE(frame=frame, length=14, column_name='REG_SLOPE')

# Time series forecast
tsf = TSF(frame=frame, length=14, column_name='TSF')

# Variance measurement
var = VAR(frame=frame, length=20, nbdev=1, column_name='VAR')

# Feed candles - all indicators update automatically
for candle in candles:
    frame.feed(candle)

# Check volatility
if stddev.is_high_volatility(threshold=2.0):
    print("High volatility detected")
elif stddev.is_low_volatility(threshold=0.5):
    print("Low volatility - potential breakout coming")

# Check if volatility is expanding or contracting
if stddev.is_expanding():
    print("Volatility expanding - trend acceleration")

# Analyze trend direction and strength
if lr_angle.is_steep_uptrend(threshold=45):
    print("Strong uptrend (angle > 45 degrees)")
elif lr_angle.is_flat(threshold=10):
    print("Market consolidating (angle < 10 degrees)")

# Check price position relative to regression
if linearreg.is_above_regression():
    distance = linearreg.get_distance_from_regression()
    print(f"Price {distance:.2f}% above regression line")

# Check if trend is accelerating
if lr_slope.is_accelerating():
    print("Trend momentum is accelerating")

# Compare price to forecast
if tsf.is_price_above_forecast():
    error = tsf.get_forecast_error()
    print(f"Price exceeding forecast by {error:.2f}%")

# Mode 2: Static utility mode (no frame required)
prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])

# Quick volatility calculation
volatility = STDDEV.compute(prices, length=5, nbdev=1)
print(f"Current volatility: {volatility[-1]:.4f}")

# Quick linear regression analysis
regression_line = LINEARREG.compute(prices, length=5)
regression_angle = LINEARREG_ANGLE.compute(prices, length=5)
regression_slope = LINEARREG_SLOPE.compute(prices, length=5)

print(f"Regression value: {regression_line[-1]:.2f}")
print(f"Trend angle: {regression_angle[-1]:.2f} degrees")
print(f"Trend slope: {regression_slope[-1]:.4f}")

# Forecast next value
forecast = TSF.compute(prices, length=5)
print(f"Next period forecast: {forecast[-1]:.2f}")

# Correlation between two assets (requires two series)
btc_prices = np.array([50000, 51000, 50500, 52000, 53000])
eth_prices = np.array([3000, 3100, 3050, 3150, 3200])
correlation = CORREL.compute(eth_prices, btc_prices, length=5)
print(f"ETH/BTC correlation: {correlation[-1]:.4f}")

# Beta coefficient (relative volatility)
beta = BETA.compute(eth_prices, btc_prices, length=5)
print(f"ETH beta vs BTC: {beta[-1]:.4f}")
```

## Integration with AssetView

```python
from trading_asset_view import AssetView
from trading_indicators import RSI, SMA, EMA, MACD, ATR

# Create AssetView with multiple timeframes
asset_view = AssetView("BTC/USDT", timeframes=["1T", "5T", "1H"])

# Add indicators to specific timeframes
rsi_1m = RSI(frame=asset_view["1T"], length=14, column_name='RSI_14')
rsi_5m = RSI(frame=asset_view["5T"], length=14, column_name='RSI_14')

sma20_1h = SMA(frame=asset_view["1H"], period=20, column_name='SMA_20')
ema50_1h = EMA(frame=asset_view["1H"], period=50, column_name='EMA_50')

macd_5m = MACD(frame=asset_view["5T"], fast=12, slow=26, signal=9)
atr_1h = ATR(frame=asset_view["1H"], length=14, column_name='ATR_14')

# Feed candles - all indicators across all timeframes update automatically
for candle in candles:
    asset_view.feed(candle)

# Access indicators
print(f"1m RSI: {rsi_1m.get_latest()}")
print(f"5m RSI: {rsi_5m.get_latest()}")
print(f"1h SMA20: {sma20_1h.get_latest()}")
print(f"1h EMA50: {ema50_1h.get_latest()}")
print(f"5m MACD: {macd_5m.get_latest()}")
print(f"1h ATR: {atr_1h.get_latest()}")
```

## Architecture

### How It Works

1. **Indicator Creation**: When you create an indicator, it binds to a TimeFrame
2. **Event Subscription**: The indicator subscribes to frame events (`new_period`, `update`, `close`)
3. **Automatic Calculation**: When frame receives candles, indicators calculate values for each period
4. **Dynamic Storage**: Values are stored in `period._data` dict with named attributes
5. **Synchronization**: Indicator periods stay synchronized with frame periods

### Period-by-Period Calculation

Instead of recalculating the entire indicator array on every update, each indicator:
- **New Period**: Creates a new IndicatorPeriod and calculates its value
- **Update**: Recalculates only the current period's value
- Uses TA-Lib efficiently by extracting only necessary historical data

### Lag Behavior

Some indicators mark previous periods when confirmation occurs:
- **Pivot Points**: Marks the pivot candle after `right_bars` confirmation
- **FVG**: Marks the middle candle when gap is confirmed

### Data Access Pattern

```python
# Same pattern as trading-frame Period
frame_period.close_price     # Frame period attribute
indicator_period.RSI_14      # Indicator period attribute
indicator_period.MACD_LINE   # Multi-value indicator

# Both use _data dict internally
frame_period._data           # {'close_price': 50000, ...}
indicator_period._data       # {'RSI_14': 65.4, 'SMA_20': 50123.5}
```

## API Reference

### BaseIndicator

Base class for all indicators.

**Methods:**
- `calculate(period)` - Calculate indicator value(s) for a period (abstract)
- `to_numpy()` - Export values as numpy array(s) (abstract)

### RSI

**Constructor:**
```python
RSI(frame, length=14, column_name='RSI', max_periods=None)
```

**Methods:**
- `get_latest()` - Get latest RSI value
- `is_overbought(threshold=70.0)` - Check if overbought
- `is_oversold(threshold=30.0)` - Check if oversold
- `to_numpy()` - Export as numpy array
- `to_normalize()` - Export normalized [0, 1]

### SMA

**Constructor:**
```python
SMA(frame, period=20, column_name='SMA', price_field='close', max_periods=None)
```

**Methods:**
- `get_latest()` - Get latest SMA value
- `to_numpy()` - Export as numpy array

### EMA

**Constructor:**
```python
EMA(frame, period=20, column_name='EMA', price_field='close', max_periods=None)
```

**Methods:**
- `get_latest()` - Get latest EMA value
- `to_numpy()` - Export as numpy array

### MACD

**Constructor:**
```python
MACD(frame, fast=12, slow=26, signal=9, column_names=['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST'], max_periods=None)
```

**Methods:**
- `get_latest()` - Get latest MACD values as dict
- `is_bullish_crossover()` - Detect bullish crossover
- `is_bearish_crossover()` - Detect bearish crossover
- `to_numpy()` - Export as dict of numpy arrays
- `to_normalize()` - Export normalized [0, 1]

### BollingerBands

**Constructor:**
```python
BollingerBands(frame, period=20, nbdevup=2.0, nbdevdn=2.0, column_names=['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'], max_periods=None)
```

**Methods:**
- `get_latest()` - Get latest BB values as dict
- `get_bandwidth()` - Get current bandwidth (volatility)
- `get_percent_b(price=None)` - Get %B indicator
- `is_squeeze(threshold=0.02)` - Detect low volatility squeeze
- `to_numpy()` - Export as dict of numpy arrays

### ATR

**Constructor:**
```python
ATR(frame, length=14, column_name='ATR', max_periods=None)
```

**Methods:**
- `get_latest()` - Get latest ATR value
- `to_numpy()` - Export as numpy array

### PivotPoints

**Constructor:**
```python
PivotPoints(frame, left_bars=5, right_bars=5, column_names=['SWING_HIGH', 'SWING_LOW'], max_periods=None)
```

**Methods:**
- `to_numpy()` - Export as dict of numpy arrays (with NaN for non-pivot periods)

### FVG

**Constructor:**
```python
FVG(frame, column_names=['FVG_TOP', 'FVG_BOTTOM', 'FVG_TYPE'], max_periods=None)
```

**Methods:**
- `is_bullish_fvg(period)` - Check if period contains bullish FVG
- `is_bearish_fvg(period)` - Check if period contains bearish FVG
- `to_numpy()` - Export as dict of numpy arrays

## Development

### Run Tests

```bash
pytest
pytest --cov=trading_indicators
```

### Project Structure

```
trading-indicators/
├── src/trading_indicators/
│   ├── __init__.py
│   ├── base.py              # BaseIndicator + IndicatorPeriod
│   │
│   ├── momentum/            # 30 momentum & oscillator indicators
│   │   ├── __init__.py
│   │   ├── adx.py           # ADX - Average Directional Movement Index
│   │   ├── adxr.py          # ADXR - ADX Rating
│   │   ├── apo.py           # APO - Absolute Price Oscillator
│   │   ├── aroon.py         # AROON - Aroon Up/Down
│   │   ├── aroonosc.py      # AROONOSC - Aroon Oscillator
│   │   ├── bop.py           # BOP - Balance of Power
│   │   ├── cci.py           # CCI - Commodity Channel Index
│   │   ├── cmo.py           # CMO - Chande Momentum Oscillator
│   │   ├── dx.py            # DX - Directional Movement Index
│   │   ├── macd.py          # MACD - Moving Avg Convergence Divergence
│   │   ├── macdext.py       # MACDEXT - MACD with controllable MA
│   │   ├── macdfix.py       # MACDFIX - MACD Fix 12/26
│   │   ├── mfi.py           # MFI - Money Flow Index
│   │   ├── minus_di.py      # MINUS_DI - Minus Directional Indicator
│   │   ├── minus_dm.py      # MINUS_DM - Minus Directional Movement
│   │   ├── mom.py           # MOM - Momentum
│   │   ├── plus_di.py       # PLUS_DI - Plus Directional Indicator
│   │   ├── plus_dm.py       # PLUS_DM - Plus Directional Movement
│   │   ├── ppo.py           # PPO - Percentage Price Oscillator
│   │   ├── roc.py           # ROC - Rate of Change
│   │   ├── rocp.py          # ROCP - Rate of Change Percentage
│   │   ├── rocr.py          # ROCR - Rate of Change Ratio
│   │   ├── rocr100.py       # ROCR100 - ROC Ratio 100 scale
│   │   ├── rsi.py           # RSI - Relative Strength Index
│   │   ├── stoch.py         # STOCH - Stochastic Oscillator
│   │   ├── stochf.py        # STOCHF - Stochastic Fast
│   │   ├── stochrsi.py      # STOCHRSI - Stochastic RSI
│   │   ├── trix.py          # TRIX - Triple EMA ROC
│   │   ├── ultosc.py        # ULTOSC - Ultimate Oscillator
│   │   └── willr.py         # WILLR - Williams %R
│   │
│   ├── trend/               # 14 trend indicators
│   │   ├── __init__.py
│   │   ├── sma.py           # SMA - Simple Moving Average
│   │   ├── ema.py           # EMA - Exponential Moving Average
│   │   ├── dema.py          # DEMA - Double Exponential MA
│   │   ├── tema.py          # TEMA - Triple Exponential MA
│   │   ├── wma.py           # WMA - Weighted Moving Average
│   │   ├── trima.py         # TRIMA - Triangular Moving Average
│   │   ├── kama.py          # KAMA - Kaufman Adaptive MA
│   │   ├── mama.py          # MAMA - MESA Adaptive MA
│   │   ├── t3.py            # T3 - Triple Exponential MA T3
│   │   ├── ht_trendline.py  # HT_TRENDLINE - Hilbert Transform
│   │   ├── ma.py            # MA - Generic Moving Average
│   │   ├── mavp.py          # MAVP - MA Variable Period
│   │   ├── midpoint.py      # MIDPOINT - MidPoint over period
│   │   ├── midprice.py      # MIDPRICE - Midpoint Price
│   │   ├── sar.py           # SAR - Parabolic SAR
│   │   ├── sarext.py        # SAREXT - Parabolic SAR Extended
│   │   ├── bollinger.py     # BBANDS - Bollinger Bands
│   │   ├── pivot_points.py  # Swing High/Low Detection
│   │   └── fvg.py           # FVG - Fair Value Gap
│   │
│   ├── volatility/          # 1 volatility indicator
│   │   ├── __init__.py
│   │   └── atr.py           # ATR - Average True Range
│   │
│   ├── volume/              # 3 volume indicators
│   │   ├── __init__.py
│   │   ├── ad.py            # AD - Chaikin A/D Line
│   │   ├── adosc.py         # ADOSC - Chaikin A/D Oscillator
│   │   └── obv.py           # OBV - On Balance Volume
│   │
│   └── stats/               # 9 statistical indicators
│       ├── __init__.py
│       ├── beta.py          # BETA - Beta Coefficient
│       ├── correl.py        # CORREL - Pearson's Correlation
│       ├── linearreg.py     # LINEARREG - Linear Regression
│       ├── linearreg_angle.py        # LINEARREG_ANGLE - Regression Angle
│       ├── linearreg_intercept.py    # LINEARREG_INTERCEPT - Y-Intercept
│       ├── linearreg_slope.py        # LINEARREG_SLOPE - Regression Slope
│       ├── stddev.py        # STDDEV - Standard Deviation
│       ├── tsf.py           # TSF - Time Series Forecast
│       └── var.py           # VAR - Variance
│
├── tests/
│   ├── test_momentum_indicators.py  # Tests for momentum indicators
│   ├── test_volume_indicators.py    # Tests for volume indicators
│   └── ...
├── pyproject.toml
└── README.md
```

## License

MIT License - see LICENSE file for details.

## Contributing

Issues and pull requests welcome at [GitHub repository](https://github.com/Morgiver/trading-indicators)

## Related Projects

- [trading-frame](https://github.com/Morgiver/trading-frame) - Core timeframe aggregation engine
- [trading-asset-view](https://github.com/Morgiver/trading-asset-view) - Multi-timeframe orchestration layer
