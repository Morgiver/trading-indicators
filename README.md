# Trading Indicators

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Technical indicators library with automatic frame synchronization** built on top of [trading-frame](https://github.com/Morgiver/trading-frame).

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

### Momentum
- **RSI** (Relative Strength Index) - Overbought/oversold detection
- **MACD** (Moving Average Convergence Divergence) - Trend following momentum

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

### Volatility
- **ATR** (Average True Range) - Volatility measurement

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
from trading_indicators import RSI, SMA, EMA, MACD, BollingerBands, ATR

# Create a frame
frame = TimeFrame('5T', max_periods=100)

# Create indicators (automatically bind to frame)
rsi = RSI(frame=frame, length=14, column_name='RSI_14')
sma20 = SMA(frame=frame, period=20, column_name='SMA_20')
ema20 = EMA(frame=frame, period=20, column_name='EMA_20')
macd = MACD(frame=frame, fast=12, slow=26, signal=9)
bb = BollingerBands(frame=frame, period=20)
atr = ATR(frame=frame, length=14, column_name='ATR_14')

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
│   ├── momentum/
│   │   ├── __init__.py
│   │   ├── rsi.py           # RSI indicator
│   │   └── macd.py          # MACD indicator
│   ├── trend/
│   │   ├── __init__.py
│   │   ├── sma.py           # SMA indicator
│   │   ├── ema.py           # EMA indicator
│   │   ├── dema.py          # DEMA indicator
│   │   ├── tema.py          # TEMA indicator
│   │   ├── wma.py           # WMA indicator
│   │   ├── kama.py          # KAMA indicator
│   │   ├── mama.py          # MAMA indicator
│   │   ├── t3.py            # T3 indicator
│   │   ├── sar.py           # Parabolic SAR
│   │   ├── bollinger.py     # Bollinger Bands
│   │   ├── pivot_points.py  # Swing High/Low
│   │   └── fvg.py           # Fair Value Gap
│   └── volatility/
│       ├── __init__.py
│       └── atr.py           # ATR indicator
├── tests/
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
