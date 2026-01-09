"""
CDLDRAGONFLYDOJI - Dragonfly Doji Candlestick Pattern

Dragonfly Doji is a bullish reversal pattern with a T-shaped candle where
the open, high, and close are the same or very close, with a long lower shadow.

Returns:
- 0: Pattern not detected
- 100: Bullish Dragonfly Doji
- -100: Bearish Dragonfly Doji
"""

from typing import Optional
import numpy as np
import talib
from ..base import BaseIndicator


class CDLDRAGONFLYDOJI(BaseIndicator):
    """
    CDLDRAGONFLYDOJI - Dragonfly Doji Candlestick Pattern

    A bullish reversal pattern with a T-shaped candle showing strong buying
    pressure after initial selling, indicating potential upward reversal.

    Usage:
        # Auto-sync mode
        dragonfly_doji = CDLDRAGONFLYDOJI(frame=frame)
        if dragonfly_doji.is_bullish():
            print("Bullish Dragonfly Doji detected")

        # Utility mode
        pattern = CDLDRAGONFLYDOJI.compute(open_prices, high_prices, low_prices, close_prices)
    """

    def __init__(self, frame=None, column_name='CDLDRAGONFLYDOJI', max_periods=None):
        """
        Initialize CDLDRAGONFLYDOJI indicator.

        Args:
            frame: TradingFrame instance for auto-sync mode (optional)
            column_name: Column name for the indicator values
            max_periods: Maximum number of periods to store
        """
        self.column_name = column_name

        if frame:
            super().__init__(frame, max_periods)
        else:
            self.frame = None
            self.periods = []

    @staticmethod
    def compute(open_prices, high_prices, low_prices, close_prices):
        """
        Compute CDLDRAGONFLYDOJI using TA-Lib (utility mode).

        Args:
            open_prices: NumPy array of open prices
            high_prices: NumPy array of high prices
            low_prices: NumPy array of low prices
            close_prices: NumPy array of close prices

        Returns:
            NumPy array of pattern signals (0, 100, -100)
        """
        open_prices = np.asarray(open_prices, dtype=np.float64)
        high_prices = np.asarray(high_prices, dtype=np.float64)
        low_prices = np.asarray(low_prices, dtype=np.float64)
        close_prices = np.asarray(close_prices, dtype=np.float64)

        return talib.CDLDRAGONFLYDOJI(open_prices, high_prices, low_prices, close_prices)

    def calculate(self, period):
        """Calculate CDLDRAGONFLYDOJI for the current period (auto-sync mode)."""
        if not self.frame or len(self.frame.periods) < 1:
            return None

        # Get OHLC data from frame
        open_prices = np.array([p.open_price for p in self.frame.periods], dtype=np.float64)
        high_prices = np.array([p.high_price for p in self.frame.periods], dtype=np.float64)
        low_prices = np.array([p.low_price for p in self.frame.periods], dtype=np.float64)
        close_prices = np.array([p.close_price for p in self.frame.periods], dtype=np.float64)

        # Calculate pattern
        result = self.compute(open_prices, high_prices, low_prices, close_prices)

        # Return the latest value
        return int(result[-1]) if not np.isnan(result[-1]) else 0


    def to_numpy(self) -> np.ndarray:
        """
        Export pattern signals as numpy array.

        Returns:
            NumPy array with pattern signals (0, 100, -100; NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[int]:
        """
        Get the latest pattern signal.

        Returns:
            Latest pattern signal (0, 100, or -100) or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """Check if bullish Dragonfly Doji pattern is detected."""
        latest = self.get_latest()
        return latest == 100 if latest is not None else False

    def is_bearish(self) -> bool:
        """Check if bearish Dragonfly Doji pattern is detected."""
        latest = self.get_latest()
        return latest == -100 if latest is not None else False

    def is_detected(self) -> bool:
        """Check if any Dragonfly Doji pattern is detected (bullish or bearish)."""
        latest = self.get_latest()
        return latest != 0 if latest is not None else False

    def get_signal(self) -> str:
        """
        Get human-readable signal.

        Returns:
            'BULLISH', 'BEARISH', or 'NONE'
        """
        latest = self.get_latest()
        if latest is None or latest == 0:
            return 'NONE'
        return 'BULLISH' if latest > 0 else 'BEARISH'
