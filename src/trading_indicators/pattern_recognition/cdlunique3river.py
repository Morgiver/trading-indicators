"""
CDLUNIQUE3RIVER - Unique 3 River Candlestick Pattern

Unique 3 River is a rare bullish reversal pattern with three candles showing
specific characteristics that signal a potential bottom reversal.

Returns:
- 0: Pattern not detected
- 100: Bullish Unique 3 River
- -100: Bearish Unique 3 River
"""

from typing import Optional
import numpy as np
import talib
from ..base import BaseIndicator


class CDLUNIQUE3RIVER(BaseIndicator):
    """
    CDLUNIQUE3RIVER - Unique 3 River Candlestick Pattern

    A rare bullish reversal pattern with three specific candles indicating
    a potential bottom and upward reversal from a downtrend.

    Usage:
        # Auto-sync mode
        unique_3_river = CDLUNIQUE3RIVER(frame=frame)
        if unique_3_river.is_bullish():
            print("Bullish Unique 3 River detected")

        # Utility mode
        pattern = CDLUNIQUE3RIVER.compute(open_prices, high_prices, low_prices, close_prices)
    """

    def __init__(self, frame=None, column_name='CDLUNIQUE3RIVER', max_periods=None):
        """
        Initialize CDLUNIQUE3RIVER indicator.

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
        Compute CDLUNIQUE3RIVER using TA-Lib (utility mode).

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

        return talib.CDLUNIQUE3RIVER(open_prices, high_prices, low_prices, close_prices)

    def calculate(self, period):
        """Calculate CDLUNIQUE3RIVER for the current period (auto-sync mode)."""
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
        """Check if bullish Unique 3 River pattern is detected."""
        latest = self.get_latest()
        return latest == 100 if latest is not None else False

    def is_bearish(self) -> bool:
        """Check if bearish Unique 3 River pattern is detected."""
        latest = self.get_latest()
        return latest == -100 if latest is not None else False

    def is_detected(self) -> bool:
        """Check if any Unique 3 River pattern is detected (bullish or bearish)."""
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
