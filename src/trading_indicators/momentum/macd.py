"""MACD (Moving Average Convergence Divergence) indicator."""

from typing import Optional, List, Dict
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class MACD(BaseIndicator):
    """
    MACD (Moving Average Convergence Divergence) indicator.

    MACD shows the relationship between two exponential moving averages.
    It consists of three components:
    - MACD Line: Difference between fast and slow EMA
    - Signal Line: EMA of MACD Line
    - Histogram: Difference between MACD Line and Signal Line

    Typical interpretation:
    - MACD crosses above signal: Bullish signal
    - MACD crosses below signal: Bearish signal
    - Histogram > 0: Bullish momentum
    - Histogram < 0: Bearish momentum

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> macd = MACD(frame=frame, fast=12, slow=26, signal=9)
        >>>
        >>> # Feed candles - MACD automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(macd.periods[-1].MACD_LINE)
        >>> print(macd.periods[-1].MACD_SIGNAL)
        >>> print(macd.periods[-1].MACD_HIST)
    """

    def __init__(
        self,
        frame: 'Frame',
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize MACD indicator.

        Args:
            frame: Frame to bind to
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            column_names: Names for [line, signal, histogram] (default: ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST'])
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.column_names = column_names or ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST']

        if len(self.column_names) != 3:
            raise ValueError("column_names must contain exactly 3 names [line, signal, histogram]")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate MACD values for a specific period.

        Args:
            period: IndicatorPeriod to populate with MACD values
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'slow + signal' periods for MACD calculation
        required_periods = self.slow + self.signal
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        # Extract close prices (use all available up to current period)
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        # Calculate MACD using TA-Lib
        macd_line, signal_line, histogram = talib.MACD(
            close_prices,
            fastperiod=self.fast,
            slowperiod=self.slow,
            signalperiod=self.signal
        )

        # The last values are for our period
        if not np.isnan(macd_line[-1]):
            setattr(period, self.column_names[0], float(macd_line[-1]))

        if not np.isnan(signal_line[-1]):
            setattr(period, self.column_names[1], float(signal_line[-1]))

        if not np.isnan(histogram[-1]):
            setattr(period, self.column_names[2], float(histogram[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Export MACD values as dictionary of numpy arrays.

        Returns:
            Dictionary with keys 'MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST'
            (or custom column names) mapping to numpy arrays
        """
        return {
            name: np.array([
                getattr(p, name) if hasattr(p, name) else np.nan
                for p in self.periods
            ])
            for name in self.column_names
        }

    def to_normalize(self) -> Dict[str, np.ndarray]:
        """
        Export normalized MACD values for ML (Min-Max normalization).

        Returns:
            Dictionary with normalized arrays [0, 1]
        """
        arrays = self.to_numpy()
        normalized = {}

        for key, arr in arrays.items():
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                min_val = np.min(valid)
                max_val = np.max(valid)
                if max_val > min_val:
                    normalized[key] = (arr - min_val) / (max_val - min_val)
                else:
                    normalized[key] = np.zeros_like(arr)
            else:
                normalized[key] = arr

        return normalized

    def get_latest(self) -> Optional[Dict[str, float]]:
        """
        Get the latest MACD values.

        Returns:
            Dictionary with latest MACD values or None if not available
        """
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def is_bullish_crossover(self) -> bool:
        """
        Check if MACD line crossed above signal line (bullish signal).

        Returns:
            True if bullish crossover detected
        """
        if len(self.periods) < 2:
            return False

        prev = self.periods[-2]
        curr = self.periods[-1]

        prev_macd = getattr(prev, self.column_names[0], None)
        prev_signal = getattr(prev, self.column_names[1], None)
        curr_macd = getattr(curr, self.column_names[0], None)
        curr_signal = getattr(curr, self.column_names[1], None)

        if all(v is not None for v in [prev_macd, prev_signal, curr_macd, curr_signal]):
            return prev_macd <= prev_signal and curr_macd > curr_signal

        return False

    def is_bearish_crossover(self) -> bool:
        """
        Check if MACD line crossed below signal line (bearish signal).

        Returns:
            True if bearish crossover detected
        """
        if len(self.periods) < 2:
            return False

        prev = self.periods[-2]
        curr = self.periods[-1]

        prev_macd = getattr(prev, self.column_names[0], None)
        prev_signal = getattr(prev, self.column_names[1], None)
        curr_macd = getattr(curr, self.column_names[0], None)
        curr_signal = getattr(curr, self.column_names[1], None)

        if all(v is not None for v in [prev_macd, prev_signal, curr_macd, curr_signal]):
            return prev_macd >= prev_signal and curr_macd < curr_signal

        return False
