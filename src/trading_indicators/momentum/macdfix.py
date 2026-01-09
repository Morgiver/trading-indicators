"""MACDFIX (Moving Average Convergence/Divergence Fix 12/26) indicator."""

from typing import Optional, List, Dict
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class MACDFIX(BaseIndicator):
    """
    MACDFIX (MACD Fix 12/26) indicator.

    Fixed MACD with 12/26 periods and adjustable signal period.
    This is a simplified version of MACD with fixed fast/slow periods.

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> macdfix = MACDFIX(frame=frame, signal=9)
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(macdfix.periods[-1].MACDFIX_LINE)
    """

    def __init__(
        self,
        frame: 'Frame',
        signal: int = 9,
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize MACDFIX indicator.

        Args:
            frame: Frame to bind to
            signal: Signal line period (default: 9)
            column_names: Names for [line, signal, histogram]
            max_periods: Maximum periods to keep
        """
        self.signal = signal
        self.column_names = column_names or ['MACDFIX_LINE', 'MACDFIX_SIGNAL', 'MACDFIX_HIST']

        if len(self.column_names) != 3:
            raise ValueError("column_names must contain exactly 3 names")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate MACDFIX values for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        required_periods = 26 + self.signal
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        macd_line, signal_line, histogram = talib.MACDFIX(
            close_prices,
            signalperiod=self.signal
        )

        if not np.isnan(macd_line[-1]):
            setattr(period, self.column_names[0], float(macd_line[-1]))
        if not np.isnan(signal_line[-1]):
            setattr(period, self.column_names[1], float(signal_line[-1]))
        if not np.isnan(histogram[-1]):
            setattr(period, self.column_names[2], float(histogram[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Export MACDFIX values as dictionary of numpy arrays."""
        return {
            name: np.array([
                getattr(p, name) if hasattr(p, name) else np.nan
                for p in self.periods
            ])
            for name in self.column_names
        }

    def get_latest(self) -> Optional[Dict[str, float]]:
        """Get the latest MACDFIX values."""
        if self.periods:
            result = {}
            for name in self.column_names:
                value = getattr(self.periods[-1], name, None)
                if value is not None:
                    result[name] = value
            return result if result else None
        return None

    def is_bullish_crossover(self) -> bool:
        """Check if MACD line crossed above signal line."""
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
        """Check if MACD line crossed below signal line."""
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
