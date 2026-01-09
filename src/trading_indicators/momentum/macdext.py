"""MACDEXT (MACD with controllable MA type) indicator."""

from typing import Optional, List, Dict
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class MACDEXT(BaseIndicator):
    """
    MACDEXT (MACD with controllable MA type) indicator.

    Extended version of MACD that allows specifying the moving average
    type for fast MA, slow MA, and signal line independently.

    MA Types (TA-Lib):
    0 = SMA, 1 = EMA, 2 = WMA, 3 = DEMA, 4 = TEMA, 5 = TRIMA,
    6 = KAMA, 7 = MAMA, 8 = T3

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> macdext = MACDEXT(frame=frame, fast=12, slow=26, signal=9,
        ...                   fast_ma_type=1, slow_ma_type=1, signal_ma_type=1)
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(macdext.periods[-1].MACD_LINE)
    """

    def __init__(
        self,
        frame: 'Frame',
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        fast_ma_type: int = 1,
        slow_ma_type: int = 1,
        signal_ma_type: int = 1,
        column_names: Optional[List[str]] = None,
        max_periods: Optional[int] = None
    ):
        """
        Initialize MACDEXT indicator.

        Args:
            frame: Frame to bind to
            fast: Fast period (default: 12)
            slow: Slow period (default: 26)
            signal: Signal line period (default: 9)
            fast_ma_type: Fast MA type (default: 1 = EMA)
            slow_ma_type: Slow MA type (default: 1 = EMA)
            signal_ma_type: Signal MA type (default: 1 = EMA)
            column_names: Names for [line, signal, histogram]
            max_periods: Maximum periods to keep
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.signal_ma_type = signal_ma_type
        self.column_names = column_names or ['MACDEXT_LINE', 'MACDEXT_SIGNAL', 'MACDEXT_HIST']

        if len(self.column_names) != 3:
            raise ValueError("column_names must contain exactly 3 names")

        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate MACDEXT values for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        required_periods = self.slow + self.signal
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        macd_line, signal_line, histogram = talib.MACDEXT(
            close_prices,
            fastperiod=self.fast,
            fastmatype=self.fast_ma_type,
            slowperiod=self.slow,
            slowmatype=self.slow_ma_type,
            signalperiod=self.signal,
            signalmatype=self.signal_ma_type
        )

        if not np.isnan(macd_line[-1]):
            setattr(period, self.column_names[0], float(macd_line[-1]))
        if not np.isnan(signal_line[-1]):
            setattr(period, self.column_names[1], float(signal_line[-1]))
        if not np.isnan(histogram[-1]):
            setattr(period, self.column_names[2], float(histogram[-1]))

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Export MACDEXT values as dictionary of numpy arrays."""
        return {
            name: np.array([
                getattr(p, name) if hasattr(p, name) else np.nan
                for p in self.periods
            ])
            for name in self.column_names
        }

    def get_latest(self) -> Optional[Dict[str, float]]:
        """Get the latest MACDEXT values."""
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
