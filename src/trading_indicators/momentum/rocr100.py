"""ROCR100 (Rate of change ratio 100 scale) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class ROCR100(BaseIndicator):
    """
    ROCR100 (Rate of change ratio 100 scale) indicator.

    Formula: (price/prevPrice)*100

    Measures the ratio of current price to previous price, scaled by 100.

    Typical interpretation:
    - ROCR100 > 100: Price increasing
    - ROCR100 < 100: Price decreasing
    - ROCR100 = 100: No change

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> rocr100 = ROCR100(frame=frame, length=10, column_name='ROCR100_10')
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(rocr100.periods[-1].ROCR100_10)
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 10,
        column_name: str = 'ROCR100',
        max_periods: Optional[int] = None
    ):
        """Initialize ROCR100 indicator."""
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate ROCR100 value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None or len(self.frame.periods) < self.length + 1:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        rocr100_values = talib.ROCR100(close_prices, timeperiod=self.length)
        rocr100_value = rocr100_values[-1]

        if not np.isnan(rocr100_value):
            setattr(period, self.column_name, float(rocr100_value))

    def to_numpy(self) -> np.ndarray:
        """Export ROCR100 values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """Get the latest ROCR100 value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """Check if ROCR100 indicates bullish condition (ROCR100 > 100)."""
        latest = self.get_latest()
        return latest is not None and latest > 100.0

    def is_bearish(self) -> bool:
        """Check if ROCR100 indicates bearish condition (ROCR100 < 100)."""
        latest = self.get_latest()
        return latest is not None and latest < 100.0
