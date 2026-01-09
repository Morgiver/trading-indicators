"""ROCR (Rate of change ratio) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class ROCR(BaseIndicator):
    """
    ROCR (Rate of change ratio) indicator.

    Formula: (price/prevPrice)

    Measures the ratio of current price to previous price.

    Typical interpretation:
    - ROCR > 1: Price increasing
    - ROCR < 1: Price decreasing
    - ROCR = 1: No change

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> rocr = ROCR(frame=frame, length=10, column_name='ROCR_10')
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(rocr.periods[-1].ROCR_10)
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 10,
        column_name: str = 'ROCR',
        max_periods: Optional[int] = None
    ):
        """Initialize ROCR indicator."""
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate ROCR value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None or len(self.frame.periods) < self.length + 1:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        rocr_values = talib.ROCR(close_prices, timeperiod=self.length)
        rocr_value = rocr_values[-1]

        if not np.isnan(rocr_value):
            setattr(period, self.column_name, float(rocr_value))

    def to_numpy(self) -> np.ndarray:
        """Export ROCR values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """Get the latest ROCR value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """Check if ROCR indicates bullish condition (ROCR > 1)."""
        latest = self.get_latest()
        return latest is not None and latest > 1.0

    def is_bearish(self) -> bool:
        """Check if ROCR indicates bearish condition (ROCR < 1)."""
        latest = self.get_latest()
        return latest is not None and latest < 1.0
