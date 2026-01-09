"""ROC (Rate of change) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class ROC(BaseIndicator):
    """
    ROC (Rate of change) indicator.

    Formula: ((price/prevPrice)-1)*100

    Measures the percentage change in price over a specified period.

    Typical interpretation:
    - ROC > 0: Price increasing
    - ROC < 0: Price decreasing
    - ROC crossing above 0: Bullish signal
    - ROC crossing below 0: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> roc = ROC(frame=frame, length=10, column_name='ROC_10')
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(roc.periods[-1].ROC_10)
        >>> print(roc.is_bullish())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 10,
        column_name: str = 'ROC',
        max_periods: Optional[int] = None
    ):
        """Initialize ROC indicator."""
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate ROC value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None or len(self.frame.periods) < self.length + 1:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        roc_values = talib.ROC(close_prices, timeperiod=self.length)
        roc_value = roc_values[-1]

        if not np.isnan(roc_value):
            setattr(period, self.column_name, float(roc_value))

    def to_numpy(self) -> np.ndarray:
        """Export ROC values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """Get the latest ROC value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """Check if ROC indicates bullish condition (ROC > 0)."""
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_bearish(self) -> bool:
        """Check if ROC indicates bearish condition (ROC < 0)."""
        latest = self.get_latest()
        return latest is not None and latest < 0
