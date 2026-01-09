"""MOM (Momentum) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class MOM(BaseIndicator):
    """
    MOM (Momentum) indicator.

    Momentum measures the rate of change in price by calculating
    the difference between current price and price n periods ago.

    Typical interpretation:
    - MOM > 0: Upward momentum
    - MOM < 0: Downward momentum
    - MOM crossing above 0: Bullish signal
    - MOM crossing below 0: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> mom = MOM(frame=frame, length=10, column_name='MOM_10')
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(mom.periods[-1].MOM_10)
        >>> print(mom.is_bullish())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 10,
        column_name: str = 'MOM',
        max_periods: Optional[int] = None
    ):
        """Initialize MOM indicator."""
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate MOM value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None or len(self.frame.periods) < self.length + 1:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        mom_values = talib.MOM(close_prices, timeperiod=self.length)
        mom_value = mom_values[-1]

        if not np.isnan(mom_value):
            setattr(period, self.column_name, float(mom_value))

    def to_numpy(self) -> np.ndarray:
        """Export MOM values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """Get the latest MOM value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """Check if MOM indicates bullish condition (MOM > 0)."""
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_bearish(self) -> bool:
        """Check if MOM indicates bearish condition (MOM < 0)."""
        latest = self.get_latest()
        return latest is not None and latest < 0
