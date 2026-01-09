"""PLUS_DM (Plus Directional Movement) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class PLUS_DM(BaseIndicator):
    """
    PLUS_DM (Plus Directional Movement) indicator.

    Measures upward directional movement between successive highs.

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> plus_dm = PLUS_DM(frame=frame, length=14)
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(plus_dm.periods[-1].PLUS_DM)
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'PLUS_DM',
        max_periods: Optional[int] = None
    ):
        """Initialize PLUS_DM indicator."""
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate PLUS_DM value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        required_periods = self.length + 1
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])

        plus_dm_values = talib.PLUS_DM(high_prices, low_prices, timeperiod=self.length)
        plus_dm_value = plus_dm_values[-1]

        if not np.isnan(plus_dm_value):
            setattr(period, self.column_name, float(plus_dm_value))

    def to_numpy(self) -> np.ndarray:
        """Export PLUS_DM values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """Get the latest PLUS_DM value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
