"""PLUS_DI (Plus Directional Indicator) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class PLUS_DI(BaseIndicator):
    """
    PLUS_DI (Plus Directional Indicator) indicator.

    Part of the Directional Movement System. Measures upward price movement.

    Typical interpretation:
    - PLUS_DI > MINUS_DI: Uptrend dominant
    - PLUS_DI crossing above MINUS_DI: Bullish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> plus_di = PLUS_DI(frame=frame, length=14)
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(plus_di.periods[-1].PLUS_DI)
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'PLUS_DI',
        max_periods: Optional[int] = None
    ):
        """Initialize PLUS_DI indicator."""
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate PLUS_DI value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        required_periods = self.length * 2
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        plus_di_values = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=self.length)
        plus_di_value = plus_di_values[-1]

        if not np.isnan(plus_di_value):
            setattr(period, self.column_name, float(plus_di_value))

    def to_numpy(self) -> np.ndarray:
        """Export PLUS_DI values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """Export normalized PLUS_DI values (0-100 → 0-1)."""
        return self.to_numpy() / 100.0

    def get_latest(self) -> Optional[float]:
        """Get the latest PLUS_DI value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None
