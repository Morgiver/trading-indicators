"""TRIX (1-day Rate-Of-Change of a Triple Smooth EMA) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class TRIX(BaseIndicator):
    """
    TRIX (1-day Rate-Of-Change of a Triple Smooth EMA) indicator.

    TRIX is a momentum oscillator that displays the rate of change
    of a triple exponentially smoothed moving average.

    Typical interpretation:
    - TRIX > 0: Bullish momentum
    - TRIX < 0: Bearish momentum
    - TRIX crossing above 0: Bullish signal
    - TRIX crossing below 0: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> trix = TRIX(frame=frame, length=15, column_name='TRIX_15')
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(trix.periods[-1].TRIX_15)
        >>> print(trix.is_bullish())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 30,
        column_name: str = 'TRIX',
        max_periods: Optional[int] = None
    ):
        """Initialize TRIX indicator."""
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate TRIX value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # TRIX requires a lot of periods due to triple smoothing
        required_periods = self.length * 3
        if period_index is None or len(self.frame.periods) < required_periods:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        trix_values = talib.TRIX(close_prices, timeperiod=self.length)
        trix_value = trix_values[-1]

        if not np.isnan(trix_value):
            setattr(period, self.column_name, float(trix_value))

    def to_numpy(self) -> np.ndarray:
        """Export TRIX values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """Get the latest TRIX value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """Check if TRIX indicates bullish condition (TRIX > 0)."""
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_bearish(self) -> bool:
        """Check if TRIX indicates bearish condition (TRIX < 0)."""
        latest = self.get_latest()
        return latest is not None and latest < 0
