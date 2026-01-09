"""WILLR (Williams' %R) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class WILLR(BaseIndicator):
    """
    WILLR (Williams' %R) indicator.

    Williams %R is a momentum indicator that measures overbought
    and oversold levels. It ranges from -100 to 0.

    Typical interpretation:
    - WILLR > -20: Overbought
    - WILLR < -80: Oversold
    - WILLR crossing above -80: Bullish signal
    - WILLR crossing below -20: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> willr = WILLR(frame=frame, length=14, column_name='WILLR_14')
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(willr.periods[-1].WILLR_14)
        >>> print(willr.is_overbought())
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'WILLR',
        max_periods: Optional[int] = None
    ):
        """
        Initialize WILLR indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods for WILLR calculation (default: 14)
            column_name: Name for the indicator column (default: 'WILLR')
            max_periods: Maximum periods to keep
        """
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate WILLR value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None or len(self.frame.periods) < self.length:
            return

        high_prices = np.array([p.high_price for p in self.frame.periods[:period_index + 1]])
        low_prices = np.array([p.low_price for p in self.frame.periods[:period_index + 1]])
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        willr_values = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=self.length)
        willr_value = willr_values[-1]

        if not np.isnan(willr_value):
            setattr(period, self.column_name, float(willr_value))

    def to_numpy(self) -> np.ndarray:
        """Export WILLR values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """Export normalized WILLR values (-100 to 0 → 0-1)."""
        values = self.to_numpy()
        return (values + 100.0) / 100.0

    def get_latest(self) -> Optional[float]:
        """Get the latest WILLR value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_overbought(self, threshold: float = -20.0) -> bool:
        """
        Check if WILLR indicates overbought condition.

        Args:
            threshold: Overbought threshold (default: -20.0)

        Returns:
            True if WILLR is above threshold (closer to 0)
        """
        latest = self.get_latest()
        return latest is not None and latest > threshold

    def is_oversold(self, threshold: float = -80.0) -> bool:
        """
        Check if WILLR indicates oversold condition.

        Args:
            threshold: Oversold threshold (default: -80.0)

        Returns:
            True if WILLR is below threshold (closer to -100)
        """
        latest = self.get_latest()
        return latest is not None and latest < threshold
