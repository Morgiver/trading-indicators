"""DX (Directional Movement Index) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class DX(BaseIndicator):
    """
    DX (Directional Movement Index) indicator.

    DX is the raw directional movement index used to calculate ADX.
    It measures the strength of directional movement.

    Values range from 0 to 100:
    - DX > 25: Strong directional movement
    - DX < 25: Weak directional movement

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> dx = DX(frame=frame, length=14, column_name='DX_14')
        >>>
        >>> # Feed candles
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(dx.periods[-1].DX_14)
    """

    def __init__(
        self,
        frame: 'Frame',
        length: int = 14,
        column_name: str = 'DX',
        max_periods: Optional[int] = None
    ):
        """
        Initialize DX indicator.

        Args:
            frame: Frame to bind to
            length: Number of periods (default: 14)
            column_name: Name for the indicator column (default: 'DX')
            max_periods: Maximum periods to keep
        """
        self.length = length
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate DX value for a specific period."""
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

        dx_values = talib.DX(high_prices, low_prices, close_prices, timeperiod=self.length)
        dx_value = dx_values[-1]

        if not np.isnan(dx_value):
            setattr(period, self.column_name, float(dx_value))

    def to_numpy(self) -> np.ndarray:
        """Export DX values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def to_normalize(self) -> np.ndarray:
        """Export normalized DX values (0-100 → 0-1)."""
        return self.to_numpy() / 100.0

    def get_latest(self) -> Optional[float]:
        """Get the latest DX value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_strong_movement(self, threshold: float = 25.0) -> bool:
        """Check if DX indicates strong directional movement."""
        latest = self.get_latest()
        return latest is not None and latest > threshold
