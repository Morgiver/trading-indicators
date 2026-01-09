"""PPO (Percentage Price Oscillator) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class PPO(BaseIndicator):
    """
    PPO (Percentage Price Oscillator) indicator.

    PPO shows the percentage difference between two EMAs.
    Unlike APO which shows absolute difference, PPO shows percentage.

    Typical interpretation:
    - PPO > 0: Fast EMA above slow EMA (bullish)
    - PPO < 0: Fast EMA below slow EMA (bearish)
    - PPO crossing above 0: Bullish signal
    - PPO crossing below 0: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> ppo = PPO(frame=frame, fast=12, slow=26, column_name='PPO')
        >>>
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> print(ppo.periods[-1].PPO)
        >>> print(ppo.is_bullish())
    """

    def __init__(
        self,
        frame: 'Frame',
        fast: int = 12,
        slow: int = 26,
        ma_type: int = 0,
        column_name: str = 'PPO',
        max_periods: Optional[int] = None
    ):
        """
        Initialize PPO indicator.

        Args:
            frame: Frame to bind to
            fast: Fast period (default: 12)
            slow: Slow period (default: 26)
            ma_type: Moving average type (default: 0 = SMA)
            column_name: Name for the indicator column (default: 'PPO')
            max_periods: Maximum periods to keep
        """
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """Calculate PPO value for a specific period."""
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        if period_index is None or len(self.frame.periods) < self.slow:
            return

        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        ppo_values = talib.PPO(close_prices, fastperiod=self.fast, slowperiod=self.slow, matype=self.ma_type)
        ppo_value = ppo_values[-1]

        if not np.isnan(ppo_value):
            setattr(period, self.column_name, float(ppo_value))

    def to_numpy(self) -> np.ndarray:
        """Export PPO values as numpy array."""
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """Get the latest PPO value."""
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """Check if PPO indicates bullish condition (PPO > 0)."""
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_bearish(self) -> bool:
        """Check if PPO indicates bearish condition (PPO < 0)."""
        latest = self.get_latest()
        return latest is not None and latest < 0
