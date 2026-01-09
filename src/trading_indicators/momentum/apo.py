"""APO (Absolute Price Oscillator) indicator."""

from typing import Optional
import numpy as np
import talib

from ..base import BaseIndicator, IndicatorPeriod


class APO(BaseIndicator):
    """
    APO (Absolute Price Oscillator) indicator.

    APO shows the difference between two exponential moving averages (EMAs).
    Unlike PPO which shows percentage, APO shows absolute price difference.

    Typical interpretation:
    - APO > 0: Fast EMA above slow EMA (bullish)
    - APO < 0: Fast EMA below slow EMA (bearish)
    - APO crossing above 0: Bullish signal
    - APO crossing below 0: Bearish signal

    Example:
        >>> from trading_frame import TimeFrame
        >>> frame = TimeFrame('5T', max_periods=100)
        >>> apo = APO(frame=frame, fast=12, slow=26, column_name='APO')
        >>>
        >>> # Feed candles - APO automatically updates
        >>> for candle in candles:
        ...     frame.feed(candle)
        >>>
        >>> # Access values
        >>> print(apo.periods[-1].APO)
        >>> print(apo.is_bullish())
    """

    def __init__(
        self,
        frame: 'Frame',
        fast: int = 12,
        slow: int = 26,
        ma_type: int = 0,
        column_name: str = 'APO',
        max_periods: Optional[int] = None
    ):
        """
        Initialize APO indicator.

        Args:
            frame: Frame to bind to
            fast: Fast period (default: 12)
            slow: Slow period (default: 26)
            ma_type: Moving average type (default: 0 = SMA). See TA-Lib MA types.
            column_name: Name for the indicator column (default: 'APO')
            max_periods: Maximum periods to keep (default: frame's max_periods)
        """
        self.fast = fast
        self.slow = slow
        self.ma_type = ma_type
        self.column_name = column_name
        super().__init__(frame, max_periods)

    def calculate(self, period: IndicatorPeriod):
        """
        Calculate APO value for a specific period.

        Args:
            period: IndicatorPeriod to populate with APO value
        """
        # Find the index of this period in the frame
        period_index = None
        for i, fp in enumerate(self.frame.periods):
            if fp.open_date == period.open_date:
                period_index = i
                break

        # Need at least 'slow' periods for APO calculation
        if period_index is None or len(self.frame.periods) < self.slow:
            return

        # Extract close prices
        close_prices = np.array([p.close_price for p in self.frame.periods[:period_index + 1]])

        # Calculate APO using TA-Lib
        apo_values = talib.APO(close_prices, fastperiod=self.fast, slowperiod=self.slow, matype=self.ma_type)

        # The last value is the APO for our period
        apo_value = apo_values[-1]

        if not np.isnan(apo_value):
            setattr(period, self.column_name, float(apo_value))

    def to_numpy(self) -> np.ndarray:
        """
        Export APO values as numpy array.

        Returns:
            NumPy array with APO values (NaN for periods without values)
        """
        return np.array([
            getattr(p, self.column_name) if hasattr(p, self.column_name) else np.nan
            for p in self.periods
        ])

    def get_latest(self) -> Optional[float]:
        """
        Get the latest APO value.

        Returns:
            Latest APO value or None if not available
        """
        if self.periods:
            return getattr(self.periods[-1], self.column_name, None)
        return None

    def is_bullish(self) -> bool:
        """
        Check if APO indicates bullish condition (APO > 0).

        Returns:
            True if APO is positive
        """
        latest = self.get_latest()
        return latest is not None and latest > 0

    def is_bearish(self) -> bool:
        """
        Check if APO indicates bearish condition (APO < 0).

        Returns:
            True if APO is negative
        """
        latest = self.get_latest()
        return latest is not None and latest < 0
